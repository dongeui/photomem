import os
import sqlite3
import time
from pathlib import Path

import sqlite_vec

DEFAULT_DB_PATH = "/app/cache/photomem.db"


def _db_path() -> str:
    return os.environ.get("PHOTOMEM_DB", DEFAULT_DB_PATH)


def _load_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    _load_vec(conn)
    return conn


def init_db() -> None:
    Path(_db_path()).parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    conn.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA wal_autocheckpoint=500;

        CREATE TABLE IF NOT EXISTS photos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path   TEXT UNIQUE NOT NULL,
            file_hash   TEXT,
            file_size   INTEGER,
            modified_at INTEGER,
            created_at  INTEGER,
            latitude    REAL,
            longitude   REAL,
            city        TEXT,
            country     TEXT,
            indexed_at  INTEGER,
            status      TEXT NOT NULL DEFAULT 'pending',
            error_msg   TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_photos
            USING vec0(embedding float[512]);

        CREATE INDEX IF NOT EXISTS idx_photos_status     ON photos(status);
        CREATE INDEX IF NOT EXISTS idx_photos_created_at ON photos(created_at);
        CREATE INDEX IF NOT EXISTS idx_photos_city       ON photos(city);
    """)
    _ensure_column(conn, "photos", "file_size", "INTEGER")
    _ensure_column(conn, "photos", "modified_at", "INTEGER")
    conn.commit()
    conn.close()


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    existing = {row["name"] for row in cur.fetchall()}
    if column_name not in existing:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def requeue_pending(conn: sqlite3.Connection) -> int:
    """Return count of pending rows after container restart — they are already pending."""
    cur = conn.execute("SELECT COUNT(*) FROM photos WHERE status='pending'")
    return cur.fetchone()[0]


def get_pending_paths(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute("SELECT file_path FROM photos WHERE status='pending'")
    return [row[0] for row in cur.fetchall()]


def get_cached_file_stats(conn: sqlite3.Connection) -> dict[str, tuple[int | None, int | None, str]]:
    cur = conn.execute("SELECT file_path, file_size, modified_at, status FROM photos")
    return {
        row["file_path"]: (row["file_size"], row["modified_at"], row["status"])
        for row in cur.fetchall()
    }


def upsert_photo(
    conn: sqlite3.Connection,
    file_path: str,
    file_hash: str,
    file_size: int | None = None,
    modified_at: int | None = None,
) -> int | None:
    """Insert or re-queue photo if hash changed. Returns row id, or None if already indexed."""
    cur = conn.execute(
        "SELECT id, file_hash, file_size, modified_at, status FROM photos WHERE file_path=?",
        (file_path,),
    )
    row = cur.fetchone()
    if row:
        if row["file_hash"] == file_hash and row["status"] == "indexed":
            if row["file_size"] != file_size or row["modified_at"] != modified_at:
                conn.execute(
                    "UPDATE photos SET file_size=?, modified_at=? WHERE id=?",
                    (file_size, modified_at, row["id"]),
                )
                conn.commit()
            return None  # already up to date
        conn.execute(
            """UPDATE photos
               SET file_hash=?, file_size=?, modified_at=?, status='pending', error_msg=NULL
               WHERE id=?""",
            (file_hash, file_size, modified_at, row["id"]),
        )
        conn.commit()
        return row["id"]
    cur = conn.execute(
        """INSERT INTO photos (file_path, file_hash, file_size, modified_at, status)
           VALUES (?, ?, ?, ?, 'pending')""",
        (file_path, file_hash, file_size, modified_at),
    )
    conn.commit()
    return cur.lastrowid


def update_photo_indexed(
    conn: sqlite3.Connection,
    photo_id: int,
    created_at: int | None,
    lat: float | None,
    lon: float | None,
    city: str | None,
    country: str | None,
    embedding: bytes,
) -> None:
    conn.execute(
        """UPDATE photos SET
            created_at=?, latitude=?, longitude=?, city=?, country=?,
            indexed_at=?, status='indexed', error_msg=NULL
           WHERE id=?""",
        (created_at, lat, lon, city, country, int(time.time()), photo_id),
    )
    conn.execute("DELETE FROM vec_photos WHERE rowid=?", (photo_id,))
    conn.execute("INSERT INTO vec_photos(rowid, embedding) VALUES (?, ?)", (photo_id, embedding))
    conn.commit()


def mark_photo_error(conn: sqlite3.Connection, photo_id: int, error: str) -> None:
    conn.execute(
        "UPDATE photos SET status='error', error_msg=? WHERE id=?",
        (error[:500], photo_id),
    )
    conn.commit()


def get_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='indexed' THEN 1 ELSE 0 END) as indexed,
            SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status='error'   THEN 1 ELSE 0 END) as errors
        FROM photos
    """)
    row = cur.fetchone()
    return {
        "total":   row["total"]   or 0,
        "indexed": row["indexed"] or 0,
        "pending": row["pending"] or 0,
        "errors":  row["errors"]  or 0,
    }


def list_photos(conn: sqlite3.Connection, limit: int = 120, offset: int = 0) -> list[dict]:
    cur = conn.execute(
        """SELECT id, file_path, created_at, city, country, status, error_msg
           FROM photos
           WHERE status='indexed'
           ORDER BY COALESCE(created_at, indexed_at, 0) DESC, id DESC
           LIMIT ? OFFSET ?""",
        (limit, offset),
    )
    return [dict(row) for row in cur.fetchall()]


def search_by_embedding(
    conn: sqlite3.Connection,
    query_embedding: bytes,
    limit: int = 20,
    city_filter: str | None = None,
    date_from: int | None = None,
    date_to: int | None = None,
) -> list[dict]:
    # KNN search in vec_photos, then join with photos for metadata
    knn_limit = limit * 3 if (city_filter or date_from or date_to) else limit
    cur = conn.execute(
        """SELECT v.rowid, v.distance
           FROM vec_photos v
           WHERE v.embedding MATCH ?
           ORDER BY v.distance
           LIMIT ?""",
        (query_embedding, knn_limit),
    )
    candidate_ids = [(row["rowid"], row["distance"]) for row in cur.fetchall()]
    if not candidate_ids:
        return []

    placeholders = ",".join("?" * len(candidate_ids))
    id_list = [r[0] for r in candidate_ids]
    dist_map = {r[0]: r[1] for r in candidate_ids}

    where_clauses = [f"id IN ({placeholders})"]
    params: list = id_list[:]

    if city_filter:
        where_clauses.append("city = ?")
        params.append(city_filter)
    if date_from:
        where_clauses.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        where_clauses.append("created_at <= ?")
        params.append(date_to)

    where_sql = " AND ".join(where_clauses)
    cur = conn.execute(
        f"SELECT id, file_path, created_at, city, country FROM photos WHERE {where_sql}",
        params,
    )
    rows = cur.fetchall()
    results = [
        {
            "id":         row["id"],
            "file_path":  row["file_path"],
            "created_at": row["created_at"],
            "city":       row["city"],
            "country":    row["country"],
            "distance":   dist_map[row["id"]],
        }
        for row in rows
    ]
    results.sort(key=lambda r: r["distance"])
    return results[:limit]
