"""
Test fixtures.

Required test photos (add to tests/fixtures/ before running):
  hongkong.jpg  — JPEG with GPS EXIF pointing to Hong Kong
  corrupt.jpg   — not a valid image (e.g. `echo "bad" > corrupt.jpg`)
  no_exif.jpg   — valid JPEG with no EXIF data

Models must be available at PHOTOMEM_MODELS (downloaded on first real run).
Set PHOTOMEM_MODELS to a writable dir with downloaded models for CI.
"""
import os
import tempfile
import pytest

from app import db, models


@pytest.fixture(scope="session", autouse=True)
def tmp_env(tmp_path_factory):
    """Point all storage to a temp dir for tests."""
    tmp = tmp_path_factory.mktemp("photomem_test")
    os.environ["PHOTOMEM_DB"] = str(tmp / "test.db")
    os.environ["PHOTOMEM_CACHE"] = str(tmp)
    os.environ["PHOTOMEM_PHOTOS"] = str(tmp / "photos")
    (tmp / "photos").mkdir()
    yield tmp


@pytest.fixture(scope="session")
def initialized_db(tmp_env):
    db.init_db()
    return db.get_connection()
