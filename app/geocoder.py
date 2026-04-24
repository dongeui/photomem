"""Offline GPS → city/country using reverse_geocoder (~200MB, no network after install)."""
from __future__ import annotations

_rg = None


def _get_rg():
    global _rg
    if _rg is None:
        import reverse_geocoder as rg  # lazy-load (slow first import)
        _rg = rg
    return _rg


def reverse_geocode(lat: float, lon: float) -> tuple[str | None, str | None]:
    """Return (city, country_code) or (None, None) on failure."""
    try:
        rg = _get_rg()
        results = rg.search([(lat, lon)], verbose=False)
        if results:
            r = results[0]
            return r.get("name"), r.get("cc")
    except Exception:
        pass
    return None, None
