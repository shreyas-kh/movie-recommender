from typing import Optional, Union

import requests
import streamlit as st

TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w300"


def _api_key() -> str:
    return st.secrets.get("TMDB_API_KEY", "")


def fetch_poster_url(tmdb_id: Union[int, float]) -> Optional[str]:
    key = _api_key()
    if not key or not tmdb_id or (isinstance(tmdb_id, float) and tmdb_id != tmdb_id):
        return None
    try:
        resp = requests.get(
            f"{TMDB_BASE}/movie/{int(tmdb_id)}",
            params={"api_key": key},
            timeout=5,
        )
        resp.raise_for_status()
        path = resp.json().get("poster_path")
        return f"{POSTER_BASE}{path}" if path else None
    except Exception:
        return None
