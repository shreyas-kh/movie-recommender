"""
Pre-fetch poster URLs from the TMDB API and save them to data/poster_urls.csv.

Run this ONCE locally before deploying. The CSV is committed to the repo so the
deployed app never needs a TMDB API key at runtime — it just reads the file.

Usage:
    TMDB_API_KEY=your_key python scripts/fetch_posters.py

    # or if .streamlit/secrets.toml already has your key:
    python scripts/fetch_posters.py

Speed: fetches run concurrently across MAX_WORKERS threads (TMDB's limit is
~50 req/s, so 8 workers is comfortable). 429 responses are retried with
backoff. Movies that fail after all retries are left OUT of the CSV so a
re-run picks them up — only definitive results (a URL, or "no poster") are
written. Safe to interrupt and re-run.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
LINKS_CSV = ROOT / "data" / "raw" / "links.csv"
OUT_CSV = ROOT / "data" / "poster_urls.csv"

TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w300"
MAX_WORKERS = 8       # concurrent requests; TMDB tolerates ~50 req/s
MAX_RETRIES = 4       # attempts before giving up on a throttled/failed request
CHECKPOINT_EVERY = 200


def _api_key() -> str:
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if key:
        return key
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            line = line.strip()
            if line.startswith("TMDB_API_KEY"):
                _, _, val = line.partition("=")
                return val.strip().strip('"').strip("'")
    return ""


def _fetch_poster(tmdb_id: int, api_key: str) -> Optional[str]:
    """Return the poster URL, "" if the movie has no poster (definitive), or
    None if the request couldn't be completed after retries (caller should
    leave it out of the CSV so a re-run retries it)."""
    backoff = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                f"{TMDB_BASE}/movie/{tmdb_id}",
                params={"api_key": api_key},
                timeout=10,
            )
            if resp.status_code == 429:
                # Respect Retry-After when present, else exponential backoff.
                wait = float(resp.headers.get("Retry-After", backoff))
                time.sleep(wait)
                backoff *= 2
                continue
            if resp.status_code == 404:
                return ""  # stale TMDB ID — definitively no poster
            resp.raise_for_status()
            path = resp.json().get("poster_path")
            return f"{POSTER_BASE}{path}" if path else ""
        except Exception:
            time.sleep(backoff)
            backoff *= 2
    return None  # exhausted retries — leave out so re-run picks it up


def main() -> None:
    api_key = _api_key()
    if not api_key:
        print("Error: TMDB_API_KEY not found.")
        print("  Set it as an environment variable:  TMDB_API_KEY=xxx python scripts/fetch_posters.py")
        print("  Or add it to .streamlit/secrets.toml:  TMDB_API_KEY = \"xxx\"")
        sys.exit(1)

    if not LINKS_CSV.exists():
        print(f"Error: {LINKS_CSV} not found. Download ml-latest-small and place the CSVs in data/raw/")
        sys.exit(1)

    links = pd.read_csv(LINKS_CSV)
    links = links.dropna(subset=["tmdbId"]).copy()
    links["movieId"] = links["movieId"].astype(int)
    links["tmdbId"] = links["tmdbId"].astype(int)

    # Load existing results for resume support
    existing_rows: list = []
    fetched_ids: set = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        existing_rows = existing.to_dict("records")
        fetched_ids = set(existing["movieId"].astype(int))
        print(f"Resuming: {len(fetched_ids)} entries already in {OUT_CSV.name}")

    to_fetch = links[~links["movieId"].isin(fetched_ids)]
    total = len(to_fetch)

    if total == 0:
        print("Nothing to fetch — all movies already in the output file.")
        return

    print(f"Fetching posters for {total} movies across {MAX_WORKERS} workers...")

    new_rows: list = []
    failed = 0
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_poster, int(row["tmdbId"]), api_key): int(row["movieId"])
            for _, row in to_fetch.iterrows()
        }
        tmdb_by_movie = dict(zip(to_fetch["movieId"], to_fetch["tmdbId"]))

        # as_completed yields in the main thread, so collecting here is
        # single-threaded and needs no extra locking.
        for future in as_completed(futures):
            movie_id = futures[future]
            poster_url = future.result()
            done += 1

            if poster_url is None:
                failed += 1  # leave out of CSV; a re-run will retry it
            else:
                new_rows.append({
                    "movieId": movie_id,
                    "tmdbId": int(tmdb_by_movie[movie_id]),
                    "poster_url": poster_url,
                })

            if done % CHECKPOINT_EVERY == 0:
                _save(existing_rows + new_rows)
                found = sum(1 for r in new_rows if r["poster_url"])
                print(f"  {done}/{total} — checkpoint saved  ({found} with poster, {failed} to retry)")

    _save(existing_rows + new_rows)

    all_rows = existing_rows + new_rows
    found = sum(1 for r in all_rows if r.get("poster_url"))
    print(f"\nDone. {found}/{len(all_rows)} movies have a poster URL.")
    if failed:
        print(f"{failed} request(s) failed after retries — re-run the script to fill them in.")
    print(f"Saved to {OUT_CSV}")
    print("\nCommit data/poster_urls.csv to the repo — the app will use it automatically.")


def _save(rows: list) -> None:
    df = pd.DataFrame(rows, columns=["movieId", "tmdbId", "poster_url"])
    df = df.sort_values("movieId").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
