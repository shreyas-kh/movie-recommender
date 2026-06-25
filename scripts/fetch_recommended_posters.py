"""
Targeted poster fetch — only the movies that can actually appear in the UI.

Fetching all ~9.7k movies takes ~40 min. But the app only ever shows:
  * movies returned by recommend() for some user, and
  * movies in a user's "Because you rated these highly" section.

This script collects exactly that set across all users, skips anything already
in data/poster_urls.csv, and fetches only the remainder — a few hundred movies,
under a minute. Appends to the same CSV the app reads.

Usage:
    TMDB_API_KEY=your_key python scripts/fetch_recommended_posters.py
    # or with the key in .streamlit/secrets.toml:
    python scripts/fetch_recommended_posters.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))          # models.recommender (also needed to unpickle)
sys.path.insert(0, str(ROOT / "app"))  # utils
sys.path.insert(0, str(Path(__file__).parent))  # fetch_posters

from models.recommender import SVDRecommender
from utils import get_user_taste_profile
from fetch_posters import (
    CHECKPOINT_EVERY,
    MAX_WORKERS,
    OUT_CSV,
    _api_key,
    _fetch_poster,
    _save,
)

DATA_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "svd_model.pkl"
N_RECS = 20  # the most the UI's slider can request


def _load_model() -> SVDRecommender:
    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH.name}")
        return SVDRecommender.load(MODEL_PATH)
    print("No pickle found — training model...")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    return SVDRecommender(n_components=50).fit(ratings)


def main() -> None:
    api_key = _api_key()
    if not api_key:
        print("Error: TMDB_API_KEY not found (env var or .streamlit/secrets.toml).")
        sys.exit(1)

    model = _load_model()
    ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")
    movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    links = pd.read_csv(DATA_DIR / "links.csv").dropna(subset=["tmdbId"])
    links["movieId"] = links["movieId"].astype(int)
    links["tmdbId"] = links["tmdbId"].astype(int)
    link_map = dict(zip(links["movieId"], links["tmdbId"]))

    # Collect every movieId that could surface in the UI, for every user.
    print(f"Scanning {len(model.user_ids)} users (recommendations + top-rated)...")
    needed: set = set()
    for uid in model.user_ids:
        needed.update(int(mid) for mid, _ in model.recommend(uid, n=N_RECS))
        top_rated, _ = get_user_taste_profile(uid, ratings_df, movies_df)
        # The empty-fallback frame has no movieId column; skip those users.
        if not top_rated.empty and "movieId" in top_rated.columns:
            needed.update(int(mid) for mid in top_rated["movieId"])
    print(f"  {len(needed)} unique movies can appear in the UI.")

    # Skip what we already have.
    existing_rows: list = []
    existing_ids: set = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        existing_rows = existing.to_dict("records")
        existing_ids = set(existing["movieId"].astype(int))

    to_fetch = [
        (mid, link_map[mid])
        for mid in sorted(needed)
        if mid in link_map and mid not in existing_ids
    ]

    if not to_fetch:
        print("Nothing to fetch — every UI-visible movie is already in poster_urls.csv.")
        return

    print(f"Fetching {len(to_fetch)} new posters across {MAX_WORKERS} workers...\n")

    new_rows: list = []
    failed = 0
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_poster, tmdb_id, api_key): (movie_id, tmdb_id)
            for movie_id, tmdb_id in to_fetch
        }
        for future in as_completed(futures):
            movie_id, tmdb_id = futures[future]
            poster_url = future.result()
            done += 1

            if poster_url is None:
                failed += 1  # leave out so a re-run retries it
            else:
                new_rows.append({
                    "movieId": movie_id,
                    "tmdbId": tmdb_id,
                    "poster_url": poster_url,
                })

            if done % CHECKPOINT_EVERY == 0:
                _save(existing_rows + new_rows)
                print(f"  {done}/{len(to_fetch)} — checkpoint saved")

    _save(existing_rows + new_rows)

    found = sum(1 for r in new_rows if r["poster_url"])
    print(f"\nDone. Added {len(new_rows)} rows ({found} with a poster).")
    if failed:
        print(f"{failed} request(s) failed after retries — re-run to fill them in.")
    print(f"poster_urls.csv now has {len(existing_rows) + len(new_rows)} entries total.")
    print("Commit data/poster_urls.csv to the repo.")


if __name__ == "__main__":
    main()
