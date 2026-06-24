"""
Run from the project root:
    python models/train.py

Trains SVD on ml-latest-small ratings and saves models/svd_model.pkl.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.recommender import SVDRecommender

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).parent / "svd_model.pkl"


def main():
    print("Loading ratings...")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    print(f"  {len(ratings):,} ratings | {ratings['userId'].nunique()} users | {ratings['movieId'].nunique()} movies")

    print("Training SVD (n_components=50)...")
    model = SVDRecommender(n_components=50)
    model.fit(ratings)

    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
