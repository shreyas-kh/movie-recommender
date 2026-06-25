"""
Run from the project root:
    python models/train.py

Evaluates SVD with an 80/20 train/test split (prints RMSE), then trains the
final model on the full ml-latest-small dataset and saves models/svd_model.pkl.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.recommender import SVDRecommender

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).parent / "svd_model.pkl"
N_COMPONENTS = 50


def rmse(model: SVDRecommender, eval_df: pd.DataFrame):
    """RMSE of the model's predicted ratings against actual ratings in eval_df.

    Only (user, movie) pairs the model saw at fit time are scored; pairs whose
    movie never appeared in training (cold-start) are skipped and reported.
    """
    movie_to_col = {mid: j for j, mid in enumerate(model.movie_ids)}
    mask = eval_df["userId"].isin(model.user_index) & eval_df["movieId"].isin(movie_to_col)
    sub = eval_df[mask]

    rows = sub["userId"].map(model.user_index).to_numpy()
    cols = sub["movieId"].map(movie_to_col).to_numpy()
    preds = model.predicted[rows, cols]
    actual = sub["rating"].to_numpy()

    score = float(np.sqrt(np.mean((preds - actual) ** 2)))
    skipped = len(eval_df) - len(sub)
    return score, len(sub), skipped


def main():
    print("Loading ratings...")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    print(f"  {len(ratings):,} ratings | {ratings['userId'].nunique()} users | {ratings['movieId'].nunique()} movies")

    # --- Evaluation: fit on 80%, score held-out 20% --------------------------
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    print(f"\nEvaluating on an 80/20 split (train={len(train_df):,}, test={len(test_df):,})...")
    eval_model = SVDRecommender(n_components=N_COMPONENTS).fit(train_df)

    train_rmse, n_train, _ = rmse(eval_model, train_df)
    test_rmse, n_test, skipped = rmse(eval_model, test_df)
    print(f"  Train RMSE: {train_rmse:.4f}  (n={n_train:,})")
    print(f"  Test  RMSE: {test_rmse:.4f}  (n={n_test:,}", end="")
    print(f", {skipped:,} cold-start test pairs skipped)" if skipped else ")")

    # --- Final model: fit on ALL data and save -------------------------------
    print(f"\nTraining final SVD on full dataset (n_components={N_COMPONENTS})...")
    model = SVDRecommender(n_components=N_COMPONENTS).fit(ratings)

    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
