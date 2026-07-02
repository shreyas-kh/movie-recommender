"""
Train/test splitting strategies for the ratings data.

Two methods, selectable everywhere via --split:

* "temporal" (default) — per-user chronological split: each user's earliest
  (1 - test_size) fraction of ratings goes to train, their most recent
  test_size fraction to test. This mirrors deployment reality (you only ever
  know a user's past) and avoids the leakage of random splits, where a user's
  later ratings can inform "predictions" of their earlier ones.

* "random" — plain shuffled split (the original protocol). Kept for
  comparison: the gap between random and temporal numbers is itself a useful
  measurement of how much the random split flatters the model.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def temporal_split_per_user(
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split on the timestamp column.

    For each user, ratings are sorted by timestamp (mergesort: stable, so
    equal timestamps keep their original file order) and the most recent
    ~test_size fraction becomes test. Guards: every user keeps at least one
    train rating; users with a single rating go entirely to train.
    """
    df = ratings_df.sort_values(["userId", "timestamp"], kind="mergesort")

    pos = df.groupby("userId").cumcount()                      # 0-based rank in time
    n = df.groupby("userId")["userId"].transform("size")       # user's rating count
    n_test = np.maximum(1, np.round(n * test_size)).astype(int)
    n_test = np.minimum(n_test, n - 1)                         # >= 1 train row/user

    is_test = pos >= (n - n_test)
    return df[~is_test].copy(), df[is_test].copy()


def random_split(
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The original shuffled split (time-agnostic; leaks future ratings)."""
    return train_test_split(ratings_df, test_size=test_size, random_state=seed)


def split_ratings(
    ratings_df: pd.DataFrame,
    method: str = "temporal",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch on method: 'temporal' or 'random'. seed only affects 'random'."""
    if method == "temporal":
        return temporal_split_per_user(ratings_df, test_size=test_size)
    if method == "random":
        return random_split(ratings_df, test_size=test_size, seed=seed)
    raise ValueError(f"Unknown split method: {method!r} (use 'temporal' or 'random')")
