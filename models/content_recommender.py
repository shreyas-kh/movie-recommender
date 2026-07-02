"""
Content-based recommender built on movie genres.

Complements the collaborative-filtering SVDRecommender: because it scores movies
purely from item features (genres), it needs no rating history for the *user* —
a single liked movie is enough to make recommendations. That makes it the
cold-start half of the HybridRecommender.

Similarity is cosine similarity over multi-hot genre vectors. We never
materialise the full movie x movie similarity matrix (~9.7k^2 would be ~750 MB);
instead we L2-normalise each movie's genre vector once at fit time, so the cosine
similarity of any movie (or taste profile) against all movies is a single matvec
computed on demand.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_NO_GENRES = "(no genres listed)"


class ContentRecommender:
    def __init__(self):
        self.movie_ids: List[int] = []
        self.movie_index: Dict[int, int] = {}
        self.genres: List[str] = []
        # L2-normalised multi-hot genre matrix: (n_movies, n_genres). Rows for
        # movies with no listed genres are all-zero (norm 0), which correctly
        # yields a cosine similarity of 0 against everything.
        self.features: Optional[np.ndarray] = None

    def fit(self, movies_df: pd.DataFrame) -> "ContentRecommender":
        """Build the normalised genre feature matrix from movies.csv.

        Needs only movieId + genres; no ratings, so this is independent of both
        SVDRecommender and any particular user.
        """
        movies_df = movies_df.drop_duplicates(subset="movieId")
        self.movie_ids = [int(m) for m in movies_df["movieId"].tolist()]
        self.movie_index = {mid: i for i, mid in enumerate(self.movie_ids)}

        # Discover the genre vocabulary (excluding the "no genres" sentinel).
        genre_lists: List[List[str]] = []
        vocab: Dict[str, int] = {}
        for raw in movies_df["genres"].fillna(""):
            gs = [g for g in raw.split("|") if g and g != _NO_GENRES]
            genre_lists.append(gs)
            for g in gs:
                if g not in vocab:
                    vocab[g] = len(vocab)
        self.genres = list(vocab.keys())

        # Multi-hot encode.
        mat = np.zeros((len(self.movie_ids), len(self.genres)), dtype=np.float64)
        for row, gs in enumerate(genre_lists):
            for g in gs:
                mat[row, vocab[g]] = 1.0

        # L2-normalise rows so a dot product == cosine similarity. Guard the
        # zero-norm rows (movies with no genres) to avoid divide-by-zero.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.features = mat / norms
        return self

    # -- internals -----------------------------------------------------------

    def _taste_vector(
        self,
        liked_movie_ids: Sequence[int],
        weights: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        """Weighted mean of the liked movies' normalised genre vectors, itself
        re-normalised to unit length. Returns None if none of the liked movies
        are known or they carry no genre signal."""
        if self.features is None:
            raise RuntimeError("ContentRecommender must be fit before use.")

        rows: List[int] = []
        ws: List[float] = []
        for i, mid in enumerate(liked_movie_ids):
            idx = self.movie_index.get(int(mid))
            if idx is None:
                continue
            rows.append(idx)
            ws.append(float(weights[i]) if weights is not None else 1.0)

        if not rows:
            return None

        w = np.asarray(ws, dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones_like(w)
        profile = (self.features[rows] * w[:, np.newaxis]).sum(axis=0)

        norm = np.linalg.norm(profile)
        if norm == 0.0:  # liked movies had no genres
            return None
        return profile / norm

    # -- public API ----------------------------------------------------------

    def score_movies(
        self,
        liked_movie_ids: Sequence[int],
        weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Cosine similarity of every movie against the liked-movie taste vector.

        Returns an array aligned to self.movie_ids, in [0, 1] (genre vectors are
        non-negative, so cosine is non-negative). All zeros if there's no usable
        signal from the liked movies.
        """
        profile = self._taste_vector(liked_movie_ids, weights)
        if profile is None:
            return np.zeros(len(self.movie_ids), dtype=np.float64)
        scores = self.features.dot(profile)
        # Numerical hygiene: clip tiny negatives from float error into [0, 1].
        return np.clip(scores, 0.0, 1.0)

    def recommend(
        self,
        liked_movie_ids: Sequence[int],
        n: int = 10,
        weights: Optional[Sequence[float]] = None,
        exclude: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Top-n most similar movies to the liked set.

        The liked movies themselves are always excluded; pass `exclude` to also
        drop already-seen movies (e.g. a warm user's full rating history).
        """
        scores = self.score_movies(liked_movie_ids, weights)

        blocked = {int(m) for m in liked_movie_ids}
        if exclude is not None:
            blocked.update(int(m) for m in exclude)
        for mid in blocked:
            idx = self.movie_index.get(mid)
            if idx is not None:
                scores[idx] = -np.inf

        top = np.argsort(scores)[::-1][:n]
        return [
            (self.movie_ids[i], float(scores[i]))
            for i in top
            if np.isfinite(scores[i])
        ]

    def similar_movies(self, movie_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """'More like this' — the n movies most similar to a single movie."""
        return self.recommend([movie_id], n=n)

    # -- persistence (optional; content model is cheap to rebuild) ------------

    def save(self, path: Union[str, Path]) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ContentRecommender":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
