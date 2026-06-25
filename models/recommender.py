import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        # algorithm='arpack' avoids the randomized path's final DGEMM call
        # (U = Q @ Uhat) which triggers spurious BLAS RuntimeWarnings on macOS
        # even when all inputs and outputs are numerically clean.
        self.svd = TruncatedSVD(n_components=n_components, algorithm="arpack")
        self.user_ids: List = []
        self.movie_ids: List = []
        self.user_index: Dict = {}
        self.user_mean: Optional[np.ndarray] = None
        self.predicted: Optional[np.ndarray] = None
        self.rated_mask: Optional[np.ndarray] = None

    def fit(self, ratings_df: pd.DataFrame) -> "SVDRecommender":
        matrix = ratings_df.pivot_table(index="userId", columns="movieId", values="rating")

        self.user_ids = list(matrix.index)
        self.movie_ids = list(matrix.columns)
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}

        user_means = matrix.mean(axis=1)  # skipna=True: valid for every user
        self.user_mean = user_means.values

        # Center only the rated entries, then build a sparse matrix.
        # fillna(0) on the full pivot frame (610×9724 for ml-latest-small,
        # ~98% zeros) and passing that dense array to randomized SVD triggers
        # spurious BLAS RuntimeWarnings during the final U = Q @ Uhat step.
        # stack() drops NaN by default, so we only store the ~100k rated
        # entries and let the rest stay implicit zeros.
        centered = matrix.sub(user_means, axis=0)
        stacked = centered.stack()

        movie_to_col = {mid: j for j, mid in enumerate(self.movie_ids)}
        row_idx = np.array([self.user_index[uid] for uid, _ in stacked.index])
        col_idx = np.array([movie_to_col[mid] for _, mid in stacked.index])

        sparse_centered = csr_matrix(
            (stacked.values.astype(np.float64), (row_idx, col_idx)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )

        self.svd.fit(sparse_centered)
        self.predicted = (
            self.svd.inverse_transform(self.svd.transform(sparse_centered))
            + self.user_mean[:, np.newaxis]
        )
        self.rated_mask = ~matrix.isna().values
        return self

    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        if user_id not in self.user_index:
            raise ValueError(f"User ID {user_id} not found in training data.")

        idx = self.user_index[user_id]
        scores = self.predicted[idx].copy()
        scores[self.rated_mask[idx]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return [(self.movie_ids[i], float(scores[i])) for i in top_indices]

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SVDRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)
