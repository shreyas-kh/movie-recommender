import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
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

        self.user_mean = matrix.mean(axis=1).values
        centered = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0).values

        self.svd.fit(centered)
        reconstructed = self.svd.inverse_transform(self.svd.transform(centered))
        self.predicted = reconstructed + self.user_mean[:, np.newaxis]

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
