"""
Ranking evaluation metrics for top-k recommendation lists.

RMSE (see train.py) measures rating-prediction error, but users never see raw
predictions — they see a ranked top-k list. These metrics score the list itself:

  * Precision@k  — how much of what we showed was good?
  * Recall@k     — how much of the good stuff did we manage to show?
  * NDCG@k       — did we put the best stuff at the top?

All functions take plain movie-ID sequences so they are model-agnostic: any
recommender that outputs a ranked ID list can be scored.
"""
import math
from typing import Dict, Sequence, Set


def precision_at_k(
    recommended_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """Fraction of the top-k recommendations that are relevant.

        P@k = |top-k ∩ relevant| / k

    The denominator is k (not the list length): a model that can only produce
    fewer than k recommendations is penalised for the slots it left empty.
    """
    if k <= 0:
        return 0.0
    top_k = list(recommended_ids)[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / float(k)


def recall_at_k(
    recommended_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int,
) -> float:
    """Fraction of all relevant movies that made it into the top-k.

        R@k = |top-k ∩ relevant| / |relevant|

    Returns 0.0 when there are no relevant movies (nothing to recall).
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = list(recommended_ids)[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / float(len(relevant_ids))


def ndcg_at_k(
    recommended_ids: Sequence[int],
    relevant_ratings: Dict[int, float],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain: position-aware, rating-weighted.

    Gain of the item at rank i (1-based) is its actual test-set rating if the
    item is relevant, else 0. Gains are discounted logarithmically by position,
    so a 5.0-rated movie at rank 1 is worth far more than the same movie at
    rank 10:

        DCG@k  = Σ_{i=1..k}  gain_i / log2(i + 1)

    That raw sum is normalised by the best score any ordering could achieve —
    the ideal DCG, obtained by placing the relevant items in descending order
    of their ratings:

        IDCG@k = Σ_{i=1..k}  ideal_gain_i / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k              ∈ [0, 1]

    Returns 0.0 when there are no relevant items (IDCG would be 0).
    """
    if not relevant_ratings or k <= 0:
        return 0.0

    top_k = list(recommended_ids)[:k]
    # enumerate is 0-based, ranks are 1-based -> discount log2(i + 2).
    dcg = sum(
        relevant_ratings.get(mid, 0.0) / math.log2(i + 2)
        for i, mid in enumerate(top_k)
    )

    ideal_gains = sorted(relevant_ratings.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))

    return dcg / idcg if idcg > 0 else 0.0
