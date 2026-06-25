from typing import Dict, List, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Post-hoc explanation layer — derived from explicit ratings, independent of
# SVD's latent factor reasoning.
# ---------------------------------------------------------------------------

def get_user_taste_profile(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Set[str]]:
    """Return the user's top-rated movies and the genre set they imply.

    Tries 4.0+ first; falls back to 3.5+ when no 4.0+ ratings exist.
    The returned DataFrame has columns: title, rating, threshold_used.
    """
    user_ratings = ratings_df[ratings_df["userId"] == user_id].merge(
        movies_df[["movieId", "title", "genres"]], on="movieId", how="left"
    )

    for threshold in (4.0, 3.5):
        top = (
            user_ratings[user_ratings["rating"] >= threshold]
            .sort_values("rating", ascending=False)
            .head(5)
            .copy()
        )
        if not top.empty:
            top["threshold_used"] = threshold
            break
    else:
        top = pd.DataFrame(columns=["title", "rating", "genres", "threshold_used"])

    liked_genres: Set[str] = set()
    for genres_str in top["genres"].dropna():
        liked_genres.update(g for g in genres_str.split("|") if g != "(no genres listed)")

    return top, liked_genres


def genre_overlap(movie_genres_str: str, liked_genres: Set[str]) -> List[str]:
    """Return genres this movie shares with the user's liked-genre set."""
    if not movie_genres_str or not liked_genres:
        return []
    movie_genres = {g for g in movie_genres_str.split("|") if g != "(no genres listed)"}
    return sorted(movie_genres & liked_genres)


def get_user_profile_stats(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> Dict[str, object]:
    """Compact stats for the sidebar taste-profile card.

    Unlike get_user_taste_profile (which returns only the top 5 movies), the
    genre counts here span ALL of the user's highly-rated movies (>=4.0, or
    >=3.5 if they have none at 4.0), so the bar chart reflects real volume.

    Returns: total_rated (int), avg_rating (float),
             genre_counts (Series, descending), top_genre (str).
    """
    user_ratings = ratings_df[ratings_df["userId"] == user_id]
    total_rated = int(len(user_ratings))
    avg_rating = float(user_ratings["rating"].mean()) if total_rated else 0.0

    merged = user_ratings.merge(movies_df[["movieId", "genres"]], on="movieId", how="left")
    liked = merged.iloc[0:0]
    for threshold in (4.0, 3.5):
        liked = merged[merged["rating"] >= threshold]
        if not liked.empty:
            break

    counts: Dict[str, int] = {}
    for genres_str in liked["genres"].dropna():
        for g in genres_str.split("|"):
            if g != "(no genres listed)":
                counts[g] = counts.get(g, 0) + 1

    genre_counts = (
        pd.Series(counts, name="movies").sort_values(ascending=False)
        if counts
        else pd.Series(dtype=int, name="movies")
    )
    top_genre = str(genre_counts.index[0]) if not genre_counts.empty else "—"

    return {
        "total_rated": total_rated,
        "avg_rating": avg_rating,
        "genre_counts": genre_counts,
        "top_genre": top_genre,
    }
