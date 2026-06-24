import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.recommender import SVDRecommender
from utils import fetch_poster_url

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).parent.parent / "models" / "svd_model.pkl"


@st.cache_resource(show_spinner="Training recommender model...")
def load_model() -> SVDRecommender:
    if MODEL_PATH.exists():
        return SVDRecommender.load(MODEL_PATH)
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    model = SVDRecommender(n_components=50)
    model.fit(ratings)
    return model


@st.cache_data
def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")
    return movies, links


def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.title("🎬 Movie Recommender")
    st.caption("Collaborative filtering with SVD · MovieLens 100k")

    model = load_model()
    movies_df, links_df = load_metadata()

    movie_meta = movies_df.merge(links_df[["movieId", "tmdbId"]], on="movieId", how="left")

    with st.sidebar:
        st.header("Settings")
        user_id = st.number_input(
            "User ID (1 – 671)", min_value=1, max_value=671, value=1, step=1
        )
        n_recs = st.slider("Recommendations", min_value=5, max_value=20, value=10)
        show_posters = st.toggle("Show posters (requires TMDB key)", value=True)

    if st.button("Get Recommendations", type="primary"):
        try:
            recs = model.recommend(int(user_id), n=n_recs)
        except ValueError as exc:
            st.error(str(exc))
            return

        rec_ids = [movie_id for movie_id, _ in recs]
        rec_scores = {movie_id: score for movie_id, score in recs}
        rec_df = movie_meta[movie_meta["movieId"].isin(rec_ids)].copy()
        rec_df["score"] = rec_df["movieId"].map(rec_scores)
        rec_df = rec_df.sort_values("score", ascending=False)

        st.subheader(f"Top {n_recs} picks for User {user_id}")

        cols = st.columns(5)
        for i, (_, row) in enumerate(rec_df.iterrows()):
            with cols[i % 5]:
                if show_posters:
                    poster = fetch_poster_url(row.get("tmdbId"))
                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.markdown("🎬")
                st.markdown(f"**{row['title']}**")
                st.caption(row.get("genres", "").replace("|", " · "))


if __name__ == "__main__":
    main()
