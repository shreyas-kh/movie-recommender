import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.recommender import SVDRecommender
from utils import genre_overlap, get_user_profile_stats, get_user_taste_profile

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).parent.parent / "models" / "svd_model.pkl"

# Semi-transparent indigo badge: readable in both dark and light Streamlit themes
# because it uses rgba() rather than a hard-coded hex colour.
_CSS = """
<style>
.match-label {
    display: inline-block;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.78em;
    font-weight: 500;
    line-height: 1.8;
    margin-top: 2px;
}
</style>
"""


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


@st.cache_data
def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ratings.csv")


@st.cache_data
def load_posters() -> Dict[int, str]:
    path = DATA_DIR.parent / "poster_urls.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df["poster_url"].notna() & (df["poster_url"] != "")]
    return dict(zip(df["movieId"].astype(int), df["poster_url"]))


def _render_rec_card(
    row: pd.Series,
    liked_genres: Set[str],
    show_posters: bool,
    poster_map: Dict[int, str],
) -> None:
    # Poster is rendered ABOVE the bordered box (not inside it) so it fills the
    # full column width — st.image clamps to a thumbnail inside a container.
    # Movies with no poster simply get no image; we never show a placeholder.
    poster = poster_map.get(int(row["movieId"])) if row.get("movieId") else None
    if show_posters and poster:
        st.image(poster, width="stretch")

    with st.container(border=True):
        st.markdown(f"**{row.get('title', 'Unknown')}**")

        score = row.get("score")
        if score is not None and not pd.isna(score):
            predicted = max(0.5, min(5.0, float(score)))  # clamp to MovieLens scale
            st.markdown(f"Predicted rating: **{predicted:.1f}** ★")

        overlap = genre_overlap(row.get("genres", ""), liked_genres)
        if overlap:
            st.markdown(
                f'<div class="match-label">Matches your interest in: {", ".join(overlap)}</div>',
                unsafe_allow_html=True,
            )
        else:
            genres = row.get("genres", "")
            if genres:
                st.caption(genres.replace("|", " · "))


# Preset personas → representative MovieLens user IDs. Data-driven picks: each
# user's highly-rated history leans strongly toward the named taste. Swap freely.
PERSONAS = [
    ("🎬 Action Fan", 493),
    ("💕 Romance Lover", 358),
    ("🧠 Indie Buff", 74),
    ("👨‍👩‍👧 Family Night", 401),
]


def _render_taste_profile(uid: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
    stats = get_user_profile_stats(uid, ratings_df, movies_df)
    st.markdown("#### 🎭 User Taste Profile")
    genre_counts = stats["genre_counts"]
    if not genre_counts.empty:
        st.caption("Genres in highly-rated movies")
        st.bar_chart(genre_counts.head(8), horizontal=True, height=220)
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Movies rated", stats["total_rated"])
    mcol2.metric("Avg rating", f"{stats['avg_rating']:.1f}")
    st.caption(f"Top genre: **{stats['top_genre']}**")


def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.title("🎬 Movie Recommender")
    st.caption("Collaborative filtering with SVD · MovieLens latest-small (~100k ratings)")

    with st.expander("How it works", expanded=False):
        st.markdown(
            """
This app uses **SVD (Singular Value Decomposition)**, a matrix factorization technique, applied to the
MovieLens dataset of ~100,000 real user ratings. SVD compresses the full ratings table into a small set
of hidden "taste factors" — patterns it discovers purely from who rated what and how highly — without
any knowledge of genres, directors, or plot.

To recommend movies for a user, the model predicts how that user would rate every film they haven't seen
yet, then surfaces the highest-predicted ones. These predictions come entirely from the learned taste
factors, not from hand-crafted rules.

The **"Matches your interest in: …"** labels shown on each recommendation are a *separate post-hoc layer*
added for readability: they compare each recommended movie's genres against the genres of movies you've
rated highly. This is a transparency aid — it is **not** how SVD chose those movies.
            """
        )

    model = load_model()
    movies_df, links_df = load_metadata()
    ratings_df = load_ratings()
    poster_map = load_posters()

    min_uid = int(min(model.user_ids))
    max_uid = int(max(model.user_ids))

    movie_meta = movies_df.merge(links_df[["movieId", "tmdbId"]], on="movieId", how="left")

    # --- Session state: persona buttons + recommendation trigger -------------
    if "user_id" not in st.session_state:
        st.session_state.user_id = min_uid
    if "show_recs" not in st.session_state:
        st.session_state.show_recs = False

    def _select_persona(pid: int) -> None:
        st.session_state.user_id = pid
        st.session_state.show_recs = True  # personas auto-trigger recommendations

    def _on_user_change() -> None:
        st.session_state.show_recs = False  # manual edit: show profile, wait to recommend

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.caption(f"{len(model.user_ids)} users · {len(model.movie_ids):,} movies")
        st.divider()

        st.markdown("**Try a persona**")
        pcols = st.columns(2)
        for i, (label, pid) in enumerate(PERSONAS):
            pcols[i % 2].button(
                label,
                key=f"persona_{pid}",
                width="stretch",
                on_click=_select_persona,
                args=(pid,),
            )

        user_id = st.number_input(
            f"User ID ({min_uid}–{max_uid})",
            min_value=min_uid,
            max_value=max_uid,
            step=1,
            key="user_id",
            on_change=_on_user_change,
        )
        n_recs = st.slider("Recommendations", min_value=5, max_value=20, value=10)
        show_posters = st.toggle("Show posters", value=True)
        if show_posters and not poster_map:
            st.caption("Run `scripts/fetch_posters.py` to enable posters.")

        if st.button("Get Recommendations", type="primary", width="stretch"):
            st.session_state.show_recs = True

        # Taste profile renders immediately for the current user (context first),
        # whether or not recommendations have been requested yet.
        uid = int(user_id)
        if uid in model.user_index:
            st.divider()
            _render_taste_profile(uid, ratings_df, movies_df)

        st.divider()
        st.caption("Built with [Streamlit](https://streamlit.io) · SVD collaborative filtering")

    if not st.session_state.show_recs:
        st.info("👈 Pick a persona or enter a User ID, then click **Get Recommendations**.")
        return

    uid = int(st.session_state.user_id)
    if uid not in model.user_index:
        st.error(f"User ID {uid} isn't in the training data. Try {min_uid}–{max_uid}.")
        return

    recs = model.recommend(uid, n=n_recs)
    if not recs:
        st.warning("No new recommendations found — this user may have already rated most movies.")
        return

    # --- "Because you rated these highly" — the user's own evidence ----------
    top_rated, liked_genres = get_user_taste_profile(uid, ratings_df, movies_df)
    if not top_rated.empty:
        threshold = top_rated["threshold_used"].iloc[0]
        heading = (
            "Because you rated these highly"
            if threshold >= 4.0
            else "Your top-rated movies"
        )
        st.subheader(heading)
        taste_cols = st.columns(len(top_rated), gap="small")
        for col, (_, row) in zip(taste_cols, top_rated.iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**")
                    filled = "★" * int(row["rating"])
                    empty = "☆" * (5 - int(row["rating"]))
                    st.caption(f"{filled}{empty}  {row['rating']:.1f}")
        st.divider()

    rec_ids = [movie_id for movie_id, _ in recs]
    rec_scores = {movie_id: score for movie_id, score in recs}
    rec_df = movie_meta[movie_meta["movieId"].isin(rec_ids)].copy()
    rec_df["score"] = rec_df["movieId"].map(rec_scores)
    rec_df = rec_df.sort_values("score", ascending=False)

    # Connecting narrative: ties the taste profile above to the picks below.
    st.subheader(f"🍿 Top {len(rec_df)} picks for User {uid}")
    st.caption("Based on your taste profile, we predict you'll enjoy these:")

    cols = st.columns(5, gap="small")
    for i, (_, row) in enumerate(rec_df.iterrows()):
        with cols[i % 5]:
            _render_rec_card(row, liked_genres, show_posters, poster_map)


if __name__ == "__main__":
    main()
