"""
Streamlit UI for the Steam Game Recommendation System.

Run with:
    streamlit run app/streamlit_app.py

This UI is a thin presentation layer only -- all recommendation logic
lives in src/recommender.py and can be used identically from a CLI,
notebook, or future API. Nothing in this file computes a recommendation
itself; it only collects input and renders SteamRecommender's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config
from src.dataset_builder import build_unified_dataset, load_unified_dataset
from src.recommender import SteamRecommender, RecommendationRequest, RecommendationFilters
from src.hardware_compatibility import UserHardwareSpec, CompatibilityLabel

st.set_page_config(page_title="Steam Game Recommender", page_icon="🎮", layout="wide")


# --------------------------------------------------------------------------
# Data loading (cached so the dataset/model are built once per session)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading game dataset and building content model...")
def get_recommender() -> SteamRecommender:
    try:
        games, requirements_map = load_unified_dataset()
    except FileNotFoundError:
        games, requirements_map, _report = build_unified_dataset()
    return SteamRecommender(games, requirements_map)


def compat_badge(label: str) -> str:
    colors = {
        CompatibilityLabel.COMPATIBLE.value: "🟢",
        CompatibilityLabel.LIKELY_COMPATIBLE.value: "🟡",
        CompatibilityLabel.BORDERLINE.value: "🟠",
        CompatibilityLabel.NOT_RECOMMENDED.value: "🔴",
        CompatibilityLabel.UNKNOWN.value: "⚪",
    }
    return f"{colors.get(label, '⚪')} {label}"


# --------------------------------------------------------------------------
# Sidebar: hardware spec + filters
# --------------------------------------------------------------------------

def render_sidebar() -> tuple[UserHardwareSpec, RecommendationFilters, int]:
    st.sidebar.header("💻 Your PC Specs")
    st.sidebar.caption("Leave anything blank/zero if you don't know it — it will simply be excluded from the compatibility check rather than counted against you.")

    os_choice = st.sidebar.selectbox("Operating System", ["Unknown", "Windows 10", "Windows 11", "Windows 7", "macOS", "Linux"])
    cpu_ghz = st.sidebar.number_input("CPU clock speed (GHz)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    ram_gb = st.sidebar.number_input("RAM (GB)", min_value=0.0, max_value=256.0, value=0.0, step=1.0)
    gpu_vram_gb = st.sidebar.number_input("GPU VRAM (GB)", min_value=0.0, max_value=48.0, value=0.0, step=1.0)
    storage_gb = st.sidebar.number_input("Free storage (GB)", min_value=0.0, max_value=8000.0, value=0.0, step=10.0)

    user_hardware = UserHardwareSpec(
        os=None if os_choice == "Unknown" else os_choice,
        cpu_ghz=cpu_ghz or None,
        ram_mb=(ram_gb * 1024) or None,
        gpu_vram_mb=(gpu_vram_gb * 1024) or None,
        storage_gb=storage_gb or None,
    )

    st.sidebar.divider()
    st.sidebar.header("🔎 Filters")

    all_genres = sorted({
        g for genres in get_recommender().games["genres"].dropna()
        for g in genres.split(";")
    })
    selected_genres = st.sidebar.multiselect("Genres", all_genres)

    platform_choice = st.sidebar.multiselect("Platforms", ["windows", "mac", "linux"])

    price_mode = st.sidebar.radio("Price", ["Any", "Free only", "Paid only"], horizontal=False)
    max_price = st.sidebar.slider("Max price (£)", 0.0, 100.0, 100.0, step=1.0)

    min_compat = st.sidebar.selectbox(
        "Minimum hardware compatibility",
        ["Any", CompatibilityLabel.BORDERLINE.value, CompatibilityLabel.LIKELY_COMPATIBLE.value,
         CompatibilityLabel.COMPATIBLE.value],
    )

    filters = RecommendationFilters(
        genres=selected_genres or None,
        platforms=platform_choice or None,
        max_price=max_price if max_price < 100.0 else None,
        free_only=(price_mode == "Free only"),
        paid_only=(price_mode == "Paid only"),
        min_compatibility=None if min_compat == "Any" else min_compat,
    )

    st.sidebar.divider()
    top_n = st.sidebar.slider("Number of recommendations", 3, 25, config.DEFAULT_TOP_N)

    return user_hardware, filters, top_n


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    st.title("🎮 Steam Game Recommendation System")
    st.caption(
        "Content-based + hardware-aware. Tell it a game you like and what PC you "
        "have, and it ranks new games by similarity, quality, engagement, and "
        "whether your machine can actually run them."
    )

    recommender = get_recommender()

    st.subheader("1. Pick a game you like")
    query = st.text_input("Search for a game", placeholder="e.g. Portal 2, Skyrim, Counter-Strike...")

    selected_appid = None
    if query:
        matches = recommender.search_games(query, limit=20)
        if matches.empty:
            st.warning(f"No games found matching '{query}'. Try a different search term.")
        else:
            options = {f"{row['name']} ({row['release_date']})": int(row["appid"]) for _, row in matches.iterrows()}
            choice = st.selectbox("Select the game", list(options.keys()))
            selected_appid = options[choice]

    user_hardware, filters, top_n = render_sidebar()

    st.subheader("2. Get recommendations")
    go = st.button("Find similar games", type="primary", disabled=selected_appid is None)

    if go and selected_appid is not None:
        request = RecommendationRequest(
            reference_appid=selected_appid,
            top_n=top_n,
            user_hardware=user_hardware,
            filters=filters,
        )
        with st.spinner("Ranking candidates..."):
            recommendations = recommender.recommend(request)

        if not recommendations:
            st.warning(
                "No recommendations matched your filters. Try loosening the genre, "
                "price, or compatibility filters."
            )
            return

        ref_row = recommender.get_game(selected_appid)
        st.success(f"Showing {len(recommendations)} games similar to **{ref_row['name']}**")

        for i, rec in enumerate(recommendations, start=1):
            g = rec.ranked_game
            row = g.raw_row
            with st.container(border=True):
                cols = st.columns([3, 1, 1, 1, 1])
                cols[0].markdown(f"### {i}. {g.name}")
                cols[1].metric("Final score", f"{g.final_score:.2f}")
                cols[2].metric("Similarity", f"{g.content_similarity:.0%}")
                price_display = "Free" if row["price"] == 0 else f"£{row['price']:.2f}"
                cols[3].metric("Price", price_display)
                cols[4].markdown(f"**{compat_badge(g.hardware_label)}**")

                detail_cols = st.columns([2, 1])
                with detail_cols[0]:
                    st.markdown(f"**Genres:** {row.get('genres', 'N/A')}")
                    st.markdown(f"**Tags:** {row.get('steamspy_tags', 'N/A')}")
                    st.markdown(f"**Released:** {row.get('release_date', 'N/A')}")
                    reviews = int(row.get("total_reviews", 0))
                    ratio = row.get("raw_rating_ratio")
                    ratio_display = f"{ratio:.0%} positive" if ratio == ratio else "no reviews yet"
                    st.markdown(f"**Reviews:** {reviews:,} ({ratio_display})")
                    avg_pt = row.get("average_playtime", 0)
                    med_pt = row.get("median_playtime", 0)
                    st.markdown(f"**Playtime:** avg {avg_pt:.0f} min, median {med_pt:.0f} min")

                with detail_cols[1]:
                    st.markdown("**Why recommended:**")
                    for reason in rec.explanation.reasons:
                        st.markdown(f"- {reason}")
                    if rec.explanation.hardware_notes:
                        st.markdown("**Hardware notes:**")
                        for note in rec.explanation.hardware_notes:
                            st.markdown(f"- {note}")


if __name__ == "__main__":
    main()
