#!/usr/bin/env python3
"""
Example usage of the Steam Recommendation System's Python API, independent
of any UI. Run after `python scripts/build_dataset.py` has cached a unified
dataset.

    python scripts/example_usage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_builder import load_unified_dataset
from src.recommender import SteamRecommender, RecommendationRequest, RecommendationFilters
from src.hardware_compatibility import UserHardwareSpec
from src.explanations import format_explanation_text


def main() -> None:
    games, requirements_map = load_unified_dataset()
    recommender = SteamRecommender(games, requirements_map)

    # --- Example 1: basic recommendation ---
    matches = recommender.search_games("Portal 2")
    if matches.empty:
        print("Example game not found in this copy of the dataset -- pick another with search_games().")
        return
    reference_appid = int(matches.iloc[0]["appid"])

    my_pc = UserHardwareSpec(
        os="Windows 10",
        cpu_ghz=3.6,
        ram_mb=16 * 1024,
        gpu_vram_mb=8 * 1024,
        storage_gb=500,
    )

    request = RecommendationRequest(
        reference_appid=reference_appid,
        top_n=5,
        user_hardware=my_pc,
    )
    recommendations = recommender.recommend(request)

    print(f"=== Top {len(recommendations)} recommendations for 'Portal 2' ===\n")
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec.ranked_game.name}  (score={rec.ranked_game.final_score:.3f})")
        print(format_explanation_text(rec.explanation))
        print()

    # --- Example 2: with filters ---
    filtered_request = RecommendationRequest(
        reference_appid=reference_appid,
        top_n=5,
        user_hardware=my_pc,
        filters=RecommendationFilters(free_only=True, genres=["Action"]),
    )
    filtered_recs = recommender.recommend(filtered_request)
    print(f"=== Free Action games similar to 'Portal 2': {len(filtered_recs)} found ===")
    for rec in filtered_recs:
        print(f"  - {rec.ranked_game.name}")


if __name__ == "__main__":
    main()
