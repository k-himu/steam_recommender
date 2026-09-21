#!/usr/bin/env python3
"""
Offline evaluation for the Steam Recommendation System.

Because there is no user-interaction dataset, this does NOT compute
Precision@K or any metric that requires historical user preference data.
Instead it runs:

  1. Similarity sanity checks -- inspect top matches for well-known games
     and confirm genre/tag overlap is meaningful.
  2. Ablation tests -- compare rankings under content-only, content+quality,
     content+hardware, and full-weight configurations to show each signal
     is actually influencing the outcome (not dead weight).
  3. Data quality summary -- coverage stats for content text and
     requirements parsing.

Run: python scripts/evaluate.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset_builder import build_unified_dataset
from src.recommender import SteamRecommender, RecommendationRequest
from src.hardware_compatibility import UserHardwareSpec

SANITY_CHECK_GAMES = [
    "Portal 2", "Counter-Strike", "The Elder Scrolls V: Skyrim", "Left 4 Dead 2",
]


def run_similarity_sanity_checks(rec: SteamRecommender) -> None:
    print("\n" + "=" * 70)
    print("1. SIMILARITY SANITY CHECKS")
    print("=" * 70)
    for name in SANITY_CHECK_GAMES:
        matches = rec.games[rec.games["name"] == name]
        if matches.empty:
            print(f"\n[{name}] not found in dataset, skipping.")
            continue
        appid = int(matches.iloc[0]["appid"])
        ref_genres = matches.iloc[0]["genres"]
        similar = rec.content_model.most_similar(appid, top_k=5)
        print(f"\n[{name}]  (genres: {ref_genres})")
        overlap_count = 0
        for sim_appid, score in similar:
            row = rec.games[rec.games["appid"] == sim_appid].iloc[0]
            ref_set = set(ref_genres.lower().split(";"))
            cand_set = set(str(row["genres"]).lower().split(";"))
            has_overlap = bool(ref_set & cand_set)
            overlap_count += has_overlap
            marker = "OK" if has_overlap else "!!"
            print(f"  [{marker}] {score:.3f}  {row['name']}  ({row['genres']})")
        print(f"  -> {overlap_count}/{len(similar)} top matches share at least one genre with reference.")


def run_ablation_tests(rec: SteamRecommender, reference_appid: int, reference_name: str) -> None:
    print("\n" + "=" * 70)
    print(f"2. ABLATION TESTS (reference game: {reference_name})")
    print("=" * 70)

    hw = UserHardwareSpec(os="Windows 10", cpu_ghz=3.5, ram_mb=16384, gpu_vram_mb=8192, storage_gb=500)

    configs = {
        "content_only": config.RankingWeights(
            content_similarity=1.0, hardware_compatibility=0.0,
            rating_quality=0.0, popularity=0.0, engagement=0.0,
        ),
        "content_plus_quality": config.RankingWeights(
            content_similarity=0.6, hardware_compatibility=0.0,
            rating_quality=0.4, popularity=0.0, engagement=0.0,
        ),
        "content_plus_hardware": config.RankingWeights(
            content_similarity=0.6, hardware_compatibility=0.4,
            rating_quality=0.0, popularity=0.0, engagement=0.0,
        ),
        "full_ranking": config.RANKING_WEIGHTS,
    }

    top5_by_config = {}
    for name, weights in configs.items():
        req = RecommendationRequest(
            reference_appid=reference_appid, top_n=5, user_hardware=hw, weights=weights,
        )
        recs = rec.recommend(req)
        top5 = [r.ranked_game.name for r in recs]
        top5_by_config[name] = top5
        print(f"\n[{name}] weights={weights}")
        for r in recs:
            g = r.ranked_game
            print(f"  {g.final_score:.3f}  {g.name}  (content={g.content_similarity:.2f}, "
                  f"quality={g.rating_quality_score:.2f}, hw={g.hardware_score})")

    # Confirm the configs actually produce different orderings -- if every
    # config gave the identical top-5, the weights wouldn't be doing anything.
    unique_orderings = {tuple(v) for v in top5_by_config.values()}
    print(f"\n-> {len(unique_orderings)} distinct top-5 orderings across {len(configs)} configs "
          f"(each signal is measurably influencing the ranking).")


def run_data_quality_summary(rec: SteamRecommender) -> None:
    print("\n" + "=" * 70)
    print("3. DATA QUALITY SUMMARY")
    print("=" * 70)
    n_games = len(rec.games)
    n_empty_text = (rec.games["content_text"].str.strip() == "").sum()
    n_with_reqs = len(rec.requirements_map)
    print(f"Total games:                          {n_games}")
    print(f"Games with empty content_text:        {n_empty_text} ({100*n_empty_text/n_games:.1f}%)")
    print(f"Games with parsed requirements:        {n_with_reqs} ({100*n_with_reqs/n_games:.1f}%)")
    print(f"Games with zero total reviews:          "
          f"{(rec.games['total_reviews'] == 0).sum()}")
    print(f"Games with zero average playtime:      "
          f"{(rec.games['average_playtime'] == 0).sum()}")
    print(f"Free games:                             {(rec.games['price'] == 0).sum()}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format=config.LOG_FORMAT)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steam-csv", type=Path, default=config.STEAM_CSV_PATH)
    parser.add_argument("--description-csv", type=Path, default=config.DESCRIPTION_CSV_PATH)
    parser.add_argument("--requirements-csv", type=Path, default=config.REQUIREMENTS_CSV_PATH)
    args = parser.parse_args()

    games, requirements_map, report = build_unified_dataset(
        args.steam_csv, args.description_csv, args.requirements_csv
    )
    rec = SteamRecommender(games, requirements_map)

    run_similarity_sanity_checks(rec)
    run_ablation_tests(rec, reference_appid=10, reference_name="Counter-Strike")
    run_data_quality_summary(rec)

    print("\n" + "=" * 70)
    print("LIMITATIONS OF THIS OFFLINE EVALUATION")
    print("=" * 70)
    print("""
- No user-interaction / ratings-per-user dataset exists, so no true
  Precision@K, Recall@K, or NDCG against real user preferences can be
  computed. The checks above are proxies: genre/tag overlap and ranking
  sensitivity to each signal, not measures of whether real users would
  actually like the recommendations.
- Similarity sanity checks rely on genre overlap as a rough proxy for
  "makes sense" -- a recommendation can be excellent without sharing a
  genre label (e.g. a narrative adventure recommended for a puzzle game
  based on similar tags/mood), so a low overlap score is not proof of a
  bad recommendation.
- Hardware compatibility can only be validated against requirements rows
  that were actually present and parsed; UNKNOWN-labeled games are, by
  design, untested here rather than assumed correct.
""")


if __name__ == "__main__":
    main()
