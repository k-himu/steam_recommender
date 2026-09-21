#!/usr/bin/env python3
"""
Command-line interface for the Steam Game Recommendation System.

A minimal, dependency-free (no streamlit) alternative to the Streamlit
app, built on the exact same SteamRecommender class. Useful for quick
testing or headless environments.

Usage:
    python app/cli.py
    python app/cli.py --game "Portal 2" --top-n 10 --ram-gb 16 --cpu-ghz 3.5 --os windows
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset_builder import build_unified_dataset, load_unified_dataset
from src.recommender import SteamRecommender, RecommendationRequest, RecommendationFilters
from src.hardware_compatibility import UserHardwareSpec
from src.explanations import format_explanation_text


def get_recommender() -> SteamRecommender:
    try:
        games, requirements_map = load_unified_dataset()
    except FileNotFoundError:
        print("No cached dataset found -- building from raw CSVs (this may take a moment)...")
        games, requirements_map, report = build_unified_dataset()
        print(report.summary())
    return SteamRecommender(games, requirements_map)


def interactive_mode(rec: SteamRecommender) -> None:
    print("=== Steam Game Recommendation System (CLI) ===\n")
    query = input("Search for a game you like: ").strip()
    matches = rec.search_games(query, limit=15)
    if matches.empty:
        print(f"No games found matching '{query}'.")
        return

    print("\nMatches:")
    for i, (_, row) in enumerate(matches.iterrows(), start=1):
        print(f"  {i}. {row['name']} ({row['release_date']})")

    choice = input("\nPick a number: ").strip()
    try:
        idx = int(choice) - 1
        appid = int(matches.iloc[idx]["appid"])
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    print("\n--- Your PC specs (press Enter to skip any field) ---")
    os_name = input("Operating system (windows/mac/linux): ").strip() or None
    cpu_ghz = _parse_float(input("CPU speed in GHz: "))
    ram_gb = _parse_float(input("RAM in GB: "))
    vram_gb = _parse_float(input("GPU VRAM in GB: "))
    storage_gb = _parse_float(input("Free storage in GB: "))

    hardware = UserHardwareSpec(
        os=os_name,
        cpu_ghz=cpu_ghz,
        ram_mb=(ram_gb * 1024) if ram_gb else None,
        gpu_vram_mb=(vram_gb * 1024) if vram_gb else None,
        storage_gb=storage_gb,
    )

    top_n_input = input("\nHow many recommendations? [10]: ").strip()
    top_n = int(top_n_input) if top_n_input else 10

    request = RecommendationRequest(reference_appid=appid, top_n=top_n, user_hardware=hardware)
    recommendations = rec.recommend(request)
    print_recommendations(recommendations, matches.iloc[idx]["name"])


def _parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def print_recommendations(recommendations, reference_name: str) -> None:
    if not recommendations:
        print("\nNo recommendations found (try loosening filters or picking a different game).")
        return

    print(f"\n=== {len(recommendations)} games similar to '{reference_name}' ===\n")
    for i, rec in enumerate(recommendations, start=1):
        g = rec.ranked_game
        row = g.raw_row
        price = "Free" if row["price"] == 0 else f"£{row['price']:.2f}"
        print(f"{i}. {g.name}  (score: {g.final_score:.3f}, price: {price}, hw: {g.hardware_label})")
        print(format_explanation_text(rec.explanation))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=str, help="Game name to search for (non-interactive mode)")
    parser.add_argument("--top-n", type=int, default=config.DEFAULT_TOP_N)
    parser.add_argument("--os", type=str, default=None)
    parser.add_argument("--cpu-ghz", type=float, default=None)
    parser.add_argument("--ram-gb", type=float, default=None)
    parser.add_argument("--vram-gb", type=float, default=None)
    parser.add_argument("--storage-gb", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format=config.LOG_FORMAT)
    rec = get_recommender()

    if args.game:
        matches = rec.search_games(args.game, limit=1)
        if matches.empty:
            print(f"No game found matching '{args.game}'.")
            return
        appid = int(matches.iloc[0]["appid"])
        hardware = UserHardwareSpec(
            os=args.os,
            cpu_ghz=args.cpu_ghz,
            ram_mb=(args.ram_gb * 1024) if args.ram_gb else None,
            gpu_vram_mb=(args.vram_gb * 1024) if args.vram_gb else None,
            storage_gb=args.storage_gb,
        )
        request = RecommendationRequest(reference_appid=appid, top_n=args.top_n, user_hardware=hardware)
        recommendations = rec.recommend(request)
        print_recommendations(recommendations, matches.iloc[0]["name"])
    else:
        interactive_mode(rec)


if __name__ == "__main__":
    main()
