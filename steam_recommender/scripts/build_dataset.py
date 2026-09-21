#!/usr/bin/env python3
"""
Build the unified, cleaned dataset from the three raw Steam CSVs and cache
it to data/processed/. Run this once before starting the app.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --steam-csv path/to/steam.csv \
        --description-csv path/to/steam_description_data.csv \
        --requirements-csv path/to/steam_requirements_data.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset_builder import build_unified_dataset, save_unified_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steam-csv", type=Path, default=config.STEAM_CSV_PATH)
    parser.add_argument("--description-csv", type=Path, default=config.DESCRIPTION_CSV_PATH)
    parser.add_argument("--requirements-csv", type=Path, default=config.REQUIREMENTS_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.DATA_PROCESSED_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

    games, requirements_map, report = build_unified_dataset(
        args.steam_csv, args.description_csv, args.requirements_csv
    )

    print("\n" + report.summary() + "\n")
    print(f"Unified dataset: {len(games)} games, {len(games.columns)} columns")
    print(f"Games with parsed requirements: {len(requirements_map)}")

    save_unified_dataset(games, requirements_map, args.output_dir)
    print(f"\nDone. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
