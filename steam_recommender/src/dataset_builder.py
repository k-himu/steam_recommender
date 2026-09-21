"""
Builds the cleaned, unified game dataset and the parsed-requirements
lookup used by the rest of the system. This is the Phase 1 (Steps 1-3)
pipeline as one callable entry point, with every cleaning decision logged.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from . import data_loading
from . import data_validation
from . import feature_engineering as fe
from .requirements_parsing import GameRequirements, parse_game_requirements

logger = logging.getLogger(__name__)


def build_unified_dataset(
    steam_path: Optional[Path] = None,
    description_path: Optional[Path] = None,
    requirements_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict[int, GameRequirements], data_validation.JoinCoverageReport]:
    """
    Full Phase-1 pipeline: load -> validate -> dedupe -> join coverage ->
    feature engineer -> parse requirements.

    Returns:
        unified_df: one row per game, with content_text + quality/popularity
                     feature columns, ready for the content model and ranking.
        requirements_map: {appid: GameRequirements}, one entry per game that
                     had at least one parsable requirements row. Games with
                     no entry here must be treated as UNKNOWN compatibility
                     by the hardware engine (never assumed compatible).
        report: the join coverage report (also logged).
    """
    games, descriptions, requirements = data_loading.load_all(
        steam_path, description_path, requirements_path
    )

    # --- Dedupe defensively (steam.csv had none in our inspection, but the
    # pipeline must not assume that holds for every copy of this dataset).
    games = data_validation.deduplicate_by_id(games, config.STEAM_ID_COL, "steam.csv")
    if descriptions is not None:
        descriptions = data_validation.deduplicate_by_id(
            descriptions, config.DESCRIPTION_ID_COL, "steam_description_data.csv"
        )
    if requirements is not None:
        requirements = data_validation.deduplicate_by_id(
            requirements, config.REQUIREMENTS_ID_COL, "steam_requirements_data.csv"
        )

    report = data_validation.check_join_coverage(games, descriptions, requirements)

    # --- Numeric column sanity coercion (malformed -> NaN, logged, never dropped)
    games = data_validation.validate_numeric_columns(
        games,
        ["positive_ratings", "negative_ratings", "average_playtime", "median_playtime", "price"],
        "steam.csv",
    )

    # --- Feature engineering
    games["content_text"] = fe.build_unified_text(games, descriptions)
    games = fe.build_quality_popularity_features(games)

    # --- Requirements parsing (only for games with a requirements row)
    requirements_map: dict[int, GameRequirements] = {}
    if requirements is not None:
        n_failed = 0
        for _, row in requirements.iterrows():
            try:
                parsed = parse_game_requirements(row)
                if parsed.has_any_requirements:
                    requirements_map[parsed.appid] = parsed
            except Exception:
                n_failed += 1
                logger.exception("Failed to parse requirements for a row; skipping that row.")
        if n_failed:
            logger.warning("%d requirements row(s) failed to parse and were skipped.", n_failed)
        logger.info(
            "Parsed requirements for %d / %d games (%.1f%% of games in steam.csv).",
            len(requirements_map), len(games), 100 * len(requirements_map) / max(len(games), 1),
        )
    else:
        logger.warning(
            "No requirements file loaded -- every game will report UNKNOWN hardware compatibility."
        )

    return games, requirements_map, report


def save_unified_dataset(
    games: pd.DataFrame,
    requirements_map: dict[int, GameRequirements],
    output_dir: Path = config.DATA_PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        games.to_parquet(output_dir / "unified_games.parquet", index=False)
        logger.info("Saved unified dataset (parquet) to %s", output_dir / "unified_games.parquet")
    except Exception:
        # parquet engine may not be installed in every environment -- fall
        # back to CSV rather than failing the whole pipeline.
        games.to_csv(output_dir / "unified_games.csv", index=False)
        logger.info("Saved unified dataset (csv fallback) to %s", output_dir / "unified_games.csv")

    with open(output_dir / "requirements_map.pkl", "wb") as f:
        pickle.dump(requirements_map, f)
    logger.info("Saved requirements map (%d games) to %s", len(requirements_map), output_dir / "requirements_map.pkl")


def load_unified_dataset(
    input_dir: Path = config.DATA_PROCESSED_DIR,
) -> tuple[pd.DataFrame, dict[int, GameRequirements]]:
    parquet_path = input_dir / "unified_games.parquet"
    csv_path = input_dir / "unified_games.csv"

    if parquet_path.exists():
        games = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        games = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No unified dataset found in {input_dir}. Run the build pipeline first "
            f"(scripts/build_dataset.py)."
        )

    req_path = input_dir / "requirements_map.pkl"
    requirements_map: dict[int, GameRequirements] = {}
    if req_path.exists():
        with open(req_path, "rb") as f:
            requirements_map = pickle.load(f)

    return games, requirements_map
