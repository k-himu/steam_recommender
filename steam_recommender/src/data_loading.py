"""
Data loading for the Steam Recommendation System.

Loads the three raw datasets and performs lightweight, honest schema
verification -- it does NOT assume column names; it checks for the
columns it needs and raises a clear error naming exactly what was
expected vs. found if something doesn't match.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class SchemaError(Exception):
    """Raised when a required column is missing from a loaded dataset."""


def _require_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{source}: missing required column(s) {missing}. "
            f"Actual columns found: {list(df.columns)}"
        )


def load_steam_games(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the core steam.csv dataset (appid, name, genres, ratings, etc.).

    Raises FileNotFoundError if the file doesn't exist, SchemaError if the
    minimal columns this project depends on aren't present.
    """
    path = path or config.STEAM_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"steam.csv not found at {path}")

    df = pd.read_csv(path)

    required = [
        config.STEAM_ID_COL, "name", "release_date", "developer", "publisher",
        "platforms", "categories", "genres", "steamspy_tags",
        "positive_ratings", "negative_ratings", "average_playtime",
        "median_playtime", "owners", "price",
    ]
    _require_columns(df, required, "steam.csv")

    logger.info("Loaded steam.csv: %d rows, %d columns", len(df), len(df.columns))
    return df


def load_descriptions(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load steam_description_data.csv.

    Returns None (with a warning) if the file is not present -- the system
    is designed to degrade gracefully rather than crash when this large,
    optional-at-runtime file isn't available. Downstream code must handle
    a None return value.
    """
    path = path or config.DESCRIPTION_CSV_PATH
    if not path.exists():
        logger.warning(
            "Descriptions file not found at %s. Proceeding without game "
            "descriptions -- content text will rely on genres/tags/categories only.",
            path,
        )
        return None

    df = pd.read_csv(path)
    required = [config.DESCRIPTION_ID_COL, "short_description"]
    _require_columns(df, required, "steam_description_data.csv")

    logger.info("Loaded descriptions: %d rows, %d columns", len(df), len(df.columns))
    return df


def load_requirements(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load steam_requirements_data.csv.

    Returns None (with a warning) if the file is not present. Downstream
    code (hardware compatibility engine) must treat every game as
    UNKNOWN compatibility in that case, never as compatible or incompatible.
    """
    path = path or config.REQUIREMENTS_CSV_PATH
    if not path.exists():
        logger.warning(
            "Requirements file not found at %s. Proceeding without hardware "
            "requirements -- all games will report UNKNOWN hardware compatibility.",
            path,
        )
        return None

    df = pd.read_csv(path)
    required = [config.REQUIREMENTS_ID_COL, "minimum"]
    _require_columns(df, required, "steam_requirements_data.csv")

    logger.info("Loaded requirements: %d rows, %d columns", len(df), len(df.columns))
    return df


def load_all(
    steam_path: Optional[Path] = None,
    description_path: Optional[Path] = None,
    requirements_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Convenience loader for all three datasets."""
    games = load_steam_games(steam_path)
    descriptions = load_descriptions(description_path)
    requirements = load_requirements(requirements_path)
    return games, descriptions, requirements
