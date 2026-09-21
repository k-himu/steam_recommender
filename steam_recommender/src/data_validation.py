"""
Data validation and join-coverage reporting.

This module never silently drops data. Every cleaning decision it makes
is counted and reported so the caller can see exactly what happened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


@dataclass
class JoinCoverageReport:
    total_games: int
    games_with_description: int
    games_missing_description: int
    games_with_requirements: int
    games_missing_requirements: int
    description_rows_unmatched: int   # rows in description file with no matching appid
    requirements_rows_unmatched: int
    duplicate_appid_in_games: int
    duplicate_appid_in_descriptions: int
    duplicate_appid_in_requirements: int

    def summary(self) -> str:
        lines = [
            "=== Join Coverage Report ===",
            f"Total games (steam.csv):              {self.total_games}",
            f"  with description:                   {self.games_with_description} "
            f"({self._pct(self.games_with_description)})",
            f"  missing description:                {self.games_missing_description} "
            f"({self._pct(self.games_missing_description)})",
            f"  with requirements:                  {self.games_with_requirements} "
            f"({self._pct(self.games_with_requirements)})",
            f"  missing requirements:               {self.games_missing_requirements} "
            f"({self._pct(self.games_missing_requirements)})",
            f"Description rows with no matching appid in steam.csv: {self.description_rows_unmatched}",
            f"Requirements rows with no matching appid in steam.csv: {self.requirements_rows_unmatched}",
            f"Duplicate appid rows -- games: {self.duplicate_appid_in_games}, "
            f"descriptions: {self.duplicate_appid_in_descriptions}, "
            f"requirements: {self.duplicate_appid_in_requirements}",
        ]
        return "\n".join(lines)

    def _pct(self, n: int) -> str:
        if self.total_games == 0:
            return "0.0%"
        return f"{100 * n / self.total_games:.1f}%"


def check_join_coverage(
    games: pd.DataFrame,
    descriptions: Optional[pd.DataFrame],
    requirements: Optional[pd.DataFrame],
) -> JoinCoverageReport:
    """
    Determine how well the three datasets actually join together, without
    performing the join yet. Used to decide/report data quality before
    building the unified table.
    """
    game_ids = set(games[config.STEAM_ID_COL])

    if descriptions is not None:
        desc_ids = set(descriptions[config.DESCRIPTION_ID_COL])
        games_with_desc = len(game_ids & desc_ids)
        desc_unmatched = len(desc_ids - game_ids)
        dup_desc = int(descriptions[config.DESCRIPTION_ID_COL].duplicated().sum())
    else:
        games_with_desc = 0
        desc_unmatched = 0
        dup_desc = 0

    if requirements is not None:
        req_ids = set(requirements[config.REQUIREMENTS_ID_COL])
        games_with_req = len(game_ids & req_ids)
        req_unmatched = len(req_ids - game_ids)
        dup_req = int(requirements[config.REQUIREMENTS_ID_COL].duplicated().sum())
    else:
        games_with_req = 0
        req_unmatched = 0
        dup_req = 0

    report = JoinCoverageReport(
        total_games=len(games),
        games_with_description=games_with_desc,
        games_missing_description=len(games) - games_with_desc,
        games_with_requirements=games_with_req,
        games_missing_requirements=len(games) - games_with_req,
        description_rows_unmatched=desc_unmatched,
        requirements_rows_unmatched=req_unmatched,
        duplicate_appid_in_games=int(games[config.STEAM_ID_COL].duplicated().sum()),
        duplicate_appid_in_descriptions=dup_desc,
        duplicate_appid_in_requirements=dup_req,
    )
    logger.info("\n%s", report.summary())
    return report


def deduplicate_by_id(df: pd.DataFrame, id_col: str, source_name: str) -> pd.DataFrame:
    """
    Drop duplicate rows by id_col, keeping the first occurrence.

    Logs exactly how many rows were dropped rather than silently
    discarding them.
    """
    before = len(df)
    deduped = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    dropped = before - len(deduped)
    if dropped:
        logger.info(
            "%s: dropped %d duplicate row(s) on '%s' (kept first occurrence).",
            source_name, dropped, id_col,
        )
    return deduped


def validate_numeric_columns(df: pd.DataFrame, numeric_cols: list[str], source_name: str) -> pd.DataFrame:
    """
    Coerce expected-numeric columns to numeric, logging how many values
    were malformed (became NaN) instead of silently failing. Malformed
    values are set to NaN, not dropped as rows -- downstream feature
    engineering handles NaN explicitly.
    """
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        before_na = df[col].isna().sum()
        coerced = pd.to_numeric(df[col], errors="coerce")
        new_na = coerced.isna().sum() - before_na
        if new_na > 0:
            logger.warning(
                "%s: column '%s' had %d value(s) that could not be parsed as "
                "numeric; set to NaN.", source_name, col, new_na,
            )
        df[col] = coerced
    return df
