"""
Feature engineering: builds the unified content-text representation and
the normalized quality/popularity numeric signals used for ranking.

Nothing here invents data. Every feature is derived from columns that
were verified to exist in Phase 1 (data_loading / data_validation).
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from . import config
from .text_utils import clean_text, strip_html, normalize_whitespace

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Unified text blob for TF-IDF
# --------------------------------------------------------------------------

def _tokenize_delimited(value: object) -> list[str]:
    """Split a semicolon-delimited categorical field into clean tokens."""
    if not isinstance(value, str) or not value.strip():
        return []
    tokens = [t.strip() for t in value.split(";") if t.strip()]
    # Replace internal spaces with underscores so multi-word tags/genres
    # ("Massively Multiplayer") stay as a single TF-IDF token instead of
    # being split into generic words that dilute meaning.
    return [re.sub(r"\s+", "_", t.lower()) for t in tokens]


def build_unified_text(
    games: pd.DataFrame,
    descriptions: pd.DataFrame | None,
) -> pd.Series:
    """
    Build one text blob per game for TF-IDF, combining genres, tags,
    categories, developer, publisher, and (if available) short_description.

    Fields are repeated per TEXT_FIELD_REPEATS to bias TF-IDF weight
    toward curated metadata over free-text description, without requiring
    a custom similarity kernel.
    """
    merged = games.copy()

    if descriptions is not None:
        desc_slim = descriptions[[config.DESCRIPTION_ID_COL, "short_description"]].copy()
        desc_slim = desc_slim.rename(columns={config.DESCRIPTION_ID_COL: config.STEAM_ID_COL})
        desc_slim["short_description"] = desc_slim["short_description"].fillna("")
        merged = merged.merge(desc_slim, on=config.STEAM_ID_COL, how="left")
        merged["short_description"] = merged["short_description"].fillna("")
    else:
        merged["short_description"] = ""

    parts = []
    for _, row in merged.iterrows():
        blob_tokens: list[str] = []

        genres_tokens = _tokenize_delimited(row.get("genres"))
        tags_tokens = _tokenize_delimited(row.get("steamspy_tags"))
        categories_tokens = _tokenize_delimited(row.get("categories"))

        blob_tokens += genres_tokens * config.TEXT_FIELD_REPEATS["genres"]
        blob_tokens += tags_tokens * config.TEXT_FIELD_REPEATS["steamspy_tags"]
        blob_tokens += categories_tokens * config.TEXT_FIELD_REPEATS["categories"]

        developer = row.get("developer")
        if isinstance(developer, str) and developer.strip():
            dev_tokens = [re.sub(r"\s+", "_", d.strip().lower()) for d in developer.split(";") if d.strip()]
            blob_tokens += dev_tokens * config.TEXT_FIELD_REPEATS["developer"]

        publisher = row.get("publisher")
        if isinstance(publisher, str) and publisher.strip():
            pub_tokens = [re.sub(r"\s+", "_", p.strip().lower()) for p in publisher.split(";") if p.strip()]
            blob_tokens += pub_tokens * config.TEXT_FIELD_REPEATS["publisher"]

        desc = row.get("short_description", "")
        desc_clean = clean_text(desc) if desc else ""

        text_blob = " ".join(blob_tokens)
        if desc_clean:
            text_blob = text_blob + " " + (desc_clean + " ") * config.TEXT_FIELD_REPEATS["short_description"]

        parts.append(text_blob.strip())

    return pd.Series(parts, index=games.index, name="content_text")


# --------------------------------------------------------------------------
# Quality / popularity numeric signals
# --------------------------------------------------------------------------

def compute_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      total_reviews          = positive + negative
      raw_rating_ratio        = positive / total_reviews (NaN if 0 reviews)
      smoothed_rating_ratio  = Bayesian-smoothed ratio, robust to low review counts
      log_total_reviews      = log1p(total_reviews), for scale-robust ranking use
    """
    df = df.copy()
    df["total_reviews"] = df["positive_ratings"].fillna(0) + df["negative_ratings"].fillna(0)

    with np.errstate(invalid="ignore", divide="ignore"):
        df["raw_rating_ratio"] = np.where(
            df["total_reviews"] > 0,
            df["positive_ratings"] / df["total_reviews"],
            np.nan,
        )

    prior_count = config.RATING_PRIOR_COUNT
    prior_mean = config.RATING_PRIOR_MEAN
    df["smoothed_rating_ratio"] = (
        df["positive_ratings"].fillna(0) + prior_count * prior_mean
    ) / (df["total_reviews"] + prior_count)

    df["log_total_reviews"] = np.log1p(df["total_reviews"])
    return df


def compute_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      log_average_playtime, log_median_playtime -- log1p transforms so a
      handful of extreme-playtime titles don't dominate the scale.
    """
    df = df.copy()
    df["log_average_playtime"] = np.log1p(df["average_playtime"].fillna(0).clip(lower=0))
    df["log_median_playtime"] = np.log1p(df["median_playtime"].fillna(0).clip(lower=0))
    return df


def compute_popularity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the 'owners' range string ("20000-50000") into a numeric
    midpoint estimate, then log-transforms it. Malformed owners strings
    become NaN rather than crashing or being silently zeroed.
    """
    df = df.copy()

    def _owners_midpoint(value: object) -> float:
        if not isinstance(value, str):
            return np.nan
        match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", value)
        if not match:
            return np.nan
        low, high = int(match.group(1)), int(match.group(2))
        return (low + high) / 2

    df["owners_estimate"] = df["owners"].apply(_owners_midpoint)
    n_bad = df["owners_estimate"].isna().sum()
    if n_bad:
        logger.warning("%d row(s) had an unparseable 'owners' value; owners_estimate set to NaN.", n_bad)
    df["log_owners_estimate"] = np.log1p(df["owners_estimate"].fillna(0))
    return df


def compute_release_age_features(df: pd.DataFrame, reference_date: str | None = None) -> pd.DataFrame:
    """Adds release_age_days (days between release_date and reference_date,
    default = max release_date in the dataset, so age is relative and
    doesn't depend on wall-clock 'today')."""
    df = df.copy()
    release_dt = pd.to_datetime(df["release_date"], errors="coerce")
    n_bad = release_dt.isna().sum()
    if n_bad:
        logger.warning("%d row(s) had an unparseable release_date.", n_bad)

    ref = pd.to_datetime(reference_date) if reference_date else release_dt.max()
    df["release_age_days"] = (ref - release_dt).dt.days
    return df


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Scale to [0, 1]; a constant series (or all-NaN) maps to 0.5
    everywhere (neutral, avoids divide-by-zero) rather than NaN or crash."""
    valid = series.dropna()
    if valid.empty or valid.max() == valid.min():
        return pd.Series(0.5, index=series.index)
    scaled = (series - valid.min()) / (valid.max() - valid.min())
    return scaled.fillna(0.0)  # missing values treated as "no signal" -> lowest normalized value


def build_quality_popularity_features(games: pd.DataFrame) -> pd.DataFrame:
    """Run the full quality/popularity feature pipeline and return the
    augmented dataframe (original columns + new numeric feature columns)."""
    df = games.copy()
    df = compute_rating_features(df)
    df = compute_engagement_features(df)
    df = compute_popularity_features(df)
    df = compute_release_age_features(df)
    return df
