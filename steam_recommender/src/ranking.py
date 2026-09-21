"""
Multi-signal ranking engine.

Combines content similarity, hardware compatibility, rating quality,
popularity, and engagement into one final score per candidate game.
The scoring function is deliberately a simple, transparent weighted sum
so it's easy to explain and easy to later replace with a learned model
(see RankingWeights in config.py, and `score_candidates` below, which is
the single seam a learned ranker would replace).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import config
from .hardware_compatibility import CompatibilityLabel, CompatibilityResult

logger = logging.getLogger(__name__)


@dataclass
class RankedGame:
    appid: int
    name: str
    final_score: float
    content_similarity: float
    hardware_score: Optional[float]
    hardware_label: str
    rating_quality_score: float
    popularity_score: float
    engagement_score: float
    raw_row: pd.Series = field(repr=False)


def _hardware_component_score(compat: CompatibilityResult) -> float:
    """
    Convert a CompatibilityResult into a [0, 1] contribution for ranking.
    UNKNOWN compatibility is scored NEUTRAL (0.5), not zero -- missing
    hardware data should not be treated the same as "confirmed
    incompatible". This keeps hardware as a real signal without unfairly
    burying every game that simply lacks parsed requirements.
    """
    if compat.score is not None:
        return compat.score
    return 0.5


def score_candidates(
    candidates: pd.DataFrame,
    content_scores: dict[int, float],
    compatibility_results: dict[int, CompatibilityResult],
    weights: config.RankingWeights = config.RANKING_WEIGHTS,
) -> list[RankedGame]:
    """
    candidates: dataframe of candidate games, already containing the
      normalized quality/popularity/engagement feature columns produced
      by feature_engineering.build_quality_popularity_features(), i.e.
      'norm_rating_quality', 'norm_popularity', 'norm_engagement'.
    content_scores: {appid: cosine_similarity} from the content model.
    compatibility_results: {appid: CompatibilityResult} from the hardware engine.
    """
    ranked: list[RankedGame] = []

    for _, row in candidates.iterrows():
        appid = int(row[config.STEAM_ID_COL])
        content_sim = content_scores.get(appid, 0.0)
        compat = compatibility_results.get(appid)
        hw_score = _hardware_component_score(compat) if compat is not None else 0.5
        hw_label = compat.label.value if compat is not None else CompatibilityLabel.UNKNOWN.value

        rating_quality = float(row.get("norm_rating_quality", 0.0))
        popularity = float(row.get("norm_popularity", 0.0))
        engagement = float(row.get("norm_engagement", 0.0))

        final_score = (
            weights.content_similarity * content_sim
            + weights.hardware_compatibility * hw_score
            + weights.rating_quality * rating_quality
            + weights.popularity * popularity
            + weights.engagement * engagement
        )

        ranked.append(RankedGame(
            appid=appid,
            name=str(row.get("name", "")),
            final_score=final_score,
            content_similarity=content_sim,
            hardware_score=compat.score if compat is not None else None,
            hardware_label=hw_label,
            rating_quality_score=rating_quality,
            popularity_score=popularity,
            engagement_score=engagement,
            raw_row=row,
        ))

    ranked.sort(key=lambda g: g.final_score, reverse=True)
    return ranked


def attach_normalized_ranking_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds norm_rating_quality, norm_popularity, norm_engagement columns
    (each in [0, 1], min-max normalized over the full candidate set passed
    in). Expects df already has smoothed_rating_ratio, log_owners_estimate,
    log_average_playtime, log_median_playtime from feature_engineering.
    """
    from .feature_engineering import min_max_normalize

    df = df.copy()
    df["norm_rating_quality"] = min_max_normalize(df["smoothed_rating_ratio"])
    df["norm_popularity"] = min_max_normalize(df["log_owners_estimate"])
    engagement_raw = df["log_average_playtime"].fillna(0) + df["log_median_playtime"].fillna(0)
    df["norm_engagement"] = min_max_normalize(engagement_raw)
    return df
