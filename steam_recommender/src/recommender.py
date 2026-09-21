"""
SteamRecommender: the top-level, UI-independent recommendation engine.

USER INPUT -> REFERENCE GAME -> CONTENT CANDIDATES -> HARDWARE FILTER/SCORE
-> QUALITY/POPULARITY -> FINAL RANKING -> EXPLANATIONS -> TOP-N

This class is usable entirely independently of any UI (CLI, Streamlit,
notebook, or future API) -- see app/cli.py and app/streamlit_app.py for
two different front ends built on top of the exact same class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from .content_similarity import ContentModel, TfidfContentModel
from .hardware_compatibility import (
    CompatibilityResult, UserHardwareSpec, evaluate_compatibility,
)
from .explanations import Explanation, build_explanation
from .ranking import RankedGame, attach_normalized_ranking_columns, score_candidates
from .requirements_parsing import GameRequirements

logger = logging.getLogger(__name__)


@dataclass
class RecommendationFilters:
    genres: Optional[list[str]] = None
    platforms: Optional[list[str]] = None          # e.g. ["windows"]
    max_price: Optional[float] = None
    free_only: bool = False
    paid_only: bool = False
    min_compatibility: Optional[str] = None         # one of CompatibilityLabel values
    release_after: Optional[str] = None              # "YYYY-MM-DD"
    release_before: Optional[str] = None


@dataclass
class RecommendationRequest:
    reference_appid: int
    top_n: int = config.DEFAULT_TOP_N
    user_hardware: UserHardwareSpec = field(default_factory=UserHardwareSpec)
    filters: RecommendationFilters = field(default_factory=RecommendationFilters)
    weights: config.RankingWeights = config.RANKING_WEIGHTS
    # Phase 2 hook (not used in Phase 1): a future personalized recommender
    # would accept a profile.UserProfile here instead of a single reference game.


@dataclass
class Recommendation:
    ranked_game: RankedGame
    explanation: Explanation
    compatibility: CompatibilityResult


class SteamRecommender:
    def __init__(
        self,
        games: pd.DataFrame,
        requirements_map: dict[int, GameRequirements],
        content_model: Optional[ContentModel] = None,
    ):
        self.games = games.reset_index(drop=True)
        self.requirements_map = requirements_map
        self._by_appid = self.games.set_index(config.STEAM_ID_COL, drop=False)
        self.content_model = content_model or self._fit_default_content_model()

    def _fit_default_content_model(self) -> ContentModel:
        if "content_text" not in self.games.columns:
            raise ValueError(
                "games dataframe has no 'content_text' column -- run "
                "feature_engineering.build_unified_text first, or pass a "
                "pre-fit content_model."
            )
        model = TfidfContentModel()
        model.fit(self.games["content_text"], self.games[config.STEAM_ID_COL])
        return model

    # ----------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------

    def search_games(self, query: str, limit: int = 15) -> pd.DataFrame:
        """Case-insensitive substring search over game names."""
        if not query or not query.strip():
            return self.games.iloc[0:0]
        mask = self.games["name"].str.contains(query.strip(), case=False, na=False, regex=False)
        return self.games[mask].head(limit)

    def get_game(self, appid: int) -> Optional[pd.Series]:
        if appid not in self._by_appid.index:
            return None
        return self._by_appid.loc[appid]

    # ----------------------------------------------------------------
    # Filtering
    # ----------------------------------------------------------------

    def _apply_filters(self, df: pd.DataFrame, filters: RecommendationFilters) -> pd.DataFrame:
        out = df

        if filters.genres:
            wanted = {g.lower() for g in filters.genres}
            out = out[out["genres"].fillna("").apply(
                lambda g: bool(wanted & {t.strip().lower() for t in g.split(";")})
            )]

        if filters.platforms:
            wanted_p = {p.lower() for p in filters.platforms}
            out = out[out["platforms"].fillna("").apply(
                lambda p: bool(wanted_p & {t.strip().lower() for t in p.split(";")})
            )]

        if filters.max_price is not None:
            out = out[out["price"] <= filters.max_price]

        if filters.free_only:
            out = out[out["price"] == 0]
        elif filters.paid_only:
            out = out[out["price"] > 0]

        if filters.release_after:
            out = out[out["release_date"] >= filters.release_after]
        if filters.release_before:
            out = out[out["release_date"] <= filters.release_before]

        return out

    # ----------------------------------------------------------------
    # Main recommendation flow
    # ----------------------------------------------------------------

    def recommend(self, request: RecommendationRequest) -> list[Recommendation]:
        reference_row = self.get_game(request.reference_appid)
        if reference_row is None:
            raise ValueError(f"No game found with appid {request.reference_appid}")

        # --- Candidate generation (content similarity) ---
        similar = self.content_model.most_similar(
            request.reference_appid, top_k=config.MAX_CANDIDATE_POOL
        )
        if not similar:
            logger.warning(
                "No content-similar candidates found for appid %s (%s). "
                "This can happen for games with very sparse/empty metadata.",
                request.reference_appid, reference_row.get("name"),
            )
            return []

        content_scores = dict(similar)
        candidate_ids = list(content_scores.keys())
        candidates = self.games[self.games[config.STEAM_ID_COL].isin(candidate_ids)].copy()

        # --- Filters ---
        candidates = self._apply_filters(candidates, request.filters)
        if candidates.empty:
            return []

        # --- Hardware compatibility ---
        compatibility_results: dict[int, CompatibilityResult] = {}
        for appid in candidates[config.STEAM_ID_COL]:
            game_reqs = self.requirements_map.get(int(appid))
            game_row = self._by_appid.loc[int(appid)] if int(appid) in self._by_appid.index else None
            game_platforms = None
            if game_row is not None and isinstance(game_row.get("platforms"), str):
                game_platforms = game_row["platforms"].split(";")
            compatibility_results[int(appid)] = evaluate_compatibility(
                request.user_hardware, game_reqs, game_platforms
            )

        if request.filters.min_compatibility:
            from .hardware_compatibility import CompatibilityLabel
            order = [
                CompatibilityLabel.NOT_RECOMMENDED, CompatibilityLabel.UNKNOWN,
                CompatibilityLabel.BORDERLINE, CompatibilityLabel.LIKELY_COMPATIBLE,
                CompatibilityLabel.COMPATIBLE,
            ]
            min_label = CompatibilityLabel(request.filters.min_compatibility)
            min_rank = order.index(min_label)
            allowed_ids = {
                appid for appid, result in compatibility_results.items()
                if order.index(result.label) >= min_rank
            }
            candidates = candidates[candidates[config.STEAM_ID_COL].isin(allowed_ids)]
            if candidates.empty:
                return []

        # --- Quality/popularity normalization (over this candidate set) ---
        candidates = attach_normalized_ranking_columns(candidates)

        # --- Final ranking ---
        ranked = score_candidates(
            candidates, content_scores, compatibility_results, request.weights
        )
        ranked = ranked[: request.top_n]

        # --- Explanations ---
        recommendations = []
        for ranked_game in ranked:
            compat_result = compatibility_results.get(ranked_game.appid)
            explanation = build_explanation(ranked_game, reference_row, compat_result)
            recommendations.append(Recommendation(
                ranked_game=ranked_game,
                explanation=explanation,
                compatibility=compat_result,
            ))

        return recommendations
