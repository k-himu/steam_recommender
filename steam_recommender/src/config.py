"""
Central configuration for the Steam Recommendation System.

All tunable parameters, file paths, and ranking weights live here so that
no module contains hard-coded magic numbers. Every weight below is a
starting point, not a scientifically optimized value -- see the "Why these
weights" note beneath RANKING_WEIGHTS for the reasoning behind each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STEAM_CSV_PATH = DATA_RAW_DIR / "steam.csv"
DESCRIPTION_CSV_PATH = DATA_RAW_DIR / "steam_description_data.csv"
REQUIREMENTS_CSV_PATH = DATA_RAW_DIR / "steam_requirements_data.csv"

UNIFIED_DATASET_PATH = DATA_PROCESSED_DIR / "unified_games.parquet"
UNIFIED_DATASET_CSV_FALLBACK = DATA_PROCESSED_DIR / "unified_games.csv"
TFIDF_MATRIX_PATH = DATA_PROCESSED_DIR / "tfidf_matrix.npz"
TFIDF_VOCAB_PATH = DATA_PROCESSED_DIR / "tfidf_vectorizer.pkl"

PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"  # Phase 2 (future), created lazily


# --------------------------------------------------------------------------
# Join keys (verified against actual files, not assumed)
# --------------------------------------------------------------------------

STEAM_ID_COL = "appid"
DESCRIPTION_ID_COL = "steam_appid"
REQUIREMENTS_ID_COL = "steam_appid"


# --------------------------------------------------------------------------
# Text vectorization (content model)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TfidfConfig:
    max_features: int = 30_000
    ngram_range: tuple = (1, 2)
    min_df: int = 2          # ignore terms that appear in fewer than 2 games
    max_df: float = 0.85     # ignore terms that appear in >85% of games (too generic)
    sublinear_tf: bool = True


TFIDF_CONFIG = TfidfConfig()

# Weight multipliers applied when building the unified text blob, so that
# structured, curated fields (genres/tags/categories) count for more than
# raw free-text description, which is noisier and more verbose.
# Implemented by repeating the field's tokens this many times before
# joining into the final text blob (a standard, transparent way to bias
# TF-IDF weight without a custom kernel).
TEXT_FIELD_REPEATS = {
    "genres": 3,
    "steamspy_tags": 3,
    "categories": 2,
    "developer": 1,
    "publisher": 1,
    "short_description": 1,
}


# --------------------------------------------------------------------------
# Quality / popularity signal engineering
# --------------------------------------------------------------------------

# Bayesian smoothing for rating ratio: treats every game as if it started
# with RATING_PRIOR_COUNT "neutral" votes at RATING_PRIOR_MEAN positivity.
# This prevents a game with 1 positive / 0 negative ratings from outranking
# a game with 50,000 positive / 500 negative ratings.
RATING_PRIOR_COUNT = 20
RATING_PRIOR_MEAN = 0.75  # Steam-wide ratings skew positive; used only as a shrinkage target


# --------------------------------------------------------------------------
# Ranking weights
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RankingWeights:
    content_similarity: float = 0.40
    hardware_compatibility: float = 0.20
    rating_quality: float = 0.20
    popularity: float = 0.10
    engagement: float = 0.10


RANKING_WEIGHTS = RankingWeights()

# --- Why these starting weights? ---
# content_similarity (0.40): the system's stated purpose is "games similar to
#   one you like" -- this must remain the dominant signal or the system turns
#   into a generic "best rated games" list, which the spec explicitly says
#   to avoid.
# hardware_compatibility (0.20): a strong secondary constraint -- a game the
#   user's PC can't run is a bad recommendation regardless of how similar or
#   good it is, but it should not completely override content relevance when
#   compatibility is merely "unknown" (missing data is not the same as
#   "incompatible").
# rating_quality (0.20): reflects whether the game is well-regarded, using
#   the smoothed positive ratio, not raw counts.
# popularity (0.10) and engagement (0.10): secondary signals (owners,
#   playtime) that add weight to established, actively-played titles without
#   dominating over relevance or quality.
# These are exposed as a dataclass specifically so they can be overridden
# at call time (see RecommenderConfig) or replaced by a learned model later.


# --------------------------------------------------------------------------
# Hardware compatibility scoring
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CompatibilityThresholds:
    # Internal compatibility score in [0, 1] mapped to a human label.
    compatible_min: float = 0.85
    likely_compatible_min: float = 0.65
    borderline_min: float = 0.40
    # below borderline_min -> NOT_RECOMMENDED
    # score is None (nothing parseable) -> UNKNOWN


COMPATIBILITY_THRESHOLDS = CompatibilityThresholds()


# --------------------------------------------------------------------------
# Recommendation defaults
# --------------------------------------------------------------------------

DEFAULT_TOP_N = 10
MAX_CANDIDATE_POOL = 300  # how many content-similar games to consider before ranking
RANDOM_SEED = 42


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
