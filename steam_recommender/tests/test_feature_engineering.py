"""Unit tests for src.feature_engineering."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import feature_engineering as fe


def _minimal_games_df(**overrides):
    base = {
        "appid": [1],
        "name": ["Test Game"],
        "release_date": ["2015-06-01"],
        "developer": ["Dev Studio"],
        "publisher": ["Pub Co"],
        "genres": ["Action;Indie"],
        "steamspy_tags": ["Action;Multiplayer"],
        "categories": ["Single-player"],
        "positive_ratings": [100],
        "negative_ratings": [10],
        "average_playtime": [200],
        "median_playtime": [150],
        "owners": ["20000-50000"],
        "price": [9.99],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_tokenize_delimited_handles_multiword_and_missing():
    assert fe._tokenize_delimited("Action;Massively Multiplayer") == ["action", "massively_multiplayer"]
    assert fe._tokenize_delimited(None) == []
    assert fe._tokenize_delimited(np.nan) == []
    assert fe._tokenize_delimited("") == []


def test_build_unified_text_no_descriptions():
    df = _minimal_games_df()
    text = fe.build_unified_text(df, None)
    assert len(text) == 1
    assert "action" in text.iloc[0]
    assert "dev_studio" in text.iloc[0]


def test_build_unified_text_with_descriptions():
    df = _minimal_games_df(appid=[1])
    desc = pd.DataFrame({
        "steam_appid": [1],
        "short_description": ["An exciting <b>action</b> game!"],
    })
    text = fe.build_unified_text(df, desc)
    assert "exciting" in text.iloc[0]
    assert "<b>" not in text.iloc[0]  # HTML stripped


def test_compute_rating_features_zero_reviews():
    df = _minimal_games_df(positive_ratings=[0], negative_ratings=[0])
    out = fe.compute_rating_features(df)
    assert out["total_reviews"].iloc[0] == 0
    assert np.isnan(out["raw_rating_ratio"].iloc[0])
    # smoothed ratio should still be well-defined (falls back to prior mean)
    assert out["smoothed_rating_ratio"].iloc[0] == pytest.approx(fe.config.RATING_PRIOR_MEAN)


def test_compute_rating_features_smoothing_pulls_low_count_toward_prior():
    """A game with 1 positive / 0 negative should NOT show ratio=1.0 after
    smoothing -- it should be pulled toward the prior mean."""
    df = _minimal_games_df(positive_ratings=[1], negative_ratings=[0])
    out = fe.compute_rating_features(df)
    assert out["raw_rating_ratio"].iloc[0] == 1.0
    assert out["smoothed_rating_ratio"].iloc[0] < 1.0


def test_compute_rating_features_high_volume_close_to_raw():
    """A game with a huge review count should have smoothed ratio very
    close to its raw ratio (smoothing shouldn't distort well-supported data)."""
    df = _minimal_games_df(positive_ratings=[50000], negative_ratings=[500])
    out = fe.compute_rating_features(df)
    raw = out["raw_rating_ratio"].iloc[0]
    smoothed = out["smoothed_rating_ratio"].iloc[0]
    assert abs(raw - smoothed) < 0.01


def test_compute_engagement_features_zero_playtime():
    df = _minimal_games_df(average_playtime=[0], median_playtime=[0])
    out = fe.compute_engagement_features(df)
    assert out["log_average_playtime"].iloc[0] == 0.0
    assert out["log_median_playtime"].iloc[0] == 0.0


def test_compute_popularity_features_valid_and_malformed():
    df = _minimal_games_df(owners=["20000-50000"])
    out = fe.compute_popularity_features(df)
    assert out["owners_estimate"].iloc[0] == 35000

    df_bad = _minimal_games_df(owners=["not-a-range"])
    out_bad = fe.compute_popularity_features(df_bad)
    assert np.isnan(out_bad["owners_estimate"].iloc[0])
    assert out_bad["log_owners_estimate"].iloc[0] == 0.0  # NaN filled to 0 before log


def test_compute_release_age_features_malformed_date():
    df = _minimal_games_df(release_date=["not-a-date"])
    out = fe.compute_release_age_features(df)
    assert pd.isna(out["release_age_days"].iloc[0])


def test_min_max_normalize_constant_series():
    s = pd.Series([5.0, 5.0, 5.0])
    result = fe.min_max_normalize(s)
    assert (result == 0.5).all()


def test_min_max_normalize_all_nan_series():
    s = pd.Series([np.nan, np.nan])
    result = fe.min_max_normalize(s)
    assert (result == 0.5).all()


def test_min_max_normalize_normal_case():
    s = pd.Series([0.0, 5.0, 10.0])
    result = fe.min_max_normalize(s)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.5
    assert result.iloc[2] == 1.0


def test_build_quality_popularity_features_runs_without_crashing_on_edge_rows():
    """A row with every possible edge case at once: zero reviews, zero
    playtime, malformed owners, missing release date."""
    df = _minimal_games_df(
        positive_ratings=[0], negative_ratings=[0],
        average_playtime=[0], median_playtime=[0],
        owners=["garbage"], release_date=["garbage-date"],
    )
    out = fe.build_quality_popularity_features(df)
    assert len(out) == 1  # row not dropped despite every field being degenerate
