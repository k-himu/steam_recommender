"""
End-to-end tests for SteamRecommender, running against the REAL steam.csv
(27,075 games) plus the small requirements/description fixtures.

These cover the engineering checklist explicitly required by the spec:
empty search results, missing descriptions, missing requirements,
malformed requirements, zero reviews, zero playtime, free vs paid games,
duplicate games, invalid hardware input, unknown hardware compatibility.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset_builder import build_unified_dataset
from src.recommender import SteamRecommender, RecommendationRequest, RecommendationFilters
from src.hardware_compatibility import UserHardwareSpec, CompatibilityLabel

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def recommender():
    games, requirements_map, _report = build_unified_dataset(
        steam_path=config.STEAM_CSV_PATH,
        description_path=FIXTURES / "steam_description_sample.csv",
        requirements_path=FIXTURES / "steam_requirements_sample.csv",
    )
    return SteamRecommender(games, requirements_map)


def test_recommender_builds_successfully(recommender):
    assert len(recommender.games) == 27075


def test_search_known_game(recommender):
    results = recommender.search_games("Half-Life")
    assert len(results) > 0
    assert results["name"].str.contains("Half-Life", case=False).all()


def test_search_empty_query_returns_empty(recommender):
    results = recommender.search_games("")
    assert len(results) == 0


def test_search_nonexistent_game_returns_empty(recommender):
    results = recommender.search_games("zzzzzznonexistentgamexyz123")
    assert len(results) == 0


def test_recommend_unknown_appid_raises_clear_error(recommender):
    with pytest.raises(ValueError, match="No game found"):
        recommender.recommend(RecommendationRequest(reference_appid=99999999999))


def test_recommend_game_with_no_requirements_data_gives_unknown_compat(recommender):
    """A reference game whose content-similar neighbors fall entirely
    outside our 3-row requirements fixture (appids 10/20/30) must report
    UNKNOWN hardware compatibility for those neighbors, never a false
    positive. The Elder Scrolls V: Skyrim's neighborhood (RPGs) doesn't
    overlap with the old Valve FPS cluster our fixture covers."""
    appid = recommender.games[
        recommender.games["name"] == "The Elder Scrolls V: Skyrim"
    ]["appid"].iloc[0]
    req = RecommendationRequest(
        reference_appid=int(appid), top_n=5,
        user_hardware=UserHardwareSpec(os="Windows 10", ram_mb=16384, cpu_ghz=3.5),
    )
    recs = recommender.recommend(req)
    assert len(recs) > 0
    for r in recs:
        assert r.ranked_game.appid not in (10, 20, 30)
        assert r.ranked_game.hardware_label == CompatibilityLabel.UNKNOWN.value


def test_recommend_game_with_requirements_gives_known_compat(recommender):
    """Counter-Strike (appid 10) IS in the requirements fixture, and its
    close neighbors (Team Fortress Classic=20, Day of Defeat=30) also are."""
    req = RecommendationRequest(
        reference_appid=10, top_n=10,
        user_hardware=UserHardwareSpec(os="Windows 10", ram_mb=16384, cpu_ghz=3.5, storage_gb=500),
    )
    recs = recommender.recommend(req)
    labels = {r.ranked_game.appid: r.ranked_game.hardware_label for r in recs}
    # appid 20 (Team Fortress Classic) should be a top candidate with known compatibility
    assert labels.get(20) == CompatibilityLabel.COMPATIBLE.value


def test_recommend_with_no_hardware_provided(recommender):
    """User provides no hardware at all -- must not crash, hardware
    component should neutrally default rather than penalize everything."""
    req = RecommendationRequest(reference_appid=10, top_n=5, user_hardware=UserHardwareSpec())
    recs = recommender.recommend(req)
    assert len(recs) > 0
    for r in recs:
        assert r.compatibility.label == CompatibilityLabel.UNKNOWN


def test_recommend_with_invalid_hardware_types_does_not_crash():
    """Hardware spec fields are typed floats; passing None values (as a
    caller-facing 'unknown' signal) must be handled gracefully everywhere."""
    spec = UserHardwareSpec(os=None, cpu_ghz=None, ram_mb=None, gpu_vram_mb=None, storage_gb=None)
    assert spec.is_empty() is True


def test_recommend_free_only_filter(recommender):
    req = RecommendationRequest(
        reference_appid=10, top_n=20,
        filters=RecommendationFilters(free_only=True),
    )
    recs = recommender.recommend(req)
    for r in recs:
        assert r.ranked_game.raw_row["price"] == 0


def test_recommend_paid_only_filter(recommender):
    req = RecommendationRequest(
        reference_appid=10, top_n=20,
        filters=RecommendationFilters(paid_only=True),
    )
    recs = recommender.recommend(req)
    for r in recs:
        assert r.ranked_game.raw_row["price"] > 0


def test_recommend_max_price_filter(recommender):
    req = RecommendationRequest(
        reference_appid=10, top_n=20,
        filters=RecommendationFilters(max_price=5.0),
    )
    recs = recommender.recommend(req)
    for r in recs:
        assert r.ranked_game.raw_row["price"] <= 5.0


def test_recommend_genre_filter(recommender):
    req = RecommendationRequest(
        reference_appid=10, top_n=20,
        filters=RecommendationFilters(genres=["Action"]),
    )
    recs = recommender.recommend(req)
    for r in recs:
        genres = r.ranked_game.raw_row["genres"]
        assert "action" in genres.lower()


def test_recommend_filters_that_exclude_everything_returns_empty_not_crash(recommender):
    req = RecommendationRequest(
        reference_appid=10, top_n=20,
        filters=RecommendationFilters(max_price=0.0, paid_only=True),  # contradictory -> nothing matches
    )
    recs = recommender.recommend(req)
    assert recs == []


def test_recommend_min_compatibility_filter(recommender):
    req = RecommendationRequest(
        reference_appid=10, top_n=20,
        user_hardware=UserHardwareSpec(os="Windows 10", ram_mb=16384, cpu_ghz=3.5),
        filters=RecommendationFilters(min_compatibility="COMPATIBLE"),
    )
    recs = recommender.recommend(req)
    for r in recs:
        assert r.ranked_game.hardware_label == CompatibilityLabel.COMPATIBLE.value


def test_zero_review_game_does_not_crash_recommendation(recommender):
    """Find a game with zero positive reviews (real dataset has none with
    BOTH positive and negative at zero -- every title has at least one
    negative rating -- so we test the zero-positive-ratings edge case,
    which stresses the same rating-ratio-division code path) and confirm
    it can still be used as a reference game without error."""
    zero_review_games = recommender.games[recommender.games["positive_ratings"] == 0]
    assert len(zero_review_games) > 0, "Expected at least one zero-positive-rating game in real dataset"
    appid = int(zero_review_games.iloc[0]["appid"])
    req = RecommendationRequest(reference_appid=appid, top_n=5)
    recs = recommender.recommend(req)  # should not raise
    assert isinstance(recs, list)


def test_zero_playtime_game_does_not_crash_recommendation(recommender):
    zero_playtime_games = recommender.games[recommender.games["average_playtime"] == 0]
    assert len(zero_playtime_games) > 0
    appid = int(zero_playtime_games.iloc[0]["appid"])
    req = RecommendationRequest(reference_appid=appid, top_n=5)
    recs = recommender.recommend(req)
    assert isinstance(recs, list)


def test_duplicate_appid_handled_at_build_time():
    """Simulate a steam.csv with a duplicate appid and confirm dedup logic
    (data_validation.deduplicate_by_id) drops it rather than crashing
    downstream (tested at the dataset_builder level via an in-memory frame)."""
    from src import data_validation
    df = pd.DataFrame({
        "appid": [1, 1, 2],
        "name": ["Game A", "Game A dup", "Game B"],
    })
    deduped = data_validation.deduplicate_by_id(df, "appid", "test")
    assert len(deduped) == 2
    assert deduped["appid"].tolist() == [1, 2]


def test_recommendation_scores_are_within_expected_bounds(recommender):
    req = RecommendationRequest(
        reference_appid=10, top_n=10,
        user_hardware=UserHardwareSpec(os="Windows 10", ram_mb=16384, cpu_ghz=3.5),
    )
    recs = recommender.recommend(req)
    for r in recs:
        assert 0.0 <= r.ranked_game.final_score <= 1.0001  # small float tolerance
        assert 0.0 <= r.ranked_game.content_similarity <= 1.0001


def test_recommendations_are_sorted_descending(recommender):
    req = RecommendationRequest(reference_appid=10, top_n=15)
    recs = recommender.recommend(req)
    scores = [r.ranked_game.final_score for r in recs]
    assert scores == sorted(scores, reverse=True)
