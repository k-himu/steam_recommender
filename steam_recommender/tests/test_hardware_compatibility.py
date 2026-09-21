"""Unit tests for src.hardware_compatibility."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import hardware_compatibility as hc
from src import requirements_parsing as rp


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "steam_requirements_sample.csv"


@pytest.fixture
def cs_requirements():
    """Counter-Strike (appid 10): very low, old requirements
    (500mhz CPU, 96MB RAM, Windows XP)."""
    df = pd.read_csv(FIXTURE_PATH)
    return rp.parse_game_requirements(df.iloc[0])


def test_modern_high_end_pc_is_compatible_with_old_game(cs_requirements):
    """A modern gaming PC should comfortably exceed CS 1.6's ancient
    minimum requirements -> COMPATIBLE."""
    user = hc.UserHardwareSpec(
        os="Windows 10", cpu_ghz=3.6, ram_mb=16384, gpu_vram_mb=8192, storage_gb=500,
    )
    result = hc.evaluate_compatibility(user, cs_requirements, game_platforms=["windows", "mac", "linux"])
    assert result.label == hc.CompatibilityLabel.COMPATIBLE
    assert result.score is not None
    assert result.score > 0.9


def test_underpowered_pc_is_not_recommended(cs_requirements):
    """A PC below the minimum RAM/CPU should fail -- even against an old
    game's low bar -- and be labeled accordingly rather than compatible."""
    user = hc.UserHardwareSpec(
        os="Windows XP", cpu_ghz=0.1, ram_mb=32, gpu_vram_mb=4, storage_gb=0.01,
    )
    result = hc.evaluate_compatibility(user, cs_requirements)
    assert result.label in (hc.CompatibilityLabel.NOT_RECOMMENDED, hc.CompatibilityLabel.BORDERLINE)
    assert result.score is not None
    assert result.score < 0.85


def test_no_requirements_data_returns_unknown_not_compatible():
    """Never claim a game runs when there is no data to determine that."""
    empty_reqs = rp.GameRequirements(appid=1, has_any_requirements=False)
    user = hc.UserHardwareSpec(os="Windows 10", cpu_ghz=3.6, ram_mb=16384)
    result = hc.evaluate_compatibility(user, empty_reqs)
    assert result.label == hc.CompatibilityLabel.UNKNOWN
    assert result.score is None


def test_no_user_hardware_returns_unknown(cs_requirements):
    user = hc.UserHardwareSpec()  # all fields None
    result = hc.evaluate_compatibility(user, cs_requirements)
    assert result.label == hc.CompatibilityLabel.UNKNOWN
    assert result.score is None


def test_none_game_requirements_returns_unknown():
    user = hc.UserHardwareSpec(os="Windows 10", ram_mb=16384)
    result = hc.evaluate_compatibility(user, None)
    assert result.label == hc.CompatibilityLabel.UNKNOWN
    assert result.score is None


def test_partial_user_spec_excludes_unknown_fields_from_score(cs_requirements):
    """User only provides RAM; other fields should be 'unknown_user_value'
    and excluded from scoring, not treated as failures."""
    user = hc.UserHardwareSpec(ram_mb=16384)
    result = hc.evaluate_compatibility(user, cs_requirements)
    unknown_user_fields = [c for c in result.comparisons if c.result == "unknown_user_value"]
    assert len(unknown_user_fields) >= 1
    ram_comparison = next(c for c in result.comparisons if c.field_name == "RAM (MB)")
    assert ram_comparison.result == "meets"


def test_mac_platform_uses_mac_requirements_when_available(cs_requirements):
    user = hc.UserHardwareSpec(os="macOS", ram_mb=8192, storage_gb=100)
    result = hc.evaluate_compatibility(user, cs_requirements)
    assert result.platform_used == "mac"


def test_explain_produces_readable_strings(cs_requirements):
    user = hc.UserHardwareSpec(os="Windows 10", cpu_ghz=3.6, ram_mb=16384)
    result = hc.evaluate_compatibility(user, cs_requirements)
    explanation = result.explain()
    assert isinstance(explanation, list)
    assert all(isinstance(line, str) for line in explanation)
    assert len(explanation) > 0


def test_compatibility_never_claims_compatible_with_mostly_unknown_fields():
    """If a game has requirement text but almost nothing parses, the
    system should not confidently claim COMPATIBLE."""
    reqs = rp.GameRequirements(
        appid=2,
        windows_minimum=rp.parse_requirement_text("Some vague unparseable requirement text"),
        has_any_requirements=True,
    )
    user = hc.UserHardwareSpec(os="Windows 10", ram_mb=16384)
    result = hc.evaluate_compatibility(user, reqs)
    # RAM requirement is unparseable here, OS is unparseable too -> score should be None (unknown)
    assert result.label == hc.CompatibilityLabel.UNKNOWN or result.score is None or result.score < 0.85
