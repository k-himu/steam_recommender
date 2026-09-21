"""Unit tests for src.requirements_parsing, built from the exact sample
rows the dataset owner provided (appid 10/20/30 from the Kaggle Steam
Store Games dataset) plus targeted edge cases for malformed input."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import requirements_parsing as rp


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "steam_requirements_sample.csv"


@pytest.fixture
def sample_df():
    return pd.read_csv(FIXTURE_PATH)


def test_parse_requirements_dict_valid():
    raw = "{'minimum': 'Minimum: 4GB RAM', 'recommended': '8GB RAM'}"
    result = rp.parse_requirements_dict(raw)
    assert result["minimum"] == "Minimum: 4GB RAM"
    assert result["recommended"] == "8GB RAM"


def test_parse_requirements_dict_malformed_never_raises():
    assert rp.parse_requirements_dict("{not a valid dict") == {}
    assert rp.parse_requirements_dict(None) == {}
    assert rp.parse_requirements_dict(float("nan")) == {}
    assert rp.parse_requirements_dict("") == {}


def test_parse_requirements_dict_strips_html():
    raw = "{'minimum': '<p><strong>Minimum:</strong> 4GB RAM<br /></p>'}"
    result = rp.parse_requirements_dict(raw)
    assert "<" not in result["minimum"]
    assert "4GB RAM" in result["minimum"]


def test_split_minimum_recommended_with_embedded_tail():
    text = "500 mhz processor, 96mb ram Recommended: 800 mhz processor, 128mb ram"
    min_part, rec_part = rp.split_minimum_recommended(text)
    assert "Recommended" not in min_part
    assert rec_part is not None
    assert "800 mhz" in rec_part


def test_split_minimum_recommended_no_embedded_tail():
    text = "500 mhz processor, 96mb ram"
    min_part, rec_part = rp.split_minimum_recommended(text)
    assert min_part == "500 mhz processor, 96mb ram"
    assert rec_part is None


def test_extract_ram_mb_gb_and_mb():
    assert rp.extract_ram_mb("Requires 4 GB RAM").value == 4096
    assert rp.extract_ram_mb("Requires 512 MB RAM").value == 512
    assert rp.extract_ram_mb("no memory info here").status == rp.ParseStatus.UNCERTAIN
    assert rp.extract_ram_mb("").status == rp.ParseStatus.UNAVAILABLE


def test_extract_os_windows_variants():
    assert rp.extract_os("Windows 10 64-bit").value == "Windows 10"
    assert rp.extract_os("Windows 7 SP1").value == "Windows 7"
    assert rp.extract_os("Windows XP").value == "Windows XP"


def test_extract_os_osx_version():
    field = rp.extract_os("Minimum: OS X Snow Leopard 10.6.3, 1GB RAM")
    assert field.value == "OS X 10.6.3"
    assert field.status == rp.ParseStatus.PARSED


def test_extract_os_unknown_returns_uncertain_not_none_silently():
    field = rp.extract_os("some completely unrelated text with no OS mention")
    assert field.status == rp.ParseStatus.UNCERTAIN
    assert field.value is None


def test_extract_cpu_ghz_and_mhz():
    ghz, raw = rp.extract_cpu("2.5 GHz processor required")
    assert ghz.value == 2.5
    ghz2, raw2 = rp.extract_cpu("500 mhz processor")
    assert ghz2.value == 0.5


def test_extract_cpu_does_not_false_positive_on_gpu_intel_text():
    """Regression test: 'Intel HD 3000' is a GPU, not a CPU, and must not
    be misidentified as CPU text just because it contains 'Intel'."""
    text = "NVIDIA GeForce 8 or higher, ATI X1600, or Intel HD 3000 or higher"
    ghz, raw = rp.extract_cpu(text)
    assert raw.status == rp.ParseStatus.UNCERTAIN
    assert raw.value is None


def test_extract_gpu_and_vram():
    gpu, vram = rp.extract_gpu("NVIDIA GeForce GTX 1060 with 6GB VRAM")
    assert gpu.value is not None
    assert "GeForce GTX 1060" in gpu.value
    assert vram.value == 6144


def test_extract_storage_gb():
    field = rp.extract_storage_gb("Requires 20 GB available space")
    assert field.value == 20.0
    field2 = rp.extract_storage_gb("Requires 500 MB Hard Drive Space")
    assert field2.value == pytest.approx(500 / 1024)


def test_extract_directx():
    field = rp.extract_directx("Requires DirectX Version 11")
    assert field.value == "11"


def test_parse_game_requirements_appid_10(sample_df):
    row = sample_df.iloc[0]
    result = rp.parse_game_requirements(row)
    assert result.appid == 10
    assert result.has_any_requirements is True

    assert result.windows_minimum is not None
    assert result.windows_minimum.os.value == "Windows XP"
    assert result.windows_minimum.ram_mb.value == 96.0
    assert result.windows_minimum.cpu_ghz.value == 0.5

    assert result.windows_recommended is not None
    assert result.windows_recommended.ram_mb.value == 128.0
    assert result.windows_recommended.cpu_ghz.value == 0.8

    assert result.mac_minimum is not None
    assert result.mac_minimum.os.value == "OS X 10.6.3"
    assert result.mac_minimum.storage_gb.value == 4.0

    assert result.linux_minimum is not None
    assert result.linux_minimum.os.value == "Ubuntu 12.04"
    assert result.linux_minimum.cpu_ghz.value == 2.8


def test_parse_game_requirements_missing_row_data():
    """A row with entirely missing requirement text must not crash and
    must report has_any_requirements=False, not fabricate values."""
    row = pd.Series({
        "steam_appid": 999999,
        "pc_requirements": None,
        "mac_requirements": None,
        "linux_requirements": None,
        "minimum": None,
        "recommended": None,
    })
    result = rp.parse_game_requirements(row)
    assert result.has_any_requirements is False
    assert result.windows_minimum is None
    assert result.mac_minimum is None
    assert result.linux_minimum is None
