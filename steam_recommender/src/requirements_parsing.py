"""
System requirements parsing.

The requirements dataset stores requirements two ways:
  1. `pc_requirements` / `mac_requirements` / `linux_requirements`: a
     stringified Python dict, e.g. "{'minimum': '...html...', 'recommended': '...'}"
  2. `minimum` / `recommended`: pre-split plain(er) text columns, derived
     from pc_requirements by whoever built the dataset -- but NOT reliably
     clean (observed: the 'minimum' column can still contain an embedded
     "Recommended: ..." tail, and `recommended` is null in ~48% of rows).

This module is intentionally conservative: every extracted field carries a
ParseStatus so downstream code (hardware compatibility) never has to guess
whether a `None` means "game has no requirement" or "we couldn't parse it".
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .text_utils import strip_html, normalize_whitespace

logger = logging.getLogger(__name__)


class ParseStatus(str, Enum):
    PARSED = "parsed"
    UNAVAILABLE = "unavailable"   # no source text to parse at all
    UNCERTAIN = "uncertain"       # text present but pattern didn't confidently match


@dataclass
class ParsedField:
    value: Optional[object]
    status: ParseStatus
    raw_snippet: Optional[str] = None  # the text fragment the value was extracted from, for auditability

    def is_known(self) -> bool:
        return self.status == ParseStatus.PARSED and self.value is not None


@dataclass
class ParsedRequirementSet:
    """One tier (minimum or recommended) of requirements for one platform."""
    os: ParsedField
    cpu_ghz: ParsedField          # numeric GHz if extractable, else uncertain
    cpu_raw: ParsedField          # raw CPU text snippet, always kept if any CPU text found
    ram_mb: ParsedField
    gpu_raw: ParsedField
    gpu_vram_mb: ParsedField
    storage_gb: ParsedField
    directx: ParsedField
    source_text: str = ""


# --------------------------------------------------------------------------
# Step 1: get plain text out of the raw dataset fields
# --------------------------------------------------------------------------

def parse_requirements_dict(raw: object) -> dict[str, str]:
    """
    Parse a stringified-dict field like pc_requirements / mac_requirements /
    linux_requirements into {'minimum': ..., 'recommended': ...} plain text
    (HTML stripped). Returns {} if the field is missing or malformed --
    never raises, always logs the failure.
    """
    if raw is None or (isinstance(raw, float)):  # NaN
        return {}
    if not isinstance(raw, str) or not raw.strip():
        return {}

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        logger.debug("Could not literal_eval requirements dict: %.80s...", raw)
        return {}

    if not isinstance(parsed, dict):
        return {}

    result = {}
    for key, value in parsed.items():
        if isinstance(value, str):
            result[key] = normalize_whitespace(strip_html(value))
    return result


_EMBEDDED_RECOMMENDED_RE = re.compile(
    r"\bRecommended\s*:\s*", re.IGNORECASE
)
_EMBEDDED_MINIMUM_RE = re.compile(
    r"\bMinimum\s*:\s*", re.IGNORECASE
)


def split_minimum_recommended(text: str) -> tuple[str, Optional[str]]:
    """
    Some 'minimum' text fields observed in this dataset still contain an
    embedded 'Recommended: ...' tail (the source data was not perfectly
    separated upstream). Split it out if present so it doesn't pollute
    minimum-tier parsing.

    Returns (minimum_text, recommended_text_or_None).
    """
    if not text:
        return "", None

    # Strip a leading "Minimum:" label if present.
    text = _EMBEDDED_MINIMUM_RE.sub("", text, count=1).strip()

    match = _EMBEDDED_RECOMMENDED_RE.search(text)
    if match:
        min_part = text[: match.start()].strip()
        rec_part = text[match.end():].strip()
        return min_part, (rec_part or None)
    return text, None


# --------------------------------------------------------------------------
# Step 2: field-level regex extractors
# --------------------------------------------------------------------------

# Order matters: more specific patterns first.
_OS_PATTERNS = [
    (r"windows\s*10", "Windows 10"),
    (r"windows\s*8\.?1", "Windows 8.1"),
    (r"windows\s*8", "Windows 8"),
    (r"windows\s*7", "Windows 7"),
    (r"windows\s*vista", "Windows Vista"),
    (r"windows\s*xp", "Windows XP"),
    (r"macos", "macOS"),
    (r"mac\s*os\s*x", "Mac OS X"),
    (r"ubuntu\s*[\d.]*", None),
    (r"linux", "Linux"),
    (r"steamos", "SteamOS"),
]


_OS_X_VERSION_RE = re.compile(r"os\s*x[\w\s]{0,20}?(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_os(text: str) -> ParsedField:
    if not text:
        return ParsedField(None, ParseStatus.UNAVAILABLE)

    lowered = text.lower()

    m_osx = _OS_X_VERSION_RE.search(lowered)
    if m_osx:
        return ParsedField(f"OS X {m_osx.group(1)}", ParseStatus.PARSED, raw_snippet=m_osx.group(0))

    for pattern, label in _OS_PATTERNS:
        m = re.search(pattern, lowered)
        if m:
            value = label if label else m.group(0).strip().title()
            return ParsedField(value, ParseStatus.PARSED, raw_snippet=m.group(0))

    return ParsedField(None, ParseStatus.UNCERTAIN, raw_snippet=text[:120])


_RAM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(gb|mb)\s*(?:of\s*)?(?:ram|memory|system memory)",
    re.IGNORECASE,
)


def extract_ram_mb(text: str) -> ParsedField:
    if not text:
        return ParsedField(None, ParseStatus.UNAVAILABLE)

    m = _RAM_RE.search(text)
    if not m:
        return ParsedField(None, ParseStatus.UNCERTAIN, raw_snippet=text[:120])

    amount, unit = float(m.group(1)), m.group(2).lower()
    mb = amount * 1024 if unit == "gb" else amount
    return ParsedField(mb, ParseStatus.PARSED, raw_snippet=m.group(0))


_CPU_GHZ_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ghz|mhz)",
    re.IGNORECASE,
)
_CPU_TEXT_RE = re.compile(
    r"([^.,;]{0,60}\b(?:processor|cpu|intel\s*core\s*i\d|amd\s*ryzen|core\s*i\d|ryzen|"
    r"dual-core|quad-core|athlon|pentium|celeron)\b[^.,;]{0,60})",
    re.IGNORECASE,
)
# Deliberately excludes bare "intel"/"amd" as trigger words: in this dataset
# those also appear inside GPU descriptions ("Intel HD 3000", "AMD Radeon"),
# so using them alone produces false-positive CPU matches. Only fire on
# unambiguous CPU vocabulary.


def extract_cpu(text: str) -> tuple[ParsedField, ParsedField]:
    """Returns (cpu_ghz, cpu_raw)."""
    if not text:
        return ParsedField(None, ParseStatus.UNAVAILABLE), ParsedField(None, ParseStatus.UNAVAILABLE)

    ghz_field = ParsedField(None, ParseStatus.UNCERTAIN)
    m = _CPU_GHZ_RE.search(text)
    if m:
        amount, unit = float(m.group(1)), m.group(2).lower()
        ghz = amount / 1000 if unit == "mhz" else amount
        ghz_field = ParsedField(round(ghz, 3), ParseStatus.PARSED, raw_snippet=m.group(0))

    raw_field = ParsedField(None, ParseStatus.UNCERTAIN)
    m2 = _CPU_TEXT_RE.search(text)
    if m2:
        raw_field = ParsedField(m2.group(0).strip(), ParseStatus.PARSED, raw_snippet=m2.group(0))
    elif ghz_field.is_known():
        # We at least found a clock speed even without surrounding CPU text.
        raw_field = ParsedField(ghz_field.raw_snippet, ParseStatus.PARSED)

    return ghz_field, raw_field


_GPU_TEXT_RE = re.compile(
    r"([^.,;]{0,80}\b(?:nvidia|geforce|radeon|ati|intel hd|intel uhd|iris|"
    r"video card|graphics card|gpu|directx capable)\b[^.,;]{0,80})",
    re.IGNORECASE,
)
_VRAM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(gb|mb)\s*(?:of\s*)?(?:vram|video memory|graphics memory)",
    re.IGNORECASE,
)


def extract_gpu(text: str) -> tuple[ParsedField, ParsedField]:
    """Returns (gpu_raw, gpu_vram_mb)."""
    if not text:
        return ParsedField(None, ParseStatus.UNAVAILABLE), ParsedField(None, ParseStatus.UNAVAILABLE)

    gpu_field = ParsedField(None, ParseStatus.UNCERTAIN)
    m = _GPU_TEXT_RE.search(text)
    if m:
        gpu_field = ParsedField(m.group(0).strip(), ParseStatus.PARSED, raw_snippet=m.group(0))

    vram_field = ParsedField(None, ParseStatus.UNCERTAIN)
    m2 = _VRAM_RE.search(text)
    if m2:
        amount, unit = float(m2.group(1)), m2.group(2).lower()
        mb = amount * 1024 if unit == "gb" else amount
        vram_field = ParsedField(mb, ParseStatus.PARSED, raw_snippet=m2.group(0))

    return gpu_field, vram_field


_STORAGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(gb|mb)\s*(?:of\s*)?(?:available|free)?\s*"
    r"(?:hard drive space|hd space|hdd space|disk space|storage|available space|free space|space)",
    re.IGNORECASE,
)


def extract_storage_gb(text: str) -> ParsedField:
    if not text:
        return ParsedField(None, ParseStatus.UNAVAILABLE)

    m = _STORAGE_RE.search(text)
    if not m:
        return ParsedField(None, ParseStatus.UNCERTAIN, raw_snippet=text[:120])

    amount, unit = float(m.group(1)), m.group(2).lower()
    gb = amount if unit == "gb" else amount / 1024
    return ParsedField(gb, ParseStatus.PARSED, raw_snippet=m.group(0))


_DIRECTX_RE = re.compile(r"directx\s*(?:version\s*)?(\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_directx(text: str) -> ParsedField:
    if not text:
        return ParsedField(None, ParseStatus.UNAVAILABLE)

    m = _DIRECTX_RE.search(text)
    if not m:
        return ParsedField(None, ParseStatus.UNCERTAIN)
    return ParsedField(m.group(1), ParseStatus.PARSED, raw_snippet=m.group(0))


# --------------------------------------------------------------------------
# Step 3: assemble a full ParsedRequirementSet from a text blob
# --------------------------------------------------------------------------

def parse_requirement_text(text: Optional[str]) -> ParsedRequirementSet:
    """Run all field extractors over one tier of requirement text (minimum
    or recommended, for one platform)."""
    text = text or ""
    cpu_ghz, cpu_raw = extract_cpu(text)
    gpu_raw, gpu_vram = extract_gpu(text)

    return ParsedRequirementSet(
        os=extract_os(text),
        cpu_ghz=cpu_ghz,
        cpu_raw=cpu_raw,
        ram_mb=extract_ram_mb(text),
        gpu_raw=gpu_raw,
        gpu_vram_mb=gpu_vram,
        storage_gb=extract_storage_gb(text),
        directx=extract_directx(text),
        source_text=text,
    )


@dataclass
class GameRequirements:
    appid: int
    windows_minimum: Optional[ParsedRequirementSet] = None
    windows_recommended: Optional[ParsedRequirementSet] = None
    mac_minimum: Optional[ParsedRequirementSet] = None
    linux_minimum: Optional[ParsedRequirementSet] = None
    has_any_requirements: bool = False


def parse_game_requirements(row: "pd.Series") -> GameRequirements:  # noqa: F821
    """
    Parse one row of steam_requirements_data.csv into structured
    Windows/Mac/Linux minimum & recommended requirement sets.

    Uses the pre-split `minimum` / `recommended` columns as the primary
    source for Windows (cleaning any embedded 'Recommended:' tail found
    inside `minimum`), and falls back to parsing the `pc_requirements`
    stringified dict if those columns are empty. Mac and Linux only have
    a 'minimum' tier available in this dataset.
    """
    appid = int(row.get("steam_appid"))
    result = GameRequirements(appid=appid)

    # --- Windows: prefer the pre-split 'minimum' / 'recommended' columns ---
    win_min_text = row.get("minimum")
    win_rec_text = row.get("recommended")

    win_min_text = win_min_text if isinstance(win_min_text, str) and win_min_text.strip() else None
    win_rec_text = win_rec_text if isinstance(win_rec_text, str) and win_rec_text.strip() else None

    embedded_rec = None
    if win_min_text:
        win_min_text, embedded_rec = split_minimum_recommended(normalize_whitespace(strip_html(win_min_text)))

    if not win_rec_text and embedded_rec:
        win_rec_text = embedded_rec
    elif win_rec_text:
        win_rec_text = normalize_whitespace(strip_html(win_rec_text))

    # Fallback: parse pc_requirements dict if the plain columns were empty.
    if not win_min_text or not win_rec_text:
        pc_dict = parse_requirements_dict(row.get("pc_requirements"))
        if not win_min_text and pc_dict.get("minimum"):
            win_min_text, embedded_rec2 = split_minimum_recommended(pc_dict["minimum"])
            if not win_rec_text and embedded_rec2:
                win_rec_text = embedded_rec2
        if not win_rec_text and pc_dict.get("recommended"):
            win_rec_text = pc_dict["recommended"]

    if win_min_text:
        result.windows_minimum = parse_requirement_text(win_min_text)
        result.has_any_requirements = True
    if win_rec_text:
        result.windows_recommended = parse_requirement_text(win_rec_text)
        result.has_any_requirements = True

    # --- Mac ---
    mac_dict = parse_requirements_dict(row.get("mac_requirements"))
    if mac_dict.get("minimum"):
        result.mac_minimum = parse_requirement_text(mac_dict["minimum"])
        result.has_any_requirements = True

    # --- Linux ---
    linux_dict = parse_requirements_dict(row.get("linux_requirements"))
    if linux_dict.get("minimum"):
        result.linux_minimum = parse_requirement_text(linux_dict["minimum"])
        result.has_any_requirements = True

    return result
