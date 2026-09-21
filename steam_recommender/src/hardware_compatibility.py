"""
Hardware compatibility engine.

Compares a user's PC specifications against a game's parsed requirements
and produces a compatibility label + score. Deliberately does NOT reduce
to a single "user_ram >= game_ram" style check for every field -- each
component type uses different comparison logic, and every result is
honest about what could and couldn't be determined.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from . import config
from .requirements_parsing import GameRequirements, ParsedRequirementSet, ParseStatus

logger = logging.getLogger(__name__)


class CompatibilityLabel(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    LIKELY_COMPATIBLE = "LIKELY COMPATIBLE"
    BORDERLINE = "BORDERLINE"
    NOT_RECOMMENDED = "NOT RECOMMENDED"
    UNKNOWN = "UNKNOWN"


@dataclass
class UserHardwareSpec:
    """User-provided PC specifications. All fields optional -- the engine
    must handle partial input gracefully rather than requiring everything."""
    os: Optional[str] = None                 # e.g. "Windows 10", "Windows 11", "macOS", "Linux"
    cpu_ghz: Optional[float] = None
    ram_mb: Optional[float] = None
    gpu_vram_mb: Optional[float] = None
    gpu_name: Optional[str] = None            # free text, used only for informational display
    storage_gb: Optional[float] = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (self.os, self.cpu_ghz, self.ram_mb, self.gpu_vram_mb, self.storage_gb)
        )


@dataclass
class FieldComparison:
    field_name: str
    user_value: Optional[object]
    required_value: Optional[object]
    parse_status: ParseStatus
    result: str   # "meets", "below", "unknown_requirement", "unknown_user_value"


@dataclass
class CompatibilityResult:
    label: CompatibilityLabel
    score: Optional[float]           # None only when label is UNKNOWN
    comparisons: list[FieldComparison] = field(default_factory=list)
    tier_used: Optional[str] = None  # "minimum" or "recommended"
    platform_used: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def explain(self) -> list[str]:
        """Human-readable bullet points, safe to show directly in the UI."""
        lines = []
        for c in self.comparisons:
            if c.result == "meets":
                lines.append(f"Your {c.field_name} meets the {self.tier_used or 'minimum'} requirement.")
            elif c.result == "below":
                lines.append(
                    f"Your {c.field_name} is below the {self.tier_used or 'minimum'} "
                    f"requirement ({c.user_value} vs required {c.required_value})."
                )
            elif c.result == "unknown_requirement":
                lines.append(f"{c.field_name.upper()} requirement could not be confidently determined.")
            elif c.result == "unknown_user_value":
                lines.append(f"You didn't provide your {c.field_name}, so it couldn't be checked.")
        lines.extend(self.notes)
        return lines


def _compare_numeric(
    field_name: str, user_value: Optional[float], parsed: "ParsedRequirementSet", attr: str
) -> FieldComparison:
    req_field = getattr(parsed, attr)
    if req_field.status != ParseStatus.PARSED or req_field.value is None:
        return FieldComparison(field_name, user_value, None, req_field.status, "unknown_requirement")
    if user_value is None:
        return FieldComparison(field_name, None, req_field.value, req_field.status, "unknown_user_value")
    result = "meets" if user_value >= req_field.value else "below"
    return FieldComparison(field_name, user_value, req_field.value, req_field.status, result)


_OS_FAMILY_MAP = {
    "windows": "windows",
    "os x": "mac",
    "macos": "mac",
    "mac os x": "mac",
    "ubuntu": "linux",
    "linux": "linux",
    "steamos": "linux",
}


def _os_family(os_string: Optional[str]) -> Optional[str]:
    if not os_string:
        return None
    lowered = os_string.lower()
    for key, family in _OS_FAMILY_MAP.items():
        if key in lowered:
            return family
    return None


def _compare_os(user_os: Optional[str], parsed: "ParsedRequirementSet") -> FieldComparison:
    req_field = parsed.os
    if req_field.status != ParseStatus.PARSED or req_field.value is None:
        return FieldComparison("OS", user_os, None, req_field.status, "unknown_requirement")
    if user_os is None:
        return FieldComparison("OS", None, req_field.value, req_field.status, "unknown_user_value")

    user_family = _os_family(user_os)
    req_family = _os_family(req_field.value)
    if user_family is None or req_family is None:
        # Can't confidently classify -- treat as unknown rather than guessing.
        return FieldComparison("OS", user_os, req_field.value, ParseStatus.UNCERTAIN, "unknown_requirement")

    result = "meets" if user_family == req_family else "below"
    return FieldComparison("OS", user_os, req_field.value, req_field.status, result)


def compare_hardware(
    user_spec: UserHardwareSpec,
    parsed_tier: "ParsedRequirementSet",
) -> list[FieldComparison]:
    """Run every component comparison for one requirement tier."""
    comparisons = [
        _compare_os(user_spec.os, parsed_tier),
        _compare_numeric("CPU (GHz)", user_spec.cpu_ghz, parsed_tier, "cpu_ghz"),
        _compare_numeric("RAM (MB)", user_spec.ram_mb, parsed_tier, "ram_mb"),
        _compare_numeric("GPU VRAM (MB)", user_spec.gpu_vram_mb, parsed_tier, "gpu_vram_mb"),
        _compare_numeric("Storage (GB)", user_spec.storage_gb, parsed_tier, "storage_gb"),
    ]
    return comparisons


def _score_from_comparisons(comparisons: list[FieldComparison]) -> Optional[float]:
    """
    Convert field-level comparisons into a single [0, 1] compatibility
    score. Fields with unknown requirement or unknown user value are
    excluded from the score entirely (not counted as failures) -- their
    absence lowers confidence, which is reflected in the resulting label
    thresholds and in `notes`, not by silently penalizing the score.
    """
    known = [c for c in comparisons if c.result in ("meets", "below")]
    if not known:
        return None
    passed = sum(1 for c in known if c.result == "meets")
    return passed / len(known)


def _select_tier(game_reqs: GameRequirements, prefer_platform: Optional[str]) -> tuple[Optional["ParsedRequirementSet"], Optional[str], Optional[str]]:
    """
    Pick which requirement tier to evaluate against, preferring the
    user's stated platform, then falling back to whatever exists.
    Prefers 'minimum' tier for the compatibility check (matches the
    conservative "can it run at all" framing); recommended-tier
    comparison is exposed separately by the caller if desired.
    """
    order = []
    if prefer_platform == "mac":
        order = [("mac", game_reqs.mac_minimum, "minimum")]
    elif prefer_platform == "linux":
        order = [("linux", game_reqs.linux_minimum, "minimum")]
    else:
        order = [("windows", game_reqs.windows_minimum, "minimum")]

    # Fallback: if the preferred platform has nothing, try any other platform
    # that does -- better to give an approximate compatibility read (clearly
    # labeled) than none at all.
    all_options = [
        ("windows", game_reqs.windows_minimum, "minimum"),
        ("mac", game_reqs.mac_minimum, "minimum"),
        ("linux", game_reqs.linux_minimum, "minimum"),
    ]
    for platform, tier, tier_name in order + all_options:
        if tier is not None:
            return tier, platform, tier_name
    return None, None, None


def evaluate_compatibility(
    user_spec: UserHardwareSpec,
    game_reqs: Optional[GameRequirements],
    game_platforms: Optional[list[str]] = None,
) -> CompatibilityResult:
    """
    Main entry point: evaluate one game's compatibility with one user's
    hardware. Never returns a false "compatible" -- when data is
    insufficient, returns UNKNOWN with score=None.
    """
    if game_reqs is None or not game_reqs.has_any_requirements:
        return CompatibilityResult(
            label=CompatibilityLabel.UNKNOWN,
            score=None,
            notes=["No parsable system requirements are available for this game."],
        )

    if user_spec.is_empty():
        return CompatibilityResult(
            label=CompatibilityLabel.UNKNOWN,
            score=None,
            notes=["No hardware specifications were provided."],
        )

    prefer_platform = _os_family(user_spec.os)
    tier, platform_used, tier_name = _select_tier(game_reqs, prefer_platform)

    if tier is None:
        return CompatibilityResult(
            label=CompatibilityLabel.UNKNOWN,
            score=None,
            notes=["This game has no requirements listed for any platform."],
        )

    notes = []
    if prefer_platform and platform_used and prefer_platform != platform_used:
        notes.append(
            f"No {prefer_platform} requirements were listed; showing {platform_used} "
            f"requirements instead as an approximation."
        )
    if game_platforms and prefer_platform and prefer_platform not in [p.lower() for p in game_platforms]:
        notes.append(f"This game does not list official {prefer_platform} support.")

    comparisons = compare_hardware(user_spec, tier)
    score = _score_from_comparisons(comparisons)

    known_count = sum(1 for c in comparisons if c.result in ("meets", "below"))
    unknown_count = len(comparisons) - known_count
    if unknown_count > 0:
        notes.append(
            f"{unknown_count} of {len(comparisons)} requirement field(s) could not be "
            f"confidently determined and were excluded from the score."
        )

    if score is None:
        label = CompatibilityLabel.UNKNOWN
    else:
        thresholds = config.COMPATIBILITY_THRESHOLDS
        any_hard_fail = any(c.result == "below" for c in comparisons if c.field_name in ("OS", "RAM (MB)"))
        if any_hard_fail and score < thresholds.borderline_min:
            label = CompatibilityLabel.NOT_RECOMMENDED
        elif score >= thresholds.compatible_min:
            label = CompatibilityLabel.COMPATIBLE
        elif score >= thresholds.likely_compatible_min:
            label = CompatibilityLabel.LIKELY_COMPATIBLE
        elif score >= thresholds.borderline_min:
            label = CompatibilityLabel.BORDERLINE
        else:
            label = CompatibilityLabel.NOT_RECOMMENDED

        # Low-confidence downgrade: if most fields were unknown, don't claim
        # full confidence even if the few known fields passed.
        if known_count <= 1 and label in (CompatibilityLabel.COMPATIBLE, CompatibilityLabel.LIKELY_COMPATIBLE):
            label = CompatibilityLabel.LIKELY_COMPATIBLE
            notes.append("Only limited requirement data was available; treat this as an approximate result.")

    return CompatibilityResult(
        label=label,
        score=score,
        comparisons=comparisons,
        tier_used=tier_name,
        platform_used=platform_used,
        notes=notes,
    )
