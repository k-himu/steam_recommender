"""
Recommendation explanations.

Generates human-readable reasons for each recommended game, built ONLY
from data actually available for that game -- never inventing shared
genres/tags that don't really overlap, never claiming a rating quality
that isn't backed by the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .hardware_compatibility import CompatibilityLabel, CompatibilityResult
from .ranking import RankedGame


@dataclass
class Explanation:
    headline: str
    reasons: list[str]
    hardware_notes: list[str]


def _shared_tokens(reference_value: str, candidate_value: str) -> list[str]:
    if not isinstance(reference_value, str) or not isinstance(candidate_value, str):
        return []
    ref_set = {t.strip().lower() for t in reference_value.split(";") if t.strip()}
    cand_set = {t.strip().lower() for t in candidate_value.split(";") if t.strip()}
    shared = ref_set & cand_set
    # Preserve a readable, title-cased form using the candidate's original tokens.
    original_map = {t.strip().lower(): t.strip() for t in candidate_value.split(";") if t.strip()}
    return [original_map[s] for s in shared if s in original_map]


def _rating_quality_phrase(smoothed_ratio: float) -> str | None:
    if smoothed_ratio >= 0.85:
        return "Very positive user rating"
    if smoothed_ratio >= 0.70:
        return "Generally positive user rating"
    if smoothed_ratio >= 0.50:
        return "Mixed user rating"
    if smoothed_ratio > 0:
        return "Mostly negative user rating"
    return None


def _popularity_phrase(owners_estimate: float) -> str | None:
    if owners_estimate is None or owners_estimate != owners_estimate:  # NaN check
        return None
    if owners_estimate >= 5_000_000:
        return "Very high player ownership"
    if owners_estimate >= 500_000:
        return "High player ownership"
    if owners_estimate >= 50_000:
        return "Moderate player ownership"
    return None


def _engagement_phrase(avg_playtime: float) -> str | None:
    if avg_playtime is None or avg_playtime != avg_playtime:
        return None
    if avg_playtime >= 600:
        return "High average player engagement (playtime)"
    if avg_playtime >= 60:
        return "Solid average player engagement (playtime)"
    return None


def build_explanation(
    ranked_game: RankedGame,
    reference_game_row: pd.Series | None,
    compat_result: CompatibilityResult | None,
) -> Explanation:
    reasons: list[str] = []
    row = ranked_game.raw_row

    if reference_game_row is not None:
        shared_genres = _shared_tokens(reference_game_row.get("genres", ""), row.get("genres", ""))
        shared_tags = _shared_tokens(reference_game_row.get("steamspy_tags", ""), row.get("steamspy_tags", ""))
        shared_categories = _shared_tokens(reference_game_row.get("categories", ""), row.get("categories", ""))

        if shared_genres:
            reasons.append(f"Similar genres: {', '.join(shared_genres[:4])}")
        if shared_tags:
            reasons.append(f"Similar tags: {', '.join(shared_tags[:4])}")
        if shared_categories:
            reasons.append(f"Similar categories: {', '.join(shared_categories[:3])}")

    reasons.append(f"{ranked_game.content_similarity * 100:.0f}% content similarity to your reference game")

    quality_phrase = _rating_quality_phrase(row.get("smoothed_rating_ratio", 0.0))
    if quality_phrase:
        reasons.append(quality_phrase)

    pop_phrase = _popularity_phrase(row.get("owners_estimate"))
    if pop_phrase:
        reasons.append(pop_phrase)

    eng_phrase = _engagement_phrase(row.get("average_playtime"))
    if eng_phrase:
        reasons.append(eng_phrase)

    hardware_notes: list[str] = []
    if compat_result is not None:
        if compat_result.label == CompatibilityLabel.COMPATIBLE:
            reasons.append("Your PC meets the available minimum requirements")
        elif compat_result.label == CompatibilityLabel.LIKELY_COMPATIBLE:
            reasons.append("Your PC likely meets the requirements")
        elif compat_result.label == CompatibilityLabel.BORDERLINE:
            hardware_notes.append("Hardware warning: your PC is borderline for this game's requirements")
        elif compat_result.label == CompatibilityLabel.NOT_RECOMMENDED:
            hardware_notes.append("Hardware warning: your PC likely does not meet this game's requirements")
        elif compat_result.label == CompatibilityLabel.UNKNOWN:
            hardware_notes.append("Hardware compatibility could not be determined for this game")

        hardware_notes.extend(compat_result.explain())

    headline = f"Recommended because it's similar to a game you like"
    return Explanation(headline=headline, reasons=reasons, hardware_notes=hardware_notes)


def format_explanation_text(explanation: Explanation) -> str:
    lines = [f"{explanation.headline}:"]
    for reason in explanation.reasons:
        lines.append(f"  - {reason}")
    if explanation.hardware_notes:
        lines.append("Hardware notes:")
        for note in explanation.hardware_notes:
            lines.append(f"  - {note}")
    return "\n".join(lines)
