"""
Phase 2 (FUTURE, NOT IMPLEMENTED): Local user profile system.

This module defines the data shape a future personalized-profile feature
will use, so the Phase 1 recommendation engine can be built to accept a
`preferences`/`profile` object today without needing to be rewritten later.

Nothing in this file is wired into the Phase 1 recommendation flow.
`RecommendationRequest` in recommender.py optionally accepts a
`UserProfile`, and if one is provided in the future, its `favorite_games`
/ `excluded_games` / hardware fields can be read by the candidate
generation and ranking stages -- but as of Phase 1, no code path creates,
saves, loads, or requires a UserProfile. Do not build UI for this yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .hardware_compatibility import UserHardwareSpec


@dataclass
class UserProfile:
    """
    Conceptual shape for a future local profile (e.g. "My Gaming PC",
    "Laptop", "Low-End PC"). Intentionally plain data -- no behavior --
    so it can be trivially serialized to a local JSON file per the
    project's "no remote database" requirement.
    """
    name: str
    favorite_games: list[int] = field(default_factory=list)   # appids
    excluded_games: list[int] = field(default_factory=list)   # appids
    hardware: Optional[UserHardwareSpec] = None
    preferred_genres: list[str] = field(default_factory=list)
    max_price: Optional[float] = None
    preferred_platforms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Local persistence (future) -- sketched, not called from anywhere in Phase 1.
# --------------------------------------------------------------------------

def save_profile_locally(profile: UserProfile, directory: Path) -> Path:
    """(FUTURE) Save a profile as local JSON. Not called in Phase 1."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile.name}.json"
    with open(path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
    return path


def load_profile_locally(path: Path) -> dict:
    """(FUTURE) Load a profile's raw dict from local JSON. Not called in Phase 1."""
    with open(path) as f:
        return json.load(f)


def list_local_profiles(directory: Path) -> list[str]:
    """(FUTURE) List available local profile names. Not called in Phase 1."""
    if not directory.exists():
        return []
    return [p.stem for p in directory.glob("*.json")]
