"""Validate an episode builder against a Phase-2 EpisodeRegistry (pure Python).

No global registry and no automatic runtime registration: the caller passes a
registry explicitly and receives structured errors.
"""
from __future__ import annotations

from typing import Any

from ..registry import EpisodeRegistry, RawTrackRegistry
from .base import BUILDER_VERSION


class BuilderRegistryError(Exception):
    """Raised when a builder is inconsistent with its registered definition."""


def validate_builder_against_registry(
    builder: Any,
    episode_registry: EpisodeRegistry,
    raw_registry: RawTrackRegistry | None = None,
) -> list[str]:
    """Return a list of error strings; empty means the builder is consistent."""
    errors: list[str] = []

    if not episode_registry.contains(builder.episode_type):
        errors.append(f"episode type '{builder.episode_type.value}' is not registered")
        return errors

    definition = episode_registry.get(builder.episode_type)

    if definition.builder_version != BUILDER_VERSION:
        errors.append(
            f"builder version {BUILDER_VERSION} != definition "
            f"{definition.builder_version}")
    if definition.max_trajectory_points != builder.trajectory_cap:
        errors.append(
            f"trajectory cap {builder.trajectory_cap} != definition "
            f"{definition.max_trajectory_points}")
    if definition.materialized != builder.materialized:
        errors.append("materialized flag mismatch")

    if raw_registry is not None:
        for track in definition.consumed_raw_tracks:
            if not raw_registry.contains(track):
                errors.append(f"consumed raw track '{track.value}' is not registered")

    return errors
