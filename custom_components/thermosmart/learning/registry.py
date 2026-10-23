"""Episode registry for Learning (pure Python, no Home Assistant).

``EpisodeRegistry`` holds the *declarations* of the materialised episode types
(schema, versions, trajectory limits, retention). An entry never performs
storage or runtime actions. The registry is a plain object with deterministic
(insertion-ordered) iteration and no hidden global mutable singleton.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .episode_schemas import (
    AfterheatEpisode,
    EpisodeType,
    HeatingEpisode,
    OutcomeEpisode,
    PassiveCoolingEpisode,
    WindowCoolingEpisode,
)


# -- canonical schema pairings (authoritative name <-> schema type) -----------

_CANONICAL_EPISODE_SCHEMA: dict[EpisodeType, type] = {
    EpisodeType.HEATING: HeatingEpisode,
    EpisodeType.AFTERHEAT: AfterheatEpisode,
    EpisodeType.PASSIVE_COOLING: PassiveCoolingEpisode,
    EpisodeType.WINDOW_COOLING: WindowCoolingEpisode,
    EpisodeType.OUTCOME: OutcomeEpisode,
}


# -- error contract -----------------------------------------------------------

class RegistryError(Exception):
    """Base class for all registry errors."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a name is registered twice in the same registry."""


class UnknownDefinitionError(RegistryError):
    """Raised when a lookup targets an unregistered name."""


class InvalidDefinitionError(RegistryError):
    """Raised when a single definition violates its own invariants."""


# -- retention ----------------------------------------------------------------

@dataclass(frozen=True)
class RetentionPolicy:
    """Declarative retention configuration (no pruning engine here).

    Only describes intent. Concrete production values are deliberately left to
    later calibration; tests may use small example values.
    """

    max_records: Optional[int] = None
    max_age_days: Optional[int] = None
    segmented: bool = False
    quality_prioritized: bool = False
    seasonal_stratification: bool = False

    def __post_init__(self) -> None:
        if self.max_records is not None and self.max_records <= 0:
            raise InvalidDefinitionError("max_records must be > 0 when set")
        if self.max_age_days is not None and self.max_age_days <= 0:
            raise InvalidDefinitionError("max_age_days must be > 0 when set")


# -- episode definition -------------------------------------------------------

@dataclass(frozen=True)
class EpisodeDefinition:
    """Declaration of one materialised episode type.

    ``open_predicate`` / ``close_predicate`` / ``abort_predicate`` are optional
    and purely declarative here — no builder logic runs in Phase 2.
    """

    episode_type: EpisodeType
    schema_type: type
    episode_schema_version: int
    builder_version: int
    max_trajectory_points: int
    max_duration_seconds: int
    retention: RetentionPolicy
    trajectory_required: bool
    materialized: bool
    open_predicate: Optional[Callable[..., Any]] = None
    close_predicate: Optional[Callable[..., Any]] = None
    abort_predicate: Optional[Callable[..., Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_type, EpisodeType):
            raise InvalidDefinitionError("episode_type must be an EpisodeType")
        if not isinstance(self.schema_type, type):
            raise InvalidDefinitionError("schema_type must be a class")
        if self.episode_schema_version <= 0:
            raise InvalidDefinitionError("episode_schema_version must be > 0")
        if self.builder_version <= 0:
            raise InvalidDefinitionError("builder_version must be > 0")
        if self.max_trajectory_points <= 0:
            raise InvalidDefinitionError("max_trajectory_points must be > 0")
        if self.max_duration_seconds <= 0:
            raise InvalidDefinitionError("max_duration_seconds must be > 0")
        canonical = _CANONICAL_EPISODE_SCHEMA.get(self.episode_type)
        if canonical is None or self.schema_type is not canonical:
            raise InvalidDefinitionError(
                f"schema_type does not match the canonical schema for episode "
                f"'{self.episode_type.value}'"
            )


# -- registries ---------------------------------------------------------------


class EpisodeRegistry:
    """Holds episode definitions, keyed by ``EpisodeType``."""

    def __init__(self) -> None:
        self._defs: dict[EpisodeType, EpisodeDefinition] = {}

    def register(self, definition: EpisodeDefinition) -> None:
        if not isinstance(definition, EpisodeDefinition):
            raise InvalidDefinitionError("expected an EpisodeDefinition")
        if definition.episode_type in self._defs:
            raise DuplicateRegistrationError(
                f"episode '{definition.episode_type.value}' already registered"
            )
        self._defs[definition.episode_type] = definition

    def get(self, episode_type: EpisodeType) -> EpisodeDefinition:
        try:
            return self._defs[episode_type]
        except KeyError:
            raise UnknownDefinitionError(
                f"unknown episode '{getattr(episode_type, 'value', episode_type)}'"
            ) from None

    def contains(self, episode_type: EpisodeType) -> bool:
        return episode_type in self._defs

    def definitions(self) -> tuple[EpisodeDefinition, ...]:
        return tuple(self._defs.values())

    def names(self) -> tuple[EpisodeType, ...]:
        return tuple(self._defs.keys())

    def __len__(self) -> int:
        return len(self._defs)
