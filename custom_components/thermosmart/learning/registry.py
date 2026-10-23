"""Registry foundations for Learning (pure Python, no Home Assistant).

Two explicit, instantiable registries hold *declarations* (not active
services):

* ``EpisodeRegistry``  — materialised episode definitions,
* ``ModelRegistry``    — model definitions (factories + declared capabilities).

A registry entry never performs storage or runtime actions. Registries are
plain objects with deterministic (insertion-ordered) iteration and no hidden
global mutable singletons. Cross-registry consistency is checked by the pure
``validate_registry_graph`` function, which reports structured issues rather
than raising or auto-repairing.

TRV-only invariant: the registry describes *schemas and declared capabilities*.
It never decides whether a zone owns a capability, and a model declared
``min_trv_only`` may not require any environmental/telemetry feature or any
capability flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from .contracts import (
    CapabilitySet,
    Model,
    PredictionType,
)
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

# Optional environmental / telemetry features that a TRV-only model must never
# declare as required.
ENVIRONMENTAL_OPTIONAL_FEATURES: frozenset[str] = frozenset({
    "outdoor_temp", "outdoor_humidity", "wind_speed", "solar_radiation", "rain",
    "humidity", "weather", "forecast", "forecast_high", "window_open", "window",
    "valve_opening", "hvac_action", "room_sensor",
})

# The concrete capability flag names (everything except the lossy ``tier``).
CAPABILITY_FLAG_NAMES: frozenset[str] = frozenset(f.name for f in fields(CapabilitySet))


# -- error contract -----------------------------------------------------------

class RegistryError(Exception):
    """Base class for all registry errors."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a name is registered twice in the same registry."""


class UnknownDefinitionError(RegistryError):
    """Raised when a lookup targets an unregistered name."""


class InvalidDefinitionError(RegistryError):
    """Raised when a single definition violates its own invariants."""


class InvalidCrossReferenceError(RegistryError):
    """Raised when an eager cross-reference resolution fails."""


# -- privacy + retention ------------------------------------------------------


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


# -- model definition ---------------------------------------------------------

@dataclass(frozen=True)
class ModelDefinition:
    """Declaration of one model: factory plus declared capabilities.

    Capability prerequisites are expressed exclusively through concrete flag
    names (``required_capability_flags``). The numeric ``tier`` can never be a
    prerequisite because it is not a CapabilitySet field.
    """

    model_name: str
    model_factory: Callable[[], Model]
    model_version: int
    parameter_version: int
    consumed_episode_types: tuple[EpisodeType, ...]
    supported_prediction_types: tuple[PredictionType, ...]
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...]
    required_capability_flags: tuple[str, ...]
    regime_dependent: bool
    advisory: bool
    control_relevant: bool
    rebuildable: bool
    can_be_not_available: bool
    min_trv_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_name:
            raise InvalidDefinitionError("model_name must be a non-empty string")
        if not callable(self.model_factory):
            raise InvalidDefinitionError("model_factory must be callable")
        if self.model_version <= 0:
            raise InvalidDefinitionError("model_version must be > 0")
        if self.parameter_version <= 0:
            raise InvalidDefinitionError("parameter_version must be > 0")
        if self.advisory and self.control_relevant:
            raise InvalidDefinitionError(
                "a model cannot be both advisory and control_relevant"
            )
        for flag in self.required_capability_flags:
            if flag == "tier":
                raise InvalidDefinitionError(
                    "the numeric tier must never be a capability prerequisite"
                )
            if flag not in CAPABILITY_FLAG_NAMES:
                raise InvalidDefinitionError(
                    f"'{flag}' is not a concrete CapabilitySet flag"
                )
        if self.min_trv_only:
            forbidden = set(self.required_features) & ENVIRONMENTAL_OPTIONAL_FEATURES
            if forbidden:
                raise InvalidDefinitionError(
                    f"TRV-only model '{self.model_name}' requires optional "
                    f"environmental features: {sorted(forbidden)}"
                )
            if self.required_capability_flags:
                raise InvalidDefinitionError(
                    f"TRV-only model '{self.model_name}' must not require any "
                    f"capability flag"
                )

    def create(self) -> Model:
        """Instantiate the model and verify it satisfies the Model protocol."""
        instance = self.model_factory()
        if not isinstance(instance, Model):
            raise InvalidDefinitionError(
                f"factory for '{self.model_name}' did not produce a Model"
            )
        return instance


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


class ModelRegistry:
    """Holds model definitions, keyed by ``model_name``."""

    def __init__(self) -> None:
        self._defs: dict[str, ModelDefinition] = {}

    def register(self, definition: ModelDefinition) -> None:
        if not isinstance(definition, ModelDefinition):
            raise InvalidDefinitionError("expected a ModelDefinition")
        if definition.model_name in self._defs:
            raise DuplicateRegistrationError(
                f"model '{definition.model_name}' already registered"
            )
        self._defs[definition.model_name] = definition

    def get(self, model_name: str) -> ModelDefinition:
        try:
            return self._defs[model_name]
        except KeyError:
            raise UnknownDefinitionError(f"unknown model '{model_name}'") from None

    def contains(self, model_name: str) -> bool:
        return model_name in self._defs

    def definitions(self) -> tuple[ModelDefinition, ...]:
        return tuple(self._defs.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._defs.keys())

    def by_prediction_type(
        self, prediction_type: PredictionType
    ) -> tuple[ModelDefinition, ...]:
        return tuple(
            d for d in self._defs.values()
            if prediction_type in d.supported_prediction_types
        )

    def advisory(self) -> tuple[ModelDefinition, ...]:
        return tuple(d for d in self._defs.values() if d.advisory)

    def control_relevant(self) -> tuple[ModelDefinition, ...]:
        return tuple(d for d in self._defs.values() if d.control_relevant)

    def __len__(self) -> int:
        return len(self._defs)


# -- cross-registry validation ------------------------------------------------

class ValidationCode(Enum):
    MODEL_UNKNOWN_EPISODE = "model_unknown_episode"
    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_VERSION = "invalid_version"
    ADVISORY_CONTROL_CONFLICT = "advisory_control_conflict"
    TIER_AS_CAPABILITY = "tier_as_capability"
    TRV_ONLY_REQUIRES_ENV = "trv_only_requires_env"


@dataclass(frozen=True)
class ValidationIssue:
    code: ValidationCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)


def validate_registry_graph(
    episode_registry: EpisodeRegistry,
    model_registry: ModelRegistry,
) -> list[ValidationIssue]:
    """Pure, non-raising cross-registry validation.

    Reports structured issues; never repairs. Single-definition invariants are
    enforced eagerly at construction, but are re-checked defensively here so a
    hand-crafted or future definition cannot slip an inconsistency past the
    graph check.
    """
    issues: list[ValidationIssue] = []

    # Episodes.
    for definition in episode_registry.definitions():
        canonical = _CANONICAL_EPISODE_SCHEMA.get(definition.episode_type)
        if canonical is not None and definition.schema_type is not canonical:
            issues.append(ValidationIssue(
                ValidationCode.SCHEMA_MISMATCH,
                f"episode '{definition.episode_type.value}' has a mismatched schema",
                {"episode_type": definition.episode_type.value},
            ))

    # Models -> episode types, plus defensive single-def checks.
    for definition in model_registry.definitions():
        for episode_type in definition.consumed_episode_types:
            if not episode_registry.contains(episode_type):
                issues.append(ValidationIssue(
                    ValidationCode.MODEL_UNKNOWN_EPISODE,
                    f"model '{definition.model_name}' consumes unregistered "
                    f"episode '{episode_type.value}'",
                    {"model": definition.model_name, "episode": episode_type.value},
                ))
        if definition.model_version <= 0 or definition.parameter_version <= 0:
            issues.append(ValidationIssue(
                ValidationCode.INVALID_VERSION,
                f"model '{definition.model_name}' has an invalid version",
                {"model": definition.model_name},
            ))
        if definition.advisory and definition.control_relevant:
            issues.append(ValidationIssue(
                ValidationCode.ADVISORY_CONTROL_CONFLICT,
                f"model '{definition.model_name}' is both advisory and control",
                {"model": definition.model_name},
            ))
        if "tier" in definition.required_capability_flags:
            issues.append(ValidationIssue(
                ValidationCode.TIER_AS_CAPABILITY,
                f"model '{definition.model_name}' uses tier as a capability",
                {"model": definition.model_name},
            ))
        if definition.min_trv_only:
            forbidden = set(definition.required_features) & ENVIRONMENTAL_OPTIONAL_FEATURES
            if forbidden:
                issues.append(ValidationIssue(
                    ValidationCode.TRV_ONLY_REQUIRES_ENV,
                    f"TRV-only model '{definition.model_name}' requires "
                    f"environmental features",
                    {"model": definition.model_name, "features": sorted(forbidden)},
                ))

    return issues
