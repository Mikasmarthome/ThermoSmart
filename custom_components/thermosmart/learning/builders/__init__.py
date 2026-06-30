"""LE 2.0 episode builder layer (Phase 6, pure Python).

Builders consume a time-ordered stream of raw observations, controller/TRV
events, HeatingDriveEnd events, RegimeResults and FeatureExtractor results and
emit materialisable Phase-1 episode schemas. They hold per-zone state, are
deterministic for an identical event sequence, mutate no input, persist nothing,
and depend on no Home Assistant / storage / model code.
"""
from .base import (
    BUILDER_VERSION,
    BuilderAction,
    BuilderInput,
    BuilderParameters,
    BuilderResult,
    BuilderStateError,
    ConfounderCode,
    DecisionContext,
    DownsamplePolicy,
    OutcomeReason,
    ReasonCode,
)
from .drive_end import DriveEndDetector, detect_drive_end
from .heating import HeatingEpisodeBuilder
from .afterheat import AfterheatEpisodeBuilder
from .cooling import PassiveCoolingEpisodeBuilder
from .window_cooling import WindowCoolingEpisodeBuilder
from .outcome import OutcomeEpisodeBuilder
from .registry_check import BuilderRegistryError, validate_builder_against_registry

__all__ = [
    "BUILDER_VERSION", "BuilderAction", "BuilderInput", "BuilderParameters",
    "BuilderResult", "BuilderStateError", "ConfounderCode", "DecisionContext",
    "DownsamplePolicy", "OutcomeReason", "ReasonCode",
    "DriveEndDetector", "detect_drive_end",
    "HeatingEpisodeBuilder", "AfterheatEpisodeBuilder",
    "PassiveCoolingEpisodeBuilder", "WindowCoolingEpisodeBuilder",
    "OutcomeEpisodeBuilder",
    "BuilderRegistryError", "validate_builder_against_registry",
]
