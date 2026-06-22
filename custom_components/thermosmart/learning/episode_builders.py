"""Public facade for the LE 2.0 episode builder layer (Phase 6).

Re-exports the builder package. Pure Python; no Home Assistant / storage / model
dependency; no runtime wiring.
"""
from __future__ import annotations

from .builders import (  # noqa: F401
    BUILDER_VERSION,
    AfterheatEpisodeBuilder,
    BuilderAction,
    BuilderInput,
    BuilderParameters,
    BuilderRegistryError,
    BuilderResult,
    BuilderStateError,
    ConfounderCode,
    DecisionContext,
    DownsamplePolicy,
    DriveEndDetector,
    HeatingEpisodeBuilder,
    OutcomeEpisodeBuilder,
    OutcomeReason,
    PassiveCoolingEpisodeBuilder,
    ReasonCode,
    WindowCoolingEpisodeBuilder,
    detect_drive_end,
    validate_builder_against_registry,
)

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
