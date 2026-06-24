"""LE 2.0 decision architecture (Phase 19A).

Typed, pure-Python internal pipeline that unifies the deterministic baseline and
LE 2.0 predictions behind a single resolver, guard layer, device adapter and
dispatch boundary. The public Home Assistant surface is unchanged; SHADOW remains
the productive default and computes the full decision without dispatching.
"""
from __future__ import annotations

from .baseline import baseline_from_recommendation, zone_input_from_recommendation
from .contracts import (
    BoostEvaluationStatus,
    ControlGuardResult,
    ControllerBaselineDecision,
    DecisionMode,
    DecisionReason,
    DecisionTrace,
    DecisionTraceEntry,
    DeviceControlCommand,
    DispatchResult,
    FallbackReason,
    LearningPrediction,
    LearningPredictionSet,
    ResolvedControlDecision,
    ZoneRuntimeInput,
)
from .device import DeviceAdapter, DeviceProfile
from .dispatch import SingleDispatcher
from .guards import GuardLayer, GuardParameters
from .legacy_map import LEGACY_INVENTORY, Disposition, LegacyEntry, by_disposition, validate_inventory
from .pipeline import DecisionPipeline
from .resolver import FinalResolver

__all__ = [
    "baseline_from_recommendation", "zone_input_from_recommendation",
    "ControllerBaselineDecision", "ZoneRuntimeInput", "LearningPrediction",
    "LearningPredictionSet", "ResolvedControlDecision", "ControlGuardResult",
    "DeviceControlCommand", "DispatchResult", "DecisionTrace", "DecisionTraceEntry",
    "DecisionMode", "DecisionReason", "FallbackReason", "BoostEvaluationStatus",
    "DeviceAdapter", "DeviceProfile", "GuardLayer", "GuardParameters",
    "SingleDispatcher", "FinalResolver", "DecisionPipeline",
    "LEGACY_INVENTORY", "Disposition", "LegacyEntry", "by_disposition", "validate_inventory",
]
