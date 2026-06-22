"""Shared builders for LE 2.0 diagnostics/export orchestration tests."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceComponent,
    ConfidenceInput,
    ConfidencePurpose,
)
from custom_components.thermosmart.learning.contracts import ConfidenceContribution
from custom_components.thermosmart.learning.export import (
    EngineSummary,
    EpisodeSummary,
    ExportInput,
    RegistrySummary,
    StorageSummary,
    ZoneExportInput,
)
from custom_components.thermosmart.learning.diagnostics import DiagnosticsInput
from custom_components.thermosmart.learning.privacy import Pseudonymizer
from custom_components.thermosmart.learning.models import (
    AfterheatModel,
    BoostModel,
    ForecastModel,
    HeatLossModel,
    HeatRateModel,
    ManualCorrectionModel,
    OutcomeModel,
)

GENERATED_AT = "2026-06-22T10:00:00+00:00"


def all_models(zone="living_room"):
    return {
        "heat_rate": HeatRateModel(zone),
        "heat_loss": HeatLossModel(zone),
        "afterheat": AfterheatModel(zone),
        "outcome": OutcomeModel(zone),
        "forecast": ForecastModel(zone),
        "boost": BoostModel(zone),
        "manual_correction": ManualCorrectionModel(zone),
    }


def heat_rate_confidence(zone="living_room"):
    hr = HeatRateModel(zone)
    agg = ConfidenceAggregator()
    return agg.aggregate(ConfidenceInput(
        ConfidencePurpose.HEAT_RATE,
        components=(ConfidenceComponent(kind=ComponentKind.HEAT_RATE,
                                        contribution=hr.confidence()),)))


def zone_input(zone_id="living_room", *, models=None, trv_only=True,
               capability_flags=None, confidence_results=None, initialized=True,
               storage=True, episodes=True, missing_evidence=()) -> ZoneExportInput:
    return ZoneExportInput(
        zone_id=zone_id, capability_flags=capability_flags or {"has_forecast": False,
                                                               "has_outdoor_temp": False},
        trv_only=trv_only, initialized=initialized,
        models=models if models is not None else all_models(zone_id),
        confidence_results=confidence_results or {"heat_rate": heat_rate_confidence(zone_id)},
        storage=StorageSummary(initialized=True, record_counts={"room": 12}) if storage else None,
        episodes=EpisodeSummary(counts_by_type={"outcome": 3}) if episodes else None,
        missing_evidence=missing_evidence)


def engine_summary(mode="advisory"):
    return EngineSummary(le_version="2.0.0", storage_schema_version=1, initialized=True,
                         mode=mode, registry_valid=True, last_capture_ts=GENERATED_AT)


def export_input(zones=None, *, salt=b"test-salt", generated_at=GENERATED_AT,
                 registry=None, budget=None) -> ExportInput:
    kw = {}
    if budget is not None:
        kw["budget"] = budget
    if zones is None:
        zones = (zone_input(),)
    return ExportInput(
        engine=engine_summary(), zones=tuple(zones),
        pseudonymizer=Pseudonymizer(salt=salt), generated_at=generated_at,
        registry=registry, ha_version="2026.6.0", integration_version="1.1.0", **kw)


def diagnostics_input(zones=None, *, salt=b"test-salt", generated_at=GENERATED_AT,
                      registry=None) -> DiagnosticsInput:
    if zones is None:
        zones = (zone_input(),)
    return DiagnosticsInput(
        engine=engine_summary(), zones=tuple(zones),
        pseudonymizer=Pseudonymizer(salt=salt), generated_at=generated_at, registry=registry)
