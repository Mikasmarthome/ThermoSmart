"""Typed legacy inventory for ThermoSmart (Phase 19A, pure Python).

A machine-checkable map of every production responsibility: its owner module, its
disposition under the new decision architecture, and (if learning) its Learning
replacement. NOTHING is deleted in Phase 19A — this is the authoritative inventory
that a later teardown (Phase 19B, gated) will act on.

Dispositions:
  KEEP      - deterministic baseline / device / safety logic stays as-is
  MOVE      - same logic, relocates behind a typed contract / adapter
  REPLACE   - LE-v1 learning logic superseded by an Learning model/prediction
  REMOVE    - safe to delete (only after its REPLACE is validated; not in 19A)
  UNCLEAR   - needs more evidence before disposition
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Disposition(Enum):
    KEEP = "keep"
    MOVE = "move"
    REPLACE = "replace"
    REMOVE = "remove"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class LegacyEntry:
    responsibility: str
    owner: str
    disposition: Disposition
    learning_replacement: str = ""
    note: str = ""


# Authoritative inventory. One row per fachliche Wahrheit.
LEGACY_INVENTORY: tuple[LegacyEntry, ...] = (
    # ---- deterministic baseline + safety + device: KEEP ----
    LegacyEntry("schedule_resolution", "coordinator._schedule_period", Disposition.KEEP,
                note="baseline schedule/comfort target"),
    LegacyEntry("target_temperature", "coordinator._compute_recommendation", Disposition.KEEP),
    LegacyEntry("effective_mode/presence", "coordinator._effective_mode/_get_presence_state",
                Disposition.KEEP),
    LegacyEntry("window_logic", "window.WindowMixin", Disposition.KEEP),
    LegacyEntry("summer_mode", "season._update_summer_mode", Disposition.KEEP),
    LegacyEntry("frost_protection", "season._apply_frost_protection", Disposition.KEEP),
    LegacyEntry("vacation_absence", "coordinator.set_vacation_override", Disposition.KEEP),
    LegacyEntry("tpi_duty", "tpi.compute_tpi", Disposition.KEEP),
    LegacyEntry("duty_to_setpoint", "tpi.duty_to_setpoint", Disposition.KEEP,
                note="additive boost semantics; bounded by TPI_MAX_BOOST_CELSIUS"),
    LegacyEntry("weather_offsets", "weather_engine.compute_forecast_suppression",
                Disposition.KEEP),
    LegacyEntry("valve_hvac_eval", "trv_control._watchdog_hvac/_async_calibrate_trvs",
                Disposition.KEEP),
    LegacyEntry("device_profiles", "trv_control._apply_temperature/quirks", Disposition.KEEP,
                note="authoritative for live writes; mirrored by decision.device.DeviceAdapter"),
    LegacyEntry("valve_maintenance", "maintenance._async_valve_maintenance", Disposition.KEEP),
    LegacyEntry("heating_failure_detect", "coordinator._check_heating_failure", Disposition.KEEP),

    # ---- dispatch + setpoint clamps: MOVE behind typed boundary ----
    LegacyEntry("setpoint_clamp", "trv_control._apply_temperature", Disposition.MOVE,
                note="mirrored by decision.guards.GuardLayer + decision.device.DeviceAdapter"),
    LegacyEntry("dispatch_path", "trv_control._apply_temperature/_async_set_valve_percent",
                Disposition.MOVE,
                note="live HA dispatch remains in trv_control; SingleDispatcher is a "
                     "shadow-only recorder with no sink — not the live dispatch boundary. "
                     "Phase B1 introduced LiveDecisionRecord to bind provenance without "
                     "routing dispatch through SingleDispatcher."),
    LegacyEntry("recommendation_dict", "coordinator._compute_recommendation", Disposition.MOVE,
                note="adapted to ZoneRuntimeInput/ControllerBaselineDecision at the edge"),

    # ---- LE v1 learning: REPLACE by Learning (remove only after validation, not in 19A) ----
    LegacyEntry("heat_rate", "learning_engine._get_heat_rate/_learned_heat_rate_multifactor",
                Disposition.REPLACE, "models.heat_rate.HeatRateModel"),
    LegacyEntry("heat_loss", "learning_engine.get_heat_loss_rate", Disposition.REPLACE,
                "models.heat_loss.HeatLossModel"),
    LegacyEntry("afterheat", "learning_engine.async_observe (residual rise)",
                Disposition.REPLACE, "models.afterheat.AfterheatModel"),
    LegacyEntry("outcome_score", "learning_engine._calc_outcome_score/get_outcome_stats",
                Disposition.REPLACE, "models.outcome.OutcomeModel"),
    LegacyEntry("preheat_minutes", "learning_engine.async_get_preheat_minutes",
                Disposition.REPLACE, "models + decision preheat integration"),
    LegacyEntry("confidence", "learning_engine.get_confidence/get_confidence_breakdown",
                Disposition.REPLACE, "confidence.ConfidenceAggregator"),
    LegacyEntry("boost_factor", "learning_engine.get_boost_factor/update_boost_factor",
                Disposition.REPLACE, "models.boost.BoostModel"),
    LegacyEntry("forecast_bias", "learning_engine.get_forecast_bias/evaluate_forecast_decisions",
                Disposition.REPLACE, "models.forecast.ForecastModel"),
    LegacyEntry("base_target_learn", "learning_engine.async_get_base_target",
                Disposition.REPLACE, "models + manual_correction.ManualCorrectionModel"),
    LegacyEntry("tpi_coefficients", "learning_engine.get_tpi_coefficients", Disposition.REPLACE,
                "models.controller_calibration (ControllerCalibration prediction)"),
    LegacyEntry("window_cooling_rate", "learning_engine.get_window_cooling_rate",
                Disposition.REPLACE, "models.heat_loss (cooling regime)"),
    LegacyEntry("manual_setpoint_learn", "learning_engine.async_observe_trv_setpoint",
                Disposition.REPLACE, "models.manual_correction.ManualCorrectionModel"),

    # ---- weather / forecast responsibility split (decided in extended 19A) ----
    LegacyEntry("weather_current_correction", "weather_engine.compute_temperature_offset",
                Disposition.KEEP,
                note="deterministic current weather correction stays authoritative"),
    LegacyEntry("forecast_trust_bias", "weather_engine.compute_forecast_suppression",
                Disposition.REPLACE, "models.forecast.ForecastModel",
                note="learned forecast trust/bias is Learning's; weather_engine keeps no learned "
                     "forecast confidence. ForecastModel must not duplicate a current "
                     "weather rule; it only gates whether a forecast-dependent path is used."),
)


def by_disposition(disposition: Disposition) -> tuple[LegacyEntry, ...]:
    return tuple(e for e in LEGACY_INVENTORY if e.disposition is disposition)


def validate_inventory() -> None:
    """Each REPLACE row must name an Learning target; no duplicate responsibilities."""
    seen: set[str] = set()
    for e in LEGACY_INVENTORY:
        if e.responsibility in seen:
            raise ValueError(f"duplicate responsibility: {e.responsibility}")
        seen.add(e.responsibility)
        if e.disposition is Disposition.REPLACE and not e.learning_replacement:
            raise ValueError(f"REPLACE without Learning target: {e.responsibility}")
        if e.disposition is Disposition.REMOVE:
            raise ValueError(f"no REMOVE permitted in Phase 19A: {e.responsibility}")
