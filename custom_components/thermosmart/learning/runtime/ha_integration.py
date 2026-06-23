"""Home Assistant shadow integration shell for LE 2.0 (Phase 17C).

The ONLY glue between the live ThermoSmart coordinator and the passive LE-2.0
runtime. It builds a typed ``RuntimeCycleInput`` from values the coordinator has
already computed, runs the passive shadow cycle behind a hard guard, and stores
only diagnostics. It NEVER calls a service, never mutates a control value, and
never lets an exception reach the heating path.

This module imports Home Assistant (it is the shell, not the pure core); it is
deliberately NOT exported from ``runtime/__init__`` so the pure runtime stays
importable without HA.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Mapping, Optional

from ...const import TPI_MAX_BOOST_CELSIUS
from ..contracts import DataQuality, Measurement
from ..confidence import ConfidencePurpose
from ..contracts import PredictionType
from .capture import ControllerDecisionInput, DecisionType, RuntimeCycleInput, ScheduleTarget
from .control import ControlContext, ControlFeature
from ..decision import DecisionMode, DecisionPipeline, FinalResolver
from ..decision.runtime_adapter import build_learning_prediction_set
from .ha_store import HomeAssistantStoreAdapter
from .lifecycle import (
    CoordinatorBridge,
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)

_LOGGER = logging.getLogger(__name__)
_ERROR_LOG_THROTTLE = 50  # log at most 1 of every N identical errors


def _zone_segment(zone_id: str) -> str:
    """Non-identifying, stable token for the store key (never an entity name)."""
    return hashlib.sha256(zone_id.encode("utf-8")).hexdigest()[:16]


def _utcnow_iso() -> str:
    from homeassistant.util import dt as dt_util
    return dt_util.utcnow().isoformat()


def _is_celsius_unit(unit: str) -> bool:
    """Return True when the unit string represents Celsius temperature.

    Accepts only forms actually produced by LE2 model contracts:
      "C"       — AfterheatModel, HeatRate, HeatLoss (short form)
      "celsius" — ForecastModel, ManualCorrection, BoostOutcome (long form)

    Rejects "K", "%", "min", empty string, or any unknown unit.
    Never performs Kelvin conversion; missing unit → conservative False.
    """
    return unit in ("C", "celsius")


def _decision_type(recommendation: Mapping[str, Any]) -> DecisionType:
    try:
        if recommendation.get("preheat_minutes", 0) and recommendation.get("preheat_minutes", 0) > 0:
            return DecisionType.PREHEAT
        if recommendation.get("boost_active"):
            return DecisionType.BOOST
        if recommendation.get("window_open") or recommendation.get("is_summer"):
            return DecisionType.IDLE
        return DecisionType.NORMAL
    except Exception:
        return DecisionType.UNKNOWN


def build_runtime_cycle_input(zone_id: str, recommendation: Mapping[str, Any], *,
                              weather: Optional[Mapping[str, Any]] = None,
                              schedule_comfort_time_utc: Optional[str] = None,
                              schedule_comfort_temperature_c: Optional[float] = None,
                              ts_iso: Optional[str] = None,
                              heating_failure: bool = False) -> RuntimeCycleInput:
    """Materialise a typed cycle input from already-computed coordinator values.

    Defensive: only reads present values, never invents a temperature, keeps 0.0,
    and uses Measurement/None for missing data. Pure given its inputs.
    """
    rec = recommendation or {}
    ts = ts_iso or _utcnow_iso()
    target = rec.get("effective_target")
    if target is None:
        target = rec.get("adjusted_target")
    setpoint = rec.get("trv_setpoint")
    current = rec.get("current_temp")
    indoor = Measurement(current, DataQuality.OK) if current is not None else None

    demand = None
    duty = rec.get("tpi_duty_cycle")
    if duty is not None:
        demand = duty > 0.0
    elif setpoint is not None and current is not None:
        demand = setpoint > current + 0.3

    boost_offset = None
    if setpoint is not None and target is not None and setpoint > target:
        boost_offset = min(setpoint - target, TPI_MAX_BOOST_CELSIUS)

    outdoor = None
    if weather:
        ot = weather.get("outdoor_temp") or weather.get("temperature")
        if ot is not None:
            outdoor = Measurement(ot, DataQuality.OK)

    decision_type = _decision_type(rec)
    return RuntimeCycleInput(
        zone_id=zone_id, ts=ts, target_c=target, trv_setpoint_c=setpoint, indoor_temp=indoor,
        schedule=ScheduleTarget(comfort_time_utc=schedule_comfort_time_utc,
                                comfort_temperature_c=schedule_comfort_temperature_c)
        if schedule_comfort_time_utc and schedule_comfort_temperature_c is not None else None,
        mode=rec.get("mode"), controller_demand=demand,
        controller_decision=ControllerDecisionInput(
            decision_type=decision_type, target_c=target, trv_setpoint_c=setpoint,
            boost_offset_c=boost_offset, boost_active=bool(rec.get("boost_active")),
            preheat_active=bool(rec.get("preheat_minutes", 0)),
            mode=rec.get("mode")),
        window_open=rec.get("window_open"), heating_failure=heating_failure,
        outdoor_temp=outdoor, forecast_high=rec.get("forecast_high"))


class LearningShadowController:
    """Owns the passive LE-2.0 runtime for one config entry / zone.

    Every public method is guarded: a learning/store/model failure is counted and
    logged (throttled), never raised. The existing heating control is untouched.
    """

    def __init__(self, hass: Any, zone_id: str, *, store: Any = None,
                 mode: LearningRuntimeMode = LearningRuntimeMode.SHADOW) -> None:
        self._zone = zone_id
        self._enabled = True
        self._errors = 0
        self._error_signatures: dict[str, int] = {}
        self._last_result = None
        adapter = store
        if adapter is None:
            try:
                adapter = HomeAssistantStoreAdapter(hass, _zone_segment(zone_id))
            except Exception:  # store construction must never break setup
                adapter = None
        self._runtime = LearningRuntime(
            LearningRuntimeConfig(mode=mode), store=adapter, clock=_utcnow_iso)
        # The new decision pipeline runs read-only every cycle (no sink => it can never
        # dispatch). The existing coordinator remains the single real dispatch path.
        self._pipeline = DecisionPipeline(
            resolver=FinalResolver(boost_runtime_limit=TPI_MAX_BOOST_CELSIUS))
        self._last_trace: Optional[dict] = None
        # Cache: schedule target time from last observe_safe call (one cycle lag is acceptable
        # for deescalation checks — TPI sufficiency uses this to compute remaining_time_to_target).
        self._last_comfort_time_utc: Optional[str] = None
        # Context snapshot at boost-apply time: compared each cycle to detect schedule transitions.
        # A change in comfort_time (same target, new period) triggers a hard schedule_change release.
        self._boost_applied_comfort_time_utc: Optional[str] = None

    @property
    def runtime(self) -> LearningRuntime:
        return self._runtime

    @property
    def errors(self) -> int:
        return self._errors

    async def async_setup(self) -> bool:
        try:
            return await self._runtime.async_setup()
        except Exception as err:  # learning setup failure must not fail the entry
            self._record_error("setup", err)
            self._enabled = False
            return False

    def observe_safe(self, recommendation: Mapping[str, Any], *,
                     weather: Optional[Mapping[str, Any]] = None,
                     schedule_comfort_time_utc: Optional[str] = None,
                     schedule_comfort_temperature_c: Optional[float] = None,
                     heating_failure: bool = False) -> None:
        """Run one passive shadow cycle. Never raises, never controls."""
        if not self._enabled:
            return
        # Cache comfort time for TPI-sufficiency check in next adjust_recommendation_safe call.
        self._last_comfort_time_utc = schedule_comfort_time_utc
        try:
            inp = build_runtime_cycle_input(
                self._zone, recommendation, weather=weather,
                schedule_comfort_time_utc=schedule_comfort_time_utc,
                schedule_comfort_temperature_c=schedule_comfort_temperature_c,
                heating_failure=heating_failure)
            self._last_result = CoordinatorBridge(self._runtime).process(inp)
        except Exception as err:
            self._record_error("cycle", err)

    @property
    def control_enabled(self) -> bool:
        return self._enabled and self._runtime.control_enabled

    def _boost_model(self):
        """Return the BoostModel for this zone, or None if unavailable."""
        try:
            zr = self._runtime._zone(self._zone)
            return zr.orchestrator.models.get("boost")
        except Exception:
            return None

    def _release_boost_lifecycle_safe(self, reason: str) -> None:
        """Release active boost lifecycle with the given reason. Never raises."""
        try:
            model = self._boost_model()
            if model is None:
                return
            from ..models.boost import BoostLifecycle
            lc = model._state.lifecycle
            if lc in (BoostLifecycle.APPLIED, BoostLifecycle.ACTIVE):
                model.release_lifecycle(reason, _utcnow_iso())
        except Exception as err:
            self._record_error("lifecycle_release", err)

    def _check_lifecycle_timeout_safe(self) -> None:
        """Release lifecycle if max_boost_duration_s has elapsed since apply. Never raises."""
        try:
            model = self._boost_model()
            if model is None:
                return
            from ..models.boost import BoostLifecycle
            s = model._state
            if s.lifecycle not in (BoostLifecycle.APPLIED, BoostLifecycle.ACTIVE):
                return
            if not s.lifecycle_start_ts:
                return
            from datetime import datetime, timezone
            start = datetime.fromisoformat(s.lifecycle_start_ts.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            elapsed = (now - start).total_seconds()
            if elapsed >= s.lifecycle_max_duration_s:
                model.release_lifecycle("timeout", _utcnow_iso())
        except Exception as err:
            self._record_error("lifecycle_timeout", err)

    def _get_le2_predictions_for_zone(self) -> dict:
        """Return last_predictions dict for the current zone. Isolated for testing."""
        try:
            zr = self._runtime._zone(self._zone)
            return getattr(zr, "last_predictions", {}) or {}
        except Exception:
            return {}

    def _check_lifecycle_deescalation_safe(self, recommendation: dict) -> None:
        """Release active boost early when real-time data shows heating is self-sufficient.

        Three ordered checks (early return on first match):
        1. released_tpi_sufficient: LE2 HEAT_RATE projection shows TPI closes gap alone,
           within the actual schedule remaining time (or fallback horizon when no schedule).
           High TPI duty alone does NOT prove sufficiency — high duty proves high demand.
        2. released_overshoot_risk: LE2 EXPECTED_OVERSHOOT (AfterheatModel residual rise)
           covers remaining gap near target (≤ 0.3°C). Safety heuristic (remaining ≤ 0.3°C
           AND slope > 0 AND active ≥ 600s) fires as fallback when prediction unavailable.
        3. released_afterheat_sufficient: LE2 EXPECTED_OVERSHOOT (AfterheatModel residual
           rise = learned physical afterheat) covers remaining deficit with safety margin.
           BOOST_OUTCOME must NOT be used here — it measures historical boost quality,
           not physical residual rise after heating stops.

        Authority map (no parallel thermal logic):
          TPI sufficient    → HEAT_RATE + remaining deficit + schedule remaining time
          Overshoot risk    → EXPECTED_OVERSHOOT (AfterheatModel) primary + heuristic fallback
          Afterheat suff.   → EXPECTED_OVERSHOOT (AfterheatModel), authoritative

        The 3600s timeout in _check_lifecycle_timeout_safe remains the absolute safety cap.
        Never raises.
        """
        try:
            model = self._boost_model()
            if model is None:
                return
            from ..models.boost import BoostLifecycle
            s = model._state
            if s.lifecycle not in (BoostLifecycle.APPLIED, BoostLifecycle.ACTIVE):
                return
            # Coordinator writes current_temp (not current_temperature) in the rec dict
            current_temp = recommendation.get("current_temp")
            target = (recommendation.get("effective_target")
                      or recommendation.get("adjusted_target"))
            if current_temp is None or target is None:
                return
            try:
                current_f = float(current_temp)
                target_f = float(target)
            except (TypeError, ValueError):
                return
            remaining = target_f - current_f
            if remaining <= 0:
                return  # target already reached; handled by target_reached check
            slope = recommendation.get("temp_slope")  # °C/min from coordinator
            ts = _utcnow_iso()
            params = model._params

            # Compute active duration since lifecycle start
            active_dur_s = 0.0
            if s.lifecycle_start_ts:
                try:
                    from datetime import datetime
                    t0 = datetime.fromisoformat(s.lifecycle_start_ts.replace("Z", "+00:00"))
                    now_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    active_dur_s = max(0.0, (now_dt - t0).total_seconds())
                except Exception:
                    pass

            # Compute remaining time to comfort schedule target (for TPI sufficiency gate)
            remaining_time_to_target_min: Optional[float] = None
            if self._last_comfort_time_utc:
                try:
                    from datetime import datetime
                    comfort_dt = datetime.fromisoformat(
                        self._last_comfort_time_utc.replace("Z", "+00:00"))
                    now_dt2 = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    rem_s = (comfort_dt - now_dt2).total_seconds()
                    if rem_s > 0:
                        remaining_time_to_target_min = rem_s / 60.0
                except Exception:
                    pass

            # Read LE2 predictions for thermal authority (never raises)
            heat_rate_c_per_h: Optional[float] = None
            # AfterheatModel residual rise: authoritative for overshoot risk AND afterheat
            afterheat_rise_c: Optional[float] = None
            afterheat_confidence: float = 0.0
            afterheat_valid: bool = False
            try:
                from ..contracts import PredictionType
                le2_preds = self._get_le2_predictions_for_zone()

                # HEAT_RATE: authoritative for TPI sufficiency projection
                hr_pred = le2_preds.get(PredictionType.HEAT_RATE)
                if hr_pred is not None and not getattr(hr_pred, "fallback_used", True):
                    hr_conf = float(getattr(hr_pred, "confidence", 0.0) or 0.0)
                    if hr_conf >= params.deescalation_tpi_heat_rate_min_confidence:
                        hr_val = hr_pred.values.get("heat_rate", 0.0)
                        if hr_val is not None:
                            hr_float = float(hr_val)
                            if hr_float > 0.0:
                                heat_rate_c_per_h = hr_float

                # EXPECTED_OVERSHOOT (from AfterheatModel): authoritative for overshoot risk
                # and afterheat sufficiency. Values key: "expected_overshoot" (unit: °C).
                # BOOST_OUTCOME is NOT used here — it is historical boost outcome quality.
                # Validity gates (all must pass before prediction is used for control):
                #   1. fallback_used=False  — non-cold-start learned prediction only
                #   2. unit must be "C" or "CELSIUS"  — physical sanity check
                #   3. no "stale"/"superseded" in warnings  — prediction currency
                #   4. source_episode_id matches current episode (if both known)
                #   5. confidence >= minimum threshold (checked in release checks below)
                eo_pred = le2_preds.get(PredictionType.EXPECTED_OVERSHOOT)
                if eo_pred is not None and not getattr(eo_pred, "fallback_used", True):
                    # Gate 2: unit — use central contract normalizer (see _is_celsius_unit)
                    _eo_units = getattr(eo_pred, "units", {}) or {}
                    _eo_unit = _eo_units.get("expected_overshoot", "")
                    if _is_celsius_unit(_eo_unit):
                        # Gate 3: stale / superseded
                        _eo_warns = getattr(eo_pred, "warnings", ()) or ()
                        if "stale" not in _eo_warns and "superseded" not in _eo_warns:
                            # Gate 4: episode binding
                            _eo_episode = getattr(eo_pred, "source_episode_id", None)
                            _cur_episode = s.current_episode_id
                            _episode_ok = (
                                not _eo_episode or not _cur_episode
                                or _eo_episode == _cur_episode)
                            if _episode_ok:
                                eo_conf = float(getattr(eo_pred, "confidence", 0.0) or 0.0)
                                eo_val = eo_pred.values.get("expected_overshoot")
                                if eo_val is not None and eo_conf > 0.0:
                                    afterheat_rise_c = float(max(0.0, eo_val))
                                    afterheat_confidence = eo_conf
                                    afterheat_valid = True
            except Exception:
                pass

            # 1. TPI sufficient: LE2 heat-rate shows TPI closes gap within the effective horizon.
            # Effective horizon: remaining_time_to_target - safety_margin (when schedule known),
            # or fallback fixed horizon (when no schedule available — conservative).
            if (heat_rate_c_per_h is not None
                    and remaining <= params.deescalation_tpi_max_remaining_c):
                try:
                    heat_rate_per_min = heat_rate_c_per_h / 60.0
                    predicted_min = remaining / heat_rate_per_min
                    if remaining_time_to_target_min is not None:
                        # Use actual remaining time minus safety margin
                        effective_horizon = (remaining_time_to_target_min
                                             - params.deescalation_tpi_safety_margin_min)
                    else:
                        # No schedule available: use conservative fixed fallback horizon
                        effective_horizon = params.deescalation_tpi_safety_horizon_min
                    slope_ok = (slope is None or float(slope) > 0.0)
                    if predicted_min <= effective_horizon and effective_horizon > 0 and slope_ok:
                        model.release_lifecycle("released_tpi_sufficient", ts)
                        return
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

            # 2. Overshoot risk: LE2 EXPECTED_OVERSHOOT (AfterheatModel residual rise) as primary.
            # Only fires near-target (remaining ≤ overshoot_deficit_c = 0.3°C) to ensure
            # separation from the wider-gap afterheat check (Check 3).
            if (afterheat_valid
                    and afterheat_confidence >= params.deescalation_overshoot_min_confidence
                    and afterheat_rise_c is not None
                    and remaining <= params.deescalation_overshoot_deficit_c):
                try:
                    if (remaining <= afterheat_rise_c + params.deescalation_overshoot_safety_buffer_c
                            and (slope is None or float(slope) > 0.0)):
                        model.release_lifecycle("released_overshoot_risk", ts)
                        return
                except (TypeError, ValueError):
                    pass
            # Heuristic fallback: only after min active time and rising slope. No prediction needed.
            if slope is not None and active_dur_s >= params.deescalation_overshoot_min_active_s:
                try:
                    if (remaining <= params.deescalation_overshoot_deficit_c
                            and float(slope) > 0.0):
                        model.release_lifecycle("released_overshoot_risk", ts)
                        return
                except (TypeError, ValueError):
                    pass

            # 3. Afterheat sufficient: AfterheatModel residual rise covers remaining gap.
            # Authority: PredictionType.EXPECTED_OVERSHOOT (not BOOST_OUTCOME).
            # No valid non-cold-start prediction → no release (slope alone is not afterheat).
            if (afterheat_valid
                    and afterheat_confidence >= params.deescalation_afterheat_min_confidence
                    and afterheat_rise_c is not None):
                try:
                    covers_gap = (
                        afterheat_rise_c >= remaining + params.deescalation_afterheat_safety_margin_c)
                    slope_supporting = (slope is None or float(slope) > 0.0)
                    if covers_gap and slope_supporting:
                        model.release_lifecycle("released_afterheat_sufficient", ts)
                        return
                except (TypeError, ValueError):
                    pass
        except Exception as err:
            self._record_error("lifecycle_deescalation", err)

    def adjust_recommendation_safe(self, recommendation: dict, *,
                                   boost_runtime_limit: Optional[float] = None) -> None:
        """In CONTROL mode only, apply LE 2.0 boost authority to the dispatched recommendation.

        Full authority: LE 2.0 determines the final boost offset (both increases and
        decreases). The existing TPI setpoint is the baseline; LE 2.0 may increase or
        reduce it, subject to confidence gate and device clamps. Never raises; a no-op
        in any non-CONTROL mode (SHADOW stays byte-identical).

        Side-effects on ``recommendation`` (Phase B1 provenance tracking):
          ``_boost_rejection_reason`` — gate that blocked boost, or None when applied.
          ``_boost_candidate_c``     — raw LE2 °C proposal (pre-guard); None = no proposal.
          ``_boost_applied_c``       — °C offset actually written; None when not applied.
        """
        recommendation["_boost_rejection_reason"] = None
        recommendation["_boost_candidate_c"] = None
        recommendation["_boost_applied_c"] = None
        if not self.control_enabled:
            return
        try:
            target = recommendation.get("effective_target")
            if target is None:
                target = recommendation.get("adjusted_target")
            setpoint = recommendation.get("trv_setpoint")
            if target is None or setpoint is None:
                recommendation["_boost_rejection_reason"] = "no_target"
                return
            # Direct valve TRVs: setpoint is always target; heating authority is valve %.
            # An additive °C offset does not apply — skip boost authority for these.
            if recommendation.get("tpi_valve_direct"):
                recommendation["_boost_rejection_reason"] = "direct_valve"
                return
            # Block during safety conditions; release lifecycle if currently applied
            if recommendation.get("window_open"):
                self._release_boost_lifecycle_safe("window_open")
                recommendation["_boost_rejection_reason"] = "window_open"
                return
            if recommendation.get("heating_failure"):
                self._release_boost_lifecycle_safe("heating_failure")
                recommendation["_boost_rejection_reason"] = "heating_failure"
                return
            # Release on mode change (non-heating mode ends boost context)
            mode = recommendation.get("mode") or recommendation.get("hvac_mode")
            if mode is not None and str(mode).lower() not in ("heat", "auto", "heat_cool"):
                self._release_boost_lifecycle_safe("mode_change")
                recommendation["_boost_rejection_reason"] = "mode_change"
                return
            # Hard release: manual override — user explicitly changed temperature during boost.
            # Outcome is marked as manually influenced; no automatic counter-correction against user.
            if recommendation.get("override_active"):
                self._release_boost_lifecycle_safe("manual_override")
                recommendation["_boost_rejection_reason"] = "manual_override"
                return
            # Hard release: effective target changed — old episode no longer matches context.
            # Prevents carrying a stale boost offset into a new schedule slot or target.
            try:
                _bm_tc = self._boost_model()
                if _bm_tc is not None and _bm_tc._state.lifecycle_base_target_c is not None:
                    if abs(float(_bm_tc._state.lifecycle_base_target_c) - float(target)) > 0.1:
                        self._release_boost_lifecycle_safe("target_change")
                        recommendation["_boost_rejection_reason"] = "target_change"
                        return
            except Exception:
                pass
            # Hard release: schedule/comfort-time context changed with target unchanged.
            # comfort_time_utc is the clearest stable context identifier: it changes when
            # the schedule period transitions (new slot, preheat→comfort, setback shift, etc.).
            # One cycle lag (from observe_safe) is acceptable — same lag as TPI sufficiency.
            try:
                _bm_ctx = self._boost_model()
                if _bm_ctx is not None:
                    _lc_ctx = (_bm_ctx._state.lifecycle.value
                               if hasattr(_bm_ctx._state.lifecycle, "value")
                               else str(_bm_ctx._state.lifecycle))
                    if _lc_ctx in ("applied", "active"):
                        # At least one side must be non-None (both None = no schedule, no change)
                        if (self._boost_applied_comfort_time_utc is not None
                                or self._last_comfort_time_utc is not None):
                            if self._last_comfort_time_utc != self._boost_applied_comfort_time_utc:
                                self._release_boost_lifecycle_safe("schedule_change")
                                recommendation["_boost_rejection_reason"] = "schedule_change"
                                return
            except Exception:
                pass
            # Release on lifecycle timeout (max_boost_duration_s elapsed)
            self._check_lifecycle_timeout_safe()
            # Hard releases must run BEFORE soft deescalation so they always take priority.
            # Hard release: early cutoff / coasting hold — Boost + EarlyCutoff must not overlap.
            early_cutoff_state = recommendation.get("early_cutoff_state")
            if early_cutoff_state in ("cutoff_applied", "coasting_hold"):
                self._release_boost_lifecycle_safe("early_cutoff")
                recommendation["_boost_rejection_reason"] = "early_cutoff"
                return  # Hard release: no boost applied this cycle
            # Hard release: target already reached — no boost benefit remaining.
            # Coordinator key is "current_temp" (not "current_temperature")
            current_temp = recommendation.get("current_temp")
            if current_temp is not None:
                try:
                    if float(current_temp) >= float(target):
                        self._release_boost_lifecycle_safe("target_reached")
                        recommendation["_boost_rejection_reason"] = "target_reached"
                        return
                except (TypeError, ValueError):
                    pass
            # Soft deescalation: release active boost mid-episode when TPI or thermal evidence
            # shows the boost is no longer needed. Only fires after all hard releases above.
            self._check_lifecycle_deescalation_safe(recommendation)
            # Deescalation dispatch: hard or soft, depending on the reason.
            # Hard deesc (overshoot/afterheat): boost is physically overfluous — immediate 0,
            #   same cycle. No step-down; TPI takes over as sole heating authority.
            # Soft deesc (TPI sufficient): controlled step-down per cycle; abrupt removal
            #   not required because TPI *will* close the gap within the schedule window.
            try:
                _bm_soft = self._boost_model()
                if _bm_soft is not None:
                    _slc = (_bm_soft._state.lifecycle.value
                            if hasattr(_bm_soft._state.lifecycle, "value")
                            else str(_bm_soft._state.lifecycle))
                    _HARD_DEESC = frozenset({
                        "released_overshoot_risk",
                        "released_afterheat_sufficient",
                    })
                    _SOFT_DEESC = frozenset({"released_tpi_sufficient"})
                    if _slc in _HARD_DEESC:
                        recommendation["_boost_rejection_reason"] = "deescalation_hard"
                        return  # hard: boost fully removed in this cycle, no step-down
                    if _slc in _SOFT_DEESC:
                        _cur_off = _bm_soft._state.applied_offset_c or 0.0
                        _step = getattr(_bm_soft._params, "deescalation_soft_step_c", 0.5)
                        _new_off = max(0.0, _cur_off - _step)
                        if _new_off > 0.0:
                            _bm_soft.update_deescalation_offset(_new_off)
                            recommendation["tpi_baseline_setpoint"] = float(setpoint)
                            recommendation["trv_setpoint"] = round(
                                float(setpoint) + _new_off, 4)
                            recommendation["_boost_applied_c"] = _new_off
                            recommendation["le2_boost_adjusted"] = True
                        else:
                            recommendation["_boost_rejection_reason"] = "deescalation_soft"
                        return  # no further boost during soft deescalation
            except Exception as err:
                self._record_error("soft_deescalation", err)
            # Cooldown guard: skip boost application during active cooldown period
            _model_cooldown = self._boost_model()
            if _model_cooldown is not None and _model_cooldown.cooldown_active(_utcnow_iso()):
                recommendation["_boost_rejection_reason"] = "cooldown_active"
                return
            # Baseline = currently applied le2 boost (not TPI offset reconstruction).
            # Anti-chatter compares against this: 0.0 when no boost active.
            baseline_offset = 0.0
            try:
                _bm = self._boost_model()
                if _bm is not None:
                    _lc = _bm._state.lifecycle.value if hasattr(
                        _bm._state.lifecycle, "value") else str(_bm._state.lifecycle)
                    if _lc in ("applied", "active"):
                        baseline_offset = float(_bm._state.applied_offset_c)
            except Exception:
                baseline_offset = max(0.0, float(setpoint) - float(target))
            zr = self._runtime._zone(self._zone)
            boost_pred = zr.last_predictions.get(PredictionType.BOOST_FACTOR)
            proposed = boost_pred.values.get("boost_factor") if boost_pred else None
            if proposed is None:
                recommendation["_boost_rejection_reason"] = "no_prediction"
                return
            recommendation["_boost_candidate_c"] = float(proposed)
            ctx = ControlContext(
                window_open=bool(recommendation.get("window_open")),
                heating_failure=bool(recommendation.get("heating_failure")),
                boost_runtime_limit=boost_runtime_limit)
            decision = self._runtime.control_decision(
                self._zone, ControlFeature.BOOST_OFFSET, baseline_offset, proposed,
                ConfidencePurpose.BOOST, context=ctx, unit="celsius_offset",
                model_version=getattr(boost_pred, "model_version", None),
                parameter_version=getattr(boost_pred, "parameter_version", None))
            # Full authority: apply any LE 2.0 decision (increase or decrease)
            if decision.applied and decision.final_value is not None:
                # Wire lifecycle: transition to APPLIED for episode tracking.
                # apply_lifecycle() returns False when episode binding blocks retry
                # (same episode that previously failed). In that case, skip the setpoint
                # update — the failed episode must not boost again in this context.
                _lifecycle_applied = False
                try:
                    model = self._boost_model()
                    if model is not None:
                        episode_id = str(getattr(zr, "last_decision_id", None) or "")
                        _lifecycle_applied = model.apply_lifecycle(
                            episode_id=episode_id,
                            applied_offset_c=decision.final_value,
                            base_target_c=float(target),
                            ts=_utcnow_iso())
                    else:
                        _lifecycle_applied = True  # no model → no binding check → allow
                except Exception as err:
                    self._record_error("lifecycle_apply", err)
                    _lifecycle_applied = True  # error path → allow (fail open for heating)
                if _lifecycle_applied:
                    # Record context at boost-apply time for schedule-change detection.
                    # Updated every cycle boost is applied; compared next cycle to detect transitions.
                    self._boost_applied_comfort_time_utc = self._last_comfort_time_utc
                    # TPI authority: final = tpi_baseline_setpoint + le2_boost_offset.
                    # Preserve original TPI baseline for audit before overwriting.
                    recommendation["tpi_baseline_setpoint"] = float(setpoint)
                    recommendation["trv_setpoint"] = round(
                        float(setpoint) + decision.final_value, 4)
                    recommendation["_boost_applied_c"] = decision.final_value
                    recommendation["le2_boost_adjusted"] = True
                else:
                    recommendation["_boost_rejection_reason"] = "lifecycle_blocked"
            else:
                recommendation["_boost_rejection_reason"] = "guard_rejected"
        except Exception as err:
            self._record_error("control", err)

    def build_live_decision_pre_dispatch(
        self,
        recommendation: Mapping[str, Any],
        *,
        ts: Optional[str] = None,
    ) -> Optional[Any]:
        """Create pre-dispatch LiveDecisionRecord from the current recommendation.

        Called after ``adjust_recommendation_safe()`` but before actual HA dispatch.
        Returns an immutable record; the caller must add dispatch fields via
        ``dataclasses.replace()`` after dispatch and then pass the completed record
        to ``compute_decision_trace_safe()``.  Returns None if the shadow is disabled
        or an internal error occurs — never raises.
        """
        if not self._enabled:
            return None
        try:
            from ..decision.contracts import LiveDecisionRecord
            from ..contracts import PredictionType as _PT
            zr = self._runtime._zone(self._zone)
            decision_id = getattr(zr, "last_decision_id", None)
            _tpi_diag = recommendation.get("tpi_coef_diag") or {}
            boost_applied = bool(recommendation.get("le2_boost_adjusted"))
            baseline_sp = (
                recommendation.get("tpi_baseline_setpoint")
                if boost_applied
                else recommendation.get("trv_setpoint")
            )
            mode_str = "control" if self._runtime.control_enabled else "shadow"
            # Onset delay — read from current predictions (same source as resolver path)
            onset_delay_min: Optional[float] = None
            onset_delay_source: Optional[str] = None
            try:
                _last_preds = getattr(zr, "last_predictions", {}) or {}
                _od_pred = _last_preds.get(_PT.ONSET_DELAY)
                if _od_pred is not None:
                    _od_fb = getattr(_od_pred, "fallback_used", True)
                    _od_conf = float(getattr(_od_pred, "confidence", 0.0) or 0.0)
                    if not _od_fb and _od_conf >= self._PREHEAT_MIN_CONFIDENCE:
                        onset_delay_min = float(
                            _od_pred.values.get("onset_delay_minutes", 0.0) or 0.0)
                        onset_delay_source = "valid"
                    else:
                        onset_delay_min = self._ONSET_DELAY_PRIOR_MIN
                        onset_delay_source = "cold_start_prior"
            except Exception:
                pass
            return LiveDecisionRecord(
                decision_id=decision_id,
                zone_id=self._zone,
                ts=ts or _utcnow_iso(),
                mode=mode_str,
                baseline_setpoint_c=(float(baseline_sp) if baseline_sp is not None else None),
                final_setpoint_c=(float(recommendation["trv_setpoint"])
                                  if recommendation.get("trv_setpoint") is not None else None),
                boost_applied=boost_applied,
                boost_candidate_c=(float(recommendation["_boost_candidate_c"])
                                   if recommendation.get("_boost_candidate_c") is not None
                                   else None),
                boost_applied_c=(float(recommendation["_boost_applied_c"])
                                 if recommendation.get("_boost_applied_c") is not None
                                 else None),
                boost_rejected_reason=recommendation.get("_boost_rejection_reason"),
                preheat_minutes=float(recommendation.get("preheat_minutes") or 0.0),
                preheat_status=recommendation.get("preheat_status"),
                early_cutoff_state=recommendation.get("early_cutoff_state"),
                heat_rate_c_per_h=_tpi_diag.get("heat_rate_c_per_h"),
                heat_rate_confidence=_tpi_diag.get("hr_confidence"),
                heat_rate_source=recommendation.get("tpi_coef_source"),
                onset_delay_min=onset_delay_min,
                onset_delay_source=onset_delay_source,
            )
        except Exception as err:
            self._record_error("live_decision_record", err)
            return None

    def _build_trace_from_live_record(
        self,
        record: Any,
        recommendation: Mapping[str, Any],
        *,
        ts: Optional[str] = None,
    ) -> Any:
        """Derive DecisionTrace from the authoritative live record — no resolver re-run."""
        from ..decision.contracts import (
            DecisionTrace, DecisionTraceEntry,
            DecisionReason, FallbackReason, DecisionMode,
        )

        boost_applied = record.boost_applied
        boost_applied_c = record.boost_applied_c or 0.0
        boost_candidate = record.boost_candidate_c
        boost_rejected = record.boost_rejected_reason

        boost_entry = DecisionTraceEntry(
            feature="boost_offset",
            baseline_value=0.0,
            le2_value=boost_candidate,
            final_value=boost_applied_c if boost_applied else 0.0,
            applied=boost_applied,
            reason=(DecisionReason.LE2_APPLIED.value if boost_applied
                    else (boost_rejected or DecisionReason.BASELINE.value)),
            fallback_reason=(FallbackReason.NONE.value if boost_applied
                             else FallbackReason.GUARD_BLOCKED.value),
            confidence=None,
            clamp_applied=0.0,
        )

        preheat_min = record.preheat_minutes
        preheat_status = record.preheat_status
        preheat_is_le2 = preheat_status in ("valid", "valid_hl_prior")
        preheat_entry = DecisionTraceEntry(
            feature="preheat_start",
            baseline_value=recommendation.get("preheat_baseline_minutes"),
            le2_value=preheat_min if preheat_is_le2 else None,
            final_value=preheat_min,
            applied=preheat_min > 0.0,
            reason=("le2_applied" if preheat_is_le2 else "deterministic_baseline"),
            fallback_reason=("none" if preheat_is_le2 else "shadow_mode"),
            confidence=record.heat_rate_confidence,
            clamp_applied=0.0,
        )

        ec_state = record.early_cutoff_state
        ec_active = ec_state in ("cutoff_applied", "coasting_hold")
        ec_contribution = recommendation.get("early_cutoff_contribution_c")
        ec_entry = DecisionTraceEntry(
            feature="early_cutoff",
            baseline_value=None,
            le2_value=ec_contribution,
            final_value=ec_contribution if ec_active else None,
            applied=ec_active,
            reason=("le2_applied" if ec_active else DecisionReason.BASELINE.value),
            fallback_reason=(FallbackReason.NONE.value if ec_active
                             else FallbackReason.GUARD_BLOCKED.value),
            confidence=None,
            clamp_applied=0.0,
        )

        reason_codes: list[str] = []
        if boost_applied:
            reason_codes.append(DecisionReason.LE2_APPLIED.value)
        if boost_rejected:
            reason_codes.append(boost_rejected)
        if not boost_applied and not boost_rejected:
            reason_codes.append(DecisionReason.BASELINE.value)

        mode_str = (DecisionMode.CONTROL.value if self._runtime.control_enabled
                    else DecisionMode.SHADOW.value)

        onset_delay = record.onset_delay_min
        room_heat_dur: Optional[float] = None
        if onset_delay is not None:
            room_heat_dur = max(0.0, preheat_min - onset_delay)

        return DecisionTrace(
            zone_id=record.zone_id,
            ts=ts or record.ts,
            mode=mode_str,
            decision_id=record.decision_id,
            baseline_setpoint_c=record.baseline_setpoint_c,
            final_setpoint_c=record.final_setpoint_c,
            applied_any=boost_applied or ec_active,
            entries=(boost_entry, preheat_entry, ec_entry),
            reason_codes=tuple(sorted(set(reason_codes))),
            preheat_minutes_le2=preheat_min if preheat_is_le2 else None,
            preheat_status=preheat_status,
            preheat_baseline_minutes=recommendation.get("preheat_baseline_minutes"),
            selected_preheat_min=preheat_min,
            heat_rate_c_per_h=record.heat_rate_c_per_h,
            heat_rate_confidence=record.heat_rate_confidence,
            heat_rate_status=record.heat_rate_source,
            effective_onset_delay_min=onset_delay or 0.0,
            onset_delay_source=record.onset_delay_source,
            onset_delay_status=record.onset_delay_source,
            effective_room_heating_duration_min=room_heat_dur,
            preheat_command_lead_time_min=preheat_min,
            early_cutoff_state=ec_state,
            early_cutoff_hold_active=ec_active,
            boost_offset_c_applied=boost_applied_c if boost_applied else None,
            boost_offset_requested_c=boost_candidate,
            boost_rejected_reason=boost_rejected,
            final_device_setpoint_c=(record.dispatch_setpoint_c
                                      if record.dispatch_setpoint_c is not None
                                      else record.final_setpoint_c),
        )

    def compute_decision_trace_safe(self, recommendation: Mapping[str, Any], *,
                                    active_control: bool = False,
                                    ts: Optional[str] = None,
                                    live_record: Optional[Any] = None) -> None:
        """Produce a DecisionTrace for this coordinator cycle.

        When ``live_record`` (a ``LiveDecisionRecord``) is provided, the trace is
        derived directly from the authoritative live path without re-running the
        resolver.  When omitted, the full pipeline is re-run for backward
        compatibility (test-only path). Never raises into the heating cycle.
        """
        if not self._enabled:
            return
        try:
            if live_record is not None:
                trace = self._build_trace_from_live_record(
                    live_record, recommendation, ts=ts or _utcnow_iso())
            else:
                # LEGACY / TEST-ONLY: backward-compat path for unit tests that call
                # compute_decision_trace_safe() without a live_record.  The production
                # coordinator always provides live_record; this branch must never run
                # in a live coordinator cycle (enforced by test_production_live_record_always_provided).
                zr = self._runtime._zone(self._zone)
                preds = build_learning_prediction_set(
                    self._zone, getattr(zr, "last_predictions", {}) or {},
                    getattr(zr, "last_confidence", {}) or {},
                    decision_id=getattr(zr, "last_decision_id", None))
                mode = (DecisionMode.CONTROL if self._runtime.control_enabled
                        else DecisionMode.SHADOW)
                trace, _ = self._pipeline.run(
                    self._zone, ts or _utcnow_iso(), recommendation, preds, mode=mode,
                    active_control=active_control,
                    decision_id=getattr(zr, "last_decision_id", None))
            trace_dict = trace.support_dict()
            # Enrich trace with TPI authority chain fields (for audit/debug)
            # These allow verification that LE 2.0 setpoint and TPI baseline are not mixed.
            # tpi_baseline_setpoint_c: written by adjust_recommendation_safe before boost.
            # Falls back to trv_setpoint in shadow mode (no boost → not overwritten).
            _tpi_diag = recommendation.get("tpi_coef_diag") or {}
            trace_dict.update({
                # decision_id: links this trace to the lifecycle capture and any outcome record.
                "decision_id": trace.decision_id,
                "comfort_target_c": recommendation.get("effective_target"),
                "tpi_duty_c": recommendation.get("tpi_duty_cycle"),
                "tpi_baseline_setpoint_c": (recommendation.get("tpi_baseline_setpoint")
                                            or recommendation.get("trv_setpoint")),
                "tpi_valve_direct": bool(recommendation.get("tpi_valve_direct", False)),
                # HeatLoss authority trace (Phase 19D — Variante B bounded adaptive heuristic)
                "tpi_coef_source": recommendation.get("tpi_coef_source"),
                # Coefficient authority chain: default → learned → blend → used
                "tpi_coef_int_default": _tpi_diag.get("coef_int_default"),
                "tpi_coef_int_learned": _tpi_diag.get("coef_int_learned"),
                "tpi_coef_blend_weight": _tpi_diag.get("blend_weight"),
                "tpi_coef_int_used": recommendation.get("tpi_coef_int"),
                "tpi_coef_ext_used": recommendation.get("tpi_coef_ext"),
                # Prediction values feeding the ratio
                "heat_rate_prediction_c_per_h": _tpi_diag.get("heat_rate_c_per_h"),
                "heat_loss_prediction_c_per_h": recommendation.get("tpi_hl_rate"),
                "heat_rate_confidence": _tpi_diag.get("hr_confidence"),
                "heat_loss_confidence": _tpi_diag.get("hl_confidence"),
                # Outdoor/context availability
                "tpi_outdoor_context_available": _tpi_diag.get("outdoor_context_available"),
                "tpi_relative_only": _tpi_diag.get("relative_only"),
                # Backward-compat alias (used by parity tests)
                "tpi_heat_loss_prediction_c_per_h": recommendation.get("tpi_hl_rate"),
                # Step-limiting transition audit fields (Phase 19D)
                "tpi_coef_int_candidate": _tpi_diag.get("coef_int_candidate"),
                "tpi_coef_int_previous": _tpi_diag.get("coef_int_previous"),
                "tpi_coef_transition_applied": _tpi_diag.get("transition_applied"),
                "tpi_coef_transition_reason": _tpi_diag.get("transition_reason"),
                "tpi_coef_max_step_up": _tpi_diag.get("coef_int_max_step_up"),
                "tpi_coef_max_step_down": _tpi_diag.get("coef_int_max_step_down"),
                # Duty-level transition audit
                "tpi_duty_previous": _tpi_diag.get("duty_previous"),
                "tpi_duty_candidate": _tpi_diag.get("duty_candidate"),
                "tpi_duty_used": _tpi_diag.get("duty_used"),
            })
            # Enrich trace with lifecycle diagnostics from BoostModel (not available in resolver)
            try:
                boost_model = self._boost_model()
                if boost_model is not None:
                    s = boost_model._state
                    ts_now = ts or _utcnow_iso()
                    active_dur: Optional[float] = None
                    cooldown_rem: Optional[float] = None
                    from ..models.boost import BoostLifecycle
                    from datetime import datetime, timezone, timedelta
                    if s.lifecycle in (BoostLifecycle.APPLIED, BoostLifecycle.ACTIVE) \
                            and s.lifecycle_start_ts:
                        try:
                            start = datetime.fromisoformat(
                                s.lifecycle_start_ts.replace("Z", "+00:00"))
                            now_dt = datetime.fromisoformat(ts_now.replace("Z", "+00:00"))
                            active_dur = round((now_dt - start).total_seconds(), 1)
                        except Exception:
                            pass
                    if s.cooldown_until_ts:
                        try:
                            until_dt = datetime.fromisoformat(
                                s.cooldown_until_ts.replace("Z", "+00:00"))
                            now_dt2 = datetime.fromisoformat(ts_now.replace("Z", "+00:00"))
                            rem = (until_dt - now_dt2).total_seconds()
                            cooldown_rem = round(max(0.0, rem), 1)
                        except Exception:
                            pass
                    # Classify deescalation type from release reason.
                    # Overshoot/afterheat deescalation are HARD (immediate 0, same cycle).
                    # Only TPI sufficient is soft (step-down per cycle).
                    _release_reason = s.lifecycle_release_reason
                    _hard_reasons = frozenset({
                        "window_open", "heating_failure", "mode_change",
                        "early_cutoff", "target_reached", "manual_override",
                        "schedule_change", "timeout", "no_response", "overshoot",
                        "target_change",
                        "released_overshoot_risk", "released_afterheat_sufficient"})
                    _soft_reasons = frozenset({"released_tpi_sufficient"})
                    _deesc_type = None
                    if _release_reason in _hard_reasons:
                        _deesc_type = "hard"
                    elif _release_reason in _soft_reasons:
                        _deesc_type = "soft"
                    # Read afterheat prediction for trace enrichment
                    _afterheat_c: Optional[float] = None
                    _afterheat_status: Optional[str] = None
                    _afterheat_conf: Optional[float] = None
                    _boost_outcome_risk: Optional[bool] = None
                    try:
                        from ..contracts import PredictionType as _PT
                        _le2p = self._get_le2_predictions_for_zone()
                        _eo_pred = _le2p.get(_PT.EXPECTED_OVERSHOOT)
                        if _eo_pred is not None:
                            _afterheat_c = float(max(0.0,
                                _eo_pred.values.get("expected_overshoot", 0.0) or 0.0))
                            _afterheat_conf = float(getattr(_eo_pred, "confidence", 0.0) or 0.0)
                            _afterheat_status = (
                                "cold_start" if getattr(_eo_pred, "fallback_used", True)
                                else "learned")
                        else:
                            _afterheat_status = "unavailable"
                        _bo_pred = _le2p.get(_PT.BOOST_OUTCOME)
                        if _bo_pred is not None and not getattr(_bo_pred, "fallback_used", True):
                            _bo_ov = _bo_pred.values.get("expected_overshoot")
                            if _bo_ov is not None:
                                _boost_outcome_risk = float(_bo_ov) > 0.0
                    except Exception:
                        pass
                    # Remaining time to target from cached comfort_time_utc
                    _remaining_tgt_min: Optional[float] = None
                    if self._last_comfort_time_utc:
                        try:
                            from datetime import datetime
                            _c_dt = datetime.fromisoformat(
                                self._last_comfort_time_utc.replace("Z", "+00:00"))
                            _n_dt = datetime.fromisoformat(ts_now.replace("Z", "+00:00"))
                            _rem_s2 = (_c_dt - _n_dt).total_seconds()
                            if _rem_s2 > 0:
                                _remaining_tgt_min = round(_rem_s2 / 60.0, 1)
                        except Exception:
                            pass
                    # Current and new offset for trace
                    _new_off_c = float(recommendation.get("trv_setpoint", 0.0) or 0.0) - \
                        float(recommendation.get("tpi_baseline_setpoint") or
                              recommendation.get("trv_setpoint") or 0.0)
                    _params_soft = boost_model._params
                    trace_dict.update({
                        "boost_lifecycle_state": s.lifecycle.value,
                        "boost_episode_id": s.current_episode_id,
                        "boost_active_duration_s": active_dur,
                        "boost_max_duration_s": s.lifecycle_max_duration_s,
                        "boost_cooldown_remaining_s": cooldown_rem,
                        "boost_release_reason": _release_reason,
                        "boost_applied_offset_c": s.applied_offset_c,
                        "boost_failed_episode_id": s.last_failed_episode_id,
                        "boost_completed_episode_id": s.last_completed_episode_id,
                        "boost_deescalation_type": _deesc_type,
                        "boost_previous_offset_c": s.applied_offset_c,
                        "boost_new_offset_c": max(0.0, _new_off_c) if _new_off_c > 0 else 0.0,
                        "final_device_setpoint_c": recommendation.get("trv_setpoint"),
                        # Afterheat authority (EXPECTED_OVERSHOOT from AfterheatModel)
                        "afterheat_prediction_c": _afterheat_c,
                        "afterheat_prediction_status": _afterheat_status,
                        "afterheat_confidence": _afterheat_conf,
                        "expected_afterheat_c": _afterheat_c,  # alias
                        "boost_outcome_overshoot_risk": _boost_outcome_risk,
                        # TPI sufficiency time fields
                        "remaining_time_to_target_min": _remaining_tgt_min,
                        "tpi_sufficiency_safety_margin_min": getattr(
                            _params_soft, "deescalation_tpi_safety_margin_min", 5.0),
                    })
            except Exception:
                pass
            self._last_trace = trace_dict
        except Exception as err:
            self._record_error("decision_trace", err)

    def clear_last_trace(self) -> None:
        """Clear the cached decision trace.

        Called by the coordinator when no authoritative live record is available
        for this cycle — prevents a stale trace from a prior cycle being read as
        current-cycle provenance.
        """
        self._last_trace = None

    @property
    def last_decision_trace(self) -> Optional[dict]:
        return self._last_trace

    def confidence_display(self) -> float:
        """Combined learning confidence [0,1] from the LE-2.0 aggregator (display only)."""
        from ..decision.confidence_adapter import combined_confidence
        try:
            zr = self._runtime._zone(self._zone)
            return combined_confidence(getattr(zr, "last_confidence", {}) or {})
        except Exception as err:
            self._record_error("confidence_display", err)
            return 0.0

    def confidence_display_attributes(self) -> dict:
        """Legacy-shaped confidence breakdown sourced purely from LE-2.0 (display only)."""
        from ..decision.confidence_adapter import confidence_breakdown
        try:
            zr = self._runtime._zone(self._zone)
            counts = self._runtime.health().model_update_counts
            return confidence_breakdown(getattr(zr, "last_confidence", {}) or {},
                                        model_update_counts=counts)
        except Exception as err:
            self._record_error("confidence_attrs", err)
            return {}

    # Maps Read Gate rejection codes to user-facing forecast trust status strings.
    _TRUST_STATUS_MAP: dict = {
        "missing": "missing",
        "stale": "stale",
        "superseded": "superseded",
    }

    def read_forecast_trust_safe(self) -> tuple:
        """Read LE 2.0 FORECAST_TRUST (dimensionless reliability [0,1]) via validated Read Gate.

        Returns ``(trust, status)`` where:
        - ``trust``: 0.0 when unavailable / cold-start / invalid; float in [0,1] when valid.
        - ``status``: one of "valid", "not_available", "missing", "cold_start", "stale",
          "superseded", "invalid".

        Status "not_available" means the shadow is disabled or an internal error occurred.
        Status "cold_start" means a prior prediction exists but has no real evidence
        (``fallback_used=True``).

        In both unavailable and cold_start cases ``trust=0.0`` so the suppression formula
        ``1 – (1–raw)·trust = 1.0`` leaves heating fully unaffected — the local thermal
        path takes over without any invented high forecast confidence.

        Semantically distinct from FORECAST_BIAS – never mixed.
        """
        if not self._enabled:
            return 0.0, "not_available"
        try:
            from ..decision.runtime_adapter import validated_prediction_read, READ_OK
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            trust_pred = preds.get(PredictionType.FORECAST_TRUST)
            if trust_pred is not None and getattr(trust_pred, "fallback_used", True):
                return 0.0, "cold_start"
            value, read_status = validated_prediction_read(
                preds,
                "forecast_trust",
                zone_id=self._zone,
                prediction_zone_id=self._zone,
                expected_unit="score",
            )
            if read_status == READ_OK and value is not None:
                return float(max(0.0, min(1.0, value))), "valid"
            return 0.0, self._TRUST_STATUS_MAP.get(read_status, "invalid")
        except Exception as err:
            self._record_error("forecast_trust_read", err)
        return 0.0, "not_available"

    def read_forecast_bias_safe(self) -> float:
        """Read LE 2.0 FORECAST_BIAS (signed °C error correction) via validated Read Gate.

        Semantically distinct from FORECAST_TRUST – never used as a reliability score.
        Returns 0.0°C when cold-start (fallback_used=True) or gate rejects.
        """
        if not self._enabled:
            return 0.0
        try:
            from ..decision.runtime_adapter import validated_prediction_read, READ_OK
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            bias_pred = preds.get(PredictionType.FORECAST_BIAS)
            if bias_pred is not None and getattr(bias_pred, "fallback_used", True):
                return 0.0
            value, status = validated_prediction_read(
                preds,
                "forecast_bias",
                zone_id=self._zone,
                prediction_zone_id=self._zone,
                expected_unit="celsius",
            )
            if status == READ_OK and value is not None:
                return float(value)
        except Exception as err:
            self._record_error("forecast_bias_read", err)
        return 0.0

    # Minimum confidence threshold for early cutoff prediction to be acted on.
    _EARLY_CUTOFF_MIN_CONFIDENCE: float = 0.35
    # Absolute maximum residual rise the model may predict (mirrors AfterheatParameters cap).
    _EARLY_CUTOFF_MAX_C: float = 3.0
    # Allowed cutoff at minimum admissible confidence: at conf=0.35 only 0.5°C is safe
    # (worst-case room deficit limited to 0.5°C if afterheat fails).  At conf=1.0 the
    # full model cap (3.0°C) applies.  Linear interpolation in between.
    _EARLY_CUTOFF_MAX_C_AT_MIN_CONF: float = 0.5

    def read_early_cutoff_safe(self) -> tuple:
        """Read LE 2.0 RECOMMENDED_EARLY_CUTOFF (expected residual rise °C) via Read Gate.

        Returns ``(residual_rise_c, status)`` where:
        - ``residual_rise_c``: 0.0 when unavailable / cold-start / low-confidence / gate-rejected;
          non-negative float capped by a confidence-proportional limit otherwise.
        - ``status``: one of "valid", "not_available", "cold_start", "low_confidence",
          "missing", "stale", "superseded", "invalid".

        A versionised prior (``fallback_used=True``) always returns ``(0.0, "cold_start")``.
        Low evidence-quality (``confidence < 0.35``) always returns ``(0.0, "low_confidence")``.
        The returned value is capped proportionally to confidence so that a marginal-confidence
        prediction can only reduce the drive target by a small, low-risk amount.
        """
        if not self._enabled:
            return 0.0, "not_available"
        try:
            from ..decision.runtime_adapter import validated_prediction_read, READ_OK
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            cutoff_pred = preds.get(PredictionType.RECOMMENDED_EARLY_CUTOFF)
            if cutoff_pred is not None and getattr(cutoff_pred, "fallback_used", True):
                return 0.0, "cold_start"
            confidence = 0.0
            if cutoff_pred is not None:
                confidence = float(getattr(cutoff_pred, "confidence", 0.0) or 0.0)
                if confidence < self._EARLY_CUTOFF_MIN_CONFIDENCE:
                    return 0.0, "low_confidence"
            value, read_status = validated_prediction_read(
                preds,
                "early_cutoff",
                zone_id=self._zone,
                prediction_zone_id=self._zone,
                expected_unit="celsius",
            )
            if read_status == READ_OK and value is not None:
                value = float(max(0.0, value))
                # Confidence-proportional cap: limit the allowed cutoff magnitude to
                # reduce control risk at lower confidence.  At minimum admissible
                # confidence (0.35) only 0.5°C is permitted; at confidence=1.0 the
                # full model cap (3.0°C) applies.  Linear interpolation in between.
                conf_range = max(0.001, 1.0 - self._EARLY_CUTOFF_MIN_CONFIDENCE)
                t = max(0.0, min(1.0, (confidence - self._EARLY_CUTOFF_MIN_CONFIDENCE) / conf_range))
                conf_cap = (self._EARLY_CUTOFF_MAX_C_AT_MIN_CONF
                            + t * (self._EARLY_CUTOFF_MAX_C - self._EARLY_CUTOFF_MAX_C_AT_MIN_CONF))
                return min(value, conf_cap), "valid"
            return 0.0, self._TRUST_STATUS_MAP.get(read_status, "invalid")
        except Exception as err:
            self._record_error("early_cutoff_read", err)
        return 0.0, "not_available"

    def read_regime_safe(self) -> Optional[str]:
        """Return the regime string from the last shadow pipeline cycle, or None.

        Returns one of the ``Regime`` enum values (e.g. "active_heating",
        "afterheat", "passive_cooling", "stable", "disturbed", "unknown") or
        ``None`` when the shadow has not yet completed a cycle.

        Used by the coordinator to gate Early Cutoff hold starts: only
        ``"active_heating"`` allows a new hold; ``None`` / ``"unknown"`` / etc.
        preserve the deterministic baseline and prevent an invented regime.
        """
        if not self._enabled:
            return None
        try:
            zr = self._runtime._zone(self._zone)
            return getattr(zr, "last_regime", None)
        except Exception as err:
            self._record_error("regime_read", err)
        return None

    # ── Preheat / HeatRate read ──────────────────────────────────────────────
    # Minimum confidence for a HEAT_RATE prediction to contribute to preheat.
    _PREHEAT_MIN_CONFIDENCE: float = 0.35
    # Onset delay cold-start PRIOR: TRV command → measurable room heating response.
    # This is a versionable prior, not a permanent truth.  Once ONSET_DELAY learning
    # accumulates evidence, read_onset_delay_safe() will return a learned value.
    # Context factors that WILL influence the learned delay (future implementation):
    #   - time bucket (morning vs afternoon)
    #   - prior setback depth and duration (deep overnight setback → longer lag)
    #   - outdoor temperature and wind (cold system → longer warm-up of pipes)
    #   - device type / zone context
    #   - schedule transition type (from night to morning vs from eco to comfort)
    _ONSET_DELAY_PRIOR_MIN: float = 5.0
    # Maximum onset delay admitted as valid (sanity gate for learned values).
    # Raised from 30 → 45 min to match OnsetDelayParameters.max_onset_delay_min:
    # real central-heating supply lags can reach 40+ min after deep overnight setbacks.
    _ONSET_DELAY_MAX_MIN: float = 45.0
    # Conservative room heating rate for the deterministic baseline (no LE needed).
    # 1.5 °C/h is intentionally slow: errs toward starting earlier, not later.
    # Real rooms typically heat at 2–4 °C/h; this under-estimates so the room is
    # warm ON TIME even for sluggish systems.  More aggressive than 0 (LE v1 cold
    # start), but NEVER invented — it's a principled physical prior.
    _DETERMINISTIC_HEAT_RATE_C_PER_H: float = 1.5
    # Safe heat-loss prior used when HEAT_LOSS_RATE prediction is absent, stale,
    # fallback-only, or low-confidence.  Choosing 0.0 would silently assume perfect
    # insulation (net_rate = heat_rate), systematically underestimating preheat
    # duration.  0.3 °C/h is a lower-bound physical prior: even well-insulated rooms
    # lose heat; using it makes the preheat estimate safer while remaining sub-linear
    # relative to the deterministic baseline net-rate (1.5 °C/h).
    _HEAT_LOSS_PRIOR_C_PER_H: float = 0.3
    # Absolute maximum preheat allowed from LE2 (prevents runaway early starts).
    _PREHEAT_MAX_MIN: float = 180.0
    # Minimum temperature deficit that justifies a preheat command (°C).
    _PREHEAT_MIN_DELTA_C: float = 0.5
    # Legacy alias kept for any external references; remove in v1.2
    _PREHEAT_FALLBACK_MIN: float = 60.0

    @classmethod
    def compute_deterministic_preheat_baseline(
        cls, current_temp: Optional[float], comfort_temp: float
    ) -> tuple:
        """Deterministic preheat baseline — no learning required.

        Inputs: ``current_temp`` (°C, may be None), ``comfort_temp`` (°C).

        Behaviour:
        - ``current_temp is None`` → ``(0.0, "unavailable")`` (cannot compute without sensor).
        - ``deficit ≤ _PREHEAT_MIN_DELTA_C`` → ``(0.0, "target_reached")``.
        - Otherwise: ``command_lead = room_heating + onset_prior`` where
          ``room_heating = deficit / _DETERMINISTIC_HEAT_RATE_C_PER_H × 60``.

        Conservative rate (1.5 °C/h) ensures the room is warm ON TIME for
        slow systems; fast rooms finish slightly early, which is safe.
        Result is capped at ``_PREHEAT_MAX_MIN``.

        TRV-only / no outdoor sensor: works identically (no outdoor input needed).
        """
        if current_temp is None:
            return 0.0, "unavailable"
        deficit = comfort_temp - current_temp
        if deficit <= cls._PREHEAT_MIN_DELTA_C:
            return 0.0, "target_reached"
        room_heating = (deficit / cls._DETERMINISTIC_HEAT_RATE_C_PER_H) * 60.0
        total = min(room_heating + cls._ONSET_DELAY_PRIOR_MIN, cls._PREHEAT_MAX_MIN)
        return round(total, 1), "deterministic_baseline"

    def read_onset_delay_safe(self) -> tuple:
        """Read LE 2.0 ONSET_DELAY (minutes): TRV command → measurable room response.

        Returns ``(delay_min, status)`` where status is:
        - ``"valid"``: learned value, confidence-gated.
        - ``"cold_start_prior"``: no evidence yet; returns ``_ONSET_DELAY_PRIOR_MIN``.
        - ``"not_available"``: shadow disabled.

        Context factors registered for future learning (not yet implemented):
        ``time_bucket``, ``setback_depth_c``, ``setback_duration_h``,
        ``outdoor_temp_c``, ``schedule_transition``, ``device_type``.
        These are tracked per episode and stored alongside onset_delay measurements
        so the model can learn situation-specific delays (e.g. morning after deep
        overnight setback is systematically longer than afternoon re-heat).
        """
        if not self._enabled:
            return self._ONSET_DELAY_PRIOR_MIN, "not_available"
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            od_pred = preds.get(PredictionType.ONSET_DELAY)
            if od_pred is None:
                return self._ONSET_DELAY_PRIOR_MIN, "cold_start_prior"
            fallback_used = getattr(od_pred, "fallback_used", True)
            if fallback_used:
                return self._ONSET_DELAY_PRIOR_MIN, "cold_start_prior"
            confidence = float(getattr(od_pred, "confidence", 0.0) or 0.0)
            if confidence < self._PREHEAT_MIN_CONFIDENCE:
                return self._ONSET_DELAY_PRIOR_MIN, "cold_start_prior"
            delay = float(od_pred.values.get("onset_delay", self._ONSET_DELAY_PRIOR_MIN) or self._ONSET_DELAY_PRIOR_MIN)
            delay = max(0.0, min(delay, self._ONSET_DELAY_MAX_MIN))
            return round(delay, 1), "valid"
        except Exception as err:
            self._record_error("onset_delay_read", err)
        return self._ONSET_DELAY_PRIOR_MIN, "cold_start_prior"

    def read_preheat_minutes_safe(
        self,
        current_temp: Optional[float],
        comfort_temp: float,
        *,
        outdoor_temp: Optional[float] = None,
    ) -> tuple:
        """Read LE 2.0 preheat duration (command lead time in minutes).

        Returns ``(minutes, status)`` where:
        - ``minutes``: command lead time ≥ 0 (always the right value for the status).
        - ``status``: one of "valid", "low_confidence", "deterministic_baseline",
          "target_reached", "unavailable", "not_available".

        Three semantically distinct durations are kept separate (A = B + C):
        - **A Command Lead Time**: returned ``minutes``
        - **B Effective Heating Onset Delay**: from ``read_onset_delay_safe()``
          (cold-start prior = 5 min; adaptive once evidence accumulates)
        - **C Effective Room Heating Duration**: ``deficit / net_heat_rate × 60``

        Fallback hierarchy (no LE v1 reads at any level):
        1. LE 2.0 valid prediction (status "valid") — adaptive, confidence-gated
        2. LE 2.0 low-confidence → deterministic baseline (status "deterministic_baseline")
        3. No LE 2.0 evidence → deterministic baseline (status "deterministic_baseline")
        4. No temperature data → (0.0, "unavailable")
        5. Target already reached → (0.0, "target_reached")
        6. Shadow disabled → deterministic baseline (status "deterministic_baseline")
        """
        if not self._enabled:
            return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}

            hr_pred = preds.get(PredictionType.HEAT_RATE)
            hl_pred = preds.get(PredictionType.HEAT_LOSS_RATE)
            af_pred = preds.get(PredictionType.EXPECTED_OVERSHOOT)

            if hr_pred is None:
                return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)

            fallback_used = getattr(hr_pred, "fallback_used", True)
            confidence = float(getattr(hr_pred, "confidence", 0.0) or 0.0)
            heat_rate_c_per_h = float(hr_pred.values.get("heat_rate", 0.0) or 0.0)

            if fallback_used or heat_rate_c_per_h <= 0:
                return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)

            if current_temp is None:
                return 0.0, "unavailable"

            deficit = comfort_temp - current_temp
            if deficit <= self._PREHEAT_MIN_DELTA_C:
                return 0.0, "target_reached"

            if confidence < self._PREHEAT_MIN_CONFIDENCE:
                return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)

            # HEAT_LOSS_RATE gate: absent / fallback / stale / low-confidence prediction
            # must NOT be treated as 0.0 (which would imply perfect insulation).
            # Use the versioned safe prior instead so preheat is never under-estimated.
            _hl_valid = (
                hl_pred is not None
                and not getattr(hl_pred, "fallback_used", True)
                and float(getattr(hl_pred, "confidence", 0.0) or 0.0) >= self._PREHEAT_MIN_CONFIDENCE
            )
            heat_loss_c_per_h = (
                float(hl_pred.values.get("heat_loss_rate", 0.0) or 0.0)
                if _hl_valid
                else self._HEAT_LOSS_PRIOR_C_PER_H
            )
            afterheat_c = float(
                af_pred.values.get("expected_overshoot", 0.0) if af_pred else 0.0) or 0.0

            # Effective Room Heating Duration (C):
            #   net_rate = heat_rate − heat_loss  (min 0.2 C/h floor)
            #   effective_deficit = deficit − afterheat (early-cutoff contribution reduces this)
            #   C = effective_deficit / net_rate × 60
            net_rate = max(heat_rate_c_per_h - heat_loss_c_per_h, 0.2)
            effective_deficit = max(0.0, deficit - max(0.0, afterheat_c))
            room_heating_min = (effective_deficit / net_rate) * 60.0

            # Confidence-proportional cap (0.35–0.6 band): interpolate between
            # deterministic_baseline and PREHEAT_MAX_MIN as confidence rises.
            if confidence < 0.6:
                conf_range = max(0.001, 0.6 - self._PREHEAT_MIN_CONFIDENCE)
                t = max(0.0, min(1.0, (confidence - self._PREHEAT_MIN_CONFIDENCE) / conf_range))
                baseline_min, _ = self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)
                cap = baseline_min + t * (self._PREHEAT_MAX_MIN - baseline_min)
                room_heating_min = min(room_heating_min, cap)

            room_heating_min = max(0.0, min(room_heating_min, self._PREHEAT_MAX_MIN))

            # Command Lead Time (A) = Room Heating Duration (C) + Onset Delay (B).
            # Onset delay comes from read_onset_delay_safe() — adaptive when learned,
            # otherwise returns the cold-start prior.  Never folded into heat rate.
            onset_delay, _ = self.read_onset_delay_safe()
            total_min = room_heating_min + onset_delay
            total_min = min(total_min, self._PREHEAT_MAX_MIN)

            # "valid" requires both HR and HL from learned LE 2.0 predictions.
            # "valid_hl_prior" signals that HR is learned but HL uses the safe prior.
            # A fallback must never be labelled as "valid".
            _status = "valid" if _hl_valid else "valid_hl_prior"
            return round(total_min, 1), _status
        except Exception as err:
            self._record_error("preheat_read", err)
        return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)

    def read_heat_rate_safe(self) -> tuple:
        """Read LE 2.0 HEAT_RATE (°C/h) for display and diagnostics.

        Returns ``(rate_c_per_h, status)`` where:
        - ``rate_c_per_h``: 0.0 when unavailable / cold-start / invalid.
        - ``status``: one of "valid", "not_available", "cold_start", "low_confidence".

        Unit is always °C/h.  Callers needing °C/min must divide by 60.
        Only the effective room heating rate is returned; onset delay is NOT
        included in this rate and must never be included in heat-rate learning.
        """
        if not self._enabled:
            return 0.0, "not_available"
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            hr_pred = preds.get(PredictionType.HEAT_RATE)
            if hr_pred is None:
                return 0.0, "cold_start"
            fallback_used = getattr(hr_pred, "fallback_used", True)
            if fallback_used:
                return 0.0, "cold_start"
            confidence = float(getattr(hr_pred, "confidence", 0.0) or 0.0)
            if confidence < self._PREHEAT_MIN_CONFIDENCE:
                return 0.0, "low_confidence"
            rate = float(hr_pred.values.get("heat_rate", 0.0) or 0.0)
            if rate <= 0:
                return 0.0, "cold_start"
            return rate, "valid"
        except Exception as err:
            self._record_error("heat_rate_read", err)
        return 0.0, "not_available"

    def read_heat_loss_rate_safe(self) -> tuple:
        """Read LE 2.0 HEAT_LOSS_RATE (°C/h) for display.

        Returns ``(rate_c_per_h, status)``.
        """
        if not self._enabled:
            return 0.0, "not_available"
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            hl_pred = preds.get(PredictionType.HEAT_LOSS_RATE)
            if hl_pred is None:
                return 0.0, "cold_start"
            if getattr(hl_pred, "fallback_used", True):
                return 0.0, "cold_start"
            rate = float(hl_pred.values.get("heat_loss_rate", 0.0) or 0.0)
            if rate < 0:
                return 0.0, "cold_start"
            return rate, "valid"
        except Exception as err:
            self._record_error("heat_loss_rate_read", err)
        return 0.0, "not_available"

    # Confidence thresholds for the coef_int blend ramp:
    #   BLEND_MIN  = gate-6 threshold (blend weight = 0 → fully default)
    #   BLEND_FULL = confidence at which blend weight reaches 1 → fully learned
    _BLEND_FULL_CONFIDENCE: float = 0.80
    # Max blend weight when HeatLoss has no outdoor context (relative_only).
    # Without outdoor context the general rate averages all episodes across ΔT — it may
    # understate losses in cold weather and overstate them in mild weather.  Capping the
    # blend at 0.5 ensures the default coef_int always contributes at least 50 %.
    _BLEND_WEIGHT_RELATIVE_ONLY_CAP: float = 0.50

    def read_tpi_coefficients_safe(self) -> tuple:
        """Return LE2-derived TPI coefficients with blend diagnostics (5-tuple).

        Returns (coef_int_used, coef_ext, heat_loss_pred_or_none, status, diag) where:
          coef_int_used  — blended coef_int applied to compute_tpi()
          coef_ext       — TPI_COEF_EXT_DEFAULT (always static)
          heat_loss_pred — learned rate in °C/h, or None for all non-"valid_le2" statuses
          status         — gate result string (see below)
          diag           — dict with fields for trace/audit (never raises)

        Classification: Variante B — Bounded Adaptive Heuristic
        -------------------------------------------------------
        coef_int is derived as heat_loss_rate / heat_rate.  This is NOT a rigorous
        derivation of a proportional control gain from first principles.

        The steady-state analysis shows: to hold the room at target against a passive
        cooling rate of heat_loss_rate, the required duty is heat_loss_rate / heat_rate.
        Using this as coef_int implies "the duty needed per °C error scales with the
        same ratio as steady-state maintenance duty."  This is directionally correct —
        lossier rooms need more aggressive control — but conflates the steady-state
        operating point with the error-correction gain.  The two are related but not
        identical.

        Safeguards that make Variante B acceptable:
          - coef_int is clamped to [0.15, 1.2] (estimate_coefficients)
          - only applied when both models exceed confidence 0.35
          - linearly blended with TPI_COEF_INT_DEFAULT over confidence [0.35 → 0.80]
          - blend is capped at 0.50 when HeatLoss has no outdoor context (relative_only)
          - reverts to deterministic defaults on stale / invalid / error

        HeatLoss outdoor context
        ------------------------
        HEAT_LOSS_RATE may be learned from episodes at varying outdoor temperatures.
        When no outdoor sensor is available (relative_only = True in reason_codes),
        the general bucket averages episodes across all observed ΔT.  This rate is a
        valid local observation but may not generalize to the current outdoor condition.
        The blend cap (_BLEND_WEIGHT_RELATIVE_ONLY_CAP) limits TPI influence of
        context-unaware heat loss estimates.

        Physical units
        --------------
        heat_rate      [°C/h] — UNIT_C_PER_H, verified by Gate 4
        heat_loss_rate [°C/h] — UNIT_C_PER_H, verified by Gate 4
        coef_int       [dimensionless = °C/h / °C/h], acts as 1/°C in TPI formula
        coef_ext       [dimensionless], static TPI_COEF_EXT_DEFAULT

        coef_ext rationale (Variante A, static)
        ----------------------------------------
        HEAT_LOSS_RATE is not normalised by the indoor-outdoor delta at episode time.
        Deriving coef_ext = coef_int / 50 and then multiplying by (target − outdoor_now)
        would implicitly double-scale the outdoor effect.  The static default avoids this.
        For TRV-only setups without an outdoor sensor, compute_tpi() skips the ext term.

        Validity gates (sequential; first failure returns deterministic defaults + diag):
          1. shadow enabled                          → le2_disabled
          2. both predictions present                → prediction_missing
          3. neither prediction has fallback_used    → cold_start
          4. UNIT_C_PER_H for both quantities        → invalid_unit
          5. no "stale"/"superseded" in warnings     → stale_or_superseded
          6. min(hr_conf, hl_conf) >= 0.35           → low_confidence
          7. both rates finite and > 0               → invalid_value

        After all gates pass, coef_int_used is computed as:
          blend       = clamp((conf − 0.35) / (0.80 − 0.35), 0, 1)
          [if relative_only: blend = min(blend, 0.50)]
          coef_int_used = blend × coef_int_raw + (1 − blend) × TPI_COEF_INT_DEFAULT

        status values:
          "valid_le2"           — all gates passed; coef_int is LE2-adaptive (possibly blended)
          "prediction_missing"  — one or both predictions absent (gate 2)
          "cold_start"          — fallback_used == True; prior-based, not learned (gate 3)
          "invalid_unit"        — unit is not UNIT_C_PER_H (gate 4)
          "stale_or_superseded" — stale or superseded warning present (gate 5)
          "low_confidence"      — confidence below 0.35 (gate 6)
          "invalid_value"       — rate ≤ 0 or non-finite (gate 7)
          "le2_disabled"        — shadow disabled
          "not_available"       — unexpected runtime error

        Never reads LE v1 _heat_loss_ema.
        """
        from ...const import TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, UNIT_C_PER_H
        _default_diag: dict = {
            "coef_int_default": TPI_COEF_INT_DEFAULT,
            "coef_int_candidate": TPI_COEF_INT_DEFAULT,  # gate failures: candidate = default
            "coef_int_learned": None, "blend_weight": 0.0,
            "relative_only": False, "outdoor_context_available": False,
            "heat_rate_c_per_h": None, "heat_loss_c_per_h": None,
            "hr_confidence": None, "hl_confidence": None,
        }
        if not self._enabled:
            return TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None, "le2_disabled", _default_diag
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            hr_pred = preds.get(PredictionType.HEAT_RATE)
            hl_pred = preds.get(PredictionType.HEAT_LOSS_RATE)
            # Gate 2: both predictions present
            if hr_pred is None or hl_pred is None:
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "prediction_missing", _default_diag)
            # Gate 3: not a fallback/cold-start prior
            if getattr(hr_pred, "fallback_used", True) or getattr(hl_pred, "fallback_used", True):
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "cold_start", _default_diag)
            # Gate 4: unit must be exactly UNIT_C_PER_H (central contract)
            _hr_units = getattr(hr_pred, "units", {}) or {}
            _hl_units = getattr(hl_pred, "units", {}) or {}
            if _hr_units.get("heat_rate") != UNIT_C_PER_H \
                    or _hl_units.get("heat_loss_rate") != UNIT_C_PER_H:
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "invalid_unit", _default_diag)
            # Gate 5: not stale or superseded
            _hr_warns = getattr(hr_pred, "warnings", ()) or ()
            _hl_warns = getattr(hl_pred, "warnings", ()) or ()
            if ("stale" in _hr_warns or "superseded" in _hr_warns
                    or "stale" in _hl_warns or "superseded" in _hl_warns):
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "stale_or_superseded", _default_diag)
            # Gate 6: confidence above threshold (min of both)
            hr_conf = float(getattr(hr_pred, "confidence", 0.0) or 0.0)
            hl_conf = float(getattr(hl_pred, "confidence", 0.0) or 0.0)
            conf = min(hr_conf, hl_conf)
            if conf < self._PREHEAT_MIN_CONFIDENCE:
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "low_confidence", {**_default_diag, "hr_confidence": hr_conf,
                                           "hl_confidence": hl_conf})
            # Gate 7: physically plausible, finite rates (nan/inf bypass <= 0 in Python)
            heat_rate = float(hr_pred.values.get("heat_rate", 0.0) or 0.0)
            heat_loss = float(hl_pred.values.get("heat_loss_rate", 0.0) or 0.0)
            if not (math.isfinite(heat_rate) and heat_rate > 0
                    and math.isfinite(heat_loss) and heat_loss > 0):
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "invalid_value", {**_default_diag, "hr_confidence": hr_conf,
                                          "hl_confidence": hl_conf,
                                          "heat_rate_c_per_h": heat_rate,
                                          "heat_loss_c_per_h": heat_loss})
            # Outdoor context: HeatLoss learned without outdoor sensor cannot claim
            # full authority because the general rate mixes all episode ΔT values.
            _hl_reasons = getattr(hl_pred, "reason_codes", ()) or ()
            relative_only = "relative_only" in _hl_reasons \
                or "outdoor_temp" in (getattr(hl_pred, "missing_evidence", ()) or ())
            # Blend: linearly ramp from 0 (at gate-6 threshold) to 1 (at full confidence).
            # This prevents abrupt step change from default→learned at the threshold.
            blend = max(0.0, min(1.0,
                (conf - self._PREHEAT_MIN_CONFIDENCE)
                / max(0.001, self._BLEND_FULL_CONFIDENCE - self._PREHEAT_MIN_CONFIDENCE)))
            if relative_only:
                blend = min(blend, self._BLEND_WEIGHT_RELATIVE_ONLY_CAP)
            from ...tpi import estimate_coefficients
            coef_int_raw, _ = estimate_coefficients(heat_rate, heat_loss)
            coef_int_used = blend * coef_int_raw + (1.0 - blend) * TPI_COEF_INT_DEFAULT
            coef_ext = TPI_COEF_EXT_DEFAULT
            diag: dict = {
                "coef_int_default": TPI_COEF_INT_DEFAULT,
                "coef_int_candidate": round(coef_int_used, 4),  # blended; before coordinator smoothing
                "coef_int_learned": round(coef_int_raw, 4),
                "blend_weight": round(blend, 4),
                "relative_only": relative_only,
                "outdoor_context_available": not relative_only,
                "heat_rate_c_per_h": round(heat_rate, 4),
                "heat_loss_c_per_h": round(heat_loss, 4),
                "hr_confidence": round(hr_conf, 4),
                "hl_confidence": round(hl_conf, 4),
            }
            return coef_int_used, coef_ext, round(heat_loss, 4), "valid_le2", diag
        except Exception as err:
            self._record_error("tpi_coefficients_read", err)
        return TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None, "not_available", _default_diag

    async def async_save_if_due(self) -> None:
        if not self._enabled:
            return
        try:
            await self._runtime.async_save_if_due()
        except Exception as err:
            self._record_error("save", err)

    async def async_flush(self) -> None:
        try:
            await self._runtime.async_flush()
        except Exception as err:
            self._record_error("flush", err)

    async def async_unload(self) -> None:
        self._enabled = False
        try:
            await self._runtime.async_unload()
        except Exception as err:
            self._record_error("unload", err)

    def diagnostics(self) -> dict:
        h = self._runtime.health()
        return {
            "mode": h.mode, "initialized": h.initialized, "dirty": h.dirty,
            "last_cycle_ts": h.last_cycle_ts, "last_save": h.last_save,
            "open_decisions": h.open_decisions, "open_comparisons": h.open_comparisons,
            "model_update_total": h.model_update_total,
            "model_update_counts": dict(h.model_update_counts),
            "prediction_snapshots": h.prediction_snapshots,
            "learning_errors": self._errors, "enabled": self._enabled,
            "control_enabled": h.control_enabled,
            "control_policy_version": h.control_policy_version,
            "control_applied": h.control_applied, "control_fallback": h.control_fallback,
            "control_rejections": dict(h.control_rejections),
            "reversion_count": h.reversion_count,
            "decision_trace": self._last_trace,
        }

    def _record_error(self, kind: str, err: Exception) -> None:
        self._errors += 1
        sig = f"{kind}:{type(err).__name__}"
        n = self._error_signatures.get(sig, 0)
        self._error_signatures[sig] = n + 1
        if n % _ERROR_LOG_THROTTLE == 0:  # throttle identical errors
            _LOGGER.warning("ThermoSmart LE2 shadow %s error (zone hidden): %s", kind,
                            type(err).__name__)
