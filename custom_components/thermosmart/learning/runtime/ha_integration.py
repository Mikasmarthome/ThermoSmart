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

    def adjust_recommendation_safe(self, recommendation: dict, *,
                                   boost_runtime_limit: Optional[float] = None) -> None:
        """In CONTROL mode only, conservatively adjust the dispatched recommendation.

        Baseline-first + single authority: the existing controller still applies
        and guards the value. Phase 18 only *reduces* an existing additive boost
        offset within clamps (the safest in-band adjustment). Never raises; a
        no-op in any non-CONTROL mode (so SHADOW stays byte-identical).
        """
        if not self.control_enabled:
            return
        try:
            target = recommendation.get("effective_target")
            if target is None:
                target = recommendation.get("adjusted_target")
            setpoint = recommendation.get("trv_setpoint")
            if target is None or setpoint is None:
                return
            baseline_offset = max(0.0, float(setpoint) - float(target))
            if baseline_offset <= 0.0:
                return  # no existing boost to adjust
            zr = self._runtime._zone(self._zone)
            boost_pred = zr.last_predictions.get(PredictionType.BOOST_FACTOR)
            proposed = boost_pred.values.get("boost_factor") if boost_pred else None
            ctx = ControlContext(
                window_open=bool(recommendation.get("window_open")),
                heating_failure=bool(recommendation.get("heating_failure")),
                boost_runtime_limit=boost_runtime_limit)
            decision = self._runtime.control_decision(
                self._zone, ControlFeature.BOOST_OFFSET, baseline_offset, proposed,
                ConfidencePurpose.BOOST, context=ctx, unit="celsius_offset",
                model_version=getattr(boost_pred, "model_version", None),
                parameter_version=getattr(boost_pred, "parameter_version", None))
            # only ever reduce an existing boost in Phase 18
            if decision.applied and decision.final_value is not None \
                    and decision.final_value < baseline_offset:
                recommendation["trv_setpoint"] = round(float(target) + decision.final_value, 4)
                recommendation["le2_boost_adjusted"] = True
        except Exception as err:
            self._record_error("control", err)

    def compute_decision_trace_safe(self, recommendation: Mapping[str, Any], *,
                                    active_control: bool = False,
                                    ts: Optional[str] = None) -> None:
        """Run the full decision pipeline read-only for this real cycle.

        Builds the real ZoneRuntimeInput + ControllerBaselineDecision + LE-2.0
        LearningPredictionSet and runs the FinalResolver/GuardLayer/DeviceAdapter to
        produce a DecisionTrace. The pipeline has NO dispatch sink, so it never sends
        a command — the existing coordinator dispatch stays the only real path and
        SHADOW remains byte-identical. Never raises into the heating cycle.
        """
        if not self._enabled:
            return
        try:
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
            self._last_trace = trace.support_dict()
        except Exception as err:
            self._record_error("decision_trace", err)

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

            heat_loss_c_per_h = float(
                hl_pred.values.get("heat_loss_rate", 0.0) if hl_pred else 0.0) or 0.0
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

            return round(total_min, 1), "valid"
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
