"""Typed core contracts for the LE 2.0 decision architecture (Phase 19A, pure Python).

These replace free-form recommendation dictionaries as the *internal* architecture
boundary. Existing HA-facing dicts are adapted at the edges (see baseline.py); the
public Home Assistant surface (entities/config/services) is unchanged.

Data flow:
    ZoneRuntimeInput -> ControllerBaselineDecision + LearningPredictionSet
    -> FinalResolver -> ResolvedControlDecision -> GuardLayer (ControlGuardResult)
    -> DeviceAdapter -> DeviceControlCommand -> Dispatcher -> DispatchResult
with a DecisionTrace recorded throughout. No HA imports; no dispatch here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

DECISION_SCHEMA_VERSION = 1


class DecisionReason(Enum):
    SAFETY = "safety"
    FROST_PROTECTION = "frost_protection"
    USER_LOCK = "user_lock"
    MODE_LOCK = "mode_lock"
    WINDOW_OPEN = "window_open"
    ABSENCE = "absence"
    SUMMER = "summer"
    BASELINE = "baseline"
    LE2_APPLIED = "le2_applied"
    DEVICE_LIMIT = "device_limit"
    ANTI_CHATTER = "anti_chatter"


class FallbackReason(Enum):
    NONE = "none"
    SHADOW_MODE = "shadow_mode"
    CONTROL_REJECTED = "control_rejected"
    GUARD_BLOCKED = "guard_blocked"
    NO_PREDICTION = "no_prediction"
    SAFETY_OVERRIDE = "safety_override"
    POLICY_ERROR = "policy_error"


class DecisionMode(Enum):
    SHADOW = "shadow"
    CONTROL = "control"


@dataclass(frozen=True)
class ZoneRuntimeInput:
    """Normalised, typed snapshot of the values a zone decision needs."""
    zone_id: str
    ts: str
    target_c: Optional[float] = None
    trv_setpoint_c: Optional[float] = None
    indoor_temp_c: Optional[float] = None
    indoor_temp_valid: bool = False
    comfort_time_utc: Optional[str] = None
    comfort_temperature_c: Optional[float] = None
    mode: Optional[str] = None
    controller_demand: Optional[bool] = None
    window_open: bool = False
    summer: bool = False
    frost_protection: bool = False
    absence: bool = False
    heating_failure: bool = False
    decision_id: Optional[str] = None
    decision_type: Optional[str] = None
    outdoor_temp_c: Optional[float] = None
    forecast_high_c: Optional[float] = None
    early_cutoff_state: Optional[str] = None
    # Preheat / HeatRate / OnsetDelay context (passed through for DecisionTrace)
    preheat_minutes_le2: Optional[float] = None
    preheat_status: Optional[str] = None
    deterministic_baseline_preheat_min: Optional[float] = None
    heat_rate_c_per_h: Optional[float] = None
    heat_rate_confidence: Optional[float] = None
    onset_delay_min: Optional[float] = None
    onset_delay_status: Optional[str] = None
    temperature_gap_c: Optional[float] = None        # comfort_temp - current_temp
    target_temperature_c: Optional[float] = None
    context_time_bucket: Optional[str] = None        # schedule period for onset-delay matching
    # Boost eligibility context (set by runtime; None = unknown → conservative fallback)
    boost_eligible: Optional[bool] = None            # explicit eligibility override from coordinator
    has_effective_onset: Optional[bool] = None       # whether heating onset has been detected
    boost_deficit_threshold_c: float = 0.5          # below this gap → no boost needed
    # TPI topology: True → direct valve TRV (duty % is control, °C offset does not apply)
    tpi_valve_direct: bool = False

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("zone_id required")


@dataclass(frozen=True)
class ControllerBaselineDecision:
    """The existing deterministic controller's result, side-effect free."""
    zone_id: str
    target_c: Optional[float]
    trv_setpoint_c: Optional[float]
    preheat_minutes: float = 0.0
    boost_offset_c: float = 0.0
    duty_cycle: float = 0.0
    control_reason: str = "schedule"
    summer: bool = False
    window_open: bool = False
    active_control: bool = False

    @property
    def safety_locked(self) -> bool:
        return self.summer or self.window_open


@dataclass(frozen=True)
class LearningPrediction:
    feature: str
    value: Optional[float]
    unit: str
    confidence: float
    confidence_purpose: Optional[str]
    fallback_used: bool = True

    def __post_init__(self) -> None:
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"{self.feature}: value must be finite or None")


@dataclass(frozen=True)
class LearningPredictionSet:
    """All technically-ready LE 2.0 predictions for one decision."""
    zone_id: str
    decision_id: Optional[str]
    predictions: Mapping[str, LearningPrediction] = field(default_factory=dict)
    confidence_results: Mapping[str, Any] = field(default_factory=dict)

    def get(self, feature: str) -> Optional[LearningPrediction]:
        return self.predictions.get(feature)


@dataclass(frozen=True)
class ControlGuardResult:
    allowed: bool
    final_value: Optional[float]
    clamp_applied: float
    reasons: tuple[str, ...]
    safe: bool = True


@dataclass(frozen=True)
class ResolvedControlDecision:
    zone_id: str
    feature: str
    baseline_value: Optional[float]
    le2_value: Optional[float]
    final_value: Optional[float]
    unit: str
    applied: bool
    mode: str
    reason: str
    fallback_reason: str
    confidence: Optional[float]
    clamp_applied: float
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.final_value is not None and not math.isfinite(self.final_value):
            raise ValueError("final_value must be finite or None")


@dataclass(frozen=True)
class DeviceControlCommand:
    """The single typed command the dispatcher may send (no HA call here)."""
    zone_id: str
    trv_setpoint_c: Optional[float]
    duty_cycle: Optional[float]
    hvac_mode: Optional[str] = None
    source_reason: str = "baseline"
    shadow_only: bool = True

    def __post_init__(self) -> None:
        for v in (self.trv_setpoint_c, self.duty_cycle):
            if v is not None and not math.isfinite(v):
                raise ValueError("command values must be finite")


@dataclass(frozen=True)
class DispatchResult:
    zone_id: str
    dispatched: bool
    command: Optional[DeviceControlCommand]
    shadow_only: bool
    reason: str


@dataclass(frozen=True)
class DecisionTraceEntry:
    feature: str
    baseline_value: Optional[float]
    le2_value: Optional[float]
    final_value: Optional[float]
    applied: bool
    reason: str
    fallback_reason: str
    confidence: Optional[float]
    clamp_applied: float


@dataclass(frozen=True)
class DecisionTrace:
    zone_id: str
    ts: str
    mode: str
    decision_id: Optional[str]
    baseline_setpoint_c: Optional[float]
    final_setpoint_c: Optional[float]
    applied_any: bool
    entries: tuple[DecisionTraceEntry, ...]
    reason_codes: tuple[str, ...]
    schema_version: int = DECISION_SCHEMA_VERSION
    # User's configured comfort target; separate from baseline_setpoint_c which may
    # already reflect an early-cutoff reduction (effective_target < comfort_target).
    comfort_temperature_c: Optional[float] = None
    # Coordinator's Early Cutoff hold lifecycle state (e.g. "cutoff_applied",
    # "coasting_hold", "released_target_reached", "inactive").
    early_cutoff_state: Optional[str] = None
    early_cutoff_hold_active: bool = False
    # Preheat / HeatRate / OnsetDelay trace fields.
    # Three semantically distinct time quantities (A = B + C):
    #   A = preheat_command_lead_time_min  (total, returned to coordinator)
    #   B = effective_onset_delay_min      (TRV→room; adaptive, separate from rate)
    #   C = effective_room_heating_duration_min  (only this drives HeatRate learning)
    preheat_minutes_le2: Optional[float] = None   # LE2 value before any selection
    preheat_status: Optional[str] = None           # "valid", "deterministic_baseline", …
    preheat_baseline_minutes: Optional[float] = None  # deterministic baseline value
    selected_preheat_min: Optional[float] = None   # = preheat_command_lead_time_min (alias)
    heat_rate_c_per_h: Optional[float] = None
    heat_rate_confidence: Optional[float] = None
    heat_rate_status: Optional[str] = None
    # Onset delay (B): separate from heat rate; never folded into rate learning.
    effective_onset_delay_min: float = 0.0
    onset_delay_source: Optional[str] = None       # "valid" | "cold_start_prior" | None
    onset_delay_status: Optional[str] = None       # same as source for now
    effective_room_heating_duration_min: Optional[float] = None
    preheat_command_lead_time_min: Optional[float] = None
    # Context for onset-delay / schedule matching
    temperature_gap_c: Optional[float] = None      # comfort_temp - current_temp
    target_temperature_c: Optional[float] = None
    context_time_bucket: Optional[str] = None      # schedule period
    fallback_source: Optional[str] = None          # which fallback level was used
    early_cutoff_contribution_c: Optional[float] = None
    preheat_fallback_used: bool = False
    # Boost diagnostics: internal truth, compat adapter, adaptive cap, lifecycle
    boost_offset_c_applied: Optional[float] = None   # internal additive °C (0.0 = neutral)
    boost_factor_compat: Optional[float] = None       # public attr (1.0 = neutral, compat adapter)
    boost_adaptive_cap_c: Optional[float] = None      # LE 2.0 adaptive cap in effect this cycle
    boost_lifecycle_state: Optional[str] = None       # current lifecycle state string
    # TPI authority chain: typed fields for no-double-apply verification
    tpi_duty_percent: Optional[float] = None          # TPI duty [0–100 %] this cycle
    tpi_baseline_setpoint_c: Optional[float] = None  # TPI setpoint before any LE2 boost
    boost_offset_requested_c: Optional[float] = None  # raw LE2 prediction (pre-gate)
    boost_rejected_reason: Optional[str] = None       # why boost was not applied (e.g. "direct_valve_control")
    # Deescalation trace fields (populated by compute_decision_trace_safe via lifecycle state)
    boost_release_reason: Optional[str] = None           # lifecycle release reason string
    boost_deescalation_type: Optional[str] = None        # "hard" or "soft" release category
    remaining_temperature_gap_c: Optional[float] = None  # deficit at deescalation check
    predicted_time_to_target_without_boost_min: Optional[float] = None
    remaining_time_to_target_min: Optional[float] = None  # schedule remaining time
    tpi_sufficiency_safety_margin_min: Optional[float] = None  # versioned TPI safety margin
    afterheat_prediction_c: Optional[float] = None       # LE2 EXPECTED_OVERSHOOT residual rise
    afterheat_prediction_status: Optional[str] = None    # "learned", "cold_start", "unavailable"
    afterheat_confidence: Optional[float] = None         # EXPECTED_OVERSHOOT prediction confidence
    boost_outcome_overshoot_risk: Optional[bool] = None  # BOOST_OUTCOME overshoot history flag
    expected_afterheat_c: Optional[float] = None         # alias: same as afterheat_prediction_c
    boost_previous_offset_c: Optional[float] = None      # applied offset before this cycle
    boost_new_offset_c: Optional[float] = None           # applied offset after this cycle
    final_device_setpoint_c: Optional[float] = None      # trv_setpoint after all authority

    def support_dict(self) -> dict:
        """Privacy-safe view: no entity names / internal ids."""
        return {
            "mode": self.mode,
            "comfort_temperature_c": self.comfort_temperature_c,
            "early_cutoff_state": self.early_cutoff_state,
            "early_cutoff_hold_active": self.early_cutoff_hold_active,
            "preheat_minutes_le2": self.preheat_minutes_le2,
            "preheat_status": self.preheat_status,
            "preheat_baseline_minutes": self.preheat_baseline_minutes,
            "selected_preheat_min": self.selected_preheat_min,
            "heat_rate_c_per_h": self.heat_rate_c_per_h,
            "heat_rate_confidence": self.heat_rate_confidence,
            "heat_rate_status": self.heat_rate_status,
            "effective_onset_delay_min": self.effective_onset_delay_min,
            "onset_delay_source": self.onset_delay_source,
            "onset_delay_status": self.onset_delay_status,
            "effective_room_heating_duration_min": self.effective_room_heating_duration_min,
            "preheat_command_lead_time_min": self.preheat_command_lead_time_min,
            "temperature_gap_c": self.temperature_gap_c,
            "target_temperature_c": self.target_temperature_c,
            "context_time_bucket": self.context_time_bucket,
            "fallback_source": self.fallback_source,
            "early_cutoff_contribution_c": self.early_cutoff_contribution_c,
            "preheat_fallback_used": self.preheat_fallback_used,
            "boost_offset_c_applied": self.boost_offset_c_applied,
            "boost_factor_compat": self.boost_factor_compat,
            "boost_adaptive_cap_c": self.boost_adaptive_cap_c,
            "boost_lifecycle_state": self.boost_lifecycle_state,
            # TPI authority chain (explicit typed fields for audit)
            "tpi_duty_percent": self.tpi_duty_percent,
            "tpi_baseline_setpoint_c": self.tpi_baseline_setpoint_c,
            "boost_offset_requested_c": self.boost_offset_requested_c,
            "boost_rejected_reason": self.boost_rejected_reason,
            # Deescalation fields
            "boost_release_reason": self.boost_release_reason,
            "boost_deescalation_type": self.boost_deescalation_type,
            "remaining_temperature_gap_c": self.remaining_temperature_gap_c,
            "predicted_time_to_target_without_boost_min": self.predicted_time_to_target_without_boost_min,
            "remaining_time_to_target_min": self.remaining_time_to_target_min,
            "tpi_sufficiency_safety_margin_min": self.tpi_sufficiency_safety_margin_min,
            "afterheat_prediction_c": self.afterheat_prediction_c,
            "afterheat_prediction_status": self.afterheat_prediction_status,
            "afterheat_confidence": self.afterheat_confidence,
            "boost_outcome_overshoot_risk": self.boost_outcome_overshoot_risk,
            "expected_afterheat_c": self.expected_afterheat_c,
            "boost_previous_offset_c": self.boost_previous_offset_c,
            "boost_new_offset_c": self.boost_new_offset_c,
            "final_device_setpoint_c": self.final_device_setpoint_c,
            "baseline_setpoint_c": self.baseline_setpoint_c,
            "final_setpoint_c": self.final_setpoint_c, "applied_any": self.applied_any,
            "features": [
                {"feature": e.feature, "baseline": e.baseline_value, "le2": e.le2_value,
                 "final": e.final_value, "applied": e.applied, "reason": e.reason,
                 "fallback": e.fallback_reason, "confidence": e.confidence,
                 "clamp": e.clamp_applied}
                for e in self.entries],
            "reason_codes": sorted(self.reason_codes), "schema_version": self.schema_version,
        }
