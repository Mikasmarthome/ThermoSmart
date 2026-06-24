"""Boost confounder + reliability contracts (B2b-4, pure Python).

Two separate concepts, never collapsed into one opaque score:

* Confounder = a possible alternative cause or interruption (typed, with a severity).
* Reliability = the quality/trustworthiness of the observation (multi-component).

A single authoritative evaluator canonicalises raw episode/dispatch evidence into typed
``BoostConfounder`` values (with de-duplication by root cause), and a single reliability
function combines the components via hard gates + a min-of-critical × bounded-modifier
rule. Both the non-boost baseline builder and the boost outcome classifier consume the
SAME evaluator, so a state can never be admissible for one and blocking for the other.
No control, no model mutation here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence

RELIABILITY_COMPONENT_VERSION = 1


class BoostOutcomeLifecycleState(Enum):
    """B2b-4c outcome-finalization lifecycle. The boost outcome is only finalized (and the
    model updated exactly once) after target reach is stably confirmed and the overshoot/
    afterheat observation window is satisfied."""
    HEATING_ACTIVE = "heating_active"
    TARGET_STABILITY_PENDING = "target_stability_pending"
    TARGET_REACHED_PENDING_OBSERVATION = "target_reached_pending_observation"
    FINALIZED = "finalized"
    INTERRUPTED = "interrupted"
    EXPIRED = "expired"


class ConfounderSeverity(Enum):
    INFO = "info"               # diagnostic only; no automatic reliability reduction
    DEGRADING = "degrading"     # usable, reduced attribution/weight; no aggressive positive
    BLOCKING = "blocking"       # observed but cause not attributable -> BOOST_CONFOUNDED
    INTERRUPTING = "interrupting"  # episode causally broken -> BOOST_INTERRUPTED

# severity ordering for "max"
_SEV_ORDER = {ConfounderSeverity.INFO: 0, ConfounderSeverity.DEGRADING: 1,
              ConfounderSeverity.BLOCKING: 2, ConfounderSeverity.INTERRUPTING: 3}


class BoostConfounder(Enum):
    # ── interrupting (observation incomplete / causally broken) ──────────────
    RESTART_GAP = "restart_gap"
    RELOAD_GAP = "reload_gap"
    SENSOR_GAP = "sensor_gap"
    TARGET_CHANGE = "target_change"
    SCHEDULE_CHANGE = "schedule_change"
    MODE_CHANGE = "mode_change"
    MANUAL_OVERRIDE = "manual_override"
    WINDOW_OPEN = "window_open"
    HEATING_FAILURE = "heating_failure"
    CONTROL_DISABLED = "control_disabled"
    EPISODE_TIMEOUT = "episode_timeout"
    INCOMPLETE_TRAJECTORY = "incomplete_trajectory"
    SUPERSEDED_BY_NEW_DECISION = "superseded_by_new_decision"
    # ── attribution-distorting (observed, cause ambiguous) ───────────────────
    SOLAR_GAIN = "solar_gain"
    EXTERNAL_HEAT_GAIN = "external_heat_gain"
    ADDITIONAL_HEAT_SOURCE = "additional_heat_source"
    ADDITIONAL_RADIATOR = "additional_radiator"
    OCCUPANCY_HEAT_GAIN = "occupancy_heat_gain"
    SUPPLY_DELAY_ANOMALY = "supply_delay_anomaly"
    MULTI_TRV_AMBIGUOUS = "multi_trv_ambiguous"
    PARTIAL_DEVICE_EFFECT = "partial_device_effect"
    SENSOR_SOURCE_CHANGE = "sensor_source_change"
    DEVICE_CONFIGURATION_CHANGE = "device_configuration_change"
    TPI_AUTHORITY_CHANGE = "tpi_authority_change"
    PREHEAT_INTERACTION = "preheat_interaction"
    EARLY_CUTOFF_INTERACTION = "early_cutoff_interaction"
    AFTERHEAT_INTERACTION = "afterheat_interaction"
    FORECAST_OR_OUTDOOR_SHIFT = "forecast_or_outdoor_shift"
    UNEXPLAINED_REHEATING = "unexplained_reheating"
    UNKNOWN = "unknown_confounder"             # unrecognised raw flag (safe-degrade)


_SEVERITY: Mapping[BoostConfounder, ConfounderSeverity] = {
    BoostConfounder.RESTART_GAP: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.RELOAD_GAP: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.SENSOR_GAP: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.TARGET_CHANGE: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.SCHEDULE_CHANGE: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.MODE_CHANGE: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.MANUAL_OVERRIDE: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.WINDOW_OPEN: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.HEATING_FAILURE: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.CONTROL_DISABLED: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.EPISODE_TIMEOUT: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.INCOMPLETE_TRAJECTORY: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.SUPERSEDED_BY_NEW_DECISION: ConfounderSeverity.INTERRUPTING,
    BoostConfounder.SOLAR_GAIN: ConfounderSeverity.BLOCKING,
    BoostConfounder.EXTERNAL_HEAT_GAIN: ConfounderSeverity.BLOCKING,
    BoostConfounder.ADDITIONAL_HEAT_SOURCE: ConfounderSeverity.BLOCKING,
    BoostConfounder.ADDITIONAL_RADIATOR: ConfounderSeverity.BLOCKING,
    BoostConfounder.OCCUPANCY_HEAT_GAIN: ConfounderSeverity.DEGRADING,
    BoostConfounder.SUPPLY_DELAY_ANOMALY: ConfounderSeverity.DEGRADING,
    BoostConfounder.MULTI_TRV_AMBIGUOUS: ConfounderSeverity.BLOCKING,
    BoostConfounder.PARTIAL_DEVICE_EFFECT: ConfounderSeverity.DEGRADING,
    BoostConfounder.SENSOR_SOURCE_CHANGE: ConfounderSeverity.DEGRADING,
    BoostConfounder.DEVICE_CONFIGURATION_CHANGE: ConfounderSeverity.BLOCKING,
    BoostConfounder.TPI_AUTHORITY_CHANGE: ConfounderSeverity.DEGRADING,
    BoostConfounder.PREHEAT_INTERACTION: ConfounderSeverity.INFO,
    BoostConfounder.EARLY_CUTOFF_INTERACTION: ConfounderSeverity.DEGRADING,
    BoostConfounder.AFTERHEAT_INTERACTION: ConfounderSeverity.INFO,
    BoostConfounder.FORECAST_OR_OUTDOOR_SHIFT: ConfounderSeverity.DEGRADING,
    BoostConfounder.UNEXPLAINED_REHEATING: ConfounderSeverity.BLOCKING,
    BoostConfounder.UNKNOWN: ConfounderSeverity.DEGRADING,
}

# raw episode/runtime flag -> canonical confounder (aliases collapse to one truth)
_ALIAS: Mapping[str, BoostConfounder] = {
    "restart_gap": BoostConfounder.RESTART_GAP, "reload_gap": BoostConfounder.RELOAD_GAP,
    "sensor_gap": BoostConfounder.SENSOR_GAP, "target_change": BoostConfounder.TARGET_CHANGE,
    "schedule_change": BoostConfounder.SCHEDULE_CHANGE, "mode_change": BoostConfounder.MODE_CHANGE,
    "manual_override": BoostConfounder.MANUAL_OVERRIDE, "window_open": BoostConfounder.WINDOW_OPEN,
    "heating_failure": BoostConfounder.HEATING_FAILURE,
    "control_disabled": BoostConfounder.CONTROL_DISABLED,
    "episode_timeout": BoostConfounder.EPISODE_TIMEOUT, "timeout": BoostConfounder.EPISODE_TIMEOUT,
    "incomplete_trajectory": BoostConfounder.INCOMPLETE_TRAJECTORY,
    "superseded_by_new_decision": BoostConfounder.SUPERSEDED_BY_NEW_DECISION,
    "solar_gain": BoostConfounder.SOLAR_GAIN, "solar": BoostConfounder.SOLAR_GAIN,
    "external_heat": BoostConfounder.EXTERNAL_HEAT_GAIN,
    "external_heat_gain": BoostConfounder.EXTERNAL_HEAT_GAIN,
    "additional_heat_source": BoostConfounder.ADDITIONAL_HEAT_SOURCE,
    "additional_radiator": BoostConfounder.ADDITIONAL_RADIATOR,
    "reheating": BoostConfounder.UNEXPLAINED_REHEATING,
    "unexplained_reheating": BoostConfounder.UNEXPLAINED_REHEATING,
    "occupancy_heat_gain": BoostConfounder.OCCUPANCY_HEAT_GAIN,
    "supply_delay": BoostConfounder.SUPPLY_DELAY_ANOMALY,
    "supply_delay_anomaly": BoostConfounder.SUPPLY_DELAY_ANOMALY,
    "multi_trv_ambiguous": BoostConfounder.MULTI_TRV_AMBIGUOUS,
    "partial_device_effect": BoostConfounder.PARTIAL_DEVICE_EFFECT,
    "sensor_source_change": BoostConfounder.SENSOR_SOURCE_CHANGE,
    "device_configuration_change": BoostConfounder.DEVICE_CONFIGURATION_CHANGE,
    "tpi_authority_change": BoostConfounder.TPI_AUTHORITY_CHANGE,
    "preheat_interaction": BoostConfounder.PREHEAT_INTERACTION,
    "early_cutoff_interaction": BoostConfounder.EARLY_CUTOFF_INTERACTION,
    "afterheat_interaction": BoostConfounder.AFTERHEAT_INTERACTION,
    "forecast_or_outdoor_shift": BoostConfounder.FORECAST_OR_OUTDOOR_SHIFT,
}

# Root-cause groups: a single physical event must not be punished three times. The members
# of a group collapse to the group's canonical confounder for reliability penalty purposes.
_CAUSE_GROUPS: tuple[tuple[BoostConfounder, frozenset], ...] = (
    (BoostConfounder.WINDOW_OPEN, frozenset({
        BoostConfounder.WINDOW_OPEN, BoostConfounder.TARGET_CHANGE,
        BoostConfounder.MODE_CHANGE, BoostConfounder.SCHEDULE_CHANGE})),
    (BoostConfounder.RESTART_GAP, frozenset({
        BoostConfounder.RESTART_GAP, BoostConfounder.RELOAD_GAP, BoostConfounder.SENSOR_GAP})),
    (BoostConfounder.MANUAL_OVERRIDE, frozenset({
        BoostConfounder.MANUAL_OVERRIDE, BoostConfounder.TARGET_CHANGE})),
)


def severity_of(c: BoostConfounder) -> ConfounderSeverity:
    return _SEVERITY.get(c, ConfounderSeverity.DEGRADING)  # unknown -> conservative degrade


@dataclass(frozen=True)
class ConfounderEvaluation:
    confounders: tuple[str, ...]              # canonical confounder values present
    max_severity: str                        # worst severity present
    interrupting: tuple[str, ...]
    blocking: tuple[str, ...]
    degrading: tuple[str, ...]
    distinct_causes: int                      # de-duplicated root-cause count (all severities)
    degrading_causes: int = 0                 # de-duplicated DEGRADING root causes (penalty count)
    component_version: int = RELIABILITY_COMPONENT_VERSION

    @property
    def is_interrupting(self) -> bool:
        return self.max_severity == ConfounderSeverity.INTERRUPTING.value

    @property
    def is_blocking(self) -> bool:
        return self.max_severity == ConfounderSeverity.BLOCKING.value


def evaluate_confounders(raw_flags: Sequence[str], *,
                         supply_delay_anomaly: bool = False,
                         partial_dispatch: bool = False,
                         multi_trv_ambiguous: bool = False) -> ConfounderEvaluation:
    """Canonicalise + de-duplicate raw evidence into a typed confounder evaluation.

    Unknown raw flags are mapped conservatively to DEGRADING (never silently dropped).
    Root-cause groups collapse so one event is not penalised multiple times.
    """
    canon: set[BoostConfounder] = set()
    for raw in raw_flags or ():
        c = _ALIAS.get(str(raw))
        if c is None:
            # unknown raw flag -> a typed UNKNOWN confounder (DEGRADING; safe-degrade, never dropped)
            canon.add(BoostConfounder.UNKNOWN)
        else:
            canon.add(c)
    if supply_delay_anomaly:
        canon.add(BoostConfounder.SUPPLY_DELAY_ANOMALY)
    if partial_dispatch:
        canon.add(BoostConfounder.PARTIAL_DEVICE_EFFECT)
    if multi_trv_ambiguous:
        canon.add(BoostConfounder.MULTI_TRV_AMBIGUOUS)

    interrupting = sorted(c.value for c in canon
                          if severity_of(c) is ConfounderSeverity.INTERRUPTING)
    blocking = sorted(c.value for c in canon if severity_of(c) is ConfounderSeverity.BLOCKING)
    degrading = sorted(c.value for c in canon if severity_of(c) is ConfounderSeverity.DEGRADING)
    if canon:
        max_sev = max((severity_of(c) for c in canon), key=lambda s: _SEV_ORDER[s])
    else:
        max_sev = ConfounderSeverity.INFO

    def _dedup_causes(members: set) -> int:
        remaining = set(members)
        n = 0
        for _lead, group in _CAUSE_GROUPS:
            if remaining & group:
                n += 1
                remaining -= group
        return n + len(remaining)

    causes = _dedup_causes(canon)
    # penalty count: ONLY degrading causes (INFO = no penalty; blocking/interrupting = hard gate)
    deg_causes = _dedup_causes({c for c in canon
                                if severity_of(c) is ConfounderSeverity.DEGRADING})

    return ConfounderEvaluation(
        confounders=tuple(sorted(c.value for c in canon)), max_severity=max_sev.value,
        interrupting=tuple(interrupting), blocking=tuple(blocking), degrading=tuple(degrading),
        distinct_causes=causes, degrading_causes=deg_causes)


# ── reliability ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BoostOutcomeReliability:
    sensor_reliability: float
    trajectory_completeness: float
    dispatch_reliability: float
    attribution_reliability: float
    baseline_reliability: float
    timing_reliability: float
    device_reliability: float
    context_stability: float
    confounder_reliability: float
    overall_reliability: float
    blocking_reasons: tuple[str, ...] = ()
    degrading_reasons: tuple[str, ...] = ()
    component_version: int = RELIABILITY_COMPONENT_VERSION

    def to_dict(self) -> dict:
        return {k: getattr(self, k) if not isinstance(getattr(self, k), tuple)
                else list(getattr(self, k)) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Mapping) -> "BoostOutcomeReliability":
        kw = {k: d.get(k, 1.0) for k in cls.__dataclass_fields__}
        kw["blocking_reasons"] = tuple(d.get("blocking_reasons", []) or ())
        kw["degrading_reasons"] = tuple(d.get("degrading_reasons", []) or ())
        kw["component_version"] = d.get("component_version", 0)
        return cls(**kw)


def _clamp01(x: Optional[float], default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return default


def compute_outcome_reliability(*, sensor_reliability: Optional[float],
                                trajectory_completeness: Optional[float],
                                dispatch_reliability: Optional[float],
                                attribution_reliability: Optional[float],
                                baseline_reliability: Optional[float],
                                timing_reliability: Optional[float],
                                device_reliability: Optional[float],
                                context_stability: Optional[float],
                                confounders: ConfounderEvaluation,
                                baseline_required: bool) -> BoostOutcomeReliability:
    """Final reliability: hard gates (blocking/interrupting -> 0) then
    overall = min(critical components) × bounded quality modifiers.

    A single high component can NOT mask a poor critical path (no arithmetic mean). Missing
    data is treated conservatively (low). The confounder severity is applied ONCE.
    """
    sensor = _clamp01(sensor_reliability)
    traj = _clamp01(trajectory_completeness)
    dispatch = _clamp01(dispatch_reliability)
    attribution = _clamp01(attribution_reliability)
    baseline = _clamp01(baseline_reliability)
    timing = _clamp01(timing_reliability)
    device = _clamp01(device_reliability)
    context = _clamp01(context_stability)

    # confounder_reliability: one penalty per distinct root cause; blocking/interrupting -> 0
    if confounders.is_interrupting or confounders.is_blocking:
        conf_rel = 0.0
    else:
        # one 0.7 penalty per distinct DEGRADING root cause (INFO -> no penalty)
        conf_rel = 0.7 ** confounders.degrading_causes

    # hard gates
    blocking_reasons = list(confounders.blocking) + list(confounders.interrupting)
    if blocking_reasons:
        overall = 0.0
    else:
        # critical components gate the result (min), non-critical modulate it.
        critical = [sensor, traj, dispatch, attribution, timing]
        if baseline_required:
            critical.append(baseline)
        core = min(critical)
        modifier = (0.5 + 0.5 * context) * (0.5 + 0.5 * device) * conf_rel
        overall = core * modifier

    return BoostOutcomeReliability(
        sensor_reliability=round(sensor, 4), trajectory_completeness=round(traj, 4),
        dispatch_reliability=round(dispatch, 4), attribution_reliability=round(attribution, 4),
        baseline_reliability=round(baseline, 4), timing_reliability=round(timing, 4),
        device_reliability=round(device, 4), context_stability=round(context, 4),
        confounder_reliability=round(conf_rel, 4), overall_reliability=round(overall, 4),
        blocking_reasons=tuple(sorted(set(blocking_reasons))),
        degrading_reasons=tuple(confounders.degrading))


# ── class-dependent adaptive gates ──────────────────────────────────────────
# Minimum overall reliability for an ADAPTIVE model update, per outcome class. Positive
# updates (SUCCESS) demand more than negative safety updates (OVERSHOOT). Classes that are
# themselves a confounder/interruption result are never adaptive.
_CLASS_MIN_RELIABILITY: Mapping[str, float] = {
    "boost_success": 0.6,
    "boost_no_effect": 0.5,
    "boost_unnecessary": 0.5,
    "boost_overshoot": 0.4,        # negative safety: no baseline needed, mid reliability ok
}
_NON_ADAPTIVE_CLASSES = frozenset({
    "boost_confounded", "boost_interrupted", "boost_partial_dispatch",
    "boost_failed_dispatch", "boost_not_attempted", "insufficient_comparison"})


def class_min_reliability(outcome_class: str) -> Optional[float]:
    """Min overall reliability for an adaptive update, or None if the class is never adaptive."""
    if outcome_class in _NON_ADAPTIVE_CLASSES:
        return None
    return _CLASS_MIN_RELIABILITY.get(outcome_class, 0.5)


def adaptive_allowed(outcome_class: str, reliability: BoostOutcomeReliability) -> bool:
    threshold = class_min_reliability(outcome_class)
    if threshold is None:
        return False
    if reliability.blocking_reasons:
        return False
    return reliability.overall_reliability >= threshold


# ── B2b-4a: multi-TRV per-device dispatch evidence ──────────────────────────
@dataclass(frozen=True)
class BoostDeviceDispatchEvidence:
    """Per-device (anonymised slot) dispatch evidence for one boost episode. No entity id."""
    slot: str                                  # stable anonymised slot, e.g. "trv0"
    control_type: str                          # "setpoint" | "direct_valve"
    requested_offset_c: Optional[float]
    effective_offset_c: Optional[float]
    clamp_applied_c: float = 0.0
    dispatch_succeeded: bool = True
    available_at_dispatch: bool = True
    available_during_observation: bool = True
    device_step_c: float = 0.5                 # device setpoint step (quantisation)


@dataclass(frozen=True)
class MultiTrvParams:
    offset_spread_ambiguous_c: float = 0.4     # absolute floor for ambiguous spread
    clamp_degrading_c: float = 0.2             # absolute floor for a degrading clamp
    requested_fraction: float = 0.34           # relative tolerance vs. the requested offset


def _quant_tolerance(devices, requested, p: "MultiTrvParams") -> float:
    """Combined absolute + step/requested-relative quantisation tolerance, so a coarse-step
    device (e.g. 0.5 °C) is not unfairly flagged for unavoidable quantisation."""
    step = max((d.device_step_c for d in devices), default=0.0)
    rel = abs(requested) * p.requested_fraction if requested else 0.0
    return max(p.offset_spread_ambiguous_c, step, rel)


def evaluate_multi_trv(devices: Sequence[BoostDeviceDispatchEvidence], *,
                       params: Optional[MultiTrvParams] = None) -> Optional[BoostConfounder]:
    """Per-device attribution check → a confounder (or None when homogeneous & clean).

    Step-relative: the spread/clamp tolerance is ``max(absolute_floor, device_step,
    fraction × requested_offset)`` — normal device quantisation is not flagged, a materially
    different boost effect is. Failed/unavailable device → PARTIAL_DEVICE_EFFECT; mixed
    control types or material spread → MULTI_TRV_AMBIGUOUS; material clamp → PARTIAL_DEVICE_EFFECT.
    """
    p = params or MultiTrvParams()
    relevant = [d for d in devices if d.control_type == "setpoint"] or list(devices)
    if not relevant:
        return None
    control_types = {d.control_type for d in devices}
    if len(control_types) > 1:
        return BoostConfounder.MULTI_TRV_AMBIGUOUS    # mixed setpoint + direct_valve
    if any((not d.dispatch_succeeded) or (not d.available_at_dispatch)
           or (not d.available_during_observation) for d in devices):
        return BoostConfounder.PARTIAL_DEVICE_EFFECT
    requested = next((d.requested_offset_c for d in relevant
                      if d.requested_offset_c is not None), 0.0)
    tol = _quant_tolerance(relevant, requested, p)
    offsets = [d.effective_offset_c for d in relevant if d.effective_offset_c is not None]
    if len(offsets) >= 2:
        if (max(offsets) - min(offsets)) > tol:
            return BoostConfounder.MULTI_TRV_AMBIGUOUS
    if any(d.clamp_applied_c > tol for d in relevant):
        return BoostConfounder.PARTIAL_DEVICE_EFFECT
    return None


# ── B2b-4a: target-reached stability ─────────────────────────────────────────
@dataclass(frozen=True)
class TargetStabilityParams:
    band_c: float = 0.3                # within target ± band counts as "in band"
    min_samples_in_band: int = 3
    min_dwell_s: float = 600.0         # must hold the band ≥ 10 min
    max_gap_s: float = 1800.0          # a gap > 30 min breaks continuity


@dataclass(frozen=True)
class TargetStabilityEvidence:
    samples_in_band: int
    dwell_duration_s: float
    max_gap_s: float
    stable: bool
    rejection_reason: Optional[str] = None


def evaluate_target_stability(points: Sequence, target: float, *,
                              params: Optional[TargetStabilityParams] = None
                              ) -> TargetStabilityEvidence:
    """A single in-band sample is NOT 'reached': require a stable dwell in the target band."""
    p = params or TargetStabilityParams()
    band_lo = target - p.band_c
    in_band = [pt for pt in points if not getattr(pt, "gap", False)
               and pt.value is not None and pt.value >= band_lo]
    samples = len(in_band)
    if samples == 0:
        return TargetStabilityEvidence(0, 0.0, 0.0, False, "never_in_band")
    dwell = (in_band[-1].offset_ms - in_band[0].offset_ms) / 1000.0
    # max gap between consecutive valid points overall
    valid = [pt for pt in points if not getattr(pt, "gap", False) and pt.value is not None]
    max_gap = 0.0
    for a, b in zip(valid, valid[1:]):
        max_gap = max(max_gap, (b.offset_ms - a.offset_ms) / 1000.0)
    if samples < p.min_samples_in_band:
        return TargetStabilityEvidence(samples, dwell, max_gap, False, "too_few_in_band")
    if dwell < p.min_dwell_s:
        return TargetStabilityEvidence(samples, dwell, max_gap, False, "dwell_too_short")
    if max_gap > p.max_gap_s:
        return TargetStabilityEvidence(samples, dwell, max_gap, False, "gap_too_large")
    return TargetStabilityEvidence(samples, round(dwell, 1), round(max_gap, 1), True, None)


# ── B2b-4a: overshoot stability ──────────────────────────────────────────────
@dataclass(frozen=True)
class OvershootStabilityParams:
    threshold_c: float = 0.5          # overshoot above target+threshold counts
    min_samples_above: int = 2
    min_dwell_s: float = 300.0        # sustained ≥ 5 min (a short peak is not overshoot)
    max_gap_s: float = 1800.0


@dataclass(frozen=True)
class OvershootStabilityEvidence:
    peak_overshoot_c: float
    stable_overshoot_c: float
    samples_above_threshold: int
    dwell_duration_s: float
    observation_window_s: float
    stable: bool
    afterheat_context: bool = False
    rejection_reason: Optional[str] = None


def evaluate_overshoot_stability(points: Sequence, target: float, *,
                                 afterheat_context: bool = False,
                                 params: Optional[OvershootStabilityParams] = None
                                 ) -> OvershootStabilityEvidence:
    """A short peak is NOT a stable overshoot; require sustained samples above the threshold."""
    p = params or OvershootStabilityParams()
    thr = target + p.threshold_c
    valid = [pt for pt in points if not getattr(pt, "gap", False) and pt.value is not None]
    if not valid:
        return OvershootStabilityEvidence(0.0, 0.0, 0, 0.0, 0.0, False, afterheat_context,
                                          "no_data")
    peak = max(pt.value for pt in valid) - target
    above = [pt for pt in valid if pt.value > thr]
    window = (valid[-1].offset_ms - valid[0].offset_ms) / 1000.0
    if len(above) < p.min_samples_above:
        return OvershootStabilityEvidence(round(max(peak, 0.0), 3), 0.0, len(above), 0.0,
                                          round(window, 1), False, afterheat_context,
                                          "too_few_above")
    dwell = (above[-1].offset_ms - above[0].offset_ms) / 1000.0
    stable_overshoot = min(pt.value for pt in above) - target
    if dwell < p.min_dwell_s:
        return OvershootStabilityEvidence(round(max(peak, 0.0), 3), round(stable_overshoot, 3),
                                          len(above), round(dwell, 1), round(window, 1), False,
                                          afterheat_context, "dwell_too_short")
    return OvershootStabilityEvidence(round(max(peak, 0.0), 3), round(stable_overshoot, 3),
                                      len(above), round(dwell, 1), round(window, 1), True,
                                      afterheat_context, None)


# ── B2b-4a: authority-interaction attribution rules ─────────────────────────
def afterheat_attribution_severity(expected_afterheat_rise_c: Optional[float],
                                   observed_overshoot_c: Optional[float], *,
                                   high_afterheat_c: float = 0.5) -> ConfounderSeverity:
    """Afterheat is INFO when small; DEGRADING when it explains a material part of overshoot.

    It never fully exonerates the boost: a strong overshoot still carries boost responsibility,
    but a high expected afterheat reduces (does not zero) the boost overshoot attribution.
    """
    if expected_afterheat_rise_c is None:
        return ConfounderSeverity.INFO
    if expected_afterheat_rise_c >= high_afterheat_c:
        return ConfounderSeverity.DEGRADING
    return ConfounderSeverity.INFO


def early_cutoff_interaction_severity(materially_overlapped: bool, *,
                                      severe: bool = False) -> ConfounderSeverity:
    """Early cutoff overlapping the boost phase must not yield a full NO_EFFECT/UNNECESSARY
    credit against the boost (no double negative credit to both authorities)."""
    if not materially_overlapped:
        return ConfounderSeverity.INFO
    return ConfounderSeverity.BLOCKING if severe else ConfounderSeverity.DEGRADING


@dataclass(frozen=True)
class SupplyDelayParams:
    tolerance_min: float = 10.0            # delay within expected ± tolerance is normal
    degrading_min: float = 20.0           # beyond this unexpected delay -> DEGRADING
    blocking_min: float = 45.0            # extreme delay -> BLOCKING
    no_onset_min: float = 90.0            # no onset by here -> HEATING_FAILURE / interrupted


def supply_delay_confounder(expected_onset_min: Optional[float],
                            actual_onset_min: Optional[float], *,
                            params: Optional[SupplyDelayParams] = None
                            ) -> Optional[BoostConfounder]:
    """Classify the heating-onset delay vs. the expected onset (from the onset-delay model)."""
    p = params or SupplyDelayParams()
    if actual_onset_min is None:
        return BoostConfounder.HEATING_FAILURE          # no onset observed
    if actual_onset_min >= p.no_onset_min:
        return BoostConfounder.HEATING_FAILURE
    expected = expected_onset_min if expected_onset_min is not None else 0.0
    excess = actual_onset_min - expected
    if excess <= p.tolerance_min:
        return None                                     # normal / known delay
    if excess >= p.blocking_min:
        return BoostConfounder.HEATING_FAILURE          # extreme -> interrupting
    if excess >= p.degrading_min:
        return BoostConfounder.SUPPLY_DELAY_ANOMALY     # moderate unexpected -> degrading
    return None                                         # within tolerance..degrading: tolerated
