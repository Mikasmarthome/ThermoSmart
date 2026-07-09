"""Shadow orchestration for Learning (pure Python).

Computes Learning shadow predictions (incl. a full preheat plan), compares them to
the existing ThermoSmart baseline and — once reality is observed — evaluates the
shadow outcome and aggregates metrics. STRICTLY advisory: produces only typed
shadow results that the coordinator can never mistake for a control command.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from ..contracts import DecisionId, PredictionType

SHADOW_SCHEMA_VERSION = 1


class ShadowStatus(Enum):
    OPEN = "open"
    EVALUATED = "evaluated"
    NOT_EVALUABLE = "not_evaluable"
    EXPIRED = "expired"


class ShadowReason(Enum):
    OK = "ok"
    NO_BASELINE = "no_baseline"
    NO_INDOOR_TEMP = "no_indoor_temp"
    LOW_CONFIDENCE = "low_confidence"
    FALLBACK_USED = "fallback_used"
    DISTURBED = "disturbed"
    INSUFFICIENT_REALITY = "insufficient_reality"


class ComparisonType(Enum):
    PREHEAT_START = "preheat_start"
    BOOST_OFFSET = "boost_offset"
    EARLY_CUTOFF = "early_cutoff"
    TARGET_ACHIEVEMENT = "target_achievement"


@dataclass(frozen=True)
class ShadowPrediction:
    prediction_type: str
    values: Mapping[str, float]
    units: Mapping[str, str]
    confidence: float
    reliability: float
    fallback_used: bool
    prior_contribution: float
    learned_contribution: float
    missing_evidence: tuple[str, ...]
    reason_codes: tuple[str, ...]
    model_version: int
    parameter_version: int
    decision_id: str
    # explicit, structural no-control marker
    control_effect: bool = False

    def __post_init__(self) -> None:
        if self.control_effect:
            raise ValueError("ShadowPrediction must never carry a control effect")


@dataclass(frozen=True)
class PreheatPlan:
    comfort_time_utc: str
    comfort_temperature_c: float
    current_temperature_c: Optional[float]
    deficit_c: Optional[float]
    expected_heat_rate_c_per_h: float
    expected_heat_loss_c_per_h: float
    expected_afterheat_rise_c: float
    active_duration_min: Optional[float]
    recommended_start_utc: Optional[str]
    expected_early_cutoff_utc: Optional[str]
    expected_achievement_c: Optional[float]
    confidence: float
    fallback_used: bool
    reason_codes: tuple[str, ...]
    control_effect: bool = False


@dataclass(frozen=True)
class ShadowComparison:
    zone: str
    decision_id: str
    comparison_type: str
    baseline_source: str
    baseline_value: Optional[float]
    shadow_value: Optional[float]
    difference: Optional[float]
    unit: str
    learning_confidence: float
    shadow_expected_outcome: Optional[float]
    baseline_outcome: Optional[float]
    actual_outcome: Optional[float]
    status: str
    reason_codes: tuple[str, ...]
    created_ts: str
    schema_version: int = SHADOW_SCHEMA_VERSION


@dataclass(frozen=True)
class ShadowOutcome:
    decision_id: str
    on_time: Optional[bool]
    early: Optional[bool]
    late: Optional[bool]
    reached: Optional[bool]
    overshoot_c: Optional[float]
    stability_after_target: Optional[float]
    data_quality: float
    reason_codes: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class ShadowMetrics:
    comparison_count: int
    evaluated_count: int
    mean_preheat_start_diff_min: Optional[float]
    on_time_rate: Optional[float]
    late_rate: Optional[float]
    early_rate: Optional[float]
    overshoot_rate: Optional[float]
    fallback_rate: Optional[float]
    not_evaluable_rate: Optional[float]
    confidence_coverage: Optional[float]


@dataclass(frozen=True)
class PreheatParameters:
    min_net_rate_c_per_h: float = 0.2
    start_margin_min: float = 5.0
    low_confidence_threshold: float = 0.4
    fallback_heat_rate_c_per_h: float = 1.5
    fallback_heat_loss_c_per_h: float = 0.3
    fallback_afterheat_rise_c: float = 0.3


# -- pure preheat computation -------------------------------------------------

def compute_preheat_plan(*, comfort_time_utc: str, comfort_temperature_c: float,
                         current_temperature_c: Optional[float],
                         heat_rate_c_per_h: float, heat_loss_c_per_h: float,
                         afterheat_rise_c: float, confidence: float, fallback_used: bool,
                         params: Optional[PreheatParameters] = None) -> PreheatPlan:
    """The schedule time IS the comfort-achievement time, not the heating start."""
    p = params or PreheatParameters()
    reasons: list[str] = []
    comfort_dt = _parse(comfort_time_utc)
    if current_temperature_c is None:
        reasons.append(ShadowReason.NO_INDOOR_TEMP.value)
        return PreheatPlan(
            comfort_time_utc=comfort_time_utc, comfort_temperature_c=comfort_temperature_c,
            current_temperature_c=None, deficit_c=None,
            expected_heat_rate_c_per_h=heat_rate_c_per_h,
            expected_heat_loss_c_per_h=heat_loss_c_per_h,
            expected_afterheat_rise_c=afterheat_rise_c, active_duration_min=None,
            recommended_start_utc=None, expected_early_cutoff_utc=None,
            expected_achievement_c=None, confidence=min(confidence, 0.3),
            fallback_used=True, reason_codes=tuple(reasons))

    deficit = max(0.0, comfort_temperature_c - current_temperature_c)
    net_rate = max(heat_rate_c_per_h - heat_loss_c_per_h, p.min_net_rate_c_per_h)
    # afterheat covers part of the rise after the drive ends (early cutoff)
    effective_deficit = max(0.0, deficit - max(0.0, afterheat_rise_c))
    active_duration_min = (effective_deficit / net_rate) * 60.0
    recommended_start = None
    early_cutoff = None
    if comfort_dt is not None:
        start_dt = comfort_dt - timedelta(minutes=active_duration_min + p.start_margin_min)
        recommended_start = start_dt.isoformat()
        # cutoff when the remaining rise will be supplied by afterheat
        cutoff_lead_min = (max(0.0, afterheat_rise_c) / net_rate) * 60.0
        early_cutoff = (comfort_dt - timedelta(minutes=cutoff_lead_min)).isoformat()
    expected_achievement = current_temperature_c + net_rate * (active_duration_min / 60.0) \
        + max(0.0, afterheat_rise_c)
    if fallback_used:
        reasons.append(ShadowReason.FALLBACK_USED.value)
    if confidence < p.low_confidence_threshold:
        reasons.append(ShadowReason.LOW_CONFIDENCE.value)
    if not reasons:
        reasons.append(ShadowReason.OK.value)
    return PreheatPlan(
        comfort_time_utc=comfort_time_utc, comfort_temperature_c=comfort_temperature_c,
        current_temperature_c=current_temperature_c, deficit_c=round(deficit, 3),
        expected_heat_rate_c_per_h=round(heat_rate_c_per_h, 4),
        expected_heat_loss_c_per_h=round(heat_loss_c_per_h, 4),
        expected_afterheat_rise_c=round(max(0.0, afterheat_rise_c), 4),
        active_duration_min=round(active_duration_min, 2),
        recommended_start_utc=recommended_start, expected_early_cutoff_utc=early_cutoff,
        expected_achievement_c=round(expected_achievement, 3), confidence=round(confidence, 4),
        fallback_used=fallback_used, reason_codes=tuple(reasons))


# -- orchestrator -------------------------------------------------------------

class ShadowOrchestrator:
    """Builds shadow predictions + baseline comparisons; no control effect."""

    def __init__(self, *, preheat_params: Optional[PreheatParameters] = None,
                 max_open_comparisons: int = 200) -> None:
        self._pp = preheat_params or PreheatParameters()
        self._max_open = max_open_comparisons

    def shadow_predictions(self, decision_id: str,
                           predictions: Mapping[PredictionType, Any]) -> list[ShadowPrediction]:
        out: list[ShadowPrediction] = []
        for ptype, pred in sorted(predictions.items(), key=lambda kv: kv[0].value):
            if pred is None:
                continue
            out.append(ShadowPrediction(
                prediction_type=ptype.value, values=dict(pred.values), units=dict(pred.units),
                confidence=round(pred.confidence, 6), reliability=round(pred.reliability, 6),
                fallback_used=pred.fallback_used,
                prior_contribution=round(pred.prior_contribution, 6),
                learned_contribution=round(pred.learned_contribution, 6),
                missing_evidence=tuple(pred.missing_evidence),
                reason_codes=tuple(pred.reason_codes), model_version=pred.model_version,
                parameter_version=pred.parameter_version, decision_id=decision_id))
        return out

    def build_preheat_plan(self, snapshot, predictions: Mapping[PredictionType, Any]) -> Optional[PreheatPlan]:
        if snapshot.schedule_comfort_time_utc is None \
                or snapshot.schedule_comfort_temperature_c is None:
            return None
        hr = _val(predictions.get(PredictionType.HEAT_RATE), "heat_rate",
                  self._pp.fallback_heat_rate_c_per_h)
        hl = _val(predictions.get(PredictionType.HEAT_LOSS_RATE), "heat_loss_rate",
                  self._pp.fallback_heat_loss_c_per_h)
        af = _val(predictions.get(PredictionType.EXPECTED_OVERSHOOT), "expected_overshoot",
                  self._pp.fallback_afterheat_rise_c)
        conf, fb = _conf(predictions.get(PredictionType.HEAT_RATE))
        cur = snapshot.indoor_temp.value if snapshot.indoor_temp is not None else None
        return compute_preheat_plan(
            comfort_time_utc=snapshot.schedule_comfort_time_utc,
            comfort_temperature_c=snapshot.schedule_comfort_temperature_c,
            current_temperature_c=cur, heat_rate_c_per_h=hr, heat_loss_c_per_h=hl,
            afterheat_rise_c=af, confidence=conf, fallback_used=fb, params=self._pp)

    def compare_preheat(self, snapshot, plan: Optional[PreheatPlan],
                        baseline_start_utc: Optional[str], confidence: float) -> Optional[ShadowComparison]:
        if plan is None or plan.recommended_start_utc is None:
            return None
        reasons = [ShadowReason.OK.value] if baseline_start_utc else [ShadowReason.NO_BASELINE.value]
        diff = None
        if baseline_start_utc:
            bd = _parse(baseline_start_utc)
            sd = _parse(plan.recommended_start_utc)
            if bd is not None and sd is not None:
                diff = round((sd - bd).total_seconds() / 60.0, 2)
        return ShadowComparison(
            zone=snapshot.zone_id, decision_id=snapshot.decision_id,
            comparison_type=ComparisonType.PREHEAT_START.value,
            baseline_source="existing_controller", baseline_value=_to_minutes(baseline_start_utc),
            shadow_value=_to_minutes(plan.recommended_start_utc), difference=diff, unit="minutes",
            learning_confidence=round(confidence, 4),
            shadow_expected_outcome=plan.expected_achievement_c, baseline_outcome=None,
            actual_outcome=None, status=ShadowStatus.OPEN.value, reason_codes=tuple(reasons),
            created_ts=snapshot.ts)

    def compare_scalar(self, snapshot, ctype: ComparisonType, baseline: Optional[float],
                       shadow: Optional[float], unit: str, confidence: float) -> ShadowComparison:
        diff = None
        reasons = [ShadowReason.OK.value]
        if baseline is None:
            reasons = [ShadowReason.NO_BASELINE.value]
        elif shadow is not None:
            diff = round(shadow - baseline, 4)
        return ShadowComparison(
            zone=snapshot.zone_id, decision_id=snapshot.decision_id,
            comparison_type=ctype.value, baseline_source="existing_controller",
            baseline_value=baseline, shadow_value=shadow, difference=diff, unit=unit,
            learning_confidence=round(confidence, 4), shadow_expected_outcome=None,
            baseline_outcome=None, actual_outcome=None, status=ShadowStatus.OPEN.value,
            reason_codes=tuple(reasons), created_ts=snapshot.ts)

    def complete_comparison(self, comparison: ShadowComparison, *, actual_outcome: float,
                            baseline_outcome: Optional[float] = None) -> ShadowComparison:
        from dataclasses import replace
        return replace(comparison, actual_outcome=actual_outcome,
                       baseline_outcome=baseline_outcome, status=ShadowStatus.EVALUATED.value)

    def evaluate_shadow_outcome(self, decision_id: str, *, target_c: float,
                                achieved_c: Optional[float], on_time: Optional[bool],
                                overshoot_c: Optional[float], stability: Optional[float],
                                data_quality: float, disturbed: bool = False) -> ShadowOutcome:
        if disturbed:
            return ShadowOutcome(decision_id=decision_id, on_time=None, early=None, late=None,
                                 reached=None, overshoot_c=None, stability_after_target=None,
                                 data_quality=round(data_quality, 4),
                                 reason_codes=(ShadowReason.DISTURBED.value,),
                                 status=ShadowStatus.NOT_EVALUABLE.value)
        if achieved_c is None:
            return ShadowOutcome(decision_id=decision_id, on_time=None, early=None, late=None,
                                 reached=None, overshoot_c=None, stability_after_target=None,
                                 data_quality=round(data_quality, 4),
                                 reason_codes=(ShadowReason.NO_INDOOR_TEMP.value,),
                                 status=ShadowStatus.NOT_EVALUABLE.value)
        reached = achieved_c >= target_c - 0.3
        return ShadowOutcome(
            decision_id=decision_id, on_time=on_time,
            early=(on_time is False and achieved_c >= target_c), late=(on_time is False and not reached),
            reached=reached, overshoot_c=round(max(0.0, achieved_c - target_c), 3)
            if overshoot_c is None else round(overshoot_c, 3),
            stability_after_target=None if stability is None else round(stability, 3),
            data_quality=round(data_quality, 4), reason_codes=(ShadowReason.OK.value,),
            status=ShadowStatus.EVALUATED.value)

    def aggregate_metrics(self, comparisons: Sequence[ShadowComparison],
                          outcomes: Sequence[ShadowOutcome]) -> ShadowMetrics:
        n = len(comparisons)
        evaluated = [c for c in comparisons if c.status == ShadowStatus.EVALUATED.value]
        preheat_diffs = [c.difference for c in comparisons
                         if c.comparison_type == ComparisonType.PREHEAT_START.value
                         and c.difference is not None]
        ev_outcomes = [o for o in outcomes if o.status == ShadowStatus.EVALUATED.value]
        not_eval = [o for o in outcomes if o.status == ShadowStatus.NOT_EVALUABLE.value]
        fb = [c for c in comparisons if ShadowReason.FALLBACK_USED.value in c.reason_codes]

        def rate(cond, base):
            return round(sum(1 for o in base if cond(o)) / len(base), 4) if base else None

        return ShadowMetrics(
            comparison_count=n, evaluated_count=len(evaluated),
            mean_preheat_start_diff_min=round(sum(preheat_diffs) / len(preheat_diffs), 3)
            if preheat_diffs else None,
            on_time_rate=rate(lambda o: o.on_time is True, ev_outcomes),
            late_rate=rate(lambda o: o.late is True, ev_outcomes),
            early_rate=rate(lambda o: o.early is True, ev_outcomes),
            overshoot_rate=rate(lambda o: (o.overshoot_c or 0.0) > 0.5, ev_outcomes),
            fallback_rate=round(len(fb) / n, 4) if n else None,
            not_evaluable_rate=round(len(not_eval) / len(outcomes), 4) if outcomes else None,
            confidence_coverage=round(
                sum(1 for c in comparisons if c.learning_confidence >= 0.4) / n, 4) if n else None)


# -- helpers ------------------------------------------------------------------

def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _to_minutes(ts_iso: Optional[str]) -> Optional[float]:
    dt = _parse(ts_iso)
    if dt is None:
        return None
    return round(dt.hour * 60 + dt.minute + dt.second / 60.0, 3)


def _val(pred: Any, key: str, default: float) -> float:
    if pred is None:
        return default
    try:
        v = pred.values.get(key)
        return float(v) if v is not None else default
    except Exception:
        return default


def _conf(pred: Any) -> tuple[float, bool]:
    if pred is None:
        return 0.2, True
    return float(pred.confidence), bool(pred.fallback_used)
