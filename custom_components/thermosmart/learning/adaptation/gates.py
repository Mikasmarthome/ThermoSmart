"""Passive gate evaluation for adaptation candidates.

All gates are evaluated defensively and never raise. A gate failure produces
a False entry in gate_results — no exceptions propagate.

No HA imports. No runtime state. Pure functions only.
"""
from __future__ import annotations

from .contracts import AdaptationGateResult, OutcomeSignal

_MIN_SAMPLES          = 8      # minimum accepted samples before any gate can pass
_MIN_DATA_QUALITY     = 0.40   # below → data_quality gate fails
_MAX_TIMEOUT_RATE     = 0.50   # above → instability gate fails
_MAX_OVERSHOOT_RATE   = 0.45
_MAX_PARTIAL_RATIO    = 0.70   # above → completeness gate fails (too many partial outcomes)
_MIN_RELIABILITY      = 0.30   # below → reliability gate fails
_MIN_REACHED_RATE     = 0.20   # below → system too broken to generate useful candidates

# Relaxed ceiling for candidates that explicitly address the instability signal
_RELAXED_TIMEOUT_CEILING  = _MAX_TIMEOUT_RATE
_RELAXED_OVERSHOOT_CEILING = _MAX_OVERSHOOT_RATE
_STRICT_TIMEOUT_CEILING    = _MAX_TIMEOUT_RATE  * 0.8
_STRICT_OVERSHOOT_CEILING  = _MAX_OVERSHOOT_RATE * 0.8


def evaluate_gates(
    signal: OutcomeSignal,
    *,
    candidate_mitigates_instability: bool = False,
) -> AdaptationGateResult:
    """Evaluate all safety gates for a candidate given the current outcome signal.

    Args:
        signal: distilled quality signals from OutcomeModel.
        candidate_mitigates_instability: when True the instability gate is
            relaxed (the instability IS the reason to adapt). Caller must set
            this explicitly — it defaults to False (strict).

    Returns:
        AdaptationGateResult with individual gate results and summary.
    """
    gates: dict[str, bool] = {}
    reasons: list[str] = []

    # Gate 1 — minimum sample count
    gates["min_samples"] = signal.sample_count >= _MIN_SAMPLES
    if not gates["min_samples"]:
        reasons.append(f"insufficient_samples:{signal.sample_count}<{_MIN_SAMPLES}")

    # Gate 2 — data quality
    gates["data_quality"] = signal.general_data_quality >= _MIN_DATA_QUALITY
    if not gates["data_quality"]:
        reasons.append(
            f"low_data_quality:{signal.general_data_quality:.2f}<{_MIN_DATA_QUALITY}"
        )

    # Gate 3 — outcome stability (relaxed when candidate addresses the instability)
    t_rate = signal.timeout_rate or 0.0
    o_rate = signal.overshoot_rate or 0.0
    t_ceiling = (
        _RELAXED_TIMEOUT_CEILING if candidate_mitigates_instability
        else _STRICT_TIMEOUT_CEILING
    )
    o_ceiling = (
        _RELAXED_OVERSHOOT_CEILING if candidate_mitigates_instability
        else _STRICT_OVERSHOOT_CEILING
    )
    gates["outcome_stability"] = t_rate <= t_ceiling and o_rate <= o_ceiling
    if not gates["outcome_stability"]:
        reasons.append(
            f"unstable_outcomes:timeout={t_rate:.2f},overshoot={o_rate:.2f}"
        )

    # Gate 4 — confounder contamination
    gates["no_high_confounder"] = not signal.confounder_contamination
    if not gates["no_high_confounder"]:
        reasons.append("confounder_contamination_detected")

    # Gate 5 — completeness (too many partial outcomes → unreliable directional signal)
    gates["outcome_completeness"] = signal.partial_ratio <= _MAX_PARTIAL_RATIO
    if not gates["outcome_completeness"]:
        reasons.append(
            f"too_many_partial:{signal.partial_ratio:.2f}>{_MAX_PARTIAL_RATIO}"
        )

    # Gate 6 — aggregate reliability
    gates["aggregate_reliability"] = signal.aggregate_reliability >= _MIN_RELIABILITY
    if not gates["aggregate_reliability"]:
        reasons.append(
            f"low_reliability:{signal.aggregate_reliability:.2f}<{_MIN_RELIABILITY}"
        )

    # Gate 7 — system not completely broken (reached rate floor)
    r_rate = signal.reached_rate or 0.0
    gates["min_reached"] = r_rate >= _MIN_REACHED_RATE
    if not gates["min_reached"]:
        reasons.append(
            f"too_low_reached:{r_rate:.2f}<{_MIN_REACHED_RATE}"
        )

    passed = all(gates.values())
    blocking = next((k for k, v in gates.items() if not v), None)

    return AdaptationGateResult(
        passed=passed,
        gate_results=gates,
        blocking_gate=blocking,
        reason_codes=tuple(reasons),
    )
