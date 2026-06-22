"""Guard layer (Phase 19A, pure Python).

Plausibility, anti-chatter / timing, and device-range guards applied AFTER the
resolver and BEFORE the device adapter. Guards may only ever fall back to the
baseline; they never invent a value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import ControlGuardResult


@dataclass(frozen=True)
class GuardParameters:
    min_setpoint_c: float = 5.0
    max_setpoint_c: float = 30.0
    max_abs_step_c: float = 5.0          # plausibility: max change vs. baseline in one cycle
    anti_chatter_min_delta_c: float = 0.1  # ignore sub-threshold changes (no re-dispatch)


class GuardLayer:
    """Stateless plausibility + range guard; anti-chatter is delta-based."""

    def __init__(self, params: Optional[GuardParameters] = None) -> None:
        self.params = params or GuardParameters()

    def check_setpoint(self, baseline_c: Optional[float], proposed_c: Optional[float],
                       *, device_min: Optional[float] = None,
                       device_max: Optional[float] = None) -> ControlGuardResult:
        p = self.params
        if proposed_c is None:
            return ControlGuardResult(False, baseline_c, 0.0, ("no_value",))
        reasons: list[str] = []
        value = float(proposed_c)
        lo = device_min if device_min is not None else p.min_setpoint_c
        hi = device_max if device_max is not None else p.max_setpoint_c
        # plausibility vs baseline
        if baseline_c is not None and abs(value - baseline_c) > p.max_abs_step_c:
            return ControlGuardResult(False, baseline_c, 0.0, ("implausible_step",))
        # anti-chatter: sub-threshold change -> keep baseline, do not re-dispatch
        if baseline_c is not None and abs(value - baseline_c) < p.anti_chatter_min_delta_c:
            return ControlGuardResult(False, baseline_c, 0.0, ("anti_chatter",))
        clamped = max(lo, min(hi, value))
        clamp_applied = abs(clamped - value)
        if clamp_applied > 0:
            reasons.append("device_limit")
        return ControlGuardResult(True, clamped, clamp_applied, tuple(reasons))
