"""Device adapter (Phase 19A, pure Python).

Turns a guarded final setpoint into the single typed ``DeviceControlCommand`` the
dispatcher accepts, applying device semantics (min/max/step/round/frost). Existing
device profiles in trv_control.py remain authoritative for live writes; this is the
typed boundary the new architecture funnels through.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import DeviceControlCommand


@dataclass(frozen=True)
class DeviceProfile:
    name: str = "generic"
    min_temp_c: float = 5.0
    max_temp_c: float = 30.0
    step_c: float = 0.5
    frost_temp_c: float = 5.0
    confirmed: bool = True  # unconfirmed profiles are still supported, just flagged


def _quantize(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(round(value / step) * step, 4)


class DeviceAdapter:
    def __init__(self, profile: Optional[DeviceProfile] = None) -> None:
        self.profile = profile or DeviceProfile()

    def build_command(self, zone_id: str, *, setpoint_c: Optional[float],
                      duty_cycle: Optional[float] = None, hvac_mode: Optional[str] = None,
                      source_reason: str = "baseline",
                      shadow_only: bool = True) -> DeviceControlCommand:
        sp = setpoint_c
        if sp is not None:
            sp = max(self.profile.min_temp_c, min(self.profile.max_temp_c, float(sp)))
            sp = _quantize(sp, self.profile.step_c)
        return DeviceControlCommand(
            zone_id=zone_id, trv_setpoint_c=sp,
            duty_cycle=None if duty_cycle is None else max(0.0, min(1.0, float(duty_cycle))),
            hvac_mode=hvac_mode, source_reason=source_reason, shadow_only=shadow_only)
