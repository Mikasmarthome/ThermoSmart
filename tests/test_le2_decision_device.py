"""Phase 19A: Device Adapter (min/max/step/round/frost; unconfirmed profiles)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import DeviceAdapter, DeviceProfile


class TestDeviceAdapter:
    def test_quantize_to_step(self):
        a = DeviceAdapter(DeviceProfile(step_c=0.5))
        assert a.build_command("z", setpoint_c=21.3).trv_setpoint_c == 21.5

    def test_quarter_step(self):
        a = DeviceAdapter(DeviceProfile(step_c=0.25))
        assert a.build_command("z", setpoint_c=21.3).trv_setpoint_c == 21.25

    def test_min_max_clamp(self):
        a = DeviceAdapter(DeviceProfile(min_temp_c=7.0, max_temp_c=28.0))
        assert a.build_command("z", setpoint_c=5.0).trv_setpoint_c == 7.0
        assert a.build_command("z", setpoint_c=30.0).trv_setpoint_c == 28.0

    def test_duty_clamped(self):
        a = DeviceAdapter()
        assert a.build_command("z", setpoint_c=21.0, duty_cycle=1.5).duty_cycle == 1.0

    def test_none_setpoint(self):
        assert DeviceAdapter().build_command("z", setpoint_c=None).trv_setpoint_c is None

    def test_unconfirmed_profile_still_supported(self):
        a = DeviceAdapter(DeviceProfile(name="unknown_trv", confirmed=False))
        cmd = a.build_command("z", setpoint_c=21.3)
        assert cmd.trv_setpoint_c == 21.5  # still produces a valid command

    def test_shadow_only_default(self):
        assert DeviceAdapter().build_command("z", setpoint_c=21.0).shadow_only is True

    def test_command_value_finite(self):
        cmd = DeviceAdapter().build_command("z", setpoint_c=21.0)
        assert cmd.trv_setpoint_c == 21.0
