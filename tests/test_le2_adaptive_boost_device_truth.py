"""B2c per-device applied truth — counterfactual baseline tests.

Verifies that device_applied_boost_i is computed via:

    effective_device_target_with_boost - effective_device_target_without_boost

where BOTH sides run through the same per-device transform path
(min-clamp, step-snap, re-clamp, max-clamp) via a single shared authority:
resolve_device_effective_setpoint(). Scenarios:

  TestResolveDeviceEffective  — unit tests for the pure resolver function
  TestDifferentPriors         — two devices with different min_temp (=different effective baselines)
  TestDeviceMaximum           — device max_temp / profile maximum reached; applied < approved
  TestStepQuantization        — step-snap changes effective offset
  TestMixedSteps              — two devices with different temperature_step
  TestPriorClampStepCombined  — all transforms combined for single device
"""
from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock

from custom_components.thermosmart.trv_control import TRVControlMixin

# No asyncio mark: all tests in this file are synchronous.


# ── Helper: invoke the pure resolver directly ─────────────────────────────────

class _FakeMixin(TRVControlMixin):
    """Minimal concrete class to test TRVControlMixin methods."""
    zone_name = "test"

    def __init__(self):
        self._device_profiles = {}

    @property
    def hass(self):
        return None


def _state(min_temp=5.0, step=0.5, current=21.0, max_temp=None):
    """Build a mock climate state with relevant attributes."""
    st = MagicMock()
    st.state = "heat"
    attrs = {
        "min_temp": min_temp,
        "temperature": current,
        "target_temp_step": step,
    }
    if max_temp is not None:
        attrs["max_temp"] = max_temp
    st.attributes = attrs
    return st


def _resolve(setpoint, *, min_temp=5.0, step=0.5, max_temp=None, profile=None):
    """Resolve effective setpoint via the production resolver."""
    mixin = _FakeMixin()
    state = _state(min_temp=min_temp, step=step, max_temp=max_temp)
    return mixin.resolve_device_effective_setpoint("climate.test", state, setpoint, profile)


def _applied(baseline, boost, *, min_temp=5.0, step=0.5, max_temp=None, profile=None):
    """Compute applied offset = resolve(baseline+boost) - resolve(baseline)."""
    return (
        _resolve(baseline + boost, min_temp=min_temp, step=step, max_temp=max_temp, profile=profile)
        - _resolve(baseline, min_temp=min_temp, step=step, max_temp=max_temp, profile=profile)
    )


# ── TestResolveDeviceEffective ────────────────────────────────────────────────

class TestResolveDeviceEffective:
    """Unit tests for resolve_device_effective_setpoint (pure resolver)."""

    def test_identity_no_clamp_no_step(self):
        """No clamp, no step → value passes through unchanged."""
        assert _resolve(21.0, min_temp=5.0, step=0.0) == pytest.approx(21.0)

    def test_step_snap_rounds_up(self):
        """21.3 with step=0.5 → 21.5 (round-half-up)."""
        assert _resolve(21.3, step=0.5) == pytest.approx(21.5)

    def test_step_snap_rounds_down(self):
        """21.2 with step=0.5 → 21.0."""
        assert _resolve(21.2, step=0.5) == pytest.approx(21.0)

    def test_step_snap_0_1(self):
        """21.07 with step=0.1 → 21.1."""
        assert _resolve(21.07, step=0.1) == pytest.approx(21.1, abs=0.01)

    def test_min_clamp_applied(self):
        """If setpoint < min_temp, effective = min_temp."""
        assert _resolve(4.0, min_temp=5.0, step=0.0) == pytest.approx(5.0)

    def test_min_clamp_after_step_snap(self):
        """After step-snap, re-clamp must restore min."""
        # 5.2 step=0.5 → snaps to 5.0; 5.0 >= min 5.0 → OK
        assert _resolve(5.2, min_temp=5.0, step=0.5) == pytest.approx(5.0)

    def test_passes_through_below_max(self):
        """Value below max_temp passes through unchanged."""
        assert _resolve(21.0, min_temp=5.0, step=0.0, max_temp=25.0) == pytest.approx(21.0)

    def test_max_clamp_applied(self):
        """Value above max_temp is clamped to max_temp."""
        assert _resolve(26.0, min_temp=5.0, step=0.0, max_temp=25.0) == pytest.approx(25.0)

    def test_no_max_when_attribute_absent(self):
        """When max_temp is absent from state, no upper clamp is applied."""
        assert _resolve(35.0, min_temp=5.0, step=0.0) == pytest.approx(35.0)

    def test_step_zero_treated_as_no_step(self):
        """step=0 means no step-snap (avoid division by zero)."""
        assert _resolve(21.3, step=0.0) == pytest.approx(21.3)


# ── TestDifferentPriors ───────────────────────────────────────────────────────

class TestDifferentPriors:
    """Two devices with different min_temp — prior may absorb boost partially or fully.

    In ThermoSmart, the SAME zone-level trv_setpoint is sent to all devices.
    Per-device effective baseline = resolve(zone_baseline, device).
    Per-device effective with_boost = resolve(zone_baseline + approved_boost, device).
    Applied_i = effective_with_boost_i - effective_baseline_i.

    Key insight: when min_temp > zone_baseline, the min clamp raises BOTH sides.
    The boost is absorbed completely when (baseline + boost) is still <= min_temp.
    """

    def test_device_below_min_prior_absorbs_boost_entirely(self):
        """
        Zone baseline = 21.0, approved boost = 0.5, Device B min_temp = 21.5.
        effective_baseline_B = max(21.0, 21.5) = 21.5
        effective_with_boost_B = max(21.5, 21.5) = 21.5   (21.0+0.5=21.5=min → no headroom)
        applied_B = 0.0  (prior absorbs full 0.5 boost)
        """
        applied = _applied(21.0, 0.5, min_temp=21.5, step=0.0)
        assert applied == pytest.approx(0.0, abs=0.01)

    def test_device_below_min_prior_partially_absorbs_boost(self):
        """
        Zone baseline = 21.0, approved boost = 2.0, Device B min_temp = 21.5.
        effective_baseline_B = 21.5 (clamped)
        effective_with_boost_B = max(23.0, 21.5) = 23.0
        applied_B = 23.0 - 21.5 = 1.5  (prior absorbs 0.5 of the 2.0 boost)
        """
        applied = _applied(21.0, 2.0, min_temp=21.5, step=0.0)
        assert applied == pytest.approx(1.5, abs=0.01)

    def test_prior_cancels_when_both_sides_above_min(self):
        """When zone_baseline > min_temp, min clamp doesn't affect either side.
        Device prior is irrelevant — applied = approved boost (no step).
        """
        bl = 22.0
        boost = 1.0
        # min_temp=5 (no clamp), min_temp=15 (no clamp since bl=22 > 15)
        for min_t in (5.0, 10.0, 15.0, 20.0):
            applied = _applied(bl, boost, min_temp=min_t, step=0.0)
            assert applied == pytest.approx(boost, abs=0.01), \
                f"prior min={min_t} should not affect applied when baseline {bl} > min"

    def test_two_device_zone_conservative_min(self):
        """Zone with two devices: conservative min(applied_A, applied_B) correctly captures
        the true zone-level applied offset.

        Device A: min_temp=5.0   → applied = 1.0 (full boost)
        Device B: min_temp=21.5  → applied = 0.5 (prior absorbs 0.5 of the 1.0 boost)
        conservative = min(1.0, 0.5) = 0.5
        """
        boost = 1.0
        bl = 21.0
        applied_a = _applied(bl, boost, min_temp=5.0, step=0.0)
        applied_b = _applied(bl, boost, min_temp=21.5, step=0.0)
        assert applied_a == pytest.approx(1.0, abs=0.01)
        assert applied_b == pytest.approx(0.5, abs=0.01)
        assert min(applied_a, applied_b) == pytest.approx(0.5, abs=0.01)


# ── TestDeviceMaximum ─────────────────────────────────────────────────────────

class TestDeviceMaximum:
    """Device max_temp and profile.maximum_setpoint reduce the applied boost.

    The resolver (steps 5-6) clamps the effective setpoint to the device ceiling
    AFTER step-snap.  When the boost raises both sides of the counterfactual, the
    ceiling clips the result so that applied < approved.
    """

    def test_max_clamp_reduces_applied_boost(self):
        """
        baseline=22.5, boost=1.0, max_temp=23.0, step=0.0
        effective_baseline = 22.5         (no clamp: 22.5 ≤ 23.0)
        effective_boosted  = 23.0         (23.5 clamped to 23.0)
        applied = 23.0 - 22.5 = 0.5      (not the full 1.0)
        """
        applied = _applied(22.5, 1.0, min_temp=5.0, step=0.0, max_temp=23.0)
        assert applied == pytest.approx(0.5, abs=0.01)

    def test_baseline_at_max_applied_zero(self):
        """
        baseline=23.0 already at max_temp → both sides clamp to 23.0.
        applied = 0.0
        """
        applied = _applied(23.0, 1.0, min_temp=5.0, step=0.0, max_temp=23.0)
        assert applied == pytest.approx(0.0, abs=0.01)

    def test_step_snap_over_max_reclamped(self):
        """
        baseline=22.0, boost=1.3, step=0.5, max_temp=23.0
        with_boost_input=23.3 → step-snap→23.5 → max-clamp→23.0
        baseline→step-snap→22.0
        applied = 23.0 - 22.0 = 1.0  (step would have given 1.5 without max)
        """
        applied = _applied(22.0, 1.3, min_temp=5.0, step=0.5, max_temp=23.0)
        assert applied == pytest.approx(1.0, abs=0.01)

    def test_profile_max_stricter_than_ha_max(self):
        """Profile maximum_setpoint = 22.5 overrides HA max_temp = 25.0."""
        from custom_components.thermosmart.device_profiles.capabilities import DeviceProfile
        profile = DeviceProfile(identifier="test", display_name="Test",
                                maximum_setpoint=22.5)
        applied = _applied(22.0, 1.0, min_temp=5.0, step=0.0, max_temp=25.0, profile=profile)
        # boosted=23.0 → ha-max(25.0)→23.0 → profile-max(22.5)→22.5
        # baseline=22.0 → ha-max(25.0)→22.0 → profile-max(22.5)→22.0
        # applied = 22.5 - 22.0 = 0.5
        assert applied == pytest.approx(0.5, abs=0.01)

    def test_ha_max_stricter_than_profile_max(self):
        """HA max_temp = 22.5 is stricter than profile.maximum_setpoint = 25.0."""
        from custom_components.thermosmart.device_profiles.capabilities import DeviceProfile
        profile = DeviceProfile(identifier="test", display_name="Test",
                                maximum_setpoint=25.0)
        applied = _applied(22.0, 1.0, min_temp=5.0, step=0.0, max_temp=22.5, profile=profile)
        # boosted=23.0 → ha-max(22.5)→22.5 → profile-max(25.0)→22.5
        # baseline=22.0 → ha-max(22.5)→22.0 → profile-max(25.0)→22.0
        # applied = 22.5 - 22.0 = 0.5
        assert applied == pytest.approx(0.5, abs=0.01)

    def test_min_and_max_combined(self):
        """min_temp=21.5, max_temp=23.0, baseline=21.0, boost=2.0.
        effective_baseline = max(21.0, 21.5) = 21.5  (min-clamp)
        effective_boosted  = max(23.0, 21.5) = 23.0 → min(23.0, 23.0) = 23.0  (max-clamp)
        applied = 23.0 - 21.5 = 1.5
        """
        applied = _applied(21.0, 2.0, min_temp=21.5, step=0.0, max_temp=23.0)
        assert applied == pytest.approx(1.5, abs=0.01)

    def test_two_device_zone_different_max_conservative_min(self):
        """Device A: max=25.0 → applied=2.0. Device B: max=22.5 → applied=0.5.
        Conservative min(2.0, 0.5) = 0.5.
        """
        applied_a = _applied(21.0, 2.0, min_temp=5.0, step=0.0, max_temp=25.0)
        applied_b = _applied(21.0, 2.0, min_temp=5.0, step=0.0, max_temp=22.5)
        assert applied_a == pytest.approx(2.0, abs=0.01)
        assert applied_b == pytest.approx(1.5, abs=0.01)
        assert min(applied_a, applied_b) == pytest.approx(1.5, abs=0.01)

    def test_prior_max_step_combined(self):
        """All three: min_temp=21.5 (prior), max_temp=23.0, step=0.5.
        baseline=21.0, boost=2.0
        effective_baseline:
          input=21.0 → min-clamp→21.5 → step-snap(21.5)→21.5 → re-clamp-min→21.5
                     → max-clamp(23.0)→21.5
        effective_boosted:
          input=23.0 → min-clamp→23.0 → step-snap(23.0)→23.0 → re-clamp-min→23.0
                     → max-clamp(23.0)→23.0
        applied = 23.0 - 21.5 = 1.5
        """
        applied = _applied(21.0, 2.0, min_temp=21.5, step=0.5, max_temp=23.0)
        assert applied == pytest.approx(1.5, abs=0.01)

    def test_step_rounding_at_max_boundary(self):
        """step=0.5, max=23.0: boosted=22.8 → snap→23.0 → within max → 23.0.
        baseline=22.0 → snap→22.0.
        applied = 1.0
        """
        applied = _applied(22.0, 0.8, min_temp=5.0, step=0.5, max_temp=23.0)
        # 22.8 step-snap: floor(22.8/0.5+0.5)*0.5 = floor(46.1)*0.5 = 46*0.5 = 23.0
        assert applied == pytest.approx(1.0, abs=0.01)


# ── TestStepQuantization ──────────────────────────────────────────────────────

class TestStepQuantization:
    """Step-snap changes the effective applied offset from the approved boost."""

    def test_boost_amplified_by_step(self):
        """approved=0.3, step=0.5 → snapped to 0.5 on both sides — effective=0.5."""
        # baseline=21.0 → snap→21.0; with_boost=21.3 → snap→21.5; applied=0.5
        applied = _applied(21.0, 0.3, min_temp=5.0, step=0.5)
        assert applied == pytest.approx(0.5, abs=0.01)

    def test_boost_reduced_by_step(self):
        """approved=0.8, step=0.5 → effectively 0.5 after snap."""
        # baseline=21.0 → snap→21.0; with_boost=21.8 → snap→22.0; applied=1.0
        # (21.8 rounds UP to 22.0 via floor(21.8/0.5+0.5)*0.5 = floor(36.1)*0.5=36*0.5=18.0*... )
        # Let me compute: floor(21.8/0.5 + 0.5)*0.5 = floor(43.6+0.5)*0.5 = floor(44.1)*0.5=44*0.5=22.0
        applied = _applied(21.0, 0.8, min_temp=5.0, step=0.5)
        assert applied == pytest.approx(1.0, abs=0.01)

    def test_boost_zero_after_step_if_both_sides_snap_same(self):
        """If baseline and baseline+boost both snap to the same value, applied=0."""
        # baseline=21.0, boost=0.1, step=0.5
        # baseline_snap=21.0; with_boost=21.1 → snap→21.0; applied=0.0
        applied = _applied(21.0, 0.1, min_temp=5.0, step=0.5)
        assert applied == pytest.approx(0.0, abs=0.01)

    def test_step_0_1_preserves_offset(self):
        """Fine step=0.1 preserves the applied offset accurately."""
        applied = _applied(21.0, 0.5, min_temp=5.0, step=0.1)
        assert applied == pytest.approx(0.5, abs=0.05)


# ── TestMixedSteps ───────────────────────────────────────────────────────────

class TestMixedSteps:
    """Two devices with different temperature_step — per-device truth diverges."""

    def test_device_a_fine_step_device_b_coarse_step(self):
        """baseline=21.0, boost=1.0:
        Device A (step=0.1): both snap with 0.1 → applied ≈ 1.0
        Device B (step=0.5): 21.5 → applied ≈ 0.5 (but with_boost=22.0 → applied=1.0)
        """
        applied_a = _applied(21.0, 1.0, min_temp=5.0, step=0.1)
        applied_b = _applied(21.0, 1.0, min_temp=5.0, step=0.5)
        # Both should be ≈ 1.0 since 1.0 is divisible by both steps
        assert applied_a == pytest.approx(1.0, abs=0.01)
        assert applied_b == pytest.approx(1.0, abs=0.01)

    def test_conservative_min_uses_lower_device(self):
        """min(applied_a, applied_b) is the conservative zone truth."""
        # approved=0.3, step_a=0.1, step_b=0.5
        a = _applied(21.0, 0.3, step=0.1)
        b = _applied(21.0, 0.3, step=0.5)
        # a: 21.3 → snap(0.1) → 21.3; applied=0.3
        # b: 21.3 → snap(0.5) → 21.5; applied=0.5
        # conservative = min(0.3, 0.5) = 0.3
        conservative = min(a, b)
        assert conservative == pytest.approx(a, abs=0.01)


# ── TestPriorClampStepCombined ────────────────────────────────────────────────

class TestPriorClampStepCombined:
    """Combined: device prior + clamp + step — all transforms at once."""

    def test_prior_clamp_and_step_combined(self):
        """
        Device: min_temp=21.0, step=0.5
        Zone baseline = 20.5 (below device min)
        Effective baseline = max(20.5, 21.0) = 21.0 → snap(0.5) → 21.0

        Zone approved boost = 1.0
        Input with_boost = 20.5 + 1.0 = 21.5
        Effective with_boost = max(21.5, 21.0) = 21.5 → snap(0.5) → 21.5

        Applied = 21.5 - 21.0 = 0.5 (not 1.0 — prior clamp absorbs 0.5)
        """
        applied = _applied(20.5, 1.0, min_temp=21.0, step=0.5)
        assert applied == pytest.approx(0.5, abs=0.01)

    def test_prior_absorbs_entire_boost(self):
        """
        Device: min_temp=22.5, step=0.5
        Baseline=21.0, boost=1.0 → with_boost=22.0
        effective_baseline = 22.5 (min clamp)
        effective_with_boost = 22.5 (min clamp still dominates)
        applied = 0.0
        """
        applied = _applied(21.0, 1.0, min_temp=22.5, step=0.5)
        assert applied == pytest.approx(0.0, abs=0.01)

    def test_prior_partially_absorbs_boost(self):
        """
        Device: min_temp=22.0, step=0.5
        Baseline=21.0, boost=2.0 → with_boost=23.0
        effective_baseline = max(21.0, 22.0) = 22.0 → snap → 22.0
        effective_with_boost = max(23.0, 22.0) = 23.0 → snap → 23.0
        applied = 23.0 - 22.0 = 1.0 (not 2.0 — prior absorbs 1.0)
        """
        applied = _applied(21.0, 2.0, min_temp=22.0, step=0.5)
        assert applied == pytest.approx(1.0, abs=0.01)
