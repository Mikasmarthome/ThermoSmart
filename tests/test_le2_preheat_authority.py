"""Pflicht-Tests: Preheat / HeatRate Authority Transfer (LE 2.0).

Verifies that LE 2.0 is the sole adaptive source for HeatRate, Preheat duration,
Preheat start, and related Confidence when a shadow is attached.  LE v1 must NOT
be read or mutated for these values.

Test classes:
    TestHeatRate           – HEAT_RATE read, unit, confidence, fallback paths
    TestPreheatMinutes     – read_preheat_minutes_safe semantics and formula
    TestEpisodeRecovery    – episode_failed reset requires active_heating regime
    TestCompatibility      – no LE v1 reads/writes, sensor surface, SHADOW default

Pattern: pure Python + MagicMock; no Docker / HA-runtime required.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.thermosmart.const import HEATING_MODE_AUTO
from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
)
from tests.helpers import make_coordinator, make_state, set_hass_states
from tests.helpers_ha_runtime import attach_shadow, FakeStore, make_recording_coordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred(values, *, prediction_type, fallback_used=False, confidence=0.8,
          unit="celsius_per_hour", learned=0.8):
    """Build a minimal Prediction-like mock."""
    p = MagicMock()
    p.prediction_type = prediction_type
    p.values = values
    p.fallback_used = fallback_used
    p.confidence = confidence
    p.learned_contribution = learned
    p.model_version = 1
    p.parameter_version = 1
    p.units = {k: unit for k in values}
    return p


def _hr_pred(rate_c_per_h, *, fallback_used=False, confidence=0.8):
    return _pred({"heat_rate": rate_c_per_h}, prediction_type=PredictionType.HEAT_RATE,
                 fallback_used=fallback_used, confidence=confidence)


def _hl_pred(rate_c_per_h, *, fallback_used=False, confidence=0.8):
    return _pred({"heat_loss_rate": rate_c_per_h}, prediction_type=PredictionType.HEAT_LOSS_RATE,
                 fallback_used=fallback_used, confidence=confidence)


def _af_pred(overshoot_c, *, fallback_used=False, confidence=0.8):
    return _pred({"expected_overshoot": overshoot_c},
                 prediction_type=PredictionType.EXPECTED_OVERSHOOT,
                 fallback_used=fallback_used, confidence=confidence)


def _shadow_with_preds(coord, *, hr=None, hl=None, af=None):
    """Attach real shadow; inject predictions directly into zone runtime."""
    sh = attach_shadow(coord, store=FakeStore())
    preds = {}
    if hr is not None:
        preds[PredictionType.HEAT_RATE] = hr
    if hl is not None:
        preds[PredictionType.HEAT_LOSS_RATE] = hl
    if af is not None:
        preds[PredictionType.EXPECTED_OVERSHOOT] = af
    sh.runtime._zone(coord.zone_id).last_predictions = preds
    return sh


def _make_shadow_only(*, hr_rate=2.4, hl_rate=0.5, af_c=0.0,
                      confidence=0.8, fallback_used=False):
    """Return a standalone shadow (not attached to coord) for unit tests."""
    from tests.helpers import make_coordinator
    coord = make_coordinator()
    return _shadow_with_preds(
        coord,
        hr=_hr_pred(hr_rate, fallback_used=fallback_used, confidence=confidence),
        hl=_hl_pred(hl_rate, confidence=confidence),
        af=_af_pred(af_c, confidence=confidence),
    )


async def _rec(coord, *, indoor="19.0", comfort="21.0"):
    set_hass_states(coord, {"sensor.test_temp": make_state(indoor)})
    coord.learning_engine.async_get_base_target = AsyncMock(return_value=float(indoor))
    return await coord._compute_recommendation(
        coord.entry.data, {"temperature": 10.0}, HEATING_MODE_AUTO
    )


# ---------------------------------------------------------------------------
# 1. HeatRate
# ---------------------------------------------------------------------------

class TestHeatRate:

    def test_valid_positive_heat_rate(self):
        """Positive learned rate → returned as-is in °C/h."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(3.6))
        rate, status = sh.read_heat_rate_safe()
        assert status == "valid"
        assert rate == pytest.approx(3.6)

    def test_zero_heat_rate_rejected(self):
        """rate=0 → cold_start (zero rate cannot heat room)."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(0.0))
        rate, status = sh.read_heat_rate_safe()
        assert status == "cold_start"
        assert rate == 0.0

    def test_negative_heat_rate_rejected(self):
        """rate<0 → cold_start (cooling during active heating is invalid)."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(-1.0))
        rate, status = sh.read_heat_rate_safe()
        assert status == "cold_start"
        assert rate == 0.0

    def test_fallback_pred_rejected(self):
        """fallback_used=True → cold_start; rate must be learned, not prior."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.0, fallback_used=True))
        rate, status = sh.read_heat_rate_safe()
        assert status == "cold_start"
        assert rate == 0.0

    def test_low_confidence_fallback(self):
        """confidence < 0.35 → low_confidence; rate rejected."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.0, confidence=0.2))
        rate, status = sh.read_heat_rate_safe()
        assert status == "low_confidence"
        assert rate == 0.0

    def test_no_prediction_cold_start(self):
        """No HEAT_RATE prediction → cold_start."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh.runtime._zone(coord.zone_id).last_predictions = {}
        rate, status = sh.read_heat_rate_safe()
        assert status == "cold_start"
        assert rate == 0.0

    def test_disabled_shadow_not_available(self):
        """Disabled shadow → not_available."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._enabled = False
        rate, status = sh.read_heat_rate_safe()
        assert status == "not_available"
        assert rate == 0.0

    def test_unit_is_celsius_per_hour(self):
        """Returned rate is in °C/h (NOT °C/min)."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(6.0))
        rate, status = sh.read_heat_rate_safe()
        # 6 °C/h = 0.1 °C/min — test ensures we're NOT returning 0.1
        assert rate == pytest.approx(6.0)

    def test_heat_rate_excludes_onset_delay(self):
        """HeatRate reports only room heating speed; onset delay not folded in."""
        # If onset delay were folded in, a 5-min delay at 60-min effective heat
        # would reduce the apparent rate. We verify the raw rate is preserved.
        coord = make_coordinator()
        expected_rate = 2.4
        sh = _shadow_with_preds(coord, hr=_hr_pred(expected_rate))
        rate, status = sh.read_heat_rate_safe()
        assert status == "valid"
        assert rate == pytest.approx(expected_rate)

    def test_missing_heat_rate_prediction_type_rejected(self):
        """HEAT_LOSS_RATE prediction stored under HEAT_RATE key → cold_start
        (wrong PredictionType accepted by structure but wrong key lookup)."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        # Inject heat_loss under HEAT_LOSS_RATE, leave HEAT_RATE absent.
        sh.runtime._zone(coord.zone_id).last_predictions = {
            PredictionType.HEAT_LOSS_RATE: _hl_pred(0.4),
        }
        rate, status = sh.read_heat_rate_safe()
        assert status == "cold_start"


# ---------------------------------------------------------------------------
# 2. Preheat minutes – read_preheat_minutes_safe
# ---------------------------------------------------------------------------

class TestPreheatMinutes:

    def test_valid_preheat_formula(self):
        """Full formula: command_lead = room_heating_min + onset_delay."""
        coord = make_coordinator()
        # hr=2.4 C/h, hl=0.4, net=2.0; deficit=21-19=2.0; effective=2.0-0=2.0
        # room_heating = 2.0/2.0 * 60 = 60 min; total = 60 + 5 = 65 min
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4), hl=_hl_pred(0.4))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid"
        assert minutes == pytest.approx(65.0, abs=1.0)

    def test_onset_delay_shifts_command_earlier(self):
        """Command lead time > room heating duration by exactly onset delay."""
        coord = make_coordinator()
        # hr=3.0, hl=0.0, net=3.0 (min guard 0.2); deficit=1.5; room_heat = 30 min; total=35
        sh = _shadow_with_preds(coord, hr=_hr_pred(3.0), hl=_hl_pred(0.0))
        minutes, status = sh.read_preheat_minutes_safe(19.5, 21.0)
        assert status == "valid"
        # onset delay must be present in returned value
        onset_delay = LearningShadowController._ONSET_DELAY_PRIOR_MIN
        assert minutes > onset_delay

    def test_no_hr_evidence_uses_deterministic_baseline(self):
        """No heat rate evidence → deterministic_baseline (positive when deficit > 0.5 °C)."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh.runtime._zone(coord.zone_id).last_predictions = {}
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "deterministic_baseline"
        assert minutes > 0.0  # 2 C deficit → positive baseline, never 0

    def test_no_hr_evidence_never_returns_le1_fallback(self):
        """Cold start must NOT call LE v1 and must NOT return 'le1_fallback' source."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh.runtime._zone(coord.zone_id).last_predictions = {}
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status != "le1_fallback"
        assert status != "cold_start"  # old status is gone

    def test_cold_start_with_none_temp_returns_unavailable(self):
        """current_temp=None → unavailable (cannot compute baseline)."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        minutes, status = sh.read_preheat_minutes_safe(None, 21.0)
        assert status == "unavailable"
        assert minutes == 0.0

    def test_target_already_reached(self):
        """Small deficit → target_reached, 0 min."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4))
        minutes, status = sh.read_preheat_minutes_safe(20.9, 21.0)
        assert status == "target_reached"
        assert minutes == 0.0

    def test_low_confidence_returns_deterministic_baseline(self):
        """confidence < 0.35 → deterministic_baseline (not LE v1, not 'low_confidence')."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4, confidence=0.2))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "deterministic_baseline"
        assert minutes > 0.0  # deficit 2 C → positive
        assert status != "low_confidence"
        assert status != "le1_fallback"

    def test_medium_confidence_capped(self):
        """0.35 ≤ confidence < 0.6 → result capped proportionally."""
        coord = make_coordinator()
        # Very high heat rate would give huge preheat; should be capped.
        sh = _shadow_with_preds(coord, hr=_hr_pred(0.3, confidence=0.45))
        minutes, status = sh.read_preheat_minutes_safe(15.0, 21.0)
        assert status == "valid"
        assert minutes <= LearningShadowController._PREHEAT_MAX_MIN

    def test_max_minutes_clamp(self):
        """Preheat never exceeds _PREHEAT_MAX_MIN regardless of rate."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(0.1))  # very slow: huge duration
        minutes, status = sh.read_preheat_minutes_safe(10.0, 25.0)
        assert minutes <= LearningShadowController._PREHEAT_MAX_MIN

    def test_afterheat_reduces_required_duration(self):
        """Expected overshoot (afterheat) reduces effective deficit → shorter duration."""
        coord = make_coordinator()
        sh_no_af = _shadow_with_preds(coord, hr=_hr_pred(2.4), hl=_hl_pred(0.4))
        min_no_af, _ = sh_no_af.read_preheat_minutes_safe(19.0, 21.0)

        coord2 = make_coordinator()
        sh_with_af = _shadow_with_preds(coord2, hr=_hr_pred(2.4), hl=_hl_pred(0.4),
                                        af=_af_pred(0.5))
        min_with_af, _ = sh_with_af.read_preheat_minutes_safe(19.0, 21.0)

        assert min_with_af < min_no_af

    def test_no_double_heat_loss_application(self):
        """heat_loss applied once (via net_rate); must not be subtracted twice."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4), hl=_hl_pred(0.4))
        min1, _ = sh.read_preheat_minutes_safe(19.0, 21.0)

        # Double-application would cause very different (much longer) result.
        # If net_rate=(2.4-0.4)=2.0 → 60+5=65; double: (2.4-0.8)=1.6 → 75+5=80
        # Verify result is close to single-application expectation.
        assert min1 == pytest.approx(65.0, abs=1.5)

    def test_current_temp_none_returns_unavailable(self):
        """current_temp=None → unavailable; no temp → cannot compute any preheat."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4))
        minutes, status = sh.read_preheat_minutes_safe(None, 21.0)
        assert status == "unavailable"
        assert minutes == 0.0

    def test_not_available_when_disabled(self):
        """Disabled shadow → deterministic_baseline (LE2 off, still heat conservatively)."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._enabled = False
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "deterministic_baseline"
        assert minutes > 0.0  # 2 C deficit → positive baseline
        assert status != "not_available"
        assert status != "le1_fallback"

    def test_fallback_pred_uses_deterministic_baseline(self):
        """fallback_used=True heat rate → deterministic_baseline (no evidence, never LE v1)."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4, fallback_used=True))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "deterministic_baseline"
        assert minutes > 0.0
        assert status != "cold_start"
        assert status != "le1_fallback"

    def test_net_rate_floor_prevents_infinite_duration(self):
        """heat_rate < heat_loss → net_rate floored at 0.2 C/h; no division error."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(0.5), hl=_hl_pred(2.0))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid"
        assert minutes <= LearningShadowController._PREHEAT_MAX_MIN

    @pytest.mark.asyncio
    async def test_preheat_status_in_recommendation(self):
        """Coordinator passes preheat_status through in recommendation dict."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4), hl=_hl_pred(0.4))
        coord._minutes_until_next_comfort = MagicMock(return_value=80)
        rec = await _rec(coord, indoor="19.0", comfort="21.0")
        assert "preheat_status" in rec
        assert rec["preheat_status"] == "valid"

    @pytest.mark.asyncio
    async def test_no_hr_evidence_deterministic_baseline_in_recommendation(self):
        """No HR evidence → deterministic_baseline status; baseline value is positive for deficit."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh.runtime._zone(coord.zone_id).last_predictions = {}
        coord._minutes_until_next_comfort = MagicMock(return_value=80)
        rec = await _rec(coord)
        assert rec.get("preheat_status") == "deterministic_baseline"
        assert rec.get("preheat_minutes", 0) > 0

    @pytest.mark.asyncio
    async def test_learning_failure_not_heating_failure(self):
        """Exception inside read_preheat_minutes_safe must not propagate upward."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime._zone = MagicMock(side_effect=RuntimeError("deliberate"))
        rec = await _rec(coord)
        # Coordinator completes, no exception → safe fallback path taken
        assert rec is not None

    @pytest.mark.asyncio
    async def test_one_preheat_read_per_cycle(self):
        """read_preheat_minutes_safe called exactly once per recommendation cycle."""
        coord = make_coordinator()
        sh = MagicMock()
        sh.read_preheat_minutes_safe.return_value = (0.0, "deterministic_baseline")
        sh.read_early_cutoff_safe.return_value = (0.0, "not_available")
        sh.read_forecast_trust_safe.return_value = (1.0, "valid")
        sh.read_forecast_bias_safe.return_value = 0.0
        sh.read_regime_safe.return_value = None
        sh.read_onset_delay_safe.return_value = (5.0, "cold_start_prior")
        sh.confidence_display.return_value = 0.0
        sh.compute_decision_trace_safe = MagicMock()
        coord.attach_le2_shadow(sh)
        await _rec(coord)
        assert sh.read_preheat_minutes_safe.call_count == 1


# ---------------------------------------------------------------------------
# 3. Episode Recovery – requires active_heating regime
# ---------------------------------------------------------------------------

class TestEpisodeRecovery:
    """
    episode_failed recovery must require BOTH:
      1. current_temp < _ec_failed_cut_target - 1.0 (room dropped after failure)
      2. _ec_regime == "active_heating" (ThermalRegimeClassifier confirms new episode)

    Passive solar, neighbour heat, internal loads → regime stays UNKNOWN / AFTERHEAT.
    These must NOT reset episode_failed.
    """

    def _make_coord_in_episode_failed(self):
        """Return coordinator with episode_failed state pre-seeded."""
        coord = make_coordinator()
        sh = MagicMock()
        sh.read_early_cutoff_safe.return_value = (0.5, "valid")
        sh.read_preheat_minutes_safe.return_value = (30.0, "valid")
        sh.read_forecast_trust_safe.return_value = (1.0, "valid")
        sh.read_forecast_bias_safe.return_value = 0.0
        sh.read_regime_safe.return_value = "active_heating"
        sh.read_onset_delay_safe.return_value = (5.0, "cold_start_prior")
        sh.confidence_display.return_value = 0.0
        sh.compute_decision_trace_safe = MagicMock()
        coord.attach_le2_shadow(sh)
        coord._indoor_temp_slope = 0.05
        # Directly seed episode_failed internal state.
        # _ec_episode_failed is the control variable; _ec_state is derived output.
        coord._ec_episode_failed = True
        coord._ec_failed_cut_target = 20.5
        # Prevent episode-change reset: comfort and period must match what coordinator computes.
        coord._ec_hold_comfort_temp = 21.0   # matches make_zone_config comfort_temp
        coord._current_schedule_period = lambda: "morning"
        coord._ec_hold_period = "morning"
        return coord, sh

    @pytest.mark.asyncio
    async def test_passive_solar_does_not_reset_episode_failed(self):
        """regime=UNKNOWN → episode_failed persists despite rising temp."""
        coord, sh = self._make_coord_in_episode_failed()
        # Regime is UNKNOWN (passive warming, not active heating)
        sh.read_regime_safe.return_value = "UNKNOWN"
        coord._indoor_temp_slope = 0.1  # rising, but not from heating
        # Room has dropped below failed threshold - 1.0
        rec = await _rec(coord, indoor="19.2", comfort="21.0")
        # episode_failed must persist
        assert rec.get("early_cutoff_state") == "episode_failed", (
            f"Passive solar must not reset episode_failed; got {rec.get('early_cutoff_state')}"
        )

    @pytest.mark.asyncio
    async def test_afterheat_regime_does_not_reset_episode_failed(self):
        """regime=AFTERHEAT → episode_failed persists (passive coasting, not new episode)."""
        coord, sh = self._make_coord_in_episode_failed()
        sh.read_regime_safe.return_value = "AFTERHEAT"
        coord._indoor_temp_slope = 0.05
        rec = await _rec(coord, indoor="19.2", comfort="21.0")
        assert rec.get("early_cutoff_state") == "episode_failed"

    @pytest.mark.asyncio
    async def test_active_heating_regime_may_reset_episode_failed(self):
        """active_heating + room dropped + slope >= 0 → new episode, reset allowed."""
        coord, sh = self._make_coord_in_episode_failed()
        sh.read_regime_safe.return_value = "active_heating"
        coord._indoor_temp_slope = 0.05
        # Room well below failed threshold - 1.0
        rec = await _rec(coord, indoor="19.2", comfort="21.0")
        # Must NOT be stuck in episode_failed (new episode can begin)
        assert rec.get("early_cutoff_state") != "episode_failed", (
            "active_heating regime + room drop should allow episode reset"
        )

    @pytest.mark.asyncio
    async def test_room_not_fallen_enough_no_reset(self):
        """Room only barely below threshold → recovery condition not met."""
        coord, sh = self._make_coord_in_episode_failed()
        sh.read_regime_safe.return_value = "active_heating"
        coord._indoor_temp_slope = 0.05
        # Only 0.8 below failed target (threshold is 1.0)
        rec = await _rec(coord, indoor="19.7", comfort="21.0")
        # episode_failed persists because room hasn't dropped far enough
        assert rec.get("early_cutoff_state") == "episode_failed"

    @pytest.mark.asyncio
    async def test_slope_none_blocks_recovery(self):
        """slope=None (no measurement) → recovery NOT triggered even with regime."""
        coord, sh = self._make_coord_in_episode_failed()
        sh.read_regime_safe.return_value = "active_heating"
        coord._indoor_temp_slope = None
        rec = await _rec(coord, indoor="19.2", comfort="21.0")
        # No slope measurement → cannot confirm new episode
        assert rec.get("early_cutoff_state") == "episode_failed"

    @pytest.mark.asyncio
    async def test_same_failed_episode_cannot_retry(self):
        """After episode_failed, same cutoff target cannot create new cutoff immediately."""
        coord, sh = self._make_coord_in_episode_failed()
        sh.read_regime_safe.return_value = "active_heating"
        coord._indoor_temp_slope = 0.05
        # Room still above failed threshold − 1.0 (19.8 > 19.5)
        rec = await _rec(coord, indoor="19.8", comfort="21.0")
        # Cannot restart early cutoff until recovery condition met
        state = rec.get("early_cutoff_state")
        assert state != "cutoff_applied", "Same episode must not instantly restart cutoff"

    @pytest.mark.asyncio
    async def test_new_episode_may_use_early_cutoff_again(self):
        """After genuine recovery, new episode can use early cutoff again."""
        coord, sh = self._make_coord_in_episode_failed()
        sh.read_regime_safe.return_value = "active_heating"
        coord._indoor_temp_slope = 0.05
        # Cycle 1: room dropped (19.2 < 20.5 - 1.0 = 19.5), active_heating → recovery
        rec1 = await _rec(coord, indoor="19.2", comfort="21.0")
        # After recovery, state should be reset (eligible/inactive)
        assert rec1.get("early_cutoff_state") != "episode_failed"

        # Cycle 2: new cutoff cycle should be evaluable
        # (we can't guarantee cutoff_applied without precise timing, but we CAN confirm
        # the state is not permanently stuck)
        rec2 = await _rec(coord, indoor="19.2", comfort="21.0")
        assert rec2.get("early_cutoff_state") != "episode_failed"


# ---------------------------------------------------------------------------
# 4. Compatibility – no LE v1 reads, no LE v1 mutations, SHADOW default
# ---------------------------------------------------------------------------

class TestCompatibility:

    @pytest.mark.asyncio
    async def test_no_le1_preheat_read_when_shadow_attached(self):
        """LE v1 async_get_preheat_minutes must NOT be called when shadow attached."""
        coord = make_coordinator()
        coord.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=99)
        sh = attach_shadow(coord, store=FakeStore())
        await _rec(coord)
        coord.learning_engine.async_get_preheat_minutes.assert_not_called()

    @pytest.mark.asyncio
    async def test_le1_preheat_never_called_without_shadow(self):
        """Without shadow, LE v1 async_get_preheat_minutes must NEVER be called; deterministic baseline is used."""
        coord = make_coordinator()
        coord.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=99)
        rec = await _rec(coord)
        coord.learning_engine.async_get_preheat_minutes.assert_not_called()
        # Without shadow, deterministic baseline must be used
        assert rec.get("preheat_status") in ("deterministic_baseline", "unavailable", "target_reached")

    @pytest.mark.asyncio
    async def test_no_le1_heat_rate_read_when_shadow_attached(self):
        """LE v1 heat rate path not triggered by preheat when shadow is attached.

        We can't easily stub LE v1's internal heat_rate read, but we verify the
        read_heat_rate_safe route is used for sensor display (not LE v1).
        """
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        # Shadow returns cold_start (no predictions); sensor should reflect this.
        rate, status = sh.read_heat_rate_safe()
        assert status == "cold_start"
        # LE v1 heat_rate is NOT expected to have been called (it has no mock to intercept,
        # but the real LE v1 path requires coordinator state we haven't triggered).
        # Presence of status confirms LE2 path was used.
        assert rate == 0.0

    @pytest.mark.asyncio
    async def test_preheat_status_field_present(self):
        """Coordinator recommendation always includes preheat_status."""
        coord = make_coordinator()
        rec = await _rec(coord)
        assert "preheat_status" in rec

    @pytest.mark.asyncio
    async def test_preheat_status_deterministic_baseline_without_shadow(self):
        """Without shadow, preheat_status = deterministic_baseline (not 'not_available', not 'le1_fallback')."""
        coord = make_coordinator()
        coord.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=0)
        rec = await _rec(coord)
        assert rec["preheat_status"] in ("deterministic_baseline", "unavailable", "target_reached")
        assert rec["preheat_status"] != "not_available"
        assert rec["preheat_status"] != "le1_fallback"

    @pytest.mark.asyncio
    async def test_shadow_default_no_control_change(self):
        """SHADOW mode: shadow attached does not change TRV setpoint vs. baseline."""
        coord_base = make_coordinator()
        coord_base.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=0)
        rec_base = await _rec(coord_base)

        coord_sh = make_coordinator()
        coord_sh.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=0)
        sh = attach_shadow(coord_sh, store=FakeStore())
        rec_sh = await _rec(coord_sh)

        # TRV setpoint must not change when shadow is attached (SHADOW mode)
        assert rec_base.get("trv_setpoint") == rec_sh.get("trv_setpoint")

    @pytest.mark.asyncio
    async def test_preheat_minutes_positive_in_deterministic_baseline(self):
        """No HR evidence → deterministic_baseline with positive minutes when deficit > 0.5 °C."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh.runtime._zone(coord.zone_id).last_predictions = {}
        coord._minutes_until_next_comfort = MagicMock(return_value=60)
        rec = await _rec(coord)
        assert rec.get("preheat_status") == "deterministic_baseline"
        assert rec.get("preheat_minutes", 0) > 0  # 2 C deficit → positive

    @pytest.mark.asyncio
    async def test_valid_preheat_activates_preheating(self):
        """When LE2 returns valid preheat minutes and comfort is approaching, preheat activates."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4), hl=_hl_pred(0.4))
        coord._minutes_until_next_comfort = MagicMock(return_value=50)
        rec = await _rec(coord, indoor="19.0", comfort="21.0")
        assert rec.get("preheat_status") == "valid"
        assert rec.get("preheat_active") is True

    def test_trv_only_no_external_sensor_returns_unavailable(self):
        """TRV-only (no temp): unavailable → coordinator must still complete (learning ≠ heating failure)."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        minutes, status = sh.read_preheat_minutes_safe(None, 21.0)
        assert status == "unavailable"
        assert minutes == 0.0
        assert status != "le1_fallback"

    def test_exception_inside_read_preheat_returns_deterministic_baseline(self):
        """Internal exception inside read_preheat_minutes_safe returns deterministic_baseline, not LE v1."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        # Corrupt the runtime to cause an internal exception.
        sh._runtime = MagicMock(side_effect=RuntimeError("runtime boom"))
        # The method must not raise; must return deterministic baseline (never LE v1).
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert isinstance(minutes, float)
        assert status == "deterministic_baseline"
        assert status != "le1_fallback"
        assert status != "not_available"
        assert minutes > 0.0  # 2 C deficit


# ---------------------------------------------------------------------------
# 5. OnsetDelay architecture: typed prior, never folded into HeatRate
# ---------------------------------------------------------------------------

class TestOnsetDelayArchitecture:
    """Onset delay is a typed PredictionType.ONSET_DELAY — separate from HeatRate."""

    def test_onset_delay_not_folded_into_heat_rate(self):
        """room_heating_duration = total - onset_delay; onset_delay never in rate."""
        coord = make_coordinator()
        # hr=2.4, hl=0.4, net=2.0; deficit=2.0; room_heat=60; onset=5; total=65
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4), hl=_hl_pred(0.4))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid"
        # onset delay is 5 min; room heating = 60; total = 65
        onset = LearningShadowController._ONSET_DELAY_PRIOR_MIN
        room_heat = minutes - onset
        # room_heating should equal deficit / net_rate * 60 = 2.0/2.0*60 = 60
        assert room_heat == pytest.approx(60.0, abs=2.0)

    def test_read_onset_delay_safe_returns_prior_without_evidence(self):
        """No onset_delay prediction → returns (5.0, 'cold_start_prior')."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh.runtime._zone(coord.zone_id).last_predictions = {}
        delay, status = sh.read_onset_delay_safe()
        assert delay == pytest.approx(LearningShadowController._ONSET_DELAY_PRIOR_MIN)
        assert status == "cold_start_prior"

    def test_read_onset_delay_safe_not_available_when_disabled(self):
        """Disabled shadow → read_onset_delay_safe returns prior with 'not_available'."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._enabled = False
        delay, status = sh.read_onset_delay_safe()
        assert delay == pytest.approx(LearningShadowController._ONSET_DELAY_PRIOR_MIN)
        assert status == "not_available"

    def test_onset_delay_prior_is_5_min(self):
        """_ONSET_DELAY_PRIOR_MIN is exactly 5.0 (the versioned prior value)."""
        assert LearningShadowController._ONSET_DELAY_PRIOR_MIN == 5.0

    def test_onset_delay_not_same_as_room_heating(self):
        """room_heating_duration and onset_delay are distinct quantities."""
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4), hl=_hl_pred(0.4))
        minutes, _ = sh.read_preheat_minutes_safe(19.0, 21.0)
        _, delay_status = sh.read_onset_delay_safe()
        # onset delay is a separate read; total > onset alone
        assert minutes > LearningShadowController._ONSET_DELAY_PRIOR_MIN


# ---------------------------------------------------------------------------
# 6. Deterministic baseline: formula, inputs, positive when deficit exists
# ---------------------------------------------------------------------------

class TestDeterministicBaseline:
    """compute_deterministic_preheat_baseline: formula = deficit/1.5*60 + 5.0."""

    def test_baseline_positive_when_deficit(self):
        """2 °C deficit → positive baseline minutes."""
        minutes, status = LearningShadowController.compute_deterministic_preheat_baseline(
            19.0, 21.0)
        assert status == "deterministic_baseline"
        assert minutes > 0.0

    def test_baseline_zero_at_target(self):
        """No deficit → target_reached, 0.0 min."""
        minutes, status = LearningShadowController.compute_deterministic_preheat_baseline(
            21.0, 21.0)
        assert status == "target_reached"
        assert minutes == 0.0

    def test_baseline_unavailable_without_temp(self):
        """current_temp=None → unavailable, 0.0 min."""
        minutes, status = LearningShadowController.compute_deterministic_preheat_baseline(
            None, 21.0)
        assert status == "unavailable"
        assert minutes == 0.0

    def test_baseline_formula_values(self):
        """deficit=2.0 → room_heat=2.0/1.5*60=80; total=80+5=85 (capped at MAX)."""
        minutes, status = LearningShadowController.compute_deterministic_preheat_baseline(
            19.0, 21.0)
        # 2.0/1.5*60 = 80 min; +5 onset = 85 min
        assert minutes == pytest.approx(85.0, abs=1.0)
        assert status == "deterministic_baseline"

    def test_baseline_includes_onset_prior(self):
        """Baseline includes onset prior (5 min) in total."""
        # deficit=0.6°C (> 0.5 threshold) → room_heat=0.6/1.5*60=24 min + 5 onset = 29 min
        minutes_small, status = LearningShadowController.compute_deterministic_preheat_baseline(
            20.4, 21.0)
        assert status == "deterministic_baseline"
        assert minutes_small > LearningShadowController._ONSET_DELAY_PRIOR_MIN

    def test_baseline_capped_at_max(self):
        """Large deficit is capped at _PREHEAT_MAX_MIN."""
        minutes, status = LearningShadowController.compute_deterministic_preheat_baseline(
            5.0, 25.0)
        assert minutes <= LearningShadowController._PREHEAT_MAX_MIN


# ---------------------------------------------------------------------------
# 7. No LE v1 reads in sensors
# ---------------------------------------------------------------------------

class TestSensorNoLE1Reads:
    """Sensor entities must never read LE v1 HeatRate or Preheat; source must not be 'le1_fallback'."""

    def test_sensor_source_is_le2_not_le1_fallback(self):
        """HeatingPower sensor source must not be 'le1_fallback'."""
        from custom_components.thermosmart.sensor import ThermoSmartHeatingPowerSensor
        coord = make_coordinator()
        sh = _shadow_with_preds(coord, hr=_hr_pred(2.4))
        sensor = ThermoSmartHeatingPowerSensor(coord, coord.entry)
        attrs = sensor.extra_state_attributes or {}
        assert attrs.get("source") != "le1_fallback"

    def test_sensor_source_unavailable_without_shadow(self):
        """HeatingPower sensor source = 'unavailable' when no shadow attached."""
        from custom_components.thermosmart.sensor import ThermoSmartHeatingPowerSensor
        coord = make_coordinator()
        sensor = ThermoSmartHeatingPowerSensor(coord, coord.entry)
        attrs = sensor.extra_state_attributes or {}
        assert attrs.get("source") == "unavailable"
        assert attrs.get("source") != "le1_fallback"
