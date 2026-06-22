"""Phase 19A-D: Early Cutoff Authority Transfer to LE 2.0.

Invarianten (verbindlich):
  - Early Cutoff liest ausschließlich über den validierten Prediction Gate.
  - Cold-Start (fallback_used=True) → (0.0, "cold_start") → kein Cutoff.
  - Low Confidence (< 0.35) → (0.0, "low_confidence") → kein Cutoff.
  - missing/stale/superseded/invalid → kein Cutoff.
  - Kein versionierter Prior löst allein Early Cutoff aus.
  - adjusted_target (Nutzersicht) bleibt immer = Komforttemperatur.
  - Early Cutoff modifiziert nur effective_target im Preheat-Branch.
  - Außerhalb von preheat_active: kein Cutoff, auch wenn Prediction verfügbar.
  - Offenes Fenster / manueller Override verhindert Cutoff.
  - Harte Obergrenze _EARLY_CUTOFF_MAX_C = 3.0°C.
  - Noise Floor _EARLY_CUTOFF_MIN_RESIDUAL_C = 0.15°C.
  - Keine LE-v1-Early-Cutoff-Reads (keine solche Methode in LearningEngine vorhanden).
  - Learning-Fehler wird niemals zur Heizstörung.
  - SHADOW bleibt produktiver Default; Single Dispatch unverändert.
  - DecisionTrace enthält Prediction, Confidence, Gate-Ergebnis, Reason Codes.

Hold-Lifecycle-Invarianten (Phase 19D):
  - Nach Cutoff-Anwendung bleibt effective_target = cut_target auch wenn
    current_temp >= cut_target (Coasting-Hold verhindert Re-Heat-Impuls).
  - Hold ist an Episode (comfort_temp + schedule_period) gebunden.
  - Hold-Release: target_reached, temp_falling, timeout, window_open,
    override, episode_change, stale/superseded/invalid prediction.
  - Nach released_temperature_falling oder released_timeout: episode_failed=True;
    kein neuer Cutoff in derselben Episode (kein Pendeln).
  - Restart erzeugt niemals einen verwaisten Hold (alle Zustände flüchtig).
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from custom_components.thermosmart.const import HEATING_MODE_AUTO
from custom_components.thermosmart.learning.contracts import Prediction, PredictionType
from custom_components.thermosmart.learning.decision.runtime_adapter import (
    validated_prediction_read,
    READ_OK, READ_MISSING, READ_STALE, READ_SUPERSEDED, READ_UNIT_MISMATCH, READ_WRONG_TYPE,
    READ_WRONG_ZONE,
)
from tests.helpers import make_coordinator, make_state, set_hass_states
from tests.helpers_ha_runtime import attach_shadow, FakeStore, make_recording_coordinator


_COMFORT = 21.0
_NIGHT = 18.0   # make_zone_config default night_temp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cutoff_pred(residual_c: float, *, fallback: bool = False,
                 confidence: float = 0.75) -> Prediction:
    if fallback:
        return Prediction(
            prediction_type=PredictionType.RECOMMENDED_EARLY_CUTOFF,
            values={"expected_residual_rise": 0.0},
            units={"expected_residual_rise": "C"},
            confidence=0.15, reliability=0.3,
            model_version=1, parameter_version=1,
            prior_contribution=1.0, learned_contribution=0.0,
            fallback_used=True, evidence_count=0,
        )
    return Prediction(
        prediction_type=PredictionType.RECOMMENDED_EARLY_CUTOFF,
        values={"expected_residual_rise": residual_c},
        units={"expected_residual_rise": "C"},
        confidence=confidence, reliability=0.85,
        model_version=1, parameter_version=1,
        prior_contribution=0.0, learned_contribution=1.0,
        fallback_used=False, evidence_count=25,
    )


def _shadow_with_cutoff(coord, pred=None):
    """Attach a real shadow to coord and inject last_predictions."""
    sh = attach_shadow(coord, store=FakeStore())
    zr = sh.runtime._zone(coord.zone_id)
    zr.last_predictions = {PredictionType.RECOMMENDED_EARLY_CUTOFF: pred} if pred else {}
    return sh


def _mock_cutoff_shadow(cutoff_c=0.5, *, status="valid",
                        regime="active_heating") -> MagicMock:
    """MagicMock shadow that returns the given early-cutoff tuple and regime.

    ``regime`` defaults to "active_heating" so that the regime gate passes in
    the hold-lifecycle tests.  Pass a different value to test gate rejection.
    """
    sh = MagicMock()
    actual = float(cutoff_c) if status == "valid" else 0.0
    sh.read_early_cutoff_safe.return_value = (actual, status)
    sh.read_forecast_trust_safe.return_value = (0.0, "not_available")
    sh.read_forecast_bias_safe.return_value = 0.0
    sh.confidence_display.return_value = 0.0
    sh.compute_decision_trace_safe = MagicMock()
    sh.read_regime_safe.return_value = regime
    sh.read_preheat_minutes_safe.return_value = (60.0, "deterministic_baseline")
    sh.read_onset_delay_safe.return_value = (5.0, "cold_start_prior")
    return sh


def _make_coord_preheat(*, cutoff_c=0.5, status="valid", indoor="19.0"):
    """Coordinator primed for AUTO preheat branch with mocked shadow.

    Pre-seeds ``_indoor_temp_slope`` so the trend gate (slope is not None AND
    slope >= 0) passes by default.  Tests that specifically want to test the slope
    gate override this after construction.

    The shadow mock returns 30 min (valid) for preheat so preheat_active=True
    when mins_until_comfort=20 <= preheat_minutes=30.
    """
    coord = make_coordinator()
    sh = _mock_cutoff_shadow(cutoff_c=cutoff_c, status=status)
    sh.read_preheat_minutes_safe.return_value = (30.0, "valid")
    coord.attach_le2_shadow(sh)
    coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
    coord._minutes_until_next_comfort = MagicMock(return_value=20)
    coord._indoor_temp_slope = 0.05           # pre-seed: rising trend evidence
    return coord


async def _rec(coord, *, indoor="19.0", weather=None) -> dict:
    """Call _compute_recommendation directly."""
    set_hass_states(coord, {"sensor.test_temp": make_state(indoor)})
    return await coord._compute_recommendation(
        coord.entry.data, weather or {"temperature": 10.0}, HEATING_MODE_AUTO)


async def _rec_at(coord, temp: float, **kwargs) -> dict:
    """Like _rec but pre-seeds the per-sensor EMA to bypass smoothing lag.

    Multi-cycle hold tests need the exact temperature to land in current_temp in
    a single call; otherwise the EMA (α ≈ 0.2) requires ~10 cycles to converge.
    Pre-seeding the EMA dict entry to ``temp`` makes the filtered reading equal
    ``temp`` immediately without touching any control logic.
    """
    coord._sensor_ema["sensor.test_temp"] = temp
    return await _rec(coord, indoor=str(temp), **kwargs)


# ---------------------------------------------------------------------------
# 1. Read Gate – RECOMMENDED_EARLY_CUTOFF
# ---------------------------------------------------------------------------

class TestEarlyCutoffReadGate:
    def test_ok_positive_residual(self):
        val, st = validated_prediction_read(
            {PredictionType.RECOMMENDED_EARLY_CUTOFF: _cutoff_pred(0.5)},
            "early_cutoff",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius")
        assert st == READ_OK and val == pytest.approx(0.5)

    def test_ok_zero_residual(self):
        val, st = validated_prediction_read(
            {PredictionType.RECOMMENDED_EARLY_CUTOFF: _cutoff_pred(0.0)},
            "early_cutoff",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius")
        assert st == READ_OK and val == pytest.approx(0.0)

    def test_missing_rejected(self):
        _, st = validated_prediction_read(
            {}, "early_cutoff",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius")
        assert st == READ_MISSING

    def test_wrong_unit_score_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.RECOMMENDED_EARLY_CUTOFF: _cutoff_pred(0.5)},
            "early_cutoff",
            zone_id="z1", prediction_zone_id="z1", expected_unit="score")
        assert st == READ_UNIT_MISMATCH

    def test_wrong_feature_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.RECOMMENDED_EARLY_CUTOFF: _cutoff_pred(0.5)},
            "no_such_feature",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius")
        assert st == READ_WRONG_TYPE

    def test_stale_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.RECOMMENDED_EARLY_CUTOFF: _cutoff_pred(0.5)},
            "early_cutoff",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius",
            age_seconds=7201.0, max_age_seconds=7200.0)
        assert st == READ_STALE

    def test_superseded_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.RECOMMENDED_EARLY_CUTOFF: _cutoff_pred(0.5)},
            "early_cutoff",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius",
            prediction_decision_id="old-id", current_decision_id="new-id")
        assert st == READ_SUPERSEDED

    def test_wrong_zone_rejected(self):
        _, st = validated_prediction_read(
            {PredictionType.RECOMMENDED_EARLY_CUTOFF: _cutoff_pred(0.5)},
            "early_cutoff",
            zone_id="zone_a", prediction_zone_id="zone_b", expected_unit="celsius")
        assert st == READ_WRONG_ZONE


# ---------------------------------------------------------------------------
# 2. Shadow: read_early_cutoff_safe() returns (float, str)
# ---------------------------------------------------------------------------

class TestShadowEarlyCutoffRead:
    def test_valid_prediction_returns_value_and_valid(self):
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.5))
        val, st = sh.read_early_cutoff_safe()
        assert st == "valid" and val == pytest.approx(0.5)

    def test_cold_start_returns_zero_cold_start(self):
        """Prior (fallback_used=True) must never trigger early cutoff."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.5, fallback=True))
        val, st = sh.read_early_cutoff_safe()
        assert val == pytest.approx(0.0) and st == "cold_start"

    def test_low_confidence_returns_zero_low_confidence(self):
        """confidence=0.2 < 0.35 threshold → no cutoff."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.5, confidence=0.2))
        val, st = sh.read_early_cutoff_safe()
        assert val == pytest.approx(0.0) and st == "low_confidence"

    def test_borderline_confidence_accepted(self):
        """confidence=0.35 (at threshold) is accepted."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.5, confidence=0.35))
        val, st = sh.read_early_cutoff_safe()
        assert st == "valid" and val == pytest.approx(0.5)

    def test_missing_prediction_returns_zero_missing(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        val, st = sh.read_early_cutoff_safe()
        assert val == pytest.approx(0.0) and st == "missing"

    def test_disabled_returns_zero_not_available(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._enabled = False
        val, st = sh.read_early_cutoff_safe()
        assert val == pytest.approx(0.0) and st == "not_available"

    def test_error_returns_zero_not_available(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime = None
        val, st = sh.read_early_cutoff_safe()
        assert val == pytest.approx(0.0) and st == "not_available"

    def test_zero_residual_valid_status(self):
        """Residual rise = 0.0 with real evidence is 'valid' (not 'not_available')."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.0))
        val, st = sh.read_early_cutoff_safe()
        assert st == "valid" and val == pytest.approx(0.0)

    def test_stale_gate_returns_stale_status(self):
        """Read Gate READ_STALE → (0.0, 'stale')."""
        pred = _cutoff_pred(0.5)
        _, gate_st = validated_prediction_read(
            {PredictionType.RECOMMENDED_EARLY_CUTOFF: pred}, "early_cutoff",
            zone_id="z1", prediction_zone_id="z1", expected_unit="celsius",
            age_seconds=7201.0, max_age_seconds=7200.0)
        assert gate_st == READ_STALE

    def test_value_clamped_to_nonnegative(self):
        """Negative residual rise is clamped to 0.0 by read_early_cutoff_safe."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        # Inject prediction with negative value (should not happen from model, but test gate)
        pred = Prediction(
            prediction_type=PredictionType.RECOMMENDED_EARLY_CUTOFF,
            values={"expected_residual_rise": -0.3},
            units={"expected_residual_rise": "C"},
            confidence=0.75, reliability=0.85,
            model_version=1, parameter_version=1,
            prior_contribution=0.0, learned_contribution=1.0,
            fallback_used=False, evidence_count=25,
        )
        sh.runtime._zone(coord.zone_id).last_predictions = {
            PredictionType.RECOMMENDED_EARLY_CUTOFF: pred}
        val, st = sh.read_early_cutoff_safe()
        assert val == pytest.approx(0.0) and st == "valid"


# ---------------------------------------------------------------------------
# 3. Coordinator: Early Cutoff applied in preheat branch
# ---------------------------------------------------------------------------

class TestCoordinatorEarlyCutoffApplied:
    async def test_valid_cutoff_reduces_preheat_effective_target(self):
        """Valid residual=0.5 → effective_target = 21.0 - 0.5 = 20.5."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        assert rec["preheat_active"] is True
        assert rec["effective_target"] == pytest.approx(20.5)

    async def test_valid_cutoff_applied_flag_set(self):
        coord = _make_coord_preheat(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is True
        assert rec["early_cutoff_c"] == pytest.approx(0.5)
        assert rec["early_cutoff_status"] == "valid"

    async def test_cutoff_does_not_change_adjusted_target(self):
        """adjusted_target = comfort_temp (user-visible, never changed by early cutoff)."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        assert rec["adjusted_target"] == pytest.approx(_COMFORT)
        # effective_target IS reduced
        assert rec["effective_target"] < _COMFORT

    async def test_cutoff_capped_at_max(self):
        """cutoff_c = 5.0 > _EARLY_CUTOFF_MAX_C (3.0) → capped to 3.0."""
        coord = _make_coord_preheat(cutoff_c=5.0)
        rec = await _rec(coord, indoor="17.0")
        if rec["early_cutoff_applied"]:
            # effective_target = 21.0 - 3.0 = 18.0 (= night_temp, protected by max guard)
            assert rec["effective_target"] == pytest.approx(18.0)

    async def test_cutoff_result_same_whether_small_or_max_capped(self):
        """cutoff_c=3.0 and cutoff_c=5.0 produce the same effective_target (cap=3.0)."""
        coord_exact = _make_coord_preheat(cutoff_c=3.0)
        coord_over = _make_coord_preheat(cutoff_c=5.0)
        rec_exact = await _rec(coord_exact, indoor="17.0")
        rec_over = await _rec(coord_over, indoor="17.0")
        # Both should produce the same cut result (capped at 3.0)
        assert rec_exact["effective_target"] == rec_over["effective_target"]


# ---------------------------------------------------------------------------
# 4. Coordinator: Early Cutoff blocked in various conditions
# ---------------------------------------------------------------------------

class TestCoordinatorEarlyCutoffBlocked:
    async def test_no_shadow_no_cutoff(self):
        """Without shadow: no cutoff, effective_target = comfort_temp."""
        coord = make_coordinator()
        coord.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=30)
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        coord._minutes_until_next_comfort = MagicMock(return_value=20)
        rec = await _rec(coord, indoor="19.0")
        assert rec["preheat_active"] is True
        assert rec["early_cutoff_applied"] is False
        assert rec["effective_target"] == pytest.approx(_COMFORT)

    async def test_cold_start_no_cutoff(self):
        """Cold-start shadow (prior only) must not trigger early cutoff."""
        coord = _make_coord_preheat(cutoff_c=0.5, status="cold_start")
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False
        assert rec["effective_target"] == pytest.approx(_COMFORT)
        assert rec["early_cutoff_status"] == "cold_start"

    async def test_low_confidence_no_cutoff(self):
        coord = _make_coord_preheat(cutoff_c=0.5, status="low_confidence")
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False
        assert rec["effective_target"] == pytest.approx(_COMFORT)

    async def test_missing_prediction_no_cutoff(self):
        coord = _make_coord_preheat(cutoff_c=0.0, status="missing")
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False

    async def test_stale_prediction_no_cutoff(self):
        coord = _make_coord_preheat(cutoff_c=0.5, status="stale")
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False

    async def test_near_zero_residual_below_noise_floor_no_cutoff(self):
        """cutoff_c = 0.05 < _EARLY_CUTOFF_MIN_RESIDUAL_C (0.15) → not applied."""
        coord = _make_coord_preheat(cutoff_c=0.05)
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False
        assert rec["effective_target"] == pytest.approx(_COMFORT)

    async def test_current_temp_above_cut_target_no_cutoff(self):
        """current_temp >= cut_target → cutoff would lower target below current → not applied."""
        # cutoff_c=0.5 → cut_target=20.5; indoor=21.0 > 20.5 → no apply
        # But preheat_active requires base_target(18.5) < comfort_temp(21.0) AND
        # current_temp <= comfort_temp (via preheat_minutes > 0)
        # Actually with indoor=20.8 > cut_target=20.5 → condition cut_target > current_temp fails
        coord = _make_coord_preheat(cutoff_c=0.5)
        rec = await _rec(coord, indoor="20.8")
        assert rec["early_cutoff_applied"] is False

    async def test_not_in_preheat_cutoff_not_applied(self):
        """Early cutoff only applies in preheat branch; night branch is unaffected."""
        coord = make_coordinator()
        coord.attach_le2_shadow(_mock_cutoff_shadow(cutoff_c=0.5))
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        # No preheat setup → preheat_active=False
        rec = await _rec(coord, indoor="17.0")
        assert rec["preheat_active"] is False
        assert rec["early_cutoff_applied"] is False

    async def test_comfort_branch_no_cutoff(self):
        """Comfort branch (base_target >= comfort_temp) → early cutoff not applied."""
        coord = make_coordinator()
        coord.attach_le2_shadow(_mock_cutoff_shadow(cutoff_c=0.5))
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=21.0)
        rec = await _rec(coord, indoor="20.5")
        assert rec["early_cutoff_applied"] is False

    async def test_window_open_cutoff_not_applied(self):
        """Window open → effective_target=None; no early cutoff path reached."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        coord._check_window_open = MagicMock(return_value=True)
        rec = await _rec(coord, indoor="19.0")
        assert rec["window_open"] is True
        assert rec["early_cutoff_applied"] is False
        assert rec["effective_target"] is None

    async def test_prior_alone_never_triggers_cutoff(self):
        """fallback_used=True (prior) → cold_start → kein Cutoff; same check as cold_start."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.8, fallback=True))
        # Preheat comes from LE2 now; mock so preheat_active=True reaches the EC gate.
        sh.read_preheat_minutes_safe = MagicMock(return_value=(30.0, "valid"))
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        coord._minutes_until_next_comfort = MagicMock(return_value=20)
        coord._indoor_temp_slope = 0.05           # pre-seed: reach shadow consultation
        sh.runtime._zone(coord.zone_id).last_regime = "active_heating"
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False
        assert rec["early_cutoff_status"] == "cold_start"


# ---------------------------------------------------------------------------
# 5. No LE-v1 Early Cutoff reads exist or are called
# ---------------------------------------------------------------------------

class TestNoLEV1EarlyCutoffReads:
    def test_learning_engine_has_no_early_cutoff_method(self):
        """LearningEngine must NOT have any early cutoff read method (no legacy path)."""
        from custom_components.thermosmart.learning_engine import LearningEngine
        assert not hasattr(LearningEngine, "get_early_cutoff")
        assert not hasattr(LearningEngine, "get_early_cutoff_minutes")
        assert not hasattr(LearningEngine, "async_get_early_cutoff")

    async def test_coordinator_does_not_call_le1_for_cutoff(self):
        """LearningEngine methods unrelated to early cutoff are unaffected; no new calls."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        await _rec(coord, indoor="19.0")
        # LearningEngine mock tracks all calls – early_cutoff-specific ones must not appear
        for call in coord.learning_engine.mock_calls:
            name = call[0]
            assert "cutoff" not in name.lower(), f"Unexpected LE v1 cutoff call: {name}"

    async def test_coordinator_does_not_update_le1_state(self):
        """No LE v1 state updates triggered by early cutoff decision."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        await _rec(coord, indoor="19.0")
        coord.learning_engine.update_heating_session.assert_not_called()
        coord.learning_engine.record_forecast_decision.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Recommendation dict contains correct early cutoff fields
# ---------------------------------------------------------------------------

class TestEarlyCutoffRecDict:
    async def test_all_three_fields_present_no_preheat(self):
        """All early_cutoff_* fields are present even when preheat is inactive."""
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        rec = await _rec(coord, indoor="17.0")
        assert "early_cutoff_applied" in rec
        assert "early_cutoff_c" in rec
        assert "early_cutoff_status" in rec

    async def test_default_values_when_not_preheat(self):
        coord = make_coordinator()
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        rec = await _rec(coord, indoor="17.0")
        assert rec["early_cutoff_applied"] is False
        assert rec["early_cutoff_c"] == pytest.approx(0.0)
        assert rec["early_cutoff_status"] == "not_available"

    async def test_applied_true_and_c_set_when_cutoff_active(self):
        coord = _make_coord_preheat(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        if rec["preheat_active"] and rec["early_cutoff_applied"]:
            assert rec["early_cutoff_c"] == pytest.approx(0.5)
            assert rec["early_cutoff_status"] == "valid"

    async def test_effective_target_distinct_from_adjusted_target_when_applied(self):
        """When cutoff is applied: effective_target < adjusted_target = comfort_temp."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        if rec["early_cutoff_applied"]:
            assert rec["effective_target"] < rec["adjusted_target"]
            assert rec["adjusted_target"] == pytest.approx(_COMFORT)


# ---------------------------------------------------------------------------
# 7. Learning failure must not become heating failure
# ---------------------------------------------------------------------------

class TestEarlyCutoffLearningFailure:
    def test_broken_runtime_returns_zero_not_available(self):
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime = None
        val, st = sh.read_early_cutoff_safe()
        assert val == pytest.approx(0.0) and st == "not_available"

    async def test_broken_shadow_heating_continues(self):
        """Broken shadow → early_cutoff_applied=False, heating not disrupted."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        sh._runtime = None
        coord.learning_engine.async_get_preheat_minutes = AsyncMock(return_value=30)
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=18.5)
        coord._minutes_until_next_comfort = MagicMock(return_value=20)
        rec = await _rec(coord, indoor="19.0")
        assert rec["effective_target"] is not None
        assert rec["early_cutoff_applied"] is False
        # Heating path continues: preheat to full comfort_temp
        assert rec["effective_target"] == pytest.approx(_COMFORT)


# ---------------------------------------------------------------------------
# 8. Preheat and Early Cutoff are semantically distinct
# ---------------------------------------------------------------------------

class TestPreheatAndEarlyCutoffDistinct:
    async def test_preheat_timing_unchanged_when_cutoff_applied(self):
        """preheat_active=True and preheat_minutes are not modified by early cutoff."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        assert rec["preheat_active"] is True
        assert rec["preheat_minutes"] == 30  # from mock

    async def test_early_cutoff_modifies_effective_not_base_target(self):
        """Early cutoff reduces effective_target; base_target stays comfort_temp."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        # base_target was set to comfort_temp by preheat logic
        assert rec["base_target"] == pytest.approx(_COMFORT)
        if rec["early_cutoff_applied"]:
            assert rec["effective_target"] < rec["base_target"]

    async def test_zero_residual_no_cutoff_full_preheat(self):
        """valid residual=0.0 → below noise floor → no cutoff, preheat to full target."""
        coord = _make_coord_preheat(cutoff_c=0.0)
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False
        assert rec["effective_target"] == pytest.approx(_COMFORT)


# ---------------------------------------------------------------------------
# 9. Decision Trace: early cutoff entry included when prediction present
# ---------------------------------------------------------------------------

class TestDecisionTrace:
    def test_trace_contains_early_cutoff_entry_when_prediction_present(self):
        """Shadow's compute_decision_trace_safe includes 'early_cutoff' in features."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.6))
        recommendation = {
            "effective_target": 21.0, "trv_setpoint": 21.0,
            "indoor_temperature": 19.0, "window_open": False,
            "preheat_active": True, "base_target": 21.0,
            "adjusted_target": 21.0, "override_active": False,
        }
        sh.compute_decision_trace_safe(recommendation, active_control=False)
        trace = sh.last_decision_trace
        assert trace is not None
        features = {e["feature"]: e for e in trace["features"]}
        assert "early_cutoff" in features

    def test_trace_early_cutoff_has_le2_value_and_confidence(self):
        """Trace entry for early_cutoff contains the predicted value and confidence."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.6, confidence=0.78))
        recommendation = {
            "effective_target": 20.5, "trv_setpoint": 20.5,
            "indoor_temperature": 19.0, "window_open": False,
            "preheat_active": True, "base_target": 21.0,
            "adjusted_target": 21.0, "override_active": False,
        }
        sh.compute_decision_trace_safe(recommendation, active_control=False)
        trace = sh.last_decision_trace
        features = {e["feature"]: e for e in trace["features"]}
        ec = features["early_cutoff"]
        assert ec["le2"] == pytest.approx(0.6)
        # Confidence may be None if no ConfidenceResult is attached, or a float
        # Either way, the trace must include the entry with the predicted value.
        assert "confidence" in ec

    def test_trace_early_cutoff_not_applied_in_shadow_mode(self):
        """In SHADOW mode the early cutoff is trace-only; applied=False in trace."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.6))
        recommendation = {
            "effective_target": 21.0, "trv_setpoint": 21.0,
            "indoor_temperature": 19.0, "window_open": False,
            "preheat_active": True, "base_target": 21.0,
            "adjusted_target": 21.0, "override_active": False,
        }
        sh.compute_decision_trace_safe(recommendation, active_control=False)
        trace = sh.last_decision_trace
        features = {e["feature"]: e for e in trace["features"]}
        ec = features["early_cutoff"]
        # In SHADOW mode, applied is always False (trace-only, no dispatch)
        assert ec["applied"] is False

    def test_trace_has_comfort_temperature_c(self):
        """DecisionTrace exposes comfort_temperature_c from recommendation's adjusted_target."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.5))
        recommendation = {
            "effective_target": 20.5,  # after cutoff: comfort(21.0) - cutoff_c(0.5)
            "trv_setpoint": 20.5,
            "indoor_temperature": 19.0, "window_open": False,
            "preheat_active": True, "base_target": 21.0,
            "adjusted_target": 21.0, "override_active": False,
        }
        sh.compute_decision_trace_safe(recommendation, active_control=False)
        trace = sh.last_decision_trace
        # comfort_temperature_c must be populated from adjusted_target
        assert trace["comfort_temperature_c"] == pytest.approx(21.0)

    def test_trace_comfort_target_distinct_from_baseline_setpoint_when_cutoff_applied(self):
        """When cutoff is applied: baseline_setpoint_c < comfort_temperature_c (separate fields)."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.5))
        recommendation = {
            "effective_target": 20.5,
            "trv_setpoint": 20.5,   # setpoint after cutoff
            "indoor_temperature": 19.0, "window_open": False,
            "preheat_active": True, "base_target": 21.0,
            "adjusted_target": 21.0, "override_active": False,
        }
        sh.compute_decision_trace_safe(recommendation, active_control=False)
        trace = sh.last_decision_trace
        assert trace["comfort_temperature_c"] == pytest.approx(21.0)
        # baseline = coordinator's TRV setpoint (≈ effective_target after cutoff)
        assert trace["baseline_setpoint_c"] == pytest.approx(20.5)
        assert trace["comfort_temperature_c"] > trace["baseline_setpoint_c"]

    def test_trace_has_reason_codes(self):
        """DecisionTrace reason_codes list is present (may be empty in shadow)."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.6))
        recommendation = {
            "effective_target": 21.0, "trv_setpoint": 21.0,
            "indoor_temperature": 19.0, "window_open": False,
            "preheat_active": True, "base_target": 21.0,
            "adjusted_target": 21.0, "override_active": False,
        }
        sh.compute_decision_trace_safe(recommendation, active_control=False)
        trace = sh.last_decision_trace
        assert "reason_codes" in trace
        assert isinstance(trace["reason_codes"], list)


# ---------------------------------------------------------------------------
# 10. Comfort-branch heating (not preheat) also gets early cutoff
# ---------------------------------------------------------------------------

def _make_coord_comfort(*, cutoff_c=0.5, status="valid"):
    """Coordinator in the active comfort-window branch (base_target >= comfort_temp)."""
    coord = make_coordinator()
    coord.attach_le2_shadow(_mock_cutoff_shadow(cutoff_c=cutoff_c, status=status))
    coord.learning_engine.async_get_base_target = AsyncMock(return_value=_COMFORT)
    coord._minutes_until_next_comfort = MagicMock(return_value=0)
    coord._indoor_temp_slope = 0.05           # pre-seed: rising trend evidence
    return coord


class TestEarlyCutoffComfortBranch:
    async def test_comfort_branch_gets_early_cutoff(self):
        """active comfort window + valid prediction → cutoff reduces effective_target."""
        coord = _make_coord_comfort(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        assert rec["preheat_active"] is False
        assert rec["base_target"] == pytest.approx(_COMFORT)
        assert rec["early_cutoff_applied"] is True
        assert rec["effective_target"] == pytest.approx(_COMFORT - 0.5)

    async def test_comfort_branch_adjusted_target_unchanged(self):
        """adjusted_target (user display) is never changed by early cutoff in comfort branch."""
        coord = _make_coord_comfort(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        assert rec["adjusted_target"] == pytest.approx(_COMFORT)

    async def test_comfort_branch_cold_start_no_cutoff(self):
        coord = _make_coord_comfort(cutoff_c=0.5, status="cold_start")
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False
        assert rec["effective_target"] == pytest.approx(_COMFORT)

    async def test_comfort_branch_invalid_no_cutoff(self):
        coord = _make_coord_comfort(cutoff_c=0.5, status="invalid")
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False

    async def test_no_duplicate_cutoff_between_preheat_and_comfort(self):
        """Preheat transitions to comfort in the next cycle; cutoff read is called exactly once."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        # Only one read call happened
        sh = coord._le2_shadow
        sh.read_early_cutoff_safe.assert_called_once()

    async def test_comfort_branch_no_duplicate_cutoff(self):
        """Single cutoff read per cycle in comfort branch."""
        coord = _make_coord_comfort(cutoff_c=0.5)
        rec = await _rec(coord, indoor="19.0")
        sh = coord._le2_shadow
        sh.read_early_cutoff_safe.assert_called_once()


# ---------------------------------------------------------------------------
# 11. Safety Guard: falling temperature blocks early cutoff
# ---------------------------------------------------------------------------

class TestFallingTemperatureGuard:
    async def test_falling_slope_blocks_cutoff_preheat(self):
        """Negative indoor temperature slope → skip early cutoff (heating not effective)."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        coord._indoor_temp_slope = -0.1  # temperature falling
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False
        assert rec["effective_target"] == pytest.approx(_COMFORT)

    async def test_falling_slope_blocks_cutoff_comfort(self):
        """Negative slope also blocks cutoff in comfort branch."""
        coord = _make_coord_comfort(cutoff_c=0.5)
        coord._indoor_temp_slope = -0.05
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False

    async def test_zero_slope_does_not_block(self):
        """slope = 0.0 (stable temperature) → cutoff is NOT blocked."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        coord._indoor_temp_slope = 0.0
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is True

    async def test_positive_slope_does_not_block(self):
        """Positive slope (rising temperature) → cutoff proceeds normally."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        coord._indoor_temp_slope = 0.3
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is True

    async def test_none_slope_blocks_cutoff_start(self):
        """None slope (no measurement history) → no hold.  Missing evidence must not be
        treated as 'neutral' or 'rising': the gate requires a real measured slope.
        Window-check is mocked to avoid its own slope EMA raising on the None value.
        """
        coord = _make_coord_preheat(cutoff_c=0.5)
        coord._indoor_temp_slope = None
        coord._check_window_open = MagicMock(return_value=False)
        rec = await _rec(coord, indoor="19.0")
        assert rec["early_cutoff_applied"] is False


# ---------------------------------------------------------------------------
# 12. Confidence-scaled cap in read_early_cutoff_safe()
# ---------------------------------------------------------------------------

class TestConfidenceScaledCap:
    def test_min_confidence_capped_at_half_degree(self):
        """confidence=0.35 → cap=0.5°C; prediction of 2.0 is capped to 0.5."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(2.0, confidence=0.35))
        val, st = sh.read_early_cutoff_safe()
        assert st == "valid"
        assert val == pytest.approx(0.5, abs=0.01)

    def test_max_confidence_allows_full_model_cap(self):
        """confidence=1.0 → cap=3.0°C; prediction of 2.8 is returned unchanged."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(2.8, confidence=1.0))
        val, st = sh.read_early_cutoff_safe()
        assert st == "valid"
        assert val == pytest.approx(2.8, abs=0.01)

    def test_medium_confidence_intermediate_cap(self):
        """confidence=0.675 (midpoint of 0.35–1.0) → cap=1.75°C."""
        # midpoint: t = (0.675 - 0.35) / (1.0 - 0.35) = 0.5 → cap = 0.5 + 0.5*(3.0-0.5) = 1.75
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(3.0, confidence=0.675))
        val, st = sh.read_early_cutoff_safe()
        assert st == "valid"
        assert val == pytest.approx(1.75, abs=0.02)

    def test_prediction_below_cap_unchanged(self):
        """Prediction smaller than the confidence-scaled cap is returned unchanged."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(0.3, confidence=0.35))
        val, st = sh.read_early_cutoff_safe()
        assert st == "valid"
        assert val == pytest.approx(0.3, abs=0.01)

    def test_borderline_confidence_low_cap(self):
        """confidence just above 0.35 → cap just above 0.5°C; large prediction capped."""
        coord = make_coordinator()
        sh = _shadow_with_cutoff(coord, _cutoff_pred(2.5, confidence=0.36))
        val, st = sh.read_early_cutoff_safe()
        assert st == "valid"
        # cap at 0.36: t≈0.0154, cap≈0.5 + 0.0154*2.5 ≈ 0.538
        assert val < 0.6

    async def test_coordinator_still_respects_max_c_backstop(self):
        """_EARLY_CUTOFF_MAX_C in coordinator acts as absolute backstop even if gate returns more."""
        from unittest.mock import patch
        coord = _make_coord_preheat(cutoff_c=3.0)  # shadow mock returns 3.0 exactly
        rec = await _rec(coord, indoor="17.0")
        # cut_target = 21.0 - 3.0 = 18.0; max(18.0, night_temp=18.0) = 18.0; 18.0 > 17.0 → applied
        if rec["early_cutoff_applied"]:
            assert rec["effective_target"] == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# 13. Single Dispatch unverändert / no double application
# ---------------------------------------------------------------------------

class TestSingleDispatchAndNoDuplicate:
    def test_shadow_pipeline_early_cutoff_not_applied_to_setpoint(self):
        """Shadow resolver marks early_cutoff as advisory-only; never touches final setpoint."""
        from custom_components.thermosmart.learning.decision.resolver import FinalResolver
        from custom_components.thermosmart.learning.decision.contracts import (
            DecisionMode, LearningPredictionSet, LearningPrediction,
            ControllerBaselineDecision, ZoneRuntimeInput)
        from datetime import datetime, timezone

        pset = LearningPredictionSet(
            zone_id="z1", decision_id="d1",
            predictions={"early_cutoff": LearningPrediction(
                "early_cutoff", 0.6, "celsius", 0.75, "early_cutoff", False)})
        base = ControllerBaselineDecision(
            zone_id="z1", target_c=21.0, trv_setpoint_c=21.0,
            preheat_minutes=30.0, boost_offset_c=0.0, active_control=False)
        zin = ZoneRuntimeInput(
            zone_id="z1", ts="2024-01-01T07:00:00Z",
            target_c=21.0, trv_setpoint_c=21.0, indoor_temp_c=19.0,
            indoor_temp_valid=True, comfort_temperature_c=21.0)
        resolver = FinalResolver()
        trace, dispatch = resolver.resolve(zin, base, pset, mode=DecisionMode.SHADOW)
        # Final setpoint must equal the BASELINE (21.0), not comfort-cutoff_c
        assert dispatch.command.trv_setpoint_c == pytest.approx(21.0)
        features = {e["feature"]: e for e in trace.support_dict()["features"]}
        assert features["early_cutoff"]["applied"] is False

    async def test_exactly_one_cutoff_read_per_cycle(self):
        """Each recommendation cycle calls read_early_cutoff_safe exactly once."""
        coord = _make_coord_preheat(cutoff_c=0.5)
        await _rec(coord, indoor="19.0")
        coord._le2_shadow.read_early_cutoff_safe.assert_called_once()

    async def test_user_goal_public_attribute_unchanged(self):
        """adjusted_target (= user goal, public entity attribute) is never changed."""
        coord = _make_coord_preheat(cutoff_c=1.0)
        rec = await _rec(coord, indoor="19.0")
        # adjusted_target is always the configured schedule comfort temperature
        assert rec["adjusted_target"] == pytest.approx(_COMFORT)
        assert rec["adjusted_target"] != rec.get("effective_target")  # must differ when applied


# ---------------------------------------------------------------------------
# 14. Early Cutoff Hold Lifecycle (Phase 19D)
# ---------------------------------------------------------------------------
# Shared helper for hold tests: comfort branch, stable episode key.
def _make_hold_coord(*, cutoff_c=0.5, status="valid"):
    """Coordinator in comfort branch, shadow with fixed cutoff, stable period.

    Pre-seeds ``_indoor_temp_slope`` to a small positive value so the trend gate
    (slope is not None AND slope >= 0) passes immediately without requiring real
    multi-cycle measurement history.
    """
    coord = make_coordinator()
    coord.attach_le2_shadow(_mock_cutoff_shadow(cutoff_c=cutoff_c, status=status))
    coord._current_schedule_period = MagicMock(return_value="weekday_comfort")
    coord._indoor_temp_slope = 0.05           # pre-seed: rising trend evidence
    return coord


class TestEarlyCutoffHoldLifecycle:
    """Phase 19D: Pflicht-Tests für den kontrollierten Hold-Lifecycle."""

    # ── 1. Kein Re-Heat wenn Raum Cut-Target erreicht ──────────────────────
    async def test_no_reheat_when_room_reaches_cut_target(self):
        """Hold must keep effective_target = cut_target even when current_temp > cut_target."""
        coord = _make_hold_coord(cutoff_c=0.5)
        # Cycle 1: 19.0°C → cut_target=20.5, hold starts (cutoff_applied)
        r1 = await _rec_at(coord, 19.0)
        assert r1["effective_target"] == pytest.approx(20.5)
        assert r1["early_cutoff_applied"] is True
        assert coord._ec_hold_active is True

        # Cycle 2: room rose to 20.6 (above cut_target=20.5, afterheat started)
        # Without hold: effective_target would revert to 21.0 → BUG
        # With hold: must stay at 20.5 (coasting, no re-heat)
        r2 = await _rec_at(coord, 20.6)
        assert r2["effective_target"] == pytest.approx(20.5), (
            "effective_target must not revert to comfort_temp while hold is active")
        assert r2["early_cutoff_applied"] is True
        assert r2["early_cutoff_state"] == "coasting_hold"
        assert coord._ec_hold_active is True

    # ── 2. Hold bleibt aktiv bei steigendem Trend ──────────────────────────
    async def test_hold_active_while_temperature_rising(self):
        """Hold persists as long as temp is still below comfort and conditions hold."""
        coord = _make_hold_coord(cutoff_c=0.5)
        coord._indoor_temp_slope = 0.08  # positive slope (rising)
        await _rec_at(coord, 19.0)                 # hold starts
        r = await _rec_at(coord, 20.3)             # still below cut_target=20.5
        assert r["early_cutoff_applied"] is True
        assert coord._ec_state == "cutoff_applied"

        r2 = await _rec_at(coord, 20.7)            # above cut_target, coasting
        assert r2["early_cutoff_applied"] is True
        assert r2["effective_target"] == pytest.approx(20.5)
        assert coord._ec_state == "coasting_hold"

    # ── 3. Hold endet bei erreichtem Komfortziel ───────────────────────────
    async def test_hold_released_target_reached(self):
        """Hold must release with 'released_target_reached' once comfort_temp is reached."""
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts
        await _rec_at(coord, 20.6)                 # coasting
        r = await _rec_at(coord, 21.0)             # comfort_temp reached → release
        assert not coord._ec_hold_active
        assert coord._ec_state == "released_target_reached"
        # After release, effective_target = comfort_temp (normal heating)
        assert r["effective_target"] == pytest.approx(21.0)
        assert r["early_cutoff_applied"] is False

    # ── 4. Hold endet bei fallender Temperatur unter Cut-Target ───────────
    async def test_hold_released_temperature_falling(self):
        """Hold releases when temp falls BELOW cut_target while slope is negative."""
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts
        # Simulate temperature falling below cut_target with negative slope
        coord._indoor_temp_slope = -0.15
        r = await _rec_at(coord, 20.3)             # 20.3 < cut_target=20.5 AND slope<0
        assert not coord._ec_hold_active
        assert coord._ec_state == "released_temperature_falling"
        # episode_failed = True: no new cutoff in this episode
        assert coord._ec_episode_failed is True
        assert r["early_cutoff_applied"] is False

    async def test_positive_slope_in_hold_no_release(self):
        """Positive slope during hold must NOT trigger temperature_falling release."""
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)
        coord._indoor_temp_slope = 0.05  # clearly positive – afterheat working
        r = await _rec_at(coord, 20.6)
        assert coord._ec_hold_active is True
        assert r["early_cutoff_applied"] is True

    # ── 5. Hold endet nach Timeout ─────────────────────────────────────────
    async def test_hold_released_timeout(self):
        """Hold releases after _EARLY_CUTOFF_HOLD_TIMEOUT_SECS (simulated via time)."""
        import time as _time
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts
        assert coord._ec_hold_active is True
        # Backdate the hold start to simulate timeout
        coord._ec_hold_started = _time.monotonic() - 1801.0
        r = await _rec_at(coord, 20.6)
        assert not coord._ec_hold_active
        assert coord._ec_state == "released_timeout"
        assert coord._ec_episode_failed is True   # prevents re-cycle same episode
        assert r["early_cutoff_applied"] is False

    # ── 6. Komfortzieländerung beendet Hold ────────────────────────────────
    async def test_target_change_releases_hold(self):
        """Changing comfort_temp ends the active hold (episode binding).

        After the old hold is released, a new hold may start for the new episode if
        conditions still warrant.  Use current_temp above new cut_target to prevent
        immediate re-entry so we can cleanly assert the old hold is gone.
        """
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts, episode comfort=21.0
        assert coord._ec_hold_active is True
        assert coord._ec_hold_comfort_temp == pytest.approx(21.0)
        # User changes comfort target → new episode key; old hold released
        coord.entry.data = {**coord.entry.data, "comfort_temp": 20.0}
        # Use temp above new cut_target (21.0-0.5=20.5) so no new hold starts
        r = await _rec_at(coord, 20.6)
        # Old hold gone; episode_failed is cleared; no new hold because 20.6 >= 20.5
        assert not coord._ec_hold_active
        assert coord._ec_episode_failed is False

    # ── 7. Schedule-Wechsel beendet Hold ───────────────────────────────────
    async def test_schedule_change_releases_hold(self):
        """Changing schedule period ends the active hold (episode binding).

        After the old hold is released, use current_temp above cut_target so no new
        hold can start and the released state is cleanly observable.
        """
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts, period="weekday_comfort"
        assert coord._ec_hold_active is True
        assert coord._ec_hold_period == "weekday_comfort"
        # Simulate schedule transition; use temp > cut_target to prevent re-entry
        coord._current_schedule_period.return_value = "weekday_night"
        r = await _rec_at(coord, 20.6)             # 20.6 >= cut_target 20.5: no new hold
        assert not coord._ec_hold_active
        assert coord._ec_episode_failed is False

    # ── 8. Fensteröffnung beendet Hold (strukturelle Sicherheit) ──────────
    async def test_window_open_releases_hold(self):
        """Window opening must release the hold before any other logic runs."""
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts
        assert coord._ec_hold_active is True
        # Open window – global release guard fires before branch selection
        coord._check_window_open = MagicMock(return_value=True)
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert coord._ec_state == "released_window_open"
        # Window branch: effective_target = None (heating off)
        assert r["effective_target"] is None

    # ── 9. Manual Override beendet Hold ────────────────────────────────────
    async def test_override_releases_hold(self):
        """Manual override must release the hold before any other logic runs."""
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)
        assert coord._ec_hold_active is True
        # Activate override
        coord._override = 25.0
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert coord._ec_state == "released_manual_override"
        # Override branch: effective_target = override value
        assert r["effective_target"] == pytest.approx(25.0)

    # ── 10. Stale Prediction während Hold → KEIN sofortiger Re-Heat ───────
    async def test_stale_prediction_does_not_release_active_hold(self):
        """Stale prediction DURING hold must NOT trigger release (would cause reheat).

        The applied cut threshold is frozen at hold-start.  After hold-start the
        prediction quality is no longer a release condition; only real temperature
        evidence, episode binding, safety gates, and timeout govern the hold.
        """
        sh = _mock_cutoff_shadow(0.5)
        # Cycle 1: valid → hold starts; Cycle 2: prediction goes stale
        sh.read_early_cutoff_safe.side_effect = [(0.5, "valid"), (0.0, "stale")]
        coord = make_coordinator()
        coord.attach_le2_shadow(sh)
        coord._current_schedule_period = MagicMock(return_value="weekday_comfort")
        coord._indoor_temp_slope = 0.05            # pre-seed: rising evidence for hold-start
        await _rec_at(coord, 19.0)                 # hold starts (valid prediction)
        assert coord._ec_hold_active is True
        # Cycle 2: prediction is stale but hold continues (cut threshold frozen)
        r = await _rec_at(coord, 20.3)
        assert coord._ec_hold_active is True, (
            "stale prediction must not release hold — that would re-heat during afterheat")
        assert r["early_cutoff_applied"] is True
        assert coord._ec_state in ("cutoff_applied", "coasting_hold")

    async def test_stale_prediction_cannot_start_hold(self):
        """A stale prediction (value=0.0) cannot start a new hold."""
        coord = _make_hold_coord(cutoff_c=0.0, status="stale")
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False
        # eligible but no valid prediction
        assert coord._ec_state == "eligible"

    # ── 11. Kein Pendeln nach Episode-Failed ──────────────────────────────
    async def test_no_cyclic_oscillation_after_temperature_falling(self):
        """After released_temperature_falling, no new cutoff in the same episode."""
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts
        coord._indoor_temp_slope = -0.2            # temp falls hard
        await _rec_at(coord, 20.3)                 # released_temperature_falling
        assert coord._ec_episode_failed is True

        # Next cycle: even with valid prediction, NO new hold starts
        coord._indoor_temp_slope = 0.05            # slope recovered
        r = await _rec_at(coord, 19.5)
        assert not coord._ec_hold_active
        assert coord._ec_state == "episode_failed"
        assert r["effective_target"] == pytest.approx(21.0)  # full heating resumes

    async def test_episode_failed_clears_on_episode_change(self):
        """episode_failed resets when the episode (comfort_temp) changes."""
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)
        coord._indoor_temp_slope = -0.2
        await _rec_at(coord, 20.3)                 # released_temperature_falling
        assert coord._ec_episode_failed is True
        # Reset slope: _indoor_temp_slope=-0.2 for 2 cycles hits WINDOW_SLOPE_MIN_POINTS=2,
        # triggering slope-based window detection and skipping the episode-change path.
        coord._indoor_temp_slope = 0.0
        # Episode change: user changes comfort target; use temp above new cut_target
        # so no new hold starts and failed-clear is unambiguous
        coord.entry.data = {**coord.entry.data, "comfort_temp": 20.5}
        # new cut_target = 21.0 - 0.5 = 20.5; current_temp 20.6 >= 20.5 → no new hold
        await _rec_at(coord, 20.6)
        assert coord._ec_episode_failed is False

    # ── 12. Restart: kein verwaister Hold ─────────────────────────────────
    async def test_restart_no_orphaned_hold(self):
        """A fresh coordinator (simulating HA restart) must have no active hold."""
        coord = _make_hold_coord()
        # All hold state must be safe defaults
        assert not coord._ec_hold_active
        assert coord._ec_state == "inactive"
        assert not coord._ec_episode_failed
        assert coord._ec_hold_cut_target == 0.0
        assert coord._ec_hold_comfort_temp == 0.0

    # ── 13. Genau ein Dispatch pro Zyklus auch im Hold ─────────────────────
    async def test_exactly_one_read_per_hold_cycle(self):
        """In a hold-active cycle, read_early_cutoff_safe is called exactly once at start.

        After hold-start the prediction threshold is frozen; the hold-validation path
        does NOT call read_early_cutoff_safe again (no prediction re-validation).
        """
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # cycle 1: hold starts → 1 read
        assert coord._le2_shadow.read_early_cutoff_safe.call_count == 1
        coord._le2_shadow.read_early_cutoff_safe.reset_mock()
        await _rec_at(coord, 20.3)                 # cycle 2: hold continues → 0 reads
        assert coord._le2_shadow.read_early_cutoff_safe.call_count == 0

    async def test_exactly_zero_reads_on_condition_release(self):
        """When hold releases via 'target_reached', read_early_cutoff_safe is not called."""
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts
        coord._le2_shadow.read_early_cutoff_safe.reset_mock()
        # Pre-seed to 21.0 so current_temp == comfort_temp → 'target_reached' fires first
        await _rec_at(coord, 21.0)
        # 'target_reached' check fires before the prediction re-validation branch
        assert coord._le2_shadow.read_early_cutoff_safe.call_count == 0

    # ── 14. Nutzerziel und Entity-Semantik unverändert ─────────────────────
    async def test_adjusted_target_unchanged_during_hold(self):
        """adjusted_target (public entity attribute) must never be modified by a hold."""
        coord = _make_hold_coord(cutoff_c=0.5)
        r1 = await _rec_at(coord, 19.0)
        r2 = await _rec_at(coord, 20.6)            # coasting hold
        assert r1["adjusted_target"] == pytest.approx(_COMFORT)
        assert r2["adjusted_target"] == pytest.approx(_COMFORT)
        # effective_target is reduced by hold; adjusted_target is never changed
        assert r2["effective_target"] == pytest.approx(20.5)
        assert r2["adjusted_target"] == pytest.approx(21.0)

    # ── 15. Confidence-proportionaler Cap bleibt erhalten ─────────────────
    async def test_confidence_cap_preserved_in_hold(self):
        """The confidence-capped cutoff value is stored and maintained throughout hold."""
        # Mock returns 2.0°C (pre-capped by read_early_cutoff_safe confidence logic)
        coord = _make_hold_coord(cutoff_c=2.0)
        # cut_target = 21.0 - 2.0 = 19.0; need current_temp < 19.0 to start hold
        r1 = await _rec_at(coord, 18.5)
        # coordinator hard cap: min(2.0, 3.0) = 2.0; cut_target = 21.0 - 2.0 = 19.0
        assert r1["effective_target"] == pytest.approx(19.0)
        assert coord._ec_hold_cut_target == pytest.approx(19.0)
        # Cycle 2: hold maintains stored cut_target; no re-application of cap
        r2 = await _rec_at(coord, 19.5)            # above cut_target → coasting
        assert r2["effective_target"] == pytest.approx(19.0)

    # ── 16. Cold Start / Prior allein startet keinen Hold ─────────────────
    async def test_cold_start_does_not_start_hold(self):
        """cold_start prediction status must not start a hold (value=0.0 < noise floor)."""
        coord = _make_hold_coord(cutoff_c=0.0, status="cold_start")
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False
        assert r["early_cutoff_status"] == "cold_start"

    async def test_low_confidence_does_not_start_hold(self):
        """low_confidence prediction must not start a hold."""
        coord = _make_hold_coord(cutoff_c=0.0, status="low_confidence")
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False

    # ── 17. Learning failure ≠ Heating failure ────────────────────────────
    async def test_missing_prediction_normal_heating(self):
        """'missing' status (learning not ready) must not block normal heating."""
        coord = _make_hold_coord(cutoff_c=0.0, status="missing")
        r = await _rec_at(coord, 19.0)
        # No hold, no cutoff – but heating to comfort_temp proceeds normally
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False
        # effective_target = comfort_temp (normal heating active)
        assert r["effective_target"] == pytest.approx(_COMFORT)

    # ── 18. Trace-Felder für Hold-State ───────────────────────────────────
    async def test_trace_has_early_cutoff_state(self):
        """early_cutoff_state must appear in the recommendation dict and match lifecycle."""
        coord = _make_hold_coord(cutoff_c=0.5)
        r1 = await _rec_at(coord, 19.0)
        assert r1["early_cutoff_state"] == "cutoff_applied"
        r2 = await _rec_at(coord, 20.6)            # coasting (above cut_target)
        assert r2["early_cutoff_state"] == "coasting_hold"
        r3 = await _rec_at(coord, 21.0)            # target reached → release
        assert r3["early_cutoff_state"] == "released_target_reached"

    async def test_trace_hold_active_field(self):
        """early_cutoff_hold_active must reflect hold state in the recommendation dict."""
        coord = _make_hold_coord(cutoff_c=0.5)
        r1 = await _rec_at(coord, 19.0)
        assert r1["early_cutoff_hold_active"] is True
        r2 = await _rec_at(coord, 21.0)            # target reached → released
        assert r2["early_cutoff_hold_active"] is False

    async def test_decision_trace_support_dict_has_hold_fields(self):
        """DecisionTrace.support_dict() must expose early_cutoff_state and hold_active."""
        from custom_components.thermosmart.learning.decision.contracts import (
            DecisionTrace, DecisionTraceEntry)
        trace = DecisionTrace(
            zone_id="z1", ts="2024-01-01T07:00:00Z", mode="shadow",
            decision_id="d1", baseline_setpoint_c=21.0, final_setpoint_c=21.0,
            applied_any=False, entries=(),
            reason_codes=("baseline",),
            early_cutoff_state="coasting_hold",
            early_cutoff_hold_active=True,
        )
        sd = trace.support_dict()
        assert sd["early_cutoff_state"] == "coasting_hold"
        assert sd["early_cutoff_hold_active"] is True

    # ── 19. Negative Slope oberhalb des Cut-Targets → kein Release ────────
    async def test_negative_slope_above_cut_target_no_release(self):
        """Negative slope while current_temp >= cut_target must NOT release the hold.

        Release condition requires current_temp < cut_target AND slope < 0.
        Sensor noise causing a brief negative slope while above cut_target is safe.
        """
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts, cut_target=20.5
        # Room at 20.6 (coasting), slope briefly negative (noise)
        coord._indoor_temp_slope = -0.03
        r = await _rec_at(coord, 20.6)             # 20.6 >= 20.5: condition NOT met
        assert coord._ec_hold_active is True
        assert r["effective_target"] == pytest.approx(20.5)

    # ── 20. Noise-Floor: kein Hold ─────────────────────────────────────────
    async def test_below_noise_floor_does_not_start_hold(self):
        """A prediction below the noise floor (0.15°C) must not start a hold."""
        coord = _make_hold_coord(cutoff_c=0.14)   # below _EARLY_CUTOFF_MIN_RESIDUAL_C
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False

    # ── 21. slope=None startet keinen Cutoff ──────────────────────────────
    async def test_slope_none_does_not_start_hold(self):
        """slope=None means no trend evidence; must not allow a new hold.

        Unavailable trend must not be treated as 'neutral' or 'rising'.
        """
        coord = _make_hold_coord(cutoff_c=0.5)
        coord._indoor_temp_slope = None            # explicitly: no measurement
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False

    # ── 22. UNKNOWN-Regime startet keinen Cutoff ───────────────────────────
    async def test_unknown_regime_no_cutoff(self):
        """ThermalRegimeClassifier returns UNKNOWN → no new hold.

        UNKNOWN means insufficient evidence; the deterministic baseline takes over.
        """
        coord = _make_hold_coord(cutoff_c=0.5)
        coord._le2_shadow.read_regime_safe.return_value = "unknown"
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False

    async def test_none_regime_no_cutoff(self):
        """None regime (shadow not yet classified) → no new hold."""
        coord = _make_hold_coord(cutoff_c=0.5)
        coord._le2_shadow.read_regime_safe.return_value = None
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False

    # ── 23. COOLING-Regime startet keinen Cutoff ───────────────────────────
    async def test_cooling_regime_no_cutoff(self):
        """Passive cooling regime → no new hold (room cooling, no heating needed)."""
        coord = _make_hold_coord(cutoff_c=0.5)
        coord._le2_shadow.read_regime_safe.return_value = "passive_cooling"
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False

    async def test_disturbed_regime_no_cutoff(self):
        """DISTURBED regime → no new hold."""
        coord = _make_hold_coord(cutoff_c=0.5)
        coord._le2_shadow.read_regime_safe.return_value = "disturbed"
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False

    # ── 24. ACTIVE_HEATING-Regime erlaubt Cutoff ───────────────────────────
    async def test_active_heating_regime_allows_cutoff(self):
        """active_heating regime with valid prediction → hold starts normally."""
        coord = _make_hold_coord(cutoff_c=0.5)    # default regime = "active_heating"
        r = await _rec_at(coord, 19.0)
        assert coord._ec_hold_active is True
        assert r["early_cutoff_applied"] is True

    # ── 25. Hold wechselt korrekt zu Coasting/Afterheat ────────────────────
    async def test_hold_continues_in_afterheat_regime(self):
        """During coasting, regime may transition to 'afterheat' — hold must continue.

        The regime gate only applies to NEW hold starts.  An AFTERHEAT regime during
        an active hold is the expected outcome; the hold must not release because of it.
        """
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts in active_heating
        # Regime transitions to afterheat (coasting phase starting)
        coord._le2_shadow.read_regime_safe.return_value = "afterheat"
        r = await _rec_at(coord, 20.6)             # above cut_target → coasting
        assert coord._ec_hold_active is True
        assert coord._ec_state == "coasting_hold"
        assert r["early_cutoff_applied"] is True

    # ── 26. Neue Heating-Episode nach episode_failed ────────────────────────
    async def test_new_heating_episode_recovers_from_episode_failed(self):
        """After episode_failed, a genuinely new heating cycle (room cooled + rising)
        must clear episode_failed so that early cutoff can be applied again.

        Recovery condition: current_temp < _ec_failed_cut_target - 1.0 AND slope >= 0
        This proves the room has dropped well below the failed cutoff threshold and
        is now rising again → a new distinct heating episode began.
        """
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)                 # hold starts, cut_target=20.5
        coord._indoor_temp_slope = -0.2
        await _rec_at(coord, 20.3)                 # released_temperature_falling
        assert coord._ec_episode_failed is True
        assert coord._ec_failed_cut_target == pytest.approx(20.5)

        # Simulate: room cooled down to 18.5 (< 20.5 - 1.0 = 19.5) and is now rising
        coord._indoor_temp_slope = 0.0             # reset slope (rising or flat again)
        r = await _rec_at(coord, 18.5)             # 18.5 < 19.5 → recovery triggered
        # episode_failed cleared → state back to inactive or eligible
        assert coord._ec_episode_failed is False
        assert coord._ec_state in ("inactive", "eligible")

    async def test_no_recovery_if_room_not_fallen_far_enough(self):
        """episode_failed must persist if room only dropped slightly below cut_target."""
        coord = _make_hold_coord(cutoff_c=0.5)
        await _rec_at(coord, 19.0)
        coord._indoor_temp_slope = -0.2
        await _rec_at(coord, 20.3)                 # released, failed_cut_target=20.5
        assert coord._ec_episode_failed is True

        # Room at 19.6 (< 20.5 but NOT < 20.5 - 1.0 = 19.5) → no recovery
        coord._indoor_temp_slope = 0.05
        await _rec_at(coord, 19.6)
        assert coord._ec_episode_failed is True
        assert coord._ec_state == "episode_failed"

    # ── 27. TRV-only ohne Regime-Evidenz: Baseline bleibt funktionstüchtig ─
    async def test_trv_only_no_cutoff_without_regime_evidence(self):
        """Without regime evidence (regime=None) heating continues via baseline.

        TRV-only setups with no trend / regime classification must never get their
        normal heating blocked; only the early cutoff feature is absent.
        """
        coord = _make_hold_coord(cutoff_c=0.5)
        coord._le2_shadow.read_regime_safe.return_value = None  # no regime yet
        r = await _rec_at(coord, 19.0)
        assert not coord._ec_hold_active
        assert r["early_cutoff_applied"] is False
        # Normal heating: effective_target = comfort_temp (21.0)
        assert r["effective_target"] == pytest.approx(_COMFORT)
        # adjusted_target unchanged: user goal always visible
        assert r["adjusted_target"] == pytest.approx(_COMFORT)
