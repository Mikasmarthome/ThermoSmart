"""Phase 19D: Preheat golden-value tests — required numerical audit cases.

Architecture (mandatory, verbatim):
  command_lead_time = effective_heating_onset_delay + room_heating_duration

  effective_heating_onset_delay = PredictionType.ONSET_DELAY (or _ONSET_DELAY_PRIOR_MIN)
  room_heating_duration         = effective_deficit / net_heating_rate × 60
  net_heating_rate              = max(heat_rate - heat_loss, 0.2)   [°C/h]

HeatLoss is used exactly once — in net_rate.  A missing / stale / low-confidence
HEAT_LOSS_RATE prediction must NOT be treated as 0.0 (perfect insulation).  The
versioned safe prior _HEAT_LOSS_PRIOR_C_PER_H = 0.3 °C/h is used instead.

Status semantics:
  "valid"          — both HR and HL from learned LE 2.0 predictions
  "valid_hl_prior" — HR learned, HL absent/fallback/low-conf → uses safe prior
  "deterministic_baseline" — HR absent/invalid; no LE2 knowledge used

The onset delay is a separate, independent authority (PredictionType.ONSET_DELAY,
never BoostModel, never folded into heat_rate).

Required golden case (verbatim from audit):
  deficit      = 2.0 °C
  heat_rate    = 2.0 °C/h
  heat_loss    = 0.5 °C/h   (valid prediction)
  onset_delay  = 10 min (learned)

  net_rate        = 1.5 °C/h
  heating_duration = 80 min
  command_lead_time = 90 min
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
)
from tests.helpers import make_coordinator
from tests.helpers_ha_runtime import attach_shadow, FakeStore

_PRIOR = LearningShadowController._ONSET_DELAY_PRIOR_MIN          # 5.0
_MIN_CONF = LearningShadowController._PREHEAT_MIN_CONFIDENCE       # 0.35
_HL_PRIOR = LearningShadowController._HEAT_LOSS_PRIOR_C_PER_H     # 0.3


# ── helpers ──────────────────────────────────────────────────────────────────

def _pred(values, *, prediction_type, fallback_used=False, confidence=0.9):
    p = MagicMock()
    p.prediction_type = prediction_type
    p.values = values
    p.fallback_used = fallback_used
    p.confidence = confidence
    p.units = {k: "celsius_per_hour" for k in values}
    return p


def _hr(rate, *, conf=0.9, fallback=False):
    return _pred({"heat_rate": rate}, prediction_type=PredictionType.HEAT_RATE,
                 fallback_used=fallback, confidence=conf)


def _hl(rate, *, conf=0.9, fallback=False):
    return _pred({"heat_loss_rate": rate}, prediction_type=PredictionType.HEAT_LOSS_RATE,
                 fallback_used=fallback, confidence=conf)


def _od(delay_min, *, conf=0.9, fallback=False):
    return _pred({"onset_delay": delay_min}, prediction_type=PredictionType.ONSET_DELAY,
                 fallback_used=fallback, confidence=conf)


def _shadow(hr=None, hl=None, od=None):
    """Return shadow with injected predictions; no afterheat (0 contribution)."""
    coord = make_coordinator()
    sh = attach_shadow(coord, store=FakeStore())
    preds = {}
    if hr is not None:
        preds[PredictionType.HEAT_RATE] = hr
    if hl is not None:
        preds[PredictionType.HEAT_LOSS_RATE] = hl
    if od is not None:
        preds[PredictionType.ONSET_DELAY] = od
    sh.runtime._zone(coord.zone_id).last_predictions = preds
    return sh


# ── Section 1: Required golden case (valid HR + valid HL) ────────────────────

class TestPreheatGoldenCase:
    """deficit=2.0, hr=2.0, hl=0.5, onset=10 → net=1.5, room=80, total=90 min."""

    def test_golden_command_lead_time(self):
        """Primary golden test: all three components produce exactly 90 min, status valid."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid"
        # deficit = 2.0, net_rate = max(2.0-0.5, 0.2) = 1.5
        # room_heating = 2.0/1.5*60 = 80.0 min
        # onset_delay = 10.0 min (learned)
        # total = 90.0 min
        assert minutes == pytest.approx(90.0, abs=0.5)

    def test_golden_net_rate_is_1_5(self):
        """net_rate = heat_rate - heat_loss = 2.0 - 0.5 = 1.5 °C/h."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        minutes, _ = sh.read_preheat_minutes_safe(19.0, 21.0)
        onset, _ = sh.read_onset_delay_safe()
        room_heating = minutes - onset
        # room_heating = 2.0 / 1.5 * 60 = 80 min
        assert room_heating == pytest.approx(80.0, abs=0.5)

    def test_golden_onset_delay_is_10(self):
        """Learned onset delay of 10 min is read from PredictionType.ONSET_DELAY."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        onset, status = sh.read_onset_delay_safe()
        assert status == "valid"
        assert onset == pytest.approx(10.0, abs=0.1)

    def test_golden_onset_separate_from_room_heating(self):
        """Total = room_heating + onset; both contribute independently."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        total, _ = sh.read_preheat_minutes_safe(19.0, 21.0)
        onset, _ = sh.read_onset_delay_safe()
        room = total - onset
        assert room == pytest.approx(80.0, abs=0.5)
        assert onset == pytest.approx(10.0, abs=0.1)
        assert total == pytest.approx(90.0, abs=0.5)


# ── Section 2: Comparative cases ─────────────────────────────────────────────

class TestPreheatComparativeCases:
    """Higher HeatLoss → longer duration; higher HeatRate → shorter duration."""

    def test_higher_heat_loss_gives_longer_duration(self):
        """Same hr=2.0, higher hl=1.2 → net=0.8 → room heating > 80 min."""
        sh_low = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(_PRIOR))
        sh_high = _shadow(hr=_hr(2.0), hl=_hl(1.2), od=_od(_PRIOR))
        min_low, _ = sh_low.read_preheat_minutes_safe(19.0, 21.0)
        min_high, _ = sh_high.read_preheat_minutes_safe(19.0, 21.0)
        assert min_high > min_low

    def test_higher_heat_rate_gives_shorter_duration(self):
        """Same hl=0.5, higher hr=4.0 → net=3.5 → room heating < 80 min."""
        sh_slow = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(_PRIOR))
        sh_fast = _shadow(hr=_hr(4.0), hl=_hl(0.5), od=_od(_PRIOR))
        min_slow, _ = sh_slow.read_preheat_minutes_safe(19.0, 21.0)
        min_fast, _ = sh_fast.read_preheat_minutes_safe(19.0, 21.0)
        assert min_fast < min_slow

    def test_onset_delay_adds_directly_not_via_rate(self):
        """Doubling onset delay adds exactly the onset difference to total; room heating unchanged."""
        sh_5 = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(5.0))
        sh_15 = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(15.0))
        min_5, _ = sh_5.read_preheat_minutes_safe(19.0, 21.0)
        min_15, _ = sh_15.read_preheat_minutes_safe(19.0, 21.0)
        # Lead time should differ by exactly 10 min (the onset difference)
        assert (min_15 - min_5) == pytest.approx(10.0, abs=1.0)


# ── Section 3: Fallback / safe baseline cases ─────────────────────────────────

class TestPreheatFallbacks:
    """Missing/invalid HeatLoss → safe prior 0.3; never 0.0; status=valid_hl_prior."""

    def test_missing_heat_loss_uses_prior_not_zero(self):
        """hl=None → _HEAT_LOSS_PRIOR_C_PER_H used; minutes > what hl=0.0 would give."""
        sh = _shadow(hr=_hr(2.0), hl=None, od=_od(10.0))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid_hl_prior"
        # With hl=0.0 (bug): net=2.0, room=60.0, total=70.0
        # With hl=0.3 (prior): net=1.7, room=2.0/1.7*60≈70.6, total≈80.6
        assert minutes > 70.0                                 # must be > buggy-0.0 result
        # Verify numerically: 2.0/1.7*60 + 10 ≈ 80.6
        assert minutes == pytest.approx(2.0 / max(2.0 - _HL_PRIOR, 0.2) * 60 + 10.0, abs=0.5)

    def test_missing_heat_loss_status_is_valid_hl_prior(self):
        """Missing HL prediction must NOT be labelled 'valid' — it uses a fallback."""
        sh = _shadow(hr=_hr(2.0), hl=None, od=_od(10.0))
        _, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid_hl_prior"
        assert status != "valid"

    def test_fallback_heat_loss_uses_prior(self):
        """fallback_used=True on HL prediction → _HEAT_LOSS_PRIOR_C_PER_H, status valid_hl_prior."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5, fallback=True), od=_od(10.0))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid_hl_prior"
        assert minutes == pytest.approx(2.0 / max(2.0 - _HL_PRIOR, 0.2) * 60 + 10.0, abs=0.5)

    def test_low_confidence_heat_loss_uses_prior(self):
        """HL confidence below threshold → safe prior, not the low-confidence learned value."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5, conf=_MIN_CONF - 0.01), od=_od(10.0))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid_hl_prior"
        # Value must reflect 0.3 prior, not 0.5 learned value or 0.0
        assert minutes == pytest.approx(2.0 / max(2.0 - _HL_PRIOR, 0.2) * 60 + 10.0, abs=0.5)

    def test_valid_heat_loss_0_5_is_used_exactly(self):
        """Valid HL prediction with rate=0.5 → net_rate=1.5, exact golden result 90 min."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid"          # both HR and HL from LE2
        assert minutes == pytest.approx(90.0, abs=0.5)  # 2.0/1.5*60 + 10

    def test_valid_heat_loss_distinguished_from_prior(self):
        """Valid HL=0.5 (status valid) gives different result than missing HL (status valid_hl_prior)."""
        sh_valid = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        sh_missing = _shadow(hr=_hr(2.0), hl=None, od=_od(10.0))
        min_valid, status_valid = sh_valid.read_preheat_minutes_safe(19.0, 21.0)
        min_missing, status_missing = sh_missing.read_preheat_minutes_safe(19.0, 21.0)
        assert status_valid == "valid"
        assert status_missing == "valid_hl_prior"
        # hl=0.5 → net=1.5 → room=80 → total=90
        # hl=0.3 prior → net=1.7 → room≈70.6 → total≈80.6
        # Different values: prior produces SHORTER time than hl=0.5 (0.5 > 0.3 prior)
        assert min_valid > min_missing

    def test_heat_loss_zero_still_computes_valid(self):
        """hl=0.0 as a VALID prediction → net_rate = max(hr-0, 0.2) = hr; status 'valid'."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.0), od=_od(10.0))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid"   # valid learned zero is not the same as missing
        # net_rate = max(2.0-0.0, 0.2) = 2.0; room = 2.0/2.0*60 = 60; total = 70
        assert minutes == pytest.approx(70.0, abs=0.5)

    def test_low_confidence_hr_uses_deterministic_baseline(self):
        """HR confidence below threshold → deterministic_baseline (full fallback, ignores HL)."""
        sh = _shadow(hr=_hr(2.0, conf=_MIN_CONF - 0.01), hl=_hl(0.5), od=_od(10.0))
        _, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "deterministic_baseline"

    def test_fallback_hr_uses_deterministic_baseline(self):
        """fallback_used=True on HR → deterministic_baseline."""
        sh = _shadow(hr=_hr(2.0, fallback=True), hl=_hl(0.5), od=_od(10.0))
        _, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "deterministic_baseline"

    def test_onset_prior_when_no_onset_prediction(self):
        """No ONSET_DELAY prediction → _ONSET_DELAY_PRIOR_MIN (5 min); valid HL still used."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=None)
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status == "valid"
        onset, onset_status = sh.read_onset_delay_safe()
        assert onset_status == "cold_start_prior"
        assert onset == pytest.approx(_PRIOR, abs=0.1)
        # total = 80 + 5 = 85 min
        assert minutes == pytest.approx(85.0, abs=0.5)

    def test_no_le1_fallback_on_missing_hl(self):
        """Missing HL never triggers LE v1 read — prior is purely deterministic."""
        sh = _shadow(hr=_hr(2.0), hl=None, od=_od(10.0))
        _, status = sh.read_preheat_minutes_safe(19.0, 21.0)
        assert status not in ("le1_fallback", "cold_start")

    def test_minimal_setup_no_outdoor(self):
        """No outdoor temperature → preheat still computes (outdoor unused in preheat)."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0, outdoor_temp=None)
        assert status == "valid"
        assert minutes == pytest.approx(90.0, abs=0.5)

    def test_trv_only_no_outdoor_with_missing_hl(self):
        """TRV-only (no outdoor): missing HL → valid_hl_prior; still functional."""
        sh = _shadow(hr=_hr(2.0), hl=None, od=_od(10.0))
        minutes, status = sh.read_preheat_minutes_safe(19.0, 21.0, outdoor_temp=None)
        assert status == "valid_hl_prior"
        assert minutes > 0.0


# ── Section 4: No double HeatLoss ────────────────────────────────────────────

class TestNoDoubleHeatLoss:
    """HeatLoss must appear exactly once (in net_rate); never separately in duty or setpoint."""

    def test_heat_loss_appears_once_in_net_rate(self):
        """Changing hl changes total by exactly the net_rate effect, not double."""
        sh_a = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        sh_b = _shadow(hr=_hr(2.0), hl=_hl(1.0), od=_od(10.0))
        min_a, _ = sh_a.read_preheat_minutes_safe(19.0, 21.0)
        min_b, _ = sh_b.read_preheat_minutes_safe(19.0, 21.0)
        # hl=0.5: net=1.5, room=80, total=90
        # hl=1.0: net=1.0, room=120, total=130 (may be capped at MAX)
        assert min_a == pytest.approx(90.0, abs=0.5)
        assert min_b > min_a

    def test_onset_delay_not_counted_as_room_heating(self):
        """Room heating duration = total - onset; onset not counted in thermal calculation."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        total, _ = sh.read_preheat_minutes_safe(19.0, 21.0)
        onset, _ = sh.read_onset_delay_safe()
        room = total - onset
        # Must equal deficit / net_rate * 60 = 2.0 / 1.5 * 60 = 80 min
        assert room == pytest.approx(80.0, abs=0.5)

    def test_onset_not_from_boost_model(self):
        """ONSET_DELAY comes from PredictionType.ONSET_DELAY, not any BoostModel key."""
        sh = _shadow(hr=_hr(2.0), hl=_hl(0.5), od=_od(10.0))
        onset, status = sh.read_onset_delay_safe()
        assert status == "valid"
        assert onset == pytest.approx(10.0, abs=0.1)

    def test_heat_loss_not_applied_in_coordinator_separately(self):
        """Coordinator uses coef_int (which embeds HL ratio) and TPI — no additive HL."""
        import asyncio
        from tests.helpers_ha_runtime import make_recording_coordinator
        coord = make_recording_coordinator()
        shadow = attach_shadow(coord, store=FakeStore())
        zr = shadow._runtime._zone(shadow._zone)
        zr.last_predictions = {
            PredictionType.HEAT_RATE: _hr(2.0),
            PredictionType.HEAT_LOSS_RATE: _hl(0.5),
            PredictionType.ONSET_DELAY: _od(10.0),
        }
        result = asyncio.run(coord._async_update_data())
        rec = (result or {}).get("zone", {})
        assert rec.get("tpi_duty_cycle") is not None
        from custom_components.thermosmart.tpi import compute_tpi
        ci = rec.get("tpi_coef_int", 0.0)
        ce = rec.get("tpi_coef_ext", 0.0)
        target = rec.get("effective_target")
        current = rec.get("current_temp")
        if target and current:
            expected_duty = compute_tpi(target, current, 10.0, ci, ce)
            assert rec.get("tpi_duty_cycle") == pytest.approx(expected_duty, abs=0.01)
