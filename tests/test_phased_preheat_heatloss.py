"""Phase D3 — preheat heat-loss double-subtraction correction.

``read_preheat_minutes_safe`` used ``net_rate = heat_rate - heat_loss`` for the room
heating duration. HeatRate is already the OBSERVED NET temperature rise (deg C/h over the
active heating phase, see HeatRateModel), i.e. the rise that remained AFTER the room's
simultaneous passive heat loss. Subtracting HEAT_LOSS_RATE again double-counted the loss
and inflated the preheat duration. D3 uses the net rate as-is:

    effective_deficit = max(0, deficit - afterheat)
    effective_rate    = max(net learned heat rate, 0.2 C/h floor)   # no heat-loss subtraction
    room_minutes      = effective_deficit / effective_rate * 60

These tests exercise the real ``read_preheat_minutes_safe`` with controlled predictions and
also demonstrate the OLD defect: changing only the heat-loss prediction used to change the
duration, and now must not.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from tests.test_learned_state_golden import _zone


def _p(values, *, confidence=0.9, fallback=False):
    return SimpleNamespace(values=values, confidence=confidence, fallback_used=fallback)


def _preds(*, heat_rate=4.0, heat_loss=0.8, afterheat=0.0, onset=0.0,
           hr_conf=0.9, hr_fallback=False, hl_conf=0.9, hl_fallback=False):
    d = {
        PredictionType.HEAT_RATE: _p({"heat_rate": heat_rate}, confidence=hr_conf,
                                     fallback=hr_fallback),
        PredictionType.HEAT_LOSS_RATE: _p({"heat_loss_rate": heat_loss}, confidence=hl_conf,
                                          fallback=hl_fallback),
        PredictionType.EXPECTED_OVERSHOOT: _p({"expected_overshoot": afterheat}),
    }
    if onset is not None:
        d[PredictionType.ONSET_DELAY] = _p({"onset_delay": onset})
    return d


async def _shadow_with(preds):
    """A CONTROL-mode shadow whose last_predictions are set directly (so the read exercises
    the real read_preheat_minutes_safe path with exactly the predictions under test)."""
    _coord, shadow, _clock, _ = await _zone(17.5, learned=True)
    zr = shadow._runtime._zone(shadow._zone)
    zr.last_predictions = preds
    return shadow


async def _read(preds, current=17.5, comfort=21.0):
    sh = await _shadow_with(preds)
    return sh.read_preheat_minutes_safe(current, comfort)


# ── core correction ───────────────────────────────────────────────────────────

class TestNetHeatRate:
    async def test_preheat_uses_net_heat_rate_not_minus_heat_loss(self):
        # deficit 3.5, net rate 4.0, afterheat 0 -> room 3.5/4.0*60 = 52.5 min; + onset prior
        # 5.0 -> 57.5 total. The OLD (buggy) value was 3.5/(4.0-0.8)*60 + 5.0 = 70.6 min.
        minutes, status = await _read(_preds(heat_rate=4.0, heat_loss=0.8))
        assert status == "valid"
        assert minutes == pytest.approx(57.5, abs=0.1)
        assert minutes < 70.0                      # strictly below the double-subtracted value

    async def test_changing_only_heat_loss_does_not_change_duration(self):
        # REGRESSION / old-defect demonstration: on the pre-D3 code these differed wildly
        # (net 3.5 -> 60 min vs net 1.0 -> 210 capped 180). After D3 heat loss is irrelevant.
        m_low, _ = await _read(_preds(heat_loss=0.5))
        m_high, _ = await _read(_preds(heat_loss=3.0))
        assert m_low == m_high

    async def test_learned_heat_loss_vs_prior_same_duration_only_status_differs(self):
        m_valid, s_valid = await _read(_preds(heat_loss=0.8, hl_conf=0.9, hl_fallback=False))
        m_prior, s_prior = await _read(_preds(heat_loss=0.0, hl_conf=0.0, hl_fallback=True))
        assert m_valid == m_prior                  # HL provenance does not affect duration
        assert s_valid == "valid"
        assert s_prior == "valid_hl_prior"         # compatibility status value preserved


# ── preserved behaviour ───────────────────────────────────────────────────────

class TestPreservedBehaviour:
    async def test_afterheat_shortens_duration(self):
        # room + onset prior 5.0:  3.5/4.0*60 + 5 = 57.5 ;  (3.5-1.0)/4.0*60 + 5 = 42.5
        m0, _ = await _read(_preds(afterheat=0.0))
        m1, _ = await _read(_preds(afterheat=1.0))
        assert m0 == pytest.approx(57.5, abs=0.1)
        assert m1 == pytest.approx(42.5, abs=0.1)
        assert m1 < m0

    async def test_onset_delay_adds_one_to_one(self):
        m8, _ = await _read(_preds(onset=8.0))
        m18, _ = await _read(_preds(onset=18.0))
        assert round(m18 - m8, 1) == 10.0

    async def test_fallback_heat_rate_uses_deterministic_baseline(self):
        _, status = await _read(_preds(hr_fallback=True))
        assert status == "deterministic_baseline"

    async def test_low_confidence_uses_deterministic_baseline(self):
        _, status = await _read(_preds(hr_conf=0.2))   # < 0.35 confidence gate
        assert status == "deterministic_baseline"

    async def test_confidence_proportional_cap_pulls_low_confidence_down(self):
        # Same inputs; a lower (but valid) confidence must not yield a LONGER preheat than a
        # high confidence — the cap interpolates between baseline and the 180 max.
        m_high, _ = await _read(_preds(heat_rate=1.0, hr_conf=0.9))
        m_low, _ = await _read(_preds(heat_rate=1.0, hr_conf=0.4))
        assert m_low <= m_high

    async def test_max_duration_cap(self):
        # deficit 21, rate 0.5 -> 2520 min uncapped -> capped at 180.
        minutes, status = await _read(_preds(heat_rate=0.5, hr_conf=0.9), current=0.0,
                                      comfort=21.0)
        assert minutes == pytest.approx(180.0, abs=0.1)

    async def test_target_reached(self):
        _, status = await _read(_preds(), current=20.8, comfort=21.0)   # deficit 0.2 <= 0.5
        assert status == "target_reached"

    async def test_unavailable_when_no_temperature(self):
        assert await _read(_preds(), current=None) == (0.0, "unavailable")
