"""Phase D2-C1b acceptance-audit corrections — N1 (stale stability state), N2 (stale
diagnostics), the non-finite conditioned-confidence guard, decision-trace observability,
and the remaining focused coverage gaps (all six bucket edges incl. 15 C, explicit
fallback_used rejection, and no-change-to-general-consumers).

Reuses the policy-B harness from ``test_conditioned_preheat_d2c1b``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from tests.test_conditioned_preheat_d2c1b import (
    _activate,
    _bucket,
    _published,
    _read,
    _setup,
    _ts,
)


def _fake_cond(*, rate=5.0, confidence=0.9, fallback_used=False, n=20,
               reasons=("conditioned_bucket",)):
    """A stand-in conditioned Prediction to isolate a single gate-2 clause."""
    return SimpleNamespace(values={"heat_rate": rate}, confidence=confidence,
                           fallback_used=fallback_used, evidence_count=n,
                           reason_codes=tuple(reasons))


# ── N1: a pending band-stability streak must not survive ineligible early returns ──

class TestN1StaleStability:
    async def test_target_reached_gap_does_not_satisfy_stability(self):
        # qualifying observation -> hours of target_reached -> new deficit: the FIRST new
        # eligible read must NOT activate (stability must restart from scratch).
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)                                 # band 3 streak starts (stabilizing)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"
        clk.advance(3 * 3600)                           # hours pass...
        _read(sh, clk, current=20.8, comfort=21.0)      # ...as target_reached early returns
        assert sh.read_conditioned_preheat_diag()["status"] == "target_reached"
        assert sh._cp_band is None and sh._cp_band_first_ts is None   # streak invalidated
        m, status = _read(sh, clk)                      # first new eligible read
        assert status == "valid"
        assert m == pytest.approx(57.5, abs=0.05)       # general, NOT the blended value
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"

    async def test_disabled_and_fallback_paths_reset_streak(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)                                  # streak starts
        sh._enabled = False
        sh.read_preheat_minutes_safe(17.5, 21.0, outdoor_temp=2.0, outdoor_source="sensor",
                                     outdoor_last_updated=_ts(clk, 60))
        assert sh._cp_band is None and sh._cp_band_first_ts is None
        assert sh.read_conditioned_preheat_diag()["status"] == "general:disabled"


# ── N2: diagnostics never retain a stale 'activated' after an ineligible return ────

class TestN2StaleDiagnostics:
    async def test_target_reached_after_activation_clears_activated(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _activate(sh, clk)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"
        _read(sh, clk, current=20.8, comfort=21.0)      # target_reached
        assert sh.read_conditioned_preheat_diag()["status"] == "target_reached"

    async def test_general_fallback_after_activation_clears_activated(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _activate(sh, clk)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"
        # now a cycle whose published HR is a fallback -> deterministic baseline early return
        zr = sh._runtime._zone(sh._zone)
        zr.last_predictions = _published(hr_fallback=True)
        _m, status = _read(sh, clk)
        assert status == "deterministic_baseline"
        assert sh.read_conditioned_preheat_diag()["status"] == "general:hr_fallback"

    async def test_low_general_confidence_clears_activated(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _activate(sh, clk)
        zr = sh._runtime._zone(sh._zone)
        zr.last_predictions = _published(hr_conf=0.2)   # < 0.35 general gate
        _m, status = _read(sh, clk)
        assert status == "deterministic_baseline"
        assert sh.read_conditioned_preheat_diag()["status"] == "general:low_confidence"


# ── gate-2 clause isolation: explicit fallback_used + non-finite confidence ────────

class TestConditionedGateClauses:
    async def test_explicit_fallback_used_rejected(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        sh._conditioned_heat_rate_prediction = lambda outdoor: _fake_cond(fallback_used=True)
        m, _s = _read(sh, clk)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:cond_gate"

    async def test_missing_conditioned_reason_rejected(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        sh._conditioned_heat_rate_prediction = lambda outdoor: _fake_cond(reasons=("general_rate",))
        m, _s = _read(sh, clk)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:cond_gate"

    async def test_nonfinite_confidence_rejected(self):
        # NaN confidence makes ``conf < MIN`` False — it must still be rejected by the
        # explicit isfinite guard, never activated.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        sh._conditioned_heat_rate_prediction = \
            lambda outdoor: _fake_cond(confidence=float("nan"))
        m, _s = _read(sh, clk)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:cond_gate"

    async def test_inf_confidence_rejected(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        sh._conditioned_heat_rate_prediction = \
            lambda outdoor: _fake_cond(confidence=float("inf"))
        m, _s = _read(sh, clk)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:cond_gate"


# ── all six bucket edges, including 15 C ──────────────────────────────────────────

class TestAllBucketEdges:
    # (outdoor value clear of every edge) -> expected band index over (-10,-5,0,5,10,15).
    @pytest.mark.parametrize("outdoor,band", [
        (-12.0, 0), (-7.0, 1), (-2.0, 2), (2.0, 3), (7.0, 4), (12.0, 5), (16.0, 6)])
    async def test_each_band_activates(self, outdoor, band):
        key = f"outdoor:{band}"
        _c, sh, clk, _zr = await _setup(buckets={key: _bucket(key=key, rate=5.0, n=20)})
        _m, status = _activate(sh, clk, outdoor=outdoor)
        assert status == "valid"
        diag = sh.read_conditioned_preheat_diag()
        assert diag["status"] == "activated"
        assert diag["band"] == band

    async def test_value_on_15_edge_is_edge_guarded(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:6": _bucket(key="outdoor:6", n=20)})
        m, _s = _activate(sh, clk, outdoor=15.0)        # exactly on the 15 C boundary
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:edge_guard"

    async def test_top_band_above_15_activates(self):
        _c, sh, clk, _zr = await _setup(
            buckets={"outdoor:6": _bucket(key="outdoor:6", rate=5.0, n=20)})
        _m, _s = _activate(sh, clk, outdoor=16.0)       # band 6, clear of the 15 C edge
        diag = sh.read_conditioned_preheat_diag()
        assert diag["status"] == "activated" and diag["band"] == 6


# ── decision-trace observability (additive; distinct from published general values) ──

class _FakeTrace:
    decision_id = "dec_fake"

    def support_dict(self):
        # a published general-HeatRate field is present and must be left untouched.
        return {"heat_rate_prediction_c_per_h": 4.0, "heat_rate_confidence": 0.9}


async def _trace_with(recommendation):
    _c, sh, _clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
    sh._build_trace_from_live_record = lambda *a, **k: _FakeTrace()
    sh.compute_decision_trace_safe(recommendation, live_record=object())
    return sh.last_decision_trace


class TestDecisionTrace:
    async def test_trace_reports_activation(self):
        tr = await _trace_with({"preheat_conditioned": {
            "status": "activated", "band": 3,
            "raw_duration_diff_min": -10.5, "applied_bound": "none"}})
        assert tr["preheat_conditioned_status"] == "activated"
        assert tr["preheat_conditioned_band"] == 3
        assert tr["preheat_conditioned_raw_duration_diff_min"] == pytest.approx(-10.5)
        assert tr["preheat_conditioned_applied_bound"] == "none"

    async def test_trace_reports_fallback(self):
        tr = await _trace_with({"preheat_conditioned": {
            "status": "general:hr_fallback", "band": None,
            "raw_duration_diff_min": None, "applied_bound": None}})
        assert tr["preheat_conditioned_status"] == "general:hr_fallback"
        assert tr["preheat_conditioned_band"] is None
        assert tr["preheat_conditioned_applied_bound"] is None

    async def test_trace_does_not_replace_published_general_values(self):
        # The additive conditioned block must change ONLY the preheat_conditioned_* keys:
        # the published general-HeatRate trace keys are identical whether or not the
        # conditioned diagnostics are present.
        tr_without = await _trace_with({})
        tr_with = await _trace_with({"preheat_conditioned": {
            "status": "activated", "band": 3,
            "raw_duration_diff_min": -10.5, "applied_bound": "none"}})
        assert tr_with["heat_rate_prediction_c_per_h"] == tr_without["heat_rate_prediction_c_per_h"]
        assert tr_with["heat_rate_confidence"] == tr_without["heat_rate_confidence"]
        assert tr_without["preheat_conditioned_status"] is None
        assert tr_with["preheat_conditioned_status"] == "activated"

    async def test_trace_absent_recommendation_key_is_safe(self):
        tr = await _trace_with({})     # no preheat_conditioned key -> fields present, None
        assert tr["preheat_conditioned_status"] is None
        assert tr["preheat_conditioned_band"] is None
        assert tr["preheat_conditioned_applied_bound"] is None


# ── no change to general prediction consumers under activation ────────────────────

class TestGeneralConsumersUnchanged:
    async def test_published_reads_and_dict_untouched(self):
        _c, sh, clk, zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        preds_id = id(zr.last_predictions)
        hr_before = zr.last_predictions[PredictionType.HEAT_RATE]
        hr_read_before = sh.read_heat_rate_safe()
        tpi_before = sh.read_tpi_coefficients_safe()
        _activate(sh, clk)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"
        assert id(zr.last_predictions) == preds_id
        assert zr.last_predictions[PredictionType.HEAT_RATE] is hr_before
        assert sh.read_heat_rate_safe() == hr_read_before
        assert sh.read_tpi_coefficients_safe() == tpi_before
