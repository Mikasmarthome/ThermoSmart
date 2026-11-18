"""Phase D2-C1b — outdoor-conditioned HeatRate activation for Preheat ONLY.

Audited policy B. When every activation gate holds, a LOCAL outdoor-conditioned HeatRate
prediction (never written to ``zr.last_predictions``, never routed through the shared
``collect_predictions``) nudges the Preheat DURATION within tight asymmetric bounds. On any
gate miss / error the general result is returned unchanged (deterministic fallback), the
``valid``/``valid_hl_prior`` status strings are preserved, and published predictions, TPI,
Boost, TRV and shadow planning are unaffected. D3 (no second HeatLoss subtraction) holds.

Gates (all must pass):
  1 general status valid / valid_hl_prior
  2 conditioned pred: not fallback, reason ``conditioned_bucket``, evidence>=10, conf>=0.35
  3 outdoor finite + plausible
  4 outdoor source a non-empty identifier
  5 live source ts: missing/malformed, >120 s future, or >10800 s old -> general
  6 outdoor >= 0.5 C from every boundary (-10,-5,0,5,10,15)
  7 same qualifying band stable >= 300 s of ELAPSED time (not call count); reset on failed
    gate / band change / restart

Duration:  A_blended = A_general + 0.5*(A_conditioned - A_general);
           A_final = clamp(A_blended, A_general-10, A_general+15) then clamp [0,180].
"""
from __future__ import annotations

import dataclasses
from datetime import timedelta
from types import SimpleNamespace

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.models.heat_rate import HeatRateBucket
from tests.test_learned_state_golden import _zone


# ── builders ────────────────────────────────────────────────────────────────────

def _p(values, *, confidence=0.9, fallback=False):
    return SimpleNamespace(values=values, confidence=confidence, fallback_used=fallback)


def _published(*, heat_rate=4.0, heat_loss=0.8, afterheat=0.0, onset=0.0,
               hr_conf=0.9, hr_fallback=False, hl_conf=0.9, hl_fallback=False):
    """The PUBLISHED (general-only) predictions the control path reads. Mirrors the D3
    test's shape so general-path behaviour is identical."""
    return {
        PredictionType.HEAT_RATE: _p({"heat_rate": heat_rate}, confidence=hr_conf,
                                     fallback=hr_fallback),
        PredictionType.HEAT_LOSS_RATE: _p({"heat_loss_rate": heat_loss}, confidence=hl_conf,
                                          fallback=hl_fallback),
        PredictionType.EXPECTED_OVERSHOOT: _p({"expected_overshoot": afterheat}),
        PredictionType.ONSET_DELAY: _p({"onset_delay": onset}),
    }


def _bucket(key="outdoor:3", rate=6.0, n=20, disp=0.2):
    return HeatRateBucket(key, rate, float(n), disp, int(n), None)


async def _setup(*, published=None, buckets=None, gen_rate=4.0, gen_disp=0.2, gen_n=25,
                 agg_rel=0.9):
    """A CONTROL-mode shadow with a crafted HeatRate model state (general + conditioned
    buckets) and directly-set published predictions, so the read exercises the real
    read_preheat_minutes_safe path with exactly the state under test."""
    coord, shadow, clock, _room = await _zone(17.5, learned=True)
    zr = shadow._runtime._zone(shadow._zone)
    zr.last_predictions = published if published is not None else _published()
    model = zr.orchestrator.models["heat_rate"]
    ts = shadow._utcnow_iso()
    gen = HeatRateBucket("general", gen_rate, float(gen_n), gen_disp, int(gen_n), ts)
    model._state = dataclasses.replace(
        model._state, general=gen, buckets=dict(buckets or {}),
        aggregate_reliability=agg_rel, last_update_ts=ts)
    return coord, shadow, clock, zr


def _ts(clock, age_s):
    return (clock.now_utc() - timedelta(seconds=age_s)).isoformat()


def _read(shadow, clock, *, outdoor=2.0, source="sensor", age_s=60.0,
          current=17.5, comfort=21.0):
    lu = _ts(clock, age_s) if age_s is not None else None
    return shadow.read_preheat_minutes_safe(
        current, comfort, outdoor_temp=outdoor, outdoor_source=source,
        outdoor_last_updated=lu)


def _activate(shadow, clock, **kw):
    """Observe a qualifying band once (stabilizing), let >= 300 s elapse, then read again —
    the second read activates. Returns (minutes, status) of the activating read."""
    _read(shadow, clock, **kw)
    clock.advance(300)
    return _read(shadow, clock, **kw)


# ── 1. successful activation ─────────────────────────────────────────────────────

class TestSuccessfulActivation:
    async def test_activates_and_blends_within_bounds(self):
        # general rate 4.0 -> A_general = 3.5/4*60 + 5 = 57.5. conditioned rate 5.0 ->
        # A_cond = 3.5/5*60 + 5 = 47.0. diff -10.5 -> blended 57.5 - 5.25 = 52.25 (in band).
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        minutes, status = _activate(sh, clk)
        assert status == "valid"
        assert minutes == pytest.approx(52.25, abs=0.05)
        diag = sh.read_conditioned_preheat_diag()
        assert diag["status"] == "activated"
        assert diag["band"] == 3
        assert diag["raw_duration_diff_min"] == pytest.approx(-10.5, abs=0.05)
        assert diag["applied_bound"] == "none"

    async def test_status_string_unchanged_valid_hl_prior(self):
        pub = _published(hl_fallback=True, hl_conf=0.0)
        _c, sh, clk, _zr = await _setup(published=pub,
                                        buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _m, status = _activate(sh, clk)
        assert status == "valid_hl_prior"           # activation never changes the status
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"


# ── 2. individual gate failures -> general ───────────────────────────────────────

class TestGateFailures:
    async def test_general_fallback_prediction_not_activated(self):
        # gate 1: published HR is a fallback -> deterministic_baseline, conditioned skipped.
        _c, sh, clk, _zr = await _setup(published=_published(hr_fallback=True),
                                        buckets={"outdoor:3": _bucket(n=20)})
        _m, status = _activate(sh, clk)
        assert status == "deterministic_baseline"
        assert sh.read_conditioned_preheat_diag()["status"] != "activated"

    async def test_no_conditioned_bucket_is_general(self):
        # gate 2: model has only a general bucket -> reason_codes != conditioned_bucket.
        _c, sh, clk, _zr = await _setup(buckets={})
        m, status = _activate(sh, clk)
        assert status == "valid"
        assert m == pytest.approx(57.5, abs=0.05)   # untouched general
        assert sh.read_conditioned_preheat_diag()["status"] == "general:cond_gate"

    async def test_insufficient_evidence_is_general(self):
        # gate 2: evidence_count 9 < 10. n=9 isolates the EVIDENCE floor — its confidence
        # (0.45*0.855 = 0.385) clears the 0.35 confidence gate, so only the sample-count
        # floor rejects it (and dropping that floor is caught by the mutation suite).
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=9)})
        m, _s = _activate(sh, clk)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:cond_gate"

    async def test_low_conditioned_confidence_is_general(self):
        # gate 2: n=12 (>=10) but general dispersion huge -> conditioned confidence < 0.35.
        _c, sh, clk, _zr = await _setup(
            gen_disp=4.0, buckets={"outdoor:3": _bucket(rate=5.0, n=12)})
        m, _s = _activate(sh, clk)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:cond_gate"

    async def test_implausible_outdoor_is_general(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        m, _s = _activate(sh, clk, outdoor=999.0)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:outdoor_implausible"

    async def test_nonfinite_outdoor_is_general(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        m, _s = _activate(sh, clk, outdoor=float("nan"))
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:outdoor_implausible"

    async def test_missing_source_is_general(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        m, _s = _activate(sh, clk, source=None)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:no_source"

    async def test_empty_source_is_general(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        m, _s = _activate(sh, clk, source="")
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:no_source"

    async def test_edge_guard_rejects_near_boundary(self):
        # 0.3 C from the 0.0 boundary (< 0.5) -> general.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        m, _s = _activate(sh, clk, outdoor=0.3)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:edge_guard"

    async def test_edge_guard_allows_safe_distance(self):
        # 0.6 C from the 0.0 boundary (>= 0.5) -> still band 3, activates.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _m, _s = _activate(sh, clk, outdoor=0.6)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"


# ── 3. timestamp age / future-clock tolerance (gate 5) ───────────────────────────

class TestFreshnessTimestamp:
    async def test_missing_timestamp_is_general(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        m, _s = _activate(sh, clk, age_s=None)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:ts_missing"

    async def test_malformed_timestamp_is_general(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        _read(sh, clk, age_s=60)
        clk.advance(300)
        m, _s = sh.read_preheat_minutes_safe(
            17.5, 21.0, outdoor_temp=2.0, outdoor_source="sensor",
            outdoor_last_updated="not-a-timestamp")
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:ts_missing"

    async def test_too_far_in_future_is_general(self):
        # 200 s in the future > 120 s tolerance.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        m, _s = _activate(sh, clk, age_s=-200)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:ts_future"

    async def test_small_future_skew_tolerated(self):
        # 60 s in the future (<= 120 s) is tolerated -> activates.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _m, _s = _activate(sh, clk, age_s=-60)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"

    async def test_too_old_is_general(self):
        # 4 h old > 3 h max.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        m, _s = _activate(sh, clk, age_s=14400)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:ts_stale"

    async def test_just_within_max_age_activates(self):
        # 3 h minus a margin is still live.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _m, _s = _activate(sh, clk, age_s=10700)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"


# ── 4. stuck-but-live outdoor value ──────────────────────────────────────────────

class TestStuckButLive:
    async def test_stuck_value_live_ts_still_activates_but_bounded(self):
        # A stuck outdoor value with a fresh last_updated passes the liveness gate (we never
        # claim last_updated proves a physical measurement) — safety rests on the tight
        # bound, so the result can never leave [A_general-10, A_general+15].
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=1.0, n=20)})
        m, _s = _activate(sh, clk)       # slow conditioned rate -> wants much longer
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"
        assert 57.5 - 10.0 - 0.01 <= m <= 57.5 + 15.0 + 0.01


# ── 5. stability: elapsed time, not call count; refresh bursts ───────────────────

class TestStability:
    async def test_first_observation_is_stabilizing(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        m, _s = _read(sh, clk)
        assert m == pytest.approx(57.5, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"

    async def test_refresh_burst_without_elapsed_time_stays_general(self):
        # Many reads at the SAME instant must NOT satisfy stability (elapsed time, not count).
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        for _ in range(25):
            m, _s = _read(sh, clk)       # clock not advanced
            assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"
            assert m == pytest.approx(57.5, abs=0.05)

    async def test_activates_after_300s_elapsed(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)
        clk.advance(299)                  # just short
        assert _read(sh, clk) and sh.read_conditioned_preheat_diag()["status"] \
            == "general:stabilizing"
        clk.advance(1)                    # now >= 300 total
        _read(sh, clk)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"

    async def test_failed_gate_resets_stability(self):
        # Build stability, then a failing cycle (bad source) must reset so the next good
        # cycle starts stabilizing again rather than activating immediately.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)
        clk.advance(300)
        _read(sh, clk, source=None)       # failed gate -> reset
        assert sh.read_conditioned_preheat_diag()["status"] == "general:no_source"
        clk.advance(300)
        _read(sh, clk)                    # fresh streak: stabilizing, NOT activated
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"


# ── 6. band switching and restart ────────────────────────────────────────────────

class TestBandSwitchAndRestart:
    async def test_band_change_restarts_stability(self):
        _c, sh, clk, _zr = await _setup(buckets={
            "outdoor:3": _bucket("outdoor:3", rate=5.0, n=20),
            "outdoor:4": _bucket("outdoor:4", rate=5.0, n=20)})
        _read(sh, clk, outdoor=2.0)       # band 3 streak starts
        clk.advance(300)
        _read(sh, clk, outdoor=7.0)       # band 4: change -> restart, stabilizing
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"
        assert sh.read_conditioned_preheat_diag()["band"] == 4
        clk.advance(300)
        _read(sh, clk, outdoor=7.0)       # band 4 now stable
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"

    async def test_restart_returns_to_general(self):
        # A fresh controller (simulated by clearing session state) must re-stabilize.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _activate(sh, clk)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"
        # simulate a restart: session-scoped hysteresis state is reset (never persisted).
        sh._cp_band = None
        sh._cp_band_first_ts = None
        clk.advance(300)
        _read(sh, clk)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"


# ── 7. duration blending and both asymmetric bounds ──────────────────────────────

class TestBlendingAndBounds:
    async def test_lower_bound_minus_10(self):
        # very fast conditioned rate -> A_cond tiny -> 0.5*diff < -10 -> clamp A_general-10.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=12.0, n=20)})
        m, _s = _activate(sh, clk)
        assert m == pytest.approx(57.5 - 10.0, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["applied_bound"] == "lower"

    async def test_upper_bound_plus_15(self):
        # very slow conditioned rate -> A_cond hits 180 cap -> 0.5*diff > 15 -> A_general+15.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=1.0, n=20)})
        m, _s = _activate(sh, clk)
        assert m == pytest.approx(57.5 + 15.0, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["applied_bound"] == "upper"

    async def test_unbounded_blend_half_weight(self):
        # rate 5.0 -> A_cond 47.0, diff -10.5, half -> 52.25 (no bound applied).
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        m, _s = _activate(sh, clk)
        assert m == pytest.approx(52.25, abs=0.05)
        assert sh.read_conditioned_preheat_diag()["applied_bound"] == "none"

    async def test_final_clamped_to_180(self):
        # Even at the upper bound the final value never exceeds 180.
        _c, sh, clk, _zr = await _setup(
            buckets={"outdoor:3": _bucket(rate=1.0, n=20)}, )
        m, _s = _activate(sh, clk, current=0.0, comfort=21.0)
        assert m <= 180.0 + 1e-6


# ── 8. invariance: baseline / target-reached / exact general fallback ────────────

class TestInvariance:
    async def test_target_reached_short_circuits(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        m, status = _activate(sh, clk, current=20.8, comfort=21.0)
        assert (m, status) == (0.0, "target_reached")
        assert sh.read_conditioned_preheat_diag()["status"] != "activated"

    async def test_unavailable_short_circuits(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(n=20)})
        assert _activate(sh, clk, current=None) == (0.0, "unavailable")
        assert sh.read_conditioned_preheat_diag()["status"] != "activated"

    async def test_exact_general_equivalence_without_outdoor(self):
        # No outdoor args at all -> byte-identical to the pre-D2-C1b general computation.
        _c, sh, _clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        m, status = sh.read_preheat_minutes_safe(17.5, 21.0)
        assert (round(m, 1), status) == (57.5, "valid")

    async def test_activated_equals_general_plus_half_diff(self):
        # Independent re-derivation of the blend from the two pure durations.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=4.5, n=20)})
        a_general, _ = sh.read_preheat_minutes_safe(17.5, 21.0)   # general only
        m, _s = _activate(sh, clk)
        a_cond = 3.5 / 4.5 * 60.0 + 5.0
        expected = a_general + 0.5 * (a_cond - a_general)
        assert m == pytest.approx(round(expected, 1), abs=0.1)


# ── 9. D3 HeatLoss regression under activation ───────────────────────────────────

class TestD3Regression:
    async def test_changing_only_heat_loss_does_not_change_activated_duration(self):
        _c1, sh1, clk1, _z1 = await _setup(published=_published(heat_loss=0.5),
                                           buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        m_low, _ = _activate(sh1, clk1)
        _c2, sh2, clk2, _z2 = await _setup(published=_published(heat_loss=3.0),
                                           buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        m_high, _ = _activate(sh2, clk2)
        assert m_low == m_high
        assert sh1.read_conditioned_preheat_diag()["status"] == "activated"


# ── 10. published prediction + TPI/Boost/TRV/shadow isolation ────────────────────

class TestPublishedIsolation:
    async def test_last_predictions_object_untouched(self):
        _c, sh, clk, zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        preds_id = id(zr.last_predictions)
        hr_before = zr.last_predictions[PredictionType.HEAT_RATE]
        hr_vals_before = dict(hr_before.values)
        _activate(sh, clk)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"
        assert id(zr.last_predictions) == preds_id                     # same dict object
        assert zr.last_predictions[PredictionType.HEAT_RATE] is hr_before
        assert zr.last_predictions[PredictionType.HEAT_RATE].values == hr_vals_before

    async def test_general_published_read_unchanged_by_activation(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        hr_before = sh.read_heat_rate_safe()
        tpi_before = sh.read_tpi_coefficients_safe()
        _activate(sh, clk)
        assert sh.read_heat_rate_safe() == hr_before                   # published HR read
        assert sh.read_tpi_coefficients_safe() == tpi_before           # TPI coefficients


# ── 11. additive diagnostics ──────────────────────────────────────────────────────

class TestDiagnostics:
    async def test_diag_has_required_additive_fields(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _activate(sh, clk)
        diag = sh.read_conditioned_preheat_diag()
        for key in ("status", "band", "raw_duration_diff_min", "applied_bound"):
            assert key in diag
        assert diag["status"] == "activated"
        assert diag["band"] == 3
        assert isinstance(diag["raw_duration_diff_min"], float)
        assert diag["applied_bound"] in ("none", "lower", "upper")

    async def test_diag_is_a_copy_not_internal_ref(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _activate(sh, clk)
        d1 = sh.read_conditioned_preheat_diag()
        d1["status"] = "tampered"
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"
