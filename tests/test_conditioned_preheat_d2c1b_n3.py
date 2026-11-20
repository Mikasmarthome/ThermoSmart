"""Phase D2-C1b N3 (no-read stability gap) + T1 coverage.

N3: a qualifying band observed once, then a long stretch with NO read_preheat_minutes_safe
call, must NOT activate on the next eligible read — the stability streak is continuous only
while consecutive qualifying observations are <= _CP_MAX_OBS_GAP_S (600 s) apart. This is a
CONTINUITY threshold, not a physical outdoor-measurement freshness guarantee. The 300 s
minimum stability requirement is unchanged.

Also an integration test of the real coordinator recommendation construction → decision
trace, so the conditioned diagnostics are proven to flow end to end.
"""
from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.models.heat_rate import HeatRateBucket
from tests.test_conditioned_preheat_d2c1b import (
    _activate,
    _bucket,
    _published,
    _read,
    _setup,
    _ts,
)
from tests.test_learned_state_golden import _zone


def _raise(*_a, **_k):
    raise RuntimeError("boom")


# ── N3: no-read gap continuity ────────────────────────────────────────────────────

class TestN3NoReadGap:
    async def test_hours_without_reads_do_not_activate(self):
        # Core N3 defect: one qualifying observation, then HOURS with no reads, then a new
        # eligible read. It must NOT activate — continuity is broken, streak restarts.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)                                   # t0: streak starts (stabilizing)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"
        clk.advance(3 * 3600)                            # hours pass with NO reads
        m, status = _read(sh, clk)                       # next eligible read
        assert status == "valid"
        assert m == pytest.approx(57.5, abs=0.05)        # general, not blended
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"
        # restarted at the current observation: first == last, both just set
        assert sh._cp_band == 3
        assert sh._cp_band_first_ts == sh._cp_band_last_ts

    async def test_gap_just_over_600_restarts(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)
        clk.advance(601)                                 # just over the continuity gap
        _read(sh, clk)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"
        clk.advance(300)                                 # now a continuous 300 s streak
        _read(sh, clk)
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"

    async def test_sub_600_gaps_accumulate_to_activation(self):
        # A sequence of qualifying observations each <= 600 s apart reaches activation once
        # >= 300 s of elapsed time has passed since the streak start.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)                                   # t0 (stabilizing)
        clk.advance(200)
        _read(sh, clk)                                   # gap 200, elapsed 200 (stabilizing)
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"
        clk.advance(200)
        _m, _s = _read(sh, clk)                          # gap 200, elapsed 400 -> activated
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"

    async def test_backward_clock_does_not_activate(self):
        # Build a streak whose last observation is well past first+300 s, then jump the clock
        # BACKWARD to an instant that is still >= first+300 s (so elapsed alone would say
        # "stable") but BEFORE the last observation (gap < 0). The backward-movement guard
        # must restart the streak rather than activate on a negative inter-observation gap.
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        t0 = clk.now_utc()
        _read(sh, clk)                                   # first = last = t0
        clk.advance(300)
        _read(sh, clk)                                   # t0+300: activated, last = t0+300
        clk.advance(300)
        _read(sh, clk)                                   # t0+600: activated, last = t0+600
        assert sh.read_conditioned_preheat_diag()["status"] == "activated"
        clk.set_utc(t0 + timedelta(seconds=450))         # backward: >= first+300 but < last
        m, _s = _read(sh, clk)
        assert m == pytest.approx(57.5, abs=0.05)        # general, NOT activated
        assert sh.read_conditioned_preheat_diag()["status"] == "general:stabilizing"


# ── T1: early-return tracker invalidation ─────────────────────────────────────────

class TestT1TrackerInvalidation:
    async def test_missing_general_prediction_invalidates(self):
        _c, sh, clk, zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)                                   # streak started
        assert sh._cp_band == 3
        zr.last_predictions = {}                         # no HEAT_RATE this cycle
        clk.advance(100)
        _m, status = _read(sh, clk)
        assert status == "deterministic_baseline"
        assert sh._cp_band is None and sh._cp_band_first_ts is None \
            and sh._cp_band_last_ts is None
        assert sh.read_conditioned_preheat_diag()["status"] == "general:no_prediction"

    async def test_unavailable_current_temp_invalidates(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)
        _read(sh, clk, current=None)                     # unavailable early return
        assert sh._cp_band is None and sh._cp_band_last_ts is None
        assert sh.read_conditioned_preheat_diag()["status"] == "unavailable"

    async def test_outer_exception_invalidates(self):
        _c, sh, clk, _zr = await _setup(buckets={"outdoor:3": _bucket(rate=5.0, n=20)})
        _read(sh, clk)
        assert sh._cp_band == 3
        sh.read_onset_delay_safe = _raise                # raises inside the try
        _m, status = _read(sh, clk)
        assert status == "deterministic_baseline"
        assert sh._cp_band is None and sh._cp_band_first_ts is None \
            and sh._cp_band_last_ts is None
        assert sh.read_conditioned_preheat_diag()["status"] == "general:error"


# ── coordinator → recommendation → decision-trace integration ─────────────────────

async def _zone_with_conditioned_bucket():
    coord, shadow, clock, room = await _zone(17.5, learned=True)
    zr = shadow._runtime._zone(shadow._zone)
    m = zr.orchestrator.models["heat_rate"]
    ts = shadow._utcnow_iso()
    m._state = dataclasses.replace(
        m._state, buckets={"outdoor:3": HeatRateBucket("outdoor:3", 5.0, 20.0, 0.2, 20, ts)})
    return coord, shadow, clock, room


class TestCoordinatorTraceWiring:
    async def test_activation_flows_to_trace(self):
        # Exercises the REAL coordinator recommendation construction + trace. The conditioned
        # diag is pinned to an activation result; the coordinator copies it into
        # recommendation["preheat_conditioned"] and compute_decision_trace_safe surfaces it.
        coord, shadow, _clk, _room = await _zone_with_conditioned_bucket()
        sentinel = {"status": "activated", "band": 3,
                    "raw_duration_diff_min": -7.5, "applied_bound": "none"}
        shadow.read_conditioned_preheat_diag = lambda: dict(sentinel)
        await coord._async_update_data()
        tr = shadow.last_decision_trace
        assert tr is not None
        assert tr["preheat_conditioned_status"] == "activated"
        assert tr["preheat_conditioned_band"] == 3
        assert tr["preheat_conditioned_raw_duration_diff_min"] == -7.5
        assert tr["preheat_conditioned_applied_bound"] == "none"
        # published general HeatRate fields are still present (never replaced).
        assert "heat_rate_prediction_c_per_h" in tr
        assert "heat_rate_confidence" in tr

    async def test_general_fallback_flows_to_trace(self):
        # No pin: the real diag (a general:* status this cycle) must still reach the trace.
        # This FAILS if the coordinator stops adding preheat_conditioned to the recommendation
        # (the trace field would then be None).
        coord, shadow, _clk, _room = await _zone_with_conditioned_bucket()
        await coord._async_update_data()
        tr = shadow.last_decision_trace
        assert tr is not None
        status = tr["preheat_conditioned_status"]
        assert status is not None and isinstance(status, str)
        assert status.startswith("general")
        assert "heat_rate_prediction_c_per_h" in tr
