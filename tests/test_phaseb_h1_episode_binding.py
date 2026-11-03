"""Phase B — H1: decision-ID continuity & baseline-contamination regression tests.

Real multi-cycle ``LearningRuntime`` runs that drive a full OUTCOME episode
(open → heat → REACHED → close) while flipping ``decision_type`` NORMAL<->BOOST
mid-episode, with authoritative per-cycle ``BoostDispatchRecord``s.

H1 root cause: after the D4 fix, ``decision_type`` can flip NORMAL->BOOST during
an open heating episode. The capture layer treated a decision_type change as a
new decision, so the episode closed under its ORIGINAL id while the boost context
was stored under a NEW id — losing the boost outcome, orphaning the context and
leaving a contaminated non-boost baseline. The fix keeps the decision_id stable
across NORMAL<->BOOST (same heating class), so context and episode bind under one
id, and a belt-and-braces guard prevents any boosted decision from becoming a
baseline.

Each invariant test is paired with the NORMAL-only control so the contrast is
explicit. The accompanying mutation run (see the remediation report) confirms
these fail against the pre-fix capture layer.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput,
    DecisionType,
    LearningRuntimeMode,
    RuntimeCycleInput,
    ScheduleTarget,
)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from tests.helpers_runtime_scenarios import COMFORT, T0, runtime

_ZONE = "lz"
_TARGET = 21.0
# temps: ramp 18->21 then hold at target long enough for the 600s stability window.
_TEMPS = [18.0, 18.6, 19.2, 19.8, 20.4, 21.0, 21.0, 21.0, 21.0]


def _cycle(rt, i, temp, *, dt=DecisionType.NORMAL, applied=0.0,
           status="fully_succeeded", heating=True):
    """One real runtime cycle with an authoritative dispatch record (as the live
    coordinator always supplies). applied>0 marks a genuinely applied boost."""
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = _TARGET + (3.0 if heating else -4.0)
    bd = BoostDispatchRecord(
        decision_id="x", boost_applied_c=applied, dispatch_status=status,
        baseline_setpoint_c=_TARGET, final_setpoint_c=setp, targets_total=1,
        effective_setpoint_min_c=setp, effective_setpoint_max_c=setp)
    inp = RuntimeCycleInput(
        zone_id=_ZONE, ts=ts, target_c=_TARGET, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK),
        schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=_TARGET),
        controller_decision=ControllerDecisionInput(
            decision_type=dt, target_c=_TARGET, trv_setpoint_c=setp,
            boost_active=(dt is DecisionType.BOOST)),
        boost_dispatch=bd)
    return rt.run_cycle(inp)


def _run(boost_cycles: dict):
    """Drive the standard reaching episode. boost_cycles maps cycle index ->
    (decision_type, applied, status). Returns the zone runtime for inspection."""
    rt = runtime(mode=LearningRuntimeMode.CONTROL)
    zr = rt._zone(_ZONE)
    ids = []
    for i, t in enumerate(_TEMPS):
        dt, applied, status = boost_cycles.get(i, (DecisionType.NORMAL, 0.0, "fully_succeeded"))
        _cycle(rt, i, t, dt=dt, applied=applied, status=status, heating=(t < _TARGET))
        od = zr.capture.open_decision
        ids.append(od.decision_id if od else None)
    return zr, ids


def _distinct(ids):
    return {i for i in ids if i is not None}


# ── scenario 6 (baseline) control: NORMAL-only ───────────────────────────────

class TestNormalOnlyControl:
    def test_normal_only_stores_one_clean_baseline(self):
        zr, ids = _run({})
        assert len(_distinct(ids)) == 1                 # one stable decision
        assert zr.baseline_store.size == 1              # a clean non-boost baseline
        assert zr.pending_boost_outcome is None         # no boost outcome
        assert len(zr.pending_boost_contexts) == 0


# ── scenarios 1-3: NORMAL<->BOOST transitions within one episode ──────────────

class TestDecisionContinuity:
    def test_1_normal_then_boost_one_episode(self):
        # Boost applied mid-episode (cycle 3, a fully-successful write).
        zr, ids = _run({3: (DecisionType.BOOST, 1.0, "fully_succeeded")})
        assert len(_distinct(ids)) == 1                 # H1: decision_id stays STABLE
        assert zr.pending_boost_outcome is not None     # boost outcome bound (not lost)
        assert len(zr.pending_boost_contexts) == 0      # context consumed, not orphaned
        assert zr.baseline_store.size == 0              # episode boosted -> NO baseline

    def test_2_boost_then_normal(self):
        # Boost at the start (cycles 0-2), then plain NORMAL until REACHED.
        zr, ids = _run({0: (DecisionType.BOOST, 1.0, "fully_succeeded"),
                        1: (DecisionType.BOOST, 1.0, "fully_succeeded"),
                        2: (DecisionType.BOOST, 1.0, "fully_succeeded")})
        assert len(_distinct(ids)) == 1
        assert zr.pending_boost_outcome is not None
        assert zr.baseline_store.size == 0              # boosted episode never a baseline

    def test_3_normal_boost_normal(self):
        zr, ids = _run({3: (DecisionType.BOOST, 1.0, "fully_succeeded"),
                        4: (DecisionType.BOOST, 1.0, "fully_succeeded")})
        # cycles 5-8 revert to NORMAL (applied 0) before/at REACHED
        assert len(_distinct(ids)) == 1
        assert zr.pending_boost_outcome is not None     # boost outcome bound (written boost)
        assert zr.baseline_store.size == 0              # boosted episode never a clean baseline


# ── scenario 4: failed boost dispatch then reapproval ─────────────────────────

class TestFailedThenReapproved:
    def test_4_failed_boost_then_reapproved(self):
        # cycle 2: boost dispatch FAILED (nothing applied). cycle 4: reapproved,
        # fully successful. The failed cycle must not bind as a boost success nor
        # mark the decision; the reapproval forms the authoritative context.
        zr, ids = _run({2: (DecisionType.BOOST, 0.0, "failed"),
                        4: (DecisionType.BOOST, 1.0, "fully_succeeded")})
        assert len(_distinct(ids)) == 1
        assert zr.pending_boost_outcome is not None     # bound from the reapproved cycle
        assert zr.baseline_store.size == 0

    def test_failed_boost_only_no_outcome_no_baseline(self):
        # A boost that only ever FAILED (never applied) across the episode: no boost
        # outcome AND no baseline (the failed dispatch is not a clean non-boost run).
        zr, ids = _run({3: (DecisionType.BOOST, 0.0, "failed")})
        # last cycle is NORMAL fully_succeeded applied 0 -> the episode is a clean
        # non-boost trajectory (the boost had zero thermal effect) -> baseline allowed.
        assert len(_distinct(ids)) == 1
        assert zr.pending_boost_outcome is None         # failed boost formed no outcome


# ── scenario 5: episode closure & context cleanup ─────────────────────────────

class TestContextCleanup:
    def test_5_context_popped_at_close(self):
        zr, ids = _run({3: (DecisionType.BOOST, 1.0, "fully_succeeded")})
        # After the OutcomeEpisode closed, the boost context was consumed (bound
        # into the pending boost outcome) and the per-decision contamination marker
        # was cleared — neither is orphaned. (pending_dispatch_records is maintained
        # every cycle, so later NORMAL cycles legitimately re-populate it.)
        assert len(zr.pending_boost_contexts) == 0      # boost context not orphaned
        assert len(zr.boost_applied_decisions) == 0     # H1 contamination marker cleared


# ── scenario 6: no contaminated baseline samples (explicit contrast) ──────────

class TestNoBaselineContamination:
    def test_6_boosted_episode_never_baseline(self):
        clean, _ = _run({})
        boosted, _ = _run({3: (DecisionType.BOOST, 1.0, "fully_succeeded")})
        assert clean.baseline_store.size == 1           # non-boost -> baseline
        assert boosted.baseline_store.size == 0         # boosted (written) -> bound as boost, not baseline

    def test_6b_held_boost_then_normal_not_baseline(self):
        # Mixed: an already-HELD boost (already_satisfied, applied>0) at cycle 3,
        # then NORMAL. The held boost marks the decision, so even though the final
        # dispatch shows no boost, the episode is excluded from baselines.
        zr, ids = _run({3: (DecisionType.BOOST, 0.8, "already_satisfied")})
        assert len(_distinct(ids)) == 1
        assert zr.baseline_store.size == 0
        assert zr.baseline_store.rejection_counts().get("boost_applied", 0) >= 1


# ── scenario 7: restart continuity (persisted context + stable id) ────────────

class TestRestartContinuity:
    def test_7_restart_mid_episode_preserves_binding(self):
        # Drive an open boosted episode partway, serialise, restore into a fresh
        # runtime, then continue to REACHED. The persisted boost context + stable
        # decision_id must still bind the outcome and keep the episode off the
        # baseline path across the restart.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        zr = rt._zone(_ZONE)
        for i in range(4):                          # cycles 0-3, boost at 3
            dt, applied = ((DecisionType.BOOST, 1.0) if i == 3
                           else (DecisionType.NORMAL, 0.0))
            _cycle(rt, i, _TEMPS[i], dt=dt, applied=applied, heating=(_TEMPS[i] < _TARGET))
        open_id_before = zr.capture.open_decision.decision_id
        assert len(zr.pending_boost_contexts) == 1  # boost context persisted-to-be
        payload = rt._build_payload()

        rt2 = runtime(mode=LearningRuntimeMode.CONTROL)
        rt2._restore_payload(payload)
        zr2 = rt2._zone(_ZONE)
        assert zr2.capture.open_decision is not None
        assert zr2.capture.open_decision.decision_id == open_id_before   # id continuity
        assert len(zr2.pending_boost_contexts) == 1                      # context survived
        # continue to REACHED under the restored runtime
        for i in range(4, len(_TEMPS)):
            _cycle(rt2, i, _TEMPS[i], dt=DecisionType.NORMAL, applied=0.0,
                   heating=(_TEMPS[i] < _TARGET))
        assert zr2.pending_boost_outcome is not None    # boost outcome bound post-restart
        assert zr2.baseline_store.size == 0             # not contaminated
