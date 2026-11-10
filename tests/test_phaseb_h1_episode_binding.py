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


def _mk_bd(applied, status, base, setp):
    """Build a single-device authoritative dispatch record as the live coordinator
    would: the POST-dispatch effective offset (F1) and the authoritative "any device
    boosted" exposure flag (N1) are set from the applied offset + status."""
    exposed_status = status in ("fully_succeeded", "already_satisfied", "partially_succeeded")
    effective = (applied if status in ("fully_succeeded", "already_satisfied") else 0.0)
    any_boosted = bool((applied or 0.0) > 0.0 and exposed_status)
    return BoostDispatchRecord(
        decision_id="x", boost_applied_c=applied, dispatch_status=status,
        baseline_setpoint_c=base, final_setpoint_c=setp, targets_total=1,
        effective_setpoint_min_c=setp, effective_setpoint_max_c=setp,
        effective_offset_c=effective, any_device_boosted=any_boosted)


def _cycle(rt, i, temp, *, dt=DecisionType.NORMAL, applied=0.0,
           status="fully_succeeded", heating=True):
    """One real runtime cycle with an authoritative dispatch record (as the live
    coordinator always supplies). applied>0 marks a genuinely applied boost."""
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = _TARGET + (3.0 if heating else -4.0)
    bd = _mk_bd(applied, status, _TARGET, setp)
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
        # into the pending boost outcome) — not orphaned. Boost provenance is now
        # latched on the episode itself (OutcomeEpisode.boost_exposed), so there is
        # no leftover zone-level marker to leak. (pending_dispatch_records is
        # maintained every cycle, so later NORMAL cycles legitimately re-populate it.)
        assert len(zr.pending_boost_contexts) == 0      # boost context not orphaned


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


# ── flexible driver for decision-ID-change scenarios (target jump / PREHEAT) ───

def _fcycle(rt, i, temp, *, zone=_ZONE, target=_TARGET, dt=DecisionType.NORMAL,
            applied=0.0, status="fully_succeeded", heating=True, window=None):
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = target + (3.0 if heating else -4.0)
    bd = _mk_bd(applied, status, target, setp)
    inp = RuntimeCycleInput(
        zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK), window_open=window,
        schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=target),
        controller_decision=ControllerDecisionInput(
            decision_type=dt, target_c=target, trv_setpoint_c=setp,
            boost_active=(dt is DecisionType.BOOST)),
        boost_dispatch=bd)
    return rt.run_cycle(inp)


class TestDecisionIdChangeWhileEpisodeOpen:
    def test_6_target_step_then_boost_under_new_target(self):
        # Outcome episode opens at target 21; the target steps to 23 mid-episode (a
        # NEW capture decision_id) and a Boost is applied under the new target. The
        # episode-scoped provenance is latched regardless of the decision_id change,
        # so the episode is never a clean baseline.
        rt = runtime(mode=LearningRuntimeMode.CONTROL); zr = rt._zone(_ZONE)
        _fcycle(rt, 0, 18.0, target=21.0); _fcycle(rt, 1, 19.0, target=21.0)
        _fcycle(rt, 2, 20.0, target=23.0, dt=DecisionType.BOOST, applied=1.0)
        for i, t in enumerate([21.0, 22.0, 23.0, 23.0, 23.0], start=3):
            _fcycle(rt, i, t, target=23.0, heating=(t < 23.0))
        assert zr.baseline_store.size == 0
        assert zr.baseline_store.rejection_counts().get("boost_applied", 0) >= 1

    def test_7_preheat_then_boost(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL); zr = rt._zone(_ZONE)
        _fcycle(rt, 0, 18.0, dt=DecisionType.PREHEAT)
        _fcycle(rt, 1, 19.0, dt=DecisionType.PREHEAT)
        _fcycle(rt, 2, 20.0, dt=DecisionType.BOOST, applied=1.0)
        for i, t in enumerate([20.6, 21.0, 21.0, 21.0], start=3):
            _fcycle(rt, i, t, heating=(t < 21.0))
        assert zr.baseline_store.size == 0

    def test_8_repeated_target_steps_and_boost(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL); zr = rt._zone(_ZONE)
        _fcycle(rt, 0, 18.0, target=20.0)
        _fcycle(rt, 1, 19.0, target=21.0)
        _fcycle(rt, 2, 19.8, target=22.0, dt=DecisionType.BOOST, applied=1.0)
        for i, t in enumerate([20.6, 21.4, 22.0, 22.0, 22.0], start=3):
            _fcycle(rt, i, t, target=22.0, heating=(t < 22.0))
        assert zr.baseline_store.size == 0


class TestRestartScenarios:
    def _drive_then_restart(self, *, boost_applied, boost_status):
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        _fcycle(rt, 0, 18.0); _fcycle(rt, 1, 19.0)
        _fcycle(rt, 2, 19.5, dt=DecisionType.BOOST, applied=boost_applied, status=boost_status)
        payload = rt._build_payload()
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL)
        rt2._restore_payload(payload)
        for i, t in enumerate([20.0, 20.6, 21.0, 21.0, 21.0], start=3):
            _fcycle(rt2, i, t, heating=(t < 21.0))
        return rt2._zone(_ZONE)

    def test_5_written_boost_restart_then_removed(self):
        zr = self._drive_then_restart(boost_applied=1.0, boost_status="fully_succeeded")
        assert zr.baseline_store.size == 0
        assert zr.pending_boost_outcome is not None

    def test_4_held_only_boost_restart_then_removed(self):
        # The case the former session-only marker FAILED: a held-only boost, a
        # restart, then boost removal. Episode-scoped provenance persists in the
        # builder state, so the episode is still excluded from baselines.
        zr = self._drive_then_restart(boost_applied=0.8, boost_status="already_satisfied")
        assert zr.baseline_store.size == 0
        assert zr.baseline_store.rejection_counts().get("boost_applied", 0) >= 1


class TestAbortAndCleanEpisode:
    def test_10_11_abort_then_clean_episode_stores_baseline(self):
        # A boosted episode aborted by a window-open interruption; a subsequent CLEAN
        # episode under a new target must store a baseline (flag/context did not leak).
        rt = runtime(mode=LearningRuntimeMode.CONTROL); zr = rt._zone(_ZONE)
        _fcycle(rt, 0, 18.0, dt=DecisionType.BOOST, applied=1.0)
        _fcycle(rt, 1, 19.0, window=True)
        for i, t in enumerate([18.0, 18.8, 19.6, 20.4, 22.0, 22.0, 22.0, 22.0], start=2):
            _fcycle(rt, i, t, target=22.0, heating=(t < 22.0))
        assert zr.baseline_store.size == 1
        assert zr.baseline_store.rejection_counts().get("boost_applied", 0) == 0


class TestBackwardCompatibleProvenance:
    def test_12_older_payload_without_provenance_field(self):
        # Emulate a pre-field payload: strip boost_seen from every serialized builder's
        # extra. A WRITTEN boost still has its persisted boost context, so the restored
        # episode is still excluded from baselines via the context cross-check (req 7).
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        _fcycle(rt, 0, 18.0); _fcycle(rt, 1, 19.0)
        _fcycle(rt, 2, 19.5, dt=DecisionType.BOOST, applied=1.0)
        payload = rt._build_payload()
        for zdata in payload["zones"].values():
            for bstate in (zdata.get("pipeline", {}).get("builders", {}) or {}).values():
                if isinstance(bstate, dict) and isinstance(bstate.get("extra"), dict):
                    bstate["extra"].pop("boost_seen", None)
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL)
        rt2._restore_payload(payload)
        for i, t in enumerate([20.0, 20.6, 21.0, 21.0, 21.0], start=3):
            _fcycle(rt2, i, t, heating=(t < 21.0))
        assert rt2._zone(_ZONE).baseline_store.size == 0

    def test_13_malformed_provenance_does_not_create_false_baseline(self):
        # A corrupt builder state must not resurrect an episode as a clean baseline.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        _fcycle(rt, 0, 18.0); _fcycle(rt, 1, 19.0)
        _fcycle(rt, 2, 19.5, dt=DecisionType.BOOST, applied=1.0)
        payload = rt._build_payload()
        for zdata in payload["zones"].values():
            builders = zdata.get("pipeline", {}).get("builders", {})
            if "outcome" in builders:
                builders["outcome"] = {"active": "not-a-bool", "extra": "not-a-dict"}
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL)
        rt2._restore_payload(payload)
        for i, t in enumerate([20.0, 20.6, 21.0, 21.0, 21.0], start=3):
            _fcycle(rt2, i, t, heating=(t < 21.0))
        assert rt2._zone(_ZONE).baseline_store.size == 0


class TestMultiZoneProvenanceIsolation:
    def test_14_zones_isolated(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        temps = [18.0, 18.8, 19.6, 20.4, 21.0, 21.0, 21.0, 21.0]
        for i, t in enumerate(temps):
            _fcycle(rt, i, t, zone="A",
                    dt=(DecisionType.BOOST if i == 2 else DecisionType.NORMAL),
                    applied=(1.0 if i == 2 else 0.0), heating=(t < 21.0))
            _fcycle(rt, i, t, zone="B", heating=(t < 21.0))
        assert rt._zone("A").baseline_store.size == 0
        assert rt._zone("B").baseline_store.size == 1
