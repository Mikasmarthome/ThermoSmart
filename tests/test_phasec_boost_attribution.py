"""Phase C — R1: mixed realized-Boost-offset attribution within one learning episode.

The pending BoostUpdateContext was overwritten last-write-wins each boost cycle, so an
episode whose realized (post-dispatch effective) offset changed — e.g. 1.0 -> 1.0 -> 0.5 °C
— was attributed entirely to the FINAL offset (0.5) as an unambiguous full-reliability
sample. The fix tracks the realized-offset min/max on the (persisted) pending context and,
when the range exceeds the quantisation/ambiguity tolerance, tags the closing episode with
the BLOCKING BOOST_OFFSET_CHANGED confounder so the boost model classifies it
BOOST_CONFOUNDED (stored, reduced reliability, NO factor movement) instead.

The important regressions fail on origin/develop (ddf9cb9): there the realized offset is not
tracked across cycles and BOOST_OFFSET_CHANGED is not a known BLOCKING confounder.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.episode_schemas import (
    ControllerKind, EpisodeReason, OutcomeEpisode, Trajectory, TrajectoryPoint)
from custom_components.thermosmart.learning.models.boost import (
    BoostOutcomeClass, BoostParameters, BoostUpdateContext, classify_boost_outcome)
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput, DecisionType, LearningRuntimeMode, RuntimeCycleInput,
    ScheduleTarget)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntime
from tests.helpers_runtime_scenarios import COMFORT, T0, runtime

_ZONE = "lz"
_TARGET = 21.0
_TEMPS = [18.0, 18.6, 19.2, 19.8, 20.4, 21.0, 21.0, 21.0, 21.0]


# ── builders ──────────────────────────────────────────────────────────────────

def _bd(effective, *, status="fully_succeeded", base=_TARGET, written=None, held=(),
        device_control_type="setpoint", unreliable=False, approved=None):
    """A dispatch record whose REALIZED effective offset is `effective`. `approved` is the
    PRE-dispatch APPROVED offset (boost_applied_c); defaults to `effective`. A failed boost
    is modelled as approved>0 with effective 0.0 (boost intended, dispatch did not apply)."""
    _approved = effective if approved is None else approved
    return BoostDispatchRecord(
        decision_id="x", boost_candidate_c=max(_approved or 0.0, 1.0),
        boost_applied_c=_approved, baseline_setpoint_c=base,
        final_setpoint_c=base + (effective or 0.0), dispatch_status=status,
        effective_offset_c=effective,
        effective_setpoints=tuple(written if written is not None
                                  else ((base + effective,) if effective and effective > 0
                                        else ())),
        held_effective_setpoints=tuple(held),
        any_device_boosted=bool((effective or 0.0) > 0.05 or held),
        device_control_type=device_control_type, targets_total=1,
        persistence_unreliable=unreliable)


def _cycle(rt, i, temp, *, effective=None, dt=DecisionType.NORMAL, status="fully_succeeded",
           heating=True, target=_TARGET, zone=_ZONE, written=None, held=(),
           device_control_type="setpoint", unreliable=False, window=None, approved=None):
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = target + (3.0 if heating else -4.0)
    if effective is not None:
        bd = _bd(effective, status=status, base=target, written=written, held=held,
                 device_control_type=device_control_type, unreliable=unreliable,
                 approved=approved)
    else:
        bd = _bd(0.0, status="fully_succeeded", base=target)   # clean record
    inp = RuntimeCycleInput(
        zone_id=zone, ts=ts, target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK), window_open=window,
        schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=target),
        controller_decision=ControllerDecisionInput(
            decision_type=dt, target_c=target, trv_setpoint_c=setp,
            boost_active=(dt is DecisionType.BOOST)),
        boost_dispatch=bd)
    return rt.run_cycle(inp)


def _run(offsets, *, zone=_ZONE, target=_TARGET, temps=None, device_control_type="setpoint"):
    """Drive a reaching episode. `offsets` maps cycle index -> dict(effective=…, status=…,
    held=…) for a BOOST cycle; other cycles are clean NORMAL. Captures emitted episodes."""
    cap: list = []
    rt = runtime(mode=LearningRuntimeMode.CONTROL, episode_sink=lambda ep: cap.append(ep))
    zr = rt._zone(zone)
    for i, t in enumerate(temps or _TEMPS):
        spec = offsets.get(i)
        if spec is not None:
            _cycle(rt, i, t, dt=DecisionType.BOOST, heating=(t < target), zone=zone,
                   target=target, device_control_type=device_control_type, **spec)
        else:
            _cycle(rt, i, t, heating=(t < target), zone=zone, target=target)
    return rt, zr, cap


def _outcomes(cap):
    return [e for e in cap if isinstance(e, OutcomeEpisode)]


def _has_offset_changed(cap):
    outs = _outcomes(cap)
    assert outs, "expected at least one closed OutcomeEpisode"
    return any("boost_offset_changed" in (o.confounder_flags or ()) for o in outs)


# ── pending-context min/max tracker (unit) ────────────────────────────────────

class TestRealizedOffsetTracker:
    def test_min_max_merged_across_cycles(self):
        rt, zr, _ = _run({0: dict(effective=1.0), 1: dict(effective=1.0),
                          2: dict(effective=0.5)},
                         temps=[18.0, 18.6, 19.2])   # keep episode open (no reach)
        ctxs = list(zr.pending_boost_contexts.values())
        assert len(ctxs) == 1
        assert ctxs[0].realized_offset_min_c == pytest.approx(0.5)
        assert ctxs[0].realized_offset_max_c == pytest.approx(1.0)

    def test_uniform_min_equals_max(self):
        rt, zr, _ = _run({0: dict(effective=1.0), 1: dict(effective=1.0)},
                         temps=[18.0, 18.6])
        ctx = list(zr.pending_boost_contexts.values())[0]
        assert ctx.realized_offset_min_c == pytest.approx(1.0)
        assert ctx.realized_offset_max_c == pytest.approx(1.0)

    def test_boost_offset_mixed_helper(self):
        mixed = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=0.5, realized_offset_max_c=1.0)
        uniform = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=1.0, realized_offset_max_c=1.0)
        within = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=1.0, realized_offset_max_c=1.2)   # 0.2 < 0.4 tol
        legacy = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0)  # None/None
        assert LearningRuntime._boost_offset_mixed(mixed) is True
        assert LearningRuntime._boost_offset_mixed(uniform) is False
        assert LearningRuntime._boost_offset_mixed(within) is False
        assert LearningRuntime._boost_offset_mixed(legacy) is False


# ── model-level classification (unit) ─────────────────────────────────────────

class TestConfounderClassification:
    def _classify(self, flags):
        return classify_boost_outcome(
            dispatch_status="fully_succeeded", authority="live_record",
            confounder_flags=flags, attribution=0.9, has_temp=True, overshoot_c=0.0,
            has_baseline=True, relative_benefit=0.3, target_reached=True,
            params=BoostParameters(), is_partial=False,
            overshoot_stable=False, target_stable=True)

    def test_boost_offset_changed_is_confounded(self):
        # REGRESSION (fails on ddf9cb9: the flag is unknown -> DEGRADING, not BLOCKING, so
        # the sample is NOT confounded and would move the factor).
        assert self._classify(("boost_offset_changed",)) is BoostOutcomeClass.BOOST_CONFOUNDED

    def test_clean_flags_not_confounded(self):
        assert self._classify(()) is not BoostOutcomeClass.BOOST_CONFOUNDED


# ── end-to-end attribution via the runtime (episode_sink) ─────────────────────

class TestEpisodeAttribution:
    def test_1_uniform_offset_not_flagged(self):
        _, _, cap = _run({0: dict(effective=1.0), 1: dict(effective=1.0),
                          2: dict(effective=1.0)})
        assert _has_offset_changed(cap) is False

    def test_2_material_step_down_flagged(self):
        # REGRESSION: 1.0 -> 1.0 -> 0.5 must be flagged (fails on ddf9cb9).
        _, _, cap = _run({0: dict(effective=1.0), 1: dict(effective=1.0),
                          2: dict(effective=0.5)})
        assert _has_offset_changed(cap) is True

    def test_3_material_step_up_flagged(self):
        _, _, cap = _run({0: dict(effective=0.5), 1: dict(effective=0.5),
                          2: dict(effective=1.0)})
        assert _has_offset_changed(cap) is True

    def test_4_within_tolerance_not_flagged(self):
        # Sub-tolerance drift (0.2 < 0.4 ambiguity floor) is not a material change.
        _, _, cap = _run({0: dict(effective=1.0), 1: dict(effective=1.1),
                          2: dict(effective=1.2)})
        assert _has_offset_changed(cap) is False

    def test_5_write_then_held_only_not_flagged(self):
        # Held-only cycles form NO new context (already_satisfied), so they do not change
        # the realized range -> a single-write episode stays uniform.
        _, _, cap = _run({0: dict(effective=1.0),
                          1: dict(effective=1.0, status="already_satisfied", held=(22.0,)),
                          2: dict(effective=1.0, status="already_satisfied", held=(22.0,))})
        assert _has_offset_changed(cap) is False

    def test_6_write_then_partial_material_flagged(self):
        # write 1.0 then a partial dispatch realizing 0.5 -> mixed -> flagged.
        _, _, cap = _run({0: dict(effective=1.0),
                          1: dict(effective=0.5, status="partially_succeeded")})
        assert _has_offset_changed(cap) is True

    def test_7_write_then_failed_boost_flagged(self):
        # F2: a later FAILED boost dispatch (boost INTENDED, approved>0, but effective 0.0)
        # after an earlier successful boost realizes 0.0 -> the episode range [0.0, 1.0] is
        # material -> flagged -> fail closed. (On ddf9cb9/d2314fc a failed cycle left the
        # earlier full-reliability context intact and permitted a factor update.)
        _, _, cap = _run({0: dict(effective=1.0),
                          1: dict(effective=0.0, approved=1.0, status="failed")})
        assert _has_offset_changed(cap) is True

    def test_8_multi_trv_still_flagged_and_offset_change_coexists(self):
        # A divergent held device (multi-TRV ambiguity) plus a changing realized offset:
        # both confounders attach; multi-TRV behavior is unchanged.
        _, _, cap = _run({0: dict(effective=1.0, written=(22.0,)),
                          1: dict(effective=0.5, written=(23.0,), held=(21.0,))})
        outs = _outcomes(cap)
        flags = set().union(*[set(o.confounder_flags or ()) for o in outs])
        assert "boost_offset_changed" in flags
        assert any("multi_trv" in f for f in flags)

    def test_9_restart_preserves_tracked_range(self):
        # Serialize mid-episode after a mixed boost, restore, finish -> still flagged.
        cap1: list = []
        rt = runtime(mode=LearningRuntimeMode.CONTROL,
                     episode_sink=lambda ep: cap1.append(ep))
        _cycle(rt, 0, 18.0, effective=1.0, dt=DecisionType.BOOST)
        _cycle(rt, 1, 18.6, effective=1.0, dt=DecisionType.BOOST)
        _cycle(rt, 2, 19.2, effective=0.5, dt=DecisionType.BOOST)
        payload = rt._build_payload()
        cap2: list = []
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL,
                      episode_sink=lambda ep: cap2.append(ep))
        rt2._restore_payload(payload)
        for i, t in enumerate([19.8, 20.4, 21.0, 21.0, 21.0], start=3):
            _cycle(rt2, i, t, heating=(t < _TARGET))
        assert _has_offset_changed(cap2) is True

    def test_10_genuine_target_jump_mixed_offset_flagged(self):
        # Genuine target change: boost 1.0 under target 21, the target JUMPS to 23 and a
        # boost 0.5 is applied under the new target. One OutcomeEpisode (opened at target 21)
        # spans both decisions and closes under the opening id; the EPISODE-scoped range
        # [0.5, 1.0] is material -> flagged. (On d2314fc the decision-keyed context tracker
        # closed under the opening decision with min=max=1.0 and did NOT flag it — the F1
        # episode-ownership gap.)
        cap: list = []
        rt = runtime(mode=LearningRuntimeMode.CONTROL,
                     episode_sink=lambda ep: cap.append(ep))
        _cycle(rt, 0, 18.0, effective=1.0, dt=DecisionType.BOOST, target=21.0)
        _cycle(rt, 1, 19.0, effective=1.0, dt=DecisionType.BOOST, target=21.0)
        _cycle(rt, 2, 20.0, effective=0.5, dt=DecisionType.BOOST, target=23.0)  # jump
        for i, t in enumerate([21.0, 22.0, 23.0, 23.0, 23.0], start=3):
            _cycle(rt, i, t, target=23.0, heating=(t < 23.0))
        assert _has_offset_changed(cap) is True

    def test_11_unreliable_evidence_fails_closed_no_crash(self):
        # An unreliable cycle fails closed (Phase B); the R1 tracker must not interfere or
        # crash, and the episode is not stored as a clean baseline.
        rt, zr, cap = _run({0: dict(effective=1.0),
                            1: dict(effective=1.0, unreliable=True),
                            2: dict(effective=0.5)})
        assert zr.baseline_store.size == 0   # unreliable -> not a clean baseline

    def test_12_clean_baseline_still_eligible(self):
        # A purely non-boost episode still stores a baseline; the R1 change is Boost-only.
        rt, zr, cap = _run({})
        assert zr.baseline_store.size == 1

    def test_13_direct_valve_zone_offset_change_flagged(self):
        _, _, cap = _run({0: dict(effective=1.0), 1: dict(effective=0.5)},
                         device_control_type="direct_valve")
        assert _has_offset_changed(cap) is True

    def test_14_no_control_commands_from_learning(self):
        # The learning path never dispatches: a mixed-offset sequence produces no control
        # effect / TRV writes.
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        for i, t in enumerate(_TEMPS):
            spec = {0: dict(effective=1.0), 1: dict(effective=1.0),
                    2: dict(effective=0.5)}.get(i)
            if spec is not None:
                r = _cycle(rt, i, t, dt=DecisionType.BOOST, heating=(t < _TARGET), **spec)
            else:
                r = _cycle(rt, i, t, heating=(t < _TARGET))
            assert r.control_commands == ()
            assert r.has_control_effect is False


# ── persistence compatibility ─────────────────────────────────────────────────

class TestPersistenceCompatibility:
    def test_legacy_context_without_range_fields_restores(self):
        # A context serialized before R1 (no realized_offset_* keys) restores with None and
        # is treated as uniform (no flag, no crash).
        d = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            boost_applied_c=1.0, authority="live_record").to_dict()
        d.pop("realized_offset_min_c", None)
        d.pop("realized_offset_max_c", None)
        ctx = BoostUpdateContext.from_dict(d)
        assert ctx.realized_offset_min_c is None
        assert ctx.realized_offset_max_c is None
        assert LearningRuntime._boost_offset_mixed(ctx) is False

    def test_range_round_trips(self):
        ctx = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=0.5, realized_offset_max_c=1.0)
        ctx2 = BoostUpdateContext.from_dict(ctx.to_dict())
        assert ctx2.realized_offset_min_c == pytest.approx(0.5)
        assert ctx2.realized_offset_max_c == pytest.approx(1.0)


# ── F1: episode-scoped detection across decision transitions ──────────────────

def _run_cross_decision(c0_dt, c0_eff, c1_dt, c1_eff, *, c0_target=_TARGET,
                        c1_target=_TARGET):
    """Open an OutcomeEpisode on cycle 0 (decision A, a boost), transition to decision B on
    cycle 1 (a boost), then reach. One episode spans both decisions."""
    cap: list = []
    rt = runtime(mode=LearningRuntimeMode.CONTROL, episode_sink=lambda ep: cap.append(ep))
    zr = rt._zone(_ZONE)
    _cycle(rt, 0, 18.0, effective=c0_eff, dt=c0_dt, target=c0_target)
    _cycle(rt, 1, 19.0, effective=c1_eff, dt=c1_dt, target=c1_target)
    for i, t in enumerate([19.8, 20.4, 21.0, 21.0, 21.0], start=2):
        _cycle(rt, i, t, target=c1_target, heating=(t < c1_target))
    return rt, zr, cap


class TestF1CrossDecisionEpisode:
    def test_f1_preheat_to_normal_down_flagged(self):
        # REGRESSION (fails on d2314fc): PREHEAT boost 1.0 then NORMAL boost 0.5 in ONE
        # episode. The decision-keyed context tracker closes under the opening (PREHEAT)
        # decision with min=max=1.0 and misses the 0.5; the episode-scoped tracker spans
        # both decisions -> range [0.5, 1.0] -> flagged.
        _, _, cap = _run_cross_decision(DecisionType.PREHEAT, 1.0, DecisionType.NORMAL, 0.5)
        assert _has_offset_changed(cap) is True

    def test_f1_preheat_to_normal_up_flagged(self):
        _, _, cap = _run_cross_decision(DecisionType.PREHEAT, 0.5, DecisionType.NORMAL, 1.0)
        assert _has_offset_changed(cap) is True

    def test_f1_uniform_across_decisions_not_flagged(self):
        # Same realized offset across a PREHEAT->NORMAL transition -> clean.
        _, _, cap = _run_cross_decision(DecisionType.PREHEAT, 1.0, DecisionType.NORMAL, 1.0)
        assert _has_offset_changed(cap) is False

    def test_f1_same_decision_mixed_flagged(self):
        # Same-decision mixed offsets remain flagged (original R1 path preserved).
        _, _, cap = _run({0: dict(effective=1.0), 1: dict(effective=1.0),
                          2: dict(effective=0.5)})
        assert _has_offset_changed(cap) is True

    def test_f1_cross_decision_restart_flagged(self):
        # Serialize mid-episode after the PREHEAT boost + NORMAL boost, restore, reach.
        # The episode-scoped range is persisted in builder state -> still flagged.
        cap1: list = []
        rt = runtime(mode=LearningRuntimeMode.CONTROL,
                     episode_sink=lambda ep: cap1.append(ep))
        _cycle(rt, 0, 18.0, effective=1.0, dt=DecisionType.PREHEAT)
        _cycle(rt, 1, 19.0, effective=0.5, dt=DecisionType.NORMAL)
        payload = rt._build_payload()
        cap2: list = []
        rt2 = runtime(mode=LearningRuntimeMode.CONTROL,
                      episode_sink=lambda ep: cap2.append(ep))
        rt2._restore_payload(payload)
        for i, t in enumerate([19.8, 20.4, 21.0, 21.0, 21.0], start=2):
            _cycle(rt2, i, t, heating=(t < _TARGET))
        assert _has_offset_changed(cap2) is True

    def test_f1_abort_then_clean_episode_isolated(self):
        # A mixed-offset episode aborted by a window-open; a later CLEAN uniform-boost
        # episode under a new target must NOT inherit the stale range (reset on abort).
        cap: list = []
        rt = runtime(mode=LearningRuntimeMode.CONTROL,
                     episode_sink=lambda ep: cap.append(ep))
        _cycle(rt, 0, 18.0, effective=1.0, dt=DecisionType.BOOST, target=21.0)
        _cycle(rt, 1, 19.0, effective=0.5, dt=DecisionType.BOOST, target=21.0)
        _cycle(rt, 2, 19.0, target=21.0, window=True)   # abort
        for i, t in enumerate([18.0, 18.8, 19.6, 20.4, 22.0, 22.0, 22.0, 22.0], start=3):
            eff = 1.0 if i in (3, 4, 5) else None
            _cycle(rt, i, t, effective=eff, dt=(DecisionType.BOOST if eff else DecisionType.NORMAL),
                   target=22.0, heating=(t < 22.0))
        # the clean uniform-boost episode (all 1.0) must not be flagged
        post_abort = [e for e in _outcomes(cap) if e.target == 22.0]
        assert post_abort, "expected a post-abort episode at target 22"
        assert all("boost_offset_changed" not in (e.confounder_flags or ())
                   for e in post_abort)


# ── F2: later degraded dispatch (failed vs held vs release) ───────────────────

class TestF2LaterDegraded:
    def test_f2_write_then_failed_fails_closed(self):
        # REGRESSION: write 1.0 -> failed (intended, effective 0.0) -> flagged.
        _, _, cap = _run({0: dict(effective=1.0),
                          1: dict(effective=0.0, approved=1.0, status="failed")})
        assert _has_offset_changed(cap) is True

    def test_f2_write_then_held_same_offset_legitimate(self):
        # Held-only at the SAME realized offset is legitimate -> not flagged.
        _, _, cap = _run({0: dict(effective=1.0),
                          1: dict(effective=1.0, status="already_satisfied", held=(22.0,))})
        assert _has_offset_changed(cap) is False

    def test_f2_write_then_release_legitimate(self):
        # Normal release: the later cycle has NO boost intent (approved 0) -> not tracked,
        # not flagged; the earlier boost is legitimately attributed.
        _, _, cap = _run({0: dict(effective=1.0)})   # cycles 1+ are clean NORMAL (release)
        assert _has_offset_changed(cap) is False

    def test_f2_write_then_partial_lower_flagged(self):
        _, _, cap = _run({0: dict(effective=1.0),
                          1: dict(effective=0.5, approved=1.0,
                                  status="partially_succeeded")})
        assert _has_offset_changed(cap) is True


# ── F3: corrupt persisted ranges fail closed ─────────────────────────────────

class TestF3PersistenceHardening:
    def test_inverted_range_fails_closed(self):
        ctx = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=1.0, realized_offset_max_c=0.5)   # inverted
        assert LearningRuntime._boost_offset_mixed(ctx) is True

    def test_one_sided_range_fails_closed(self):
        only_min = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=1.0, realized_offset_max_c=None)
        only_max = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=None, realized_offset_max_c=1.0)
        assert LearningRuntime._boost_offset_mixed(only_min) is True
        assert LearningRuntime._boost_offset_mixed(only_max) is True

    def test_bool_range_fails_closed(self):
        # A bool is not a numeric offset (is_finite(True) is True, so it survives the
        # context's finite check) -> the consumer must fail closed.
        ctx = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=True, realized_offset_max_c=True)
        assert LearningRuntime._boost_offset_mixed(ctx) is True

    def test_valid_and_legacy_unaffected(self):
        legacy = BoostUpdateContext(source_episode_id="e", decision_id="d",
                                    requested_offset_c=1.0)
        valid_uniform = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=1.0, realized_offset_max_c=1.0)
        assert LearningRuntime._boost_offset_mixed(legacy) is False
        assert LearningRuntime._boost_offset_mixed(valid_uniform) is False

    def test_corrupt_restore_does_not_become_clean_sample(self):
        # A persisted context with an inverted range restores and, at episode consume, is
        # treated as mixed (fail closed) rather than a clean uniform sample.
        d = BoostUpdateContext(
            source_episode_id="e", decision_id="d", requested_offset_c=1.0,
            realized_offset_min_c=0.5, realized_offset_max_c=1.0).to_dict()
        d["realized_offset_min_c"] = 1.0
        d["realized_offset_max_c"] = 0.5      # inverted after "corruption"
        ctx = BoostUpdateContext.from_dict(d)
        assert LearningRuntime._boost_offset_mixed(ctx) is True

    def test_episode_scoped_range_fail_closed(self):
        # The same fail-closed guard applies to the episode-scoped range.
        ep_inverted = SimpleNamespaceEpisode(1.0, 0.5)
        ep_onesided = SimpleNamespaceEpisode(None, 1.0)
        ep_bool = SimpleNamespaceEpisode(True, True)
        ep_legacy = SimpleNamespaceEpisode(None, None)
        ep_uniform = SimpleNamespaceEpisode(1.0, 1.0)
        assert LearningRuntime._episode_offset_mixed(ep_inverted) is True
        assert LearningRuntime._episode_offset_mixed(ep_onesided) is True
        assert LearningRuntime._episode_offset_mixed(ep_bool) is True
        assert LearningRuntime._episode_offset_mixed(ep_legacy) is False
        assert LearningRuntime._episode_offset_mixed(ep_uniform) is False


class SimpleNamespaceEpisode:
    def __init__(self, lo, hi):
        self.boost_realized_offset_min_c = lo
        self.boost_realized_offset_max_c = hi


# ── factor-update proof (model level) ─────────────────────────────────────────

class TestFactorUpdateProof:
    def _model_update(self, *, confounders=()):
        from custom_components.thermosmart.learning.models import BoostModel
        from tests.helpers_boost import boost_episode, boost_context_with_comparison
        m = BoostModel(_ZONE)
        ep = boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0, 21.0],
                           reason=EpisodeReason.REACHED, confounder_flags=confounders)
        ctx = boost_context_with_comparison(ep, requested_offset_c=1.5)
        m.update(ep, ctx)
        return m._state.general.effective_factor

    def test_uniform_high_reliability_moves_factor(self):
        # A clean, uniform, high-reliability REACHED boost moves the learned factor.
        dim = self._model_update(confounders=())
        assert dim.effective_n > 0.0

    def test_offset_changed_blocks_factor_move(self):
        # REGRESSION (fails on d2314fc: the flag is DEGRADING there, so the factor moves):
        # the SAME episode tagged boost_offset_changed is BOOST_CONFOUNDED -> no movement.
        dim = self._model_update(confounders=("boost_offset_changed",))
        assert dim.effective_n == 0.0


# ── no control effect from the learning path (scenarios 15/16) ────────────────

class TestNoControlEffect:
    def test_mixed_and_failed_sequences_issue_no_control(self):
        rt = runtime(mode=LearningRuntimeMode.CONTROL)
        seq = {0: dict(effective=1.0), 1: dict(effective=0.5),
               2: dict(effective=0.0, approved=1.0, status="failed")}
        for i, t in enumerate(_TEMPS):
            spec = seq.get(i)
            if spec is not None:
                r = _cycle(rt, i, t, dt=DecisionType.BOOST, heating=(t < _TARGET), **spec)
            else:
                r = _cycle(rt, i, t, heating=(t < _TARGET))
            assert r.control_commands == ()
            assert r.has_control_effect is False
