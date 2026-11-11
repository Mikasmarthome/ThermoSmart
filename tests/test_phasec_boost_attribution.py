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
        device_control_type="setpoint", unreliable=False):
    """A dispatch record whose REALIZED effective offset is `effective`."""
    return BoostDispatchRecord(
        decision_id="x", boost_candidate_c=max(effective or 0.0, 1.0),
        boost_applied_c=effective, baseline_setpoint_c=base,
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
           device_control_type="setpoint", unreliable=False, window=None):
    ts = (T0 + timedelta(minutes=i * 5)).isoformat()
    setp = target + (3.0 if heating else -4.0)
    if effective is not None:
        bd = _bd(effective, status=status, base=target, written=written, held=held,
                 device_control_type=device_control_type, unreliable=unreliable)
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

    def test_7_write_then_failed_not_flagged(self):
        # A failed cycle forms no context, so the realized range stays at the write's 1.0.
        _, _, cap = _run({0: dict(effective=1.0),
                          1: dict(effective=0.0, status="failed")})
        assert _has_offset_changed(cap) is False

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

    def test_10_offset_change_across_target_jump_flagged(self):
        # Mixed offsets under target 21, then a target jump to 22. The episode (o_target 21)
        # closes under the old decision carrying its tracked range -> flagged.
        cap: list = []
        rt = runtime(mode=LearningRuntimeMode.CONTROL,
                     episode_sink=lambda ep: cap.append(ep))
        _cycle(rt, 0, 18.0, effective=1.0, dt=DecisionType.BOOST, target=21.0)
        _cycle(rt, 1, 19.0, effective=1.0, dt=DecisionType.BOOST, target=21.0)
        _cycle(rt, 2, 20.0, effective=0.5, dt=DecisionType.BOOST, target=21.0)
        for i, t in enumerate([21.0, 21.0, 21.0], start=3):
            _cycle(rt, i, t, target=21.0, heating=False)
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
