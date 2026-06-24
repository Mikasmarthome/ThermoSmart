"""B2b-4c: stable target / overshoot observation driving the boost outcome + exactly-once."""
from __future__ import annotations

from dataclasses import replace

import pytest

from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
from custom_components.thermosmart.learning.models import (
    BoostModel, BoostOutcomeClass as C, BoostOutcomeLifecycleState)
from tests.helpers_boost import boost_context_with_comparison, boost_episode


def _cls(model, ep, ctx):
    model.update(ep, ctx)
    s = model._state.recent_samples + model._state.diagnostic_samples
    return s[-1].outcome_class if s else None


class TestLifecycleContract:
    def test_states_exist(self):
        for n in ("HEATING_ACTIVE", "TARGET_STABILITY_PENDING",
                  "TARGET_REACHED_PENDING_OBSERVATION", "FINALIZED", "INTERRUPTED", "EXPIRED"):
            assert hasattr(BoostOutcomeLifecycleState, n)


class TestTargetStabilityDrivesClass:
    def test_reached_is_success_eligible(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1",
                           reason=EpisodeReason.REACHED)
        cls = _cls(m, ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        assert cls in (C.BOOST_SUCCESS.value, C.BOOST_NO_EFFECT.value, C.BOOST_UNNECESSARY.value)

    def test_timeout_not_success(self):
        # a non-REACHED (timeout) episode is never SUCCESS/NO_EFFECT/UNNECESSARY
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.0, 19.5, 20.0, 20.2], decision_id="d1",
                           reason=EpisodeReason.TIMEOUT, target=21.0)
        cls = _cls(m, ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        assert cls not in (C.BOOST_SUCCESS.value, C.BOOST_NO_EFFECT.value,
                           C.BOOST_UNNECESSARY.value)


class TestOvershootStabilityDrivesClass:
    def test_sustained_overshoot_is_overshoot(self):
        m = BoostModel("lz")
        # rises to 21 then sustained >21.5 for several samples (>=300 s)
        ep = boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 23.0], decision_id="d1")
        cls = _cls(m, ep, boost_context_with_comparison(ep, requested_offset_c=2.0))
        assert cls == C.BOOST_OVERSHOOT.value

    def test_short_peak_not_overshoot(self):
        m = BoostModel("lz")
        # one brief spike above +0.5 then back in band -> NOT a stable overshoot
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 23.0, 21.0, 21.0, 21.0], decision_id="d1",
                           reason=EpisodeReason.REACHED)
        cls = _cls(m, ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        assert cls != C.BOOST_OVERSHOOT.value


class TestExactlyOnce:
    def test_same_decision_one_update(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="dup")
        ctx = boost_context_with_comparison(ep, requested_offset_c=1.5)
        r1 = m.update(ep, ctx)
        n_after_first = m._state.general.effective_factor.effective_n
        # re-deliver the same decision (retry / duplicate episode / late event)
        ep2 = boost_episode("e1b", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="dup")
        r2 = m.update(ep2, boost_context_with_comparison(ep2, requested_offset_c=1.5))
        assert r1.accepted and not r2.accepted        # second is deduped
        assert m._state.general.effective_factor.effective_n == n_after_first

    def test_dedup_survives_restart(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="dup")
        m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        m2 = BoostModel("lz")
        m2.deserialize_state(m.serialize_state())     # restart: processed ids restored
        ep2 = boost_episode("e1b", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="dup")
        r = m2.update(ep2, boost_context_with_comparison(ep2, requested_offset_c=1.5))
        assert not r.accepted                          # no double update after restart
