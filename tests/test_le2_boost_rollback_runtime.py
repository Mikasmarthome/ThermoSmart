"""B2b-5/5a: scoped degradation/rollback wired through the real BoostModel.update + predict."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
from custom_components.thermosmart.learning.models import (
    BoostLearningReadiness as R, BoostModel, BoostPredictionContext)
from tests.helpers_boost import boost_context_with_comparison, boost_episode

_OVERSHOOT = [18.0, 19.0, 20.0, 21.0, 22.5, 23.0, 23.0]   # stable overshoot, ~2.0 over target
_SUCCESS = [18.0, 19.5, 20.5, 21.0, 21.0]


def _feed(m, vals, did, *, offset=1.5):
    ep = boost_episode(did, vals, decision_id=did, reason=EpisodeReason.REACHED)
    m.update(ep, boost_context_with_comparison(ep, requested_offset_c=offset))


def _bucket(m):
    """The single context bucket created by the default helper context."""
    bs = m._state.degradation.buckets
    return next(iter(bs.values())) if bs else None


def _ctx(now="2025-01-01T00:10:00+00:00"):
    # mirrors the helper's frozen decision context so predict selects the same bucket
    return BoostPredictionContext(start_deficit_c=3.0, decision_type="ts", intensity_c=1.5,
                                  now_ts=now)


class TestBucketScopedRollback:
    def test_overshoot_reaches_bucket_tracker(self):
        m = BoostModel("lz")
        _feed(m, _OVERSHOOT, "d1")
        b = _bucket(m)
        assert b is not None and b.overshoot_evidence > 0.0 and b.adaptive_outcomes >= 1

    def test_extreme_overshoot_rolls_back_bucket_not_general(self):
        m = BoostModel("lz")
        _feed(m, _OVERSHOOT, "d1")
        b = _bucket(m)
        assert b.readiness == R.ROLLED_BACK and b.rollback_count == 1
        # general is NOT rolled back by a single local overshoot
        assert m._state.degradation.general.readiness != R.ROLLED_BACK
        assert m._state.degradation.general.rollback_count == 0

    def test_confounded_overshoot_no_rollback(self):
        m = BoostModel("lz")
        ep = boost_episode("d1", _OVERSHOOT, decision_id="d1", reason=EpisodeReason.REACHED,
                           confounder_flags=("additional_heat_source",))
        m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        b = _bucket(m)
        assert b is None or (b.readiness != R.ROLLED_BACK and b.overshoot_evidence == 0.0)


class TestPredictionScope:
    def _eligible_bucket(self, m):
        for i in range(25):
            _feed(m, _SUCCESS, f"s{i}")

    def test_bucket_rollback_falls_back_to_general(self):
        # B2b-5a/7: a rolled-back bucket with an ELIGIBLE general falls back to general (the
        # impaired bucket factor is never used).
        m = BoostModel("lz")
        self._eligible_bucket(m)                 # selectable, eligible bucket with known-good
        _feed(m, _OVERSHOOT, "bad1")
        _feed(m, _OVERSHOOT, "bad2")             # 2 reliable overshoots -> bucket rollback
        b = _bucket(m)
        assert b.readiness == R.ROLLED_BACK
        rec = m.predict_boost_factor(_ctx())
        assert rec.selected_scope == "general_fallback" and rec.fallback_used

    def test_eligible_bucket_factor_usable(self):
        m = BoostModel("lz")
        self._eligible_bucket(m)
        rec = m.predict_boost_factor(_ctx())
        assert rec.readiness == R.ELIGIBLE and rec.factor_usable
        assert rec.eligibility_reason == "bucket_eligible"


class TestKnownGoodRestore:
    def test_success_then_overshoot_restores_bucket_known_good(self):
        m = BoostModel("lz")
        for i in range(25):
            _feed(m, _SUCCESS, f"s{i}")
        b = _bucket(m)
        kg = b.known_good
        assert kg is not None and kg.factor > 0.0
        _feed(m, _OVERSHOOT, "bad1")
        _feed(m, _OVERSHOOT, "bad2")
        b2 = _bucket(m)
        assert b2.readiness == R.ROLLED_BACK
        # the bucket factor rolled back to its own known good (not zero, not general)
        bkey = next(iter(m._state.degradation.buckets))
        assert m._state.buckets[bkey].effective_factor.value == pytest.approx(kg.factor, rel=1e-6)


class TestRestart:
    def test_scoped_state_round_trips(self):
        m = BoostModel("lz")
        _feed(m, _OVERSHOOT, "d1")
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        b, b2 = _bucket(m), _bucket(m2)
        assert b2.readiness == b.readiness and b2.rollback_count == b.rollback_count
        assert b2.cooldown_until_ts == b.cooldown_until_ts
        assert b2.overshoot_evidence == pytest.approx(b.overshoot_evidence, rel=1e-9)

    def test_legacy_single_state_migrates_to_general(self):
        # a B2b-5 (pre-scoped) persisted single degradation dict -> loaded as the general scope
        m = BoostModel("lz")
        blob = dict(m.serialize_state())
        blob["degradation"] = {"readiness": "degraded", "degradation_score": 2.0,
                               "component_version": 1}
        m.deserialize_state(blob)
        assert m._state.degradation.general.readiness == R.DEGRADED

    def test_unknown_version_degrades_safe(self):
        m = BoostModel("lz")
        blob = dict(m.serialize_state())
        blob["degradation"] = {"scoped": True, "component_version": 999}
        m.deserialize_state(blob)
        assert m._state.degradation.general.readiness == R.INSUFFICIENT_DATA

    def test_rollback_not_double_processed_on_restart(self):
        m = BoostModel("lz")
        _feed(m, _OVERSHOOT, "d1")
        rc1 = _bucket(m).rollback_count
        blob = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(blob)
        ep = boost_episode("d1b", _OVERSHOOT, decision_id="d1", reason=EpisodeReason.REACHED)
        m2.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        assert _bucket(m2).rollback_count == rc1     # deduped, no second rollback


class TestDiagnostics:
    def test_diagnostics_privacy_safe_scoped(self):
        m = BoostModel("lz")
        _feed(m, _OVERSHOOT, "d1")
        diag = m.degradation_diagnostics()
        assert diag["scope"] == "general" and "buckets" in diag
        assert diag["active_bucket_states"] >= 1 and diag["degraded_bucket_count"] >= 1
        # bucket keys are hashed in the export (no raw context key / entity id)
        for h in diag["buckets"]:
            assert "|" not in h and "def" not in h and "/" not in h
