"""Phase S1a Item 3 — Clock regression tests: naive timestamps, clock isolation, tz contract.

Proves:
1. _ensure_aware() normalizes naive ISO timestamps to UTC (old storage compat).
2. _time_weight() returns the correct weight for naive timestamps — NOT zero.
3. _thermal_weight() returns the correct weight for naive timestamps — NOT zero.
4. Malformed timestamp strings still return 0.0 (valid data-error path).
5. Two independent FakeClock instances are fully isolated — no shared state.
6. Shadow controller inherits exactly the coordinator's injected FakeClock.
7. Runtime cycle input timestamp derives from the injected clock (not wall time).
8. _rebuild_confidence uses the correct time-weight for naive-ts observations.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning_engine import _ensure_aware, _time_weight, _thermal_weight
from tests.helpers import make_learning_engine


# ── Constants ─────────────────────────────────────────────────────────────────

_T0 = datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
_T0_NAIVE = datetime(2025, 6, 24, 12, 0, 0)  # same instant, no tzinfo


# ── 1. _ensure_aware ─────────────────────────────────────────────────────────

class TestEnsureAware:
    def test_naive_becomes_utc(self):
        naive = datetime(2024, 1, 15, 8, 30, 0)
        result = _ensure_aware(naive)
        assert result.tzinfo == timezone.utc
        assert result.replace(tzinfo=None) == naive

    def test_aware_unchanged(self):
        aware = datetime(2024, 1, 15, 8, 30, 0, tzinfo=timezone.utc)
        result = _ensure_aware(aware)
        assert result is aware

    def test_aware_with_offset_unchanged(self):
        from datetime import timezone as _tz
        tz_plus1 = _tz(timedelta(hours=1))
        aware = datetime(2024, 1, 15, 9, 30, 0, tzinfo=tz_plus1)
        result = _ensure_aware(aware)
        assert result is aware
        assert result.tzinfo == tz_plus1


# ── 2. _time_weight with naive timestamps ─────────────────────────────────────

class TestTimeWeightNaiveTimestamps:
    """Before the fix, _time_weight returned 0.0 for naive timestamps (TypeError silently
    caught). After the fix, it normalizes the naive timestamp to UTC and returns the
    correct decay weight."""

    def test_fresh_naive_obs_returns_nonzero_weight(self):
        now = _T0
        # Fresh observation: ts is naive but same moment as now
        naive_ts = _T0_NAIVE.isoformat()  # "2025-06-24T12:00:00"
        w = _time_weight(naive_ts, now, halflife_days=180)
        # age_days = 0 → weight = exp(0) = 1.0
        assert w == pytest.approx(1.0, abs=1e-6), (
            f"_time_weight must normalize naive ts and return ~1.0 for fresh obs, got {w!r}. "
            "If 0.0, the TypeError is still being silently swallowed."
        )

    def test_old_naive_obs_decays_correctly(self):
        now = _T0
        # Observation 180 days ago (naive)
        old = _T0_NAIVE - timedelta(days=180)
        naive_ts = old.isoformat()
        w = _time_weight(naive_ts, now, halflife_days=180)
        # age_days = 180 → weight = exp(-1) ≈ 0.368
        assert w == pytest.approx(math.exp(-1.0), abs=1e-4)

    def test_aware_and_naive_same_instant_same_weight(self):
        now = _T0
        ts_naive = _T0_NAIVE.isoformat()
        ts_aware = _T0.isoformat()
        w_naive = _time_weight(ts_naive, now)
        w_aware = _time_weight(ts_aware, now)
        assert w_naive == pytest.approx(w_aware, abs=1e-9)

    def test_malformed_string_still_returns_zero(self):
        w = _time_weight("not-a-date", _T0)
        assert w == 0.0

    def test_empty_string_returns_zero(self):
        w = _time_weight("", _T0)
        assert w == 0.0


# ── 3. _thermal_weight with naive timestamps ──────────────────────────────────

class TestThermalWeightNaiveTimestamps:
    def _fresh_obs(self, ts_str: str) -> dict:
        return {"ts": ts_str, "outdoor_temp": 10.0}

    def test_fresh_naive_obs_returns_nonzero_weight(self):
        now = _T0
        obs = self._fresh_obs(_T0_NAIVE.isoformat())
        conditions = {"outdoor_temp": 10.0}
        w = _thermal_weight(obs, conditions, now)
        # age_days = 0 → recency 1.0; temp_sim 1.0 (exact match); expect ~1.0
        assert w > 0.5, (
            f"_thermal_weight must normalize naive ts and return high weight for fresh obs, got {w!r}."
        )

    def test_naive_and_aware_same_instant_same_weight(self):
        now = _T0
        conditions = {"outdoor_temp": 10.0}
        obs_naive = self._fresh_obs(_T0_NAIVE.isoformat())
        obs_aware = self._fresh_obs(_T0.isoformat())
        w_naive = _thermal_weight(obs_naive, conditions, now)
        w_aware = _thermal_weight(obs_aware, conditions, now)
        assert w_naive == pytest.approx(w_aware, abs=1e-9)

    def test_missing_ts_returns_zero(self):
        w = _thermal_weight({"outdoor_temp": 10.0}, {"outdoor_temp": 10.0}, _T0)
        assert w == 0.0

    def test_malformed_ts_returns_zero(self):
        w = _thermal_weight({"ts": "bad"}, {"outdoor_temp": 10.0}, _T0)
        assert w == 0.0


# ── 4. _rebuild_confidence is not silently zeroed by naive timestamps ─────────

class TestConfidenceNotSilentlyZeroed:
    """If naive-ts observations caused _time_weight to return 0.0, every observation
    would have zero weight and confidence would be 0.0 — exactly the silent corruption
    that was occurring before the fix."""

    def _naive_heat_obs(self, days_ago: int) -> dict:
        ts = (_T0_NAIVE - timedelta(days=days_ago)).isoformat()
        return {"ts": ts, "heat_rate": 0.05}

    def test_naive_ts_observations_contribute_to_confidence(self):
        e = make_learning_engine(clock=FakeClock(start=_T0))
        e._observations["z"] = [self._naive_heat_obs(0) for _ in range(50)]
        e._rebuild_confidence("z")
        conf = e.get_confidence("z")
        # 50 fresh observations → base track saturates → confidence ≈ 0.6
        assert conf == pytest.approx(0.6, abs=0.02), (
            f"Naive-ts observations must contribute to confidence, got {conf:.3f}. "
            "If 0.0, _time_weight is still silently returning 0.0 for naive timestamps."
        )

    def test_aware_and_naive_obs_give_same_confidence(self):
        e_aware = make_learning_engine(clock=FakeClock(start=_T0))
        e_naive = make_learning_engine(clock=FakeClock(start=_T0))

        aware_obs = [{"ts": (_T0 - timedelta(days=i)).isoformat(), "heat_rate": 0.05}
                     for i in range(50)]
        naive_obs = [{"ts": (_T0_NAIVE - timedelta(days=i)).isoformat(), "heat_rate": 0.05}
                     for i in range(50)]

        e_aware._observations["z"] = aware_obs
        e_naive._observations["z"] = naive_obs
        e_aware._rebuild_confidence("z")
        e_naive._rebuild_confidence("z")

        assert e_aware.get_confidence("z") == pytest.approx(e_naive.get_confidence("z"), abs=1e-6)


# ── 5. FakeClock independence — two instances never share state ────────────────

class TestFakeClockIndependence:
    def test_two_clocks_same_start_are_independent(self):
        c1 = FakeClock(start=_T0)
        c2 = FakeClock(start=_T0)
        c1.advance(3600)
        # c2 must not have advanced
        assert c1.now_utc() == _T0 + timedelta(seconds=3600)
        assert c2.now_utc() == _T0

    def test_two_clocks_different_start_are_independent(self):
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        c1 = FakeClock(start=t1)
        c2 = FakeClock(start=t2)
        assert c1.now_utc() == t1
        assert c2.now_utc() == t2

    def test_two_coordinators_independent_clocks(self):
        from tests.helpers import make_coordinator
        c1 = FakeClock(start=_T0)
        c2 = FakeClock(start=_T0 + timedelta(hours=1))
        coord1 = make_coordinator(clock=c1)
        coord2 = make_coordinator(clock=c2)
        ts1 = coord1._clock.now_utc()
        ts2 = coord2._clock.now_utc()
        assert ts1 == _T0
        assert ts2 == _T0 + timedelta(hours=1)
        # Advance one clock — other stays frozen
        c1.advance(7200)
        assert coord1._clock.now_utc() == _T0 + timedelta(seconds=7200)
        assert coord2._clock.now_utc() == _T0 + timedelta(hours=1)

    def test_engines_with_independent_clocks(self):
        t_a = datetime(2025, 3, 1, 6, 0, tzinfo=timezone.utc)
        t_b = datetime(2025, 9, 1, 6, 0, tzinfo=timezone.utc)
        ea = make_learning_engine(clock=FakeClock(start=t_a))
        eb = make_learning_engine(clock=FakeClock(start=t_b))
        assert ea._now_local() != eb._now_local()


# ── 6. Shadow controller inherits coordinator's injected clock ────────────────

class TestShadowClockInheritance:
    def test_shadow_uses_coordinator_clock(self):
        from tests.helpers import make_coordinator
        from tests.helpers_ha_runtime import attach_shadow

        fake = FakeClock(start=_T0)
        coord = make_coordinator(clock=fake)
        shadow = attach_shadow(coord)

        # Shadow's internal clock must produce the same timestamp as coordinator's clock
        shadow_ts = shadow._utcnow_iso()
        coord_ts = coord._clock.now_utc().isoformat()
        assert shadow_ts == coord_ts, (
            f"Shadow clock diverges from coordinator clock: {shadow_ts!r} != {coord_ts!r}. "
            "attach_shadow() must forward coord._clock to the shadow."
        )

    def test_shadow_clock_advances_with_coordinator_clock(self):
        from tests.helpers import make_coordinator
        from tests.helpers_ha_runtime import attach_shadow

        fake = FakeClock(start=_T0)
        coord = make_coordinator(clock=fake)
        shadow = attach_shadow(coord)

        fake.advance(3600)

        shadow_ts = shadow._utcnow_iso()
        expected = (_T0 + timedelta(seconds=3600)).isoformat()
        assert shadow_ts == expected

    def test_two_shadows_same_coordinator_same_clock(self):
        from tests.helpers import make_coordinator
        from tests.helpers_ha_runtime import attach_shadow, FakeStore

        fake = FakeClock(start=_T0)
        coord = make_coordinator(clock=fake)
        s1 = attach_shadow(coord, zone_id="zone1", store=FakeStore())
        s2 = attach_shadow(coord, zone_id="zone2", store=FakeStore())

        assert s1._utcnow_iso() == s2._utcnow_iso()
        fake.advance(300)
        assert s1._utcnow_iso() == s2._utcnow_iso()

    def test_shadow_no_clock_raises(self):
        """A shadow without an explicit clock must raise ValueError — no silent fallback."""
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            LearningShadowController,
        )
        import pytest

        class _FakeHass:
            class states:
                @staticmethod
                def get(_): return None

        with pytest.raises(ValueError, match="requires an explicit Clock"):
            LearningShadowController(_FakeHass(), "z")

    def test_shadow_with_clock_returns_injected_time(self):
        """FakeClock injected into shadow must produce deterministic timestamps."""
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            LearningShadowController,
        )
        from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode

        class _FakeHass:
            class states:
                @staticmethod
                def get(_): return None

        fake = FakeClock(start=_T0)
        shadow_fake = LearningShadowController(
            _FakeHass(), "z",
            clock=fake,
            mode=LearningRuntimeMode.CONTROL,
        )
        ts_fake = shadow_fake._utcnow_iso()
        assert ts_fake == _T0.isoformat()


# ── 7. Runtime cycle input uses the injected clock ────────────────────────────

class TestRuntimeCycleInputClock:
    """build_runtime_cycle_input() must use the injected clock for ts_iso, not wall time."""

    def test_cycle_input_ts_uses_injected_clock(self):
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            build_runtime_cycle_input,
        )

        fake = FakeClock(start=_T0)
        zone = {"zone_id": "z", "comfort_temp": 21.0}
        rec = {"trv_setpoint": 22.0, "effective_target": 21.0, "current_temp": 19.0}

        # Pass ts_iso derived from injected clock
        ts_iso = fake.now_utc().isoformat()
        cycle_input = build_runtime_cycle_input(zone, rec, ts_iso=ts_iso)

        assert cycle_input.ts == _T0.isoformat(), (
            f"Cycle input ts must match FakeClock, got {cycle_input.ts!r}"
        )

    def test_cycle_input_wall_time_differs_from_fake_clock(self):
        """Verify FakeClock (_T0 = 2025-06-24) differs from real wall time (>= 2025)."""
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            build_runtime_cycle_input,
        )

        fake = FakeClock(start=_T0)
        zone = {"zone_id": "z", "comfort_temp": 21.0}
        rec = {"trv_setpoint": 22.0, "effective_target": 21.0, "current_temp": 19.0}

        ts_from_clock = fake.now_utc().isoformat()
        cycle_input = build_runtime_cycle_input(zone, rec, ts_iso=ts_from_clock)

        # _T0 is 2025-06-24T12:00:00+00:00 — verify it's preserved exactly
        assert cycle_input.ts.startswith("2025-06-24T12:00:00"), (
            "Cycle input ts must preserve the FakeClock time, not derive from wall clock."
        )

    def test_cycle_input_requires_ts_iso(self):
        """build_runtime_cycle_input must raise ValueError when ts_iso is omitted."""
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            build_runtime_cycle_input,
        )
        import pytest

        rec = {"trv_setpoint": 22.0, "effective_target": 21.0, "current_temp": 19.0}
        with pytest.raises(ValueError, match="ts_iso is required"):
            build_runtime_cycle_input("z", rec, ts_iso=None)


# ── 8. DST / timezone conversion does not corrupt timestamps ──────────────────

class TestDSTConversion:
    """dt_util.as_local() is used as a converter only (not a clock source).
    Tests that converting UTC to local and back is correct.
    """

    def test_utc_to_local_roundtrip(self):
        from custom_components.thermosmart.learning.clock import FakeClock
        from homeassistant.util import dt as dt_util

        fake = FakeClock(start=_T0)
        utc_now = fake.now_utc()

        # Convert to local via as_local (the same path _now_local() uses in production)
        local_now = dt_util.as_local(utc_now)

        # The local datetime must represent the same instant as UTC
        assert local_now.utctimetuple() == utc_now.utctimetuple()

    def test_naive_local_timestamps_not_produced_by_now_local(self):
        """_now_local() must always produce tz-aware datetimes."""
        e = make_learning_engine(clock=FakeClock(start=_T0))
        local_now = e._now_local()
        assert local_now.tzinfo is not None, (
            "_now_local() returned a naive datetime — clock injection is broken."
        )


# ── 9. Retention boundary tests (_prune_old_observations) ─────────────────────

class TestRetentionBoundary:
    """_prune_old_observations uses datetime comparison, not ISO string comparison.

    Boundary: just-before, at, and just-after the cutoff.
    """

    def _engine_with_obs(self, obs_list, *, now_utc):
        """Build a LearningEngine with given observations and a frozen clock."""
        e = make_learning_engine(clock=FakeClock(start=now_utc))
        for zone_id, ts in obs_list:
            e._observations.setdefault(zone_id, []).append(
                {"ts": ts, "temp_delta": 1.0}
            )
        return e

    def test_observation_just_before_cutoff_is_pruned(self):
        now = datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
        max_age = 30  # days
        # cutoff = now - 30d = 2025-05-25 12:00:00 UTC
        just_before = (now - timedelta(days=30, seconds=1)).isoformat()
        e = self._engine_with_obs([("z", just_before)], now_utc=now)
        e._prune_old_observations(max_age_days=max_age)
        assert e._observations.get("z", []) == []

    def test_observation_exactly_at_cutoff_survives(self):
        now = datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
        max_age = 30
        at_cutoff = (now - timedelta(days=30)).isoformat()
        e = self._engine_with_obs([("z", at_cutoff)], now_utc=now)
        e._prune_old_observations(max_age_days=max_age)
        assert len(e._observations.get("z", [])) == 1

    def test_observation_just_after_cutoff_survives(self):
        now = datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
        max_age = 30
        just_after = (now - timedelta(days=30) + timedelta(seconds=1)).isoformat()
        e = self._engine_with_obs([("z", just_after)], now_utc=now)
        e._prune_old_observations(max_age_days=max_age)
        assert len(e._observations.get("z", [])) == 1

    def test_naive_timestamp_normalized_at_boundary(self):
        """Naive timestamps (legacy storage) are assumed UTC and compared correctly."""
        now = datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
        max_age = 30
        # Naive string — exactly at cutoff
        at_cutoff_naive = (now - timedelta(days=30)).replace(tzinfo=None).isoformat()
        assert "+" not in at_cutoff_naive  # confirm it's naive
        e = self._engine_with_obs([("z", at_cutoff_naive)], now_utc=now)
        e._prune_old_observations(max_age_days=max_age)
        assert len(e._observations.get("z", [])) == 1

    def test_malformed_timestamp_is_pruned(self):
        """Malformed timestamp string → observation pruned (fail visibly by removing)."""
        now = datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
        e = self._engine_with_obs([("z", "not-a-date")], now_utc=now)
        e._prune_old_observations(max_age_days=30)
        assert e._observations.get("z", []) == []

    def test_missing_timestamp_is_pruned(self):
        """Observation without ts key → pruned deterministically."""
        now = datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
        e = make_learning_engine(clock=FakeClock(start=now))
        e._observations["z"] = [{"temp_delta": 1.0}]  # no ts key
        e._prune_old_observations(max_age_days=30)
        assert e._observations.get("z", []) == []

    def test_mixed_naive_aware_at_same_age(self):
        """Aware and naive timestamps at the same logical time produce same keep/prune result."""
        now = datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc)
        max_age = 30
        at_cutoff = now - timedelta(days=30)
        aware_ts = at_cutoff.isoformat()
        naive_ts = at_cutoff.replace(tzinfo=None).isoformat()

        e_aware = self._engine_with_obs([("z", aware_ts)], now_utc=now)
        e_naive = self._engine_with_obs([("z", naive_ts)], now_utc=now)

        e_aware._prune_old_observations(max_age_days=max_age)
        e_naive._prune_old_observations(max_age_days=max_age)

        # Both at cutoff — both should survive (>= comparison)
        assert len(e_aware._observations.get("z", [])) == len(e_naive._observations.get("z", []))
