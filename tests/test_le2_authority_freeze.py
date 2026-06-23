"""Phase 19A-B Atomic Step 0: LE-v1 learning freeze + confidence read transfer."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.learning_engine import LearningEngine


def _engine():
    hass = MagicMock()
    return LearningEngine(hass)


class TestLearningFreeze:
    def test_default_not_frozen(self):
        assert _engine().frozen is False

    def test_freeze_sets_flag(self):
        e = _engine(); e.freeze()
        assert e.frozen is True

    async def test_observe_noop_when_frozen(self):
        e = _engine(); e.set_zone_enabled("z", True); e.freeze()
        await e.async_observe("z", {"target": 21.0, "current_temp": 19.0}, {}, )
        # no observation recorded
        assert sum(len(v) for v in e._observations.values()) == 0

    def test_update_boost_factor_noop_when_frozen(self):
        e = _engine(); e.set_zone_enabled("z", True); e.freeze()
        before = dict(e._boost_factors)
        e.update_boost_factor("z", overshot=True)
        assert e._boost_factors == before

    def test_update_heating_session_noop_when_frozen(self):
        e = _engine(); e.set_zone_enabled("z", True); e.freeze()
        e.update_heating_session("z", 19.0, 21.0, True, {})
        assert e._heating_sessions == {}

    def test_record_forecast_decision_noop_when_frozen(self):
        e = _engine(); e.set_zone_enabled("z", True); e.freeze()
        e.record_forecast_decision("z", 21.0, 21.0, 10.0, 19.0, 0.0, 5.0)
        assert sum(len(v) for v in e._forecast_decisions.values()) == 0

    async def test_observe_trv_setpoint_noop_when_frozen(self):
        e = _engine(); e.set_zone_enabled("z", True); e.freeze()
        await e.async_observe_trv_setpoint("z", 22.0, 19.0, 21.0, {}, 0.05)
        assert sum(len(v) for v in e._trv_observations.values()) == 0

    async def test_observe_window_cooling_noop_when_frozen(self):
        e = _engine(); e.set_zone_enabled("z", True); e.freeze()
        await e.async_observe_window_cooling("z", 30.0, 21.0, 19.0, {})
        assert sum(len(v) for v in e._window_cooling_obs.values()) == 0

    async def test_async_save_noop_when_frozen(self):
        e = _engine(); e.freeze()
        e._store = MagicMock()
        e._store.async_save = MagicMock()
        await e.async_save()
        e._store.async_save.assert_not_called()

    def test_prune_noop_when_frozen(self):
        e = _engine(); e.freeze()
        e._observations["orphan"] = [{"x": 1}]
        e.prune_orphaned_zones({"active"})
        assert "orphan" in e._observations  # untouched

    def test_isolation_engine_boost_factor_noop(self):
        # update_boost_factor is now a no-op: LE v1 boost_factor superseded by LE 2.0
        e = _engine(); e.set_zone_enabled("z", True)
        e.update_boost_factor("z", overshot=True)
        assert not e._boost_factors  # no-op: _boost_factors stays empty
