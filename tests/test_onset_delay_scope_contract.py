"""Scope contract for the onset delay model.

The onset delay prediction feeds the preheat lead time. It must not feed a
boost supply-delay confounder: that heuristic never ran (it looked the value up
under a key the model does not publish), was judged not worth activating and has
been removed. Nothing here may bring it back by accident.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.models import boost_reliability
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntime
from tests.helpers_boost import boost_episode
from tests.test_learned_state_golden import _cycle, _zone

_LATE_ONSET = [18.0] * 10 + [18.6, 19.2, 19.8, 20.4, 21.0]   # 50 min without a response


async def _preheat_with_onset(onset_delay_min: float):
    coord, shadow, clock, _ = await _zone(17.5, learned=True, onset_delay_min=onset_delay_min)
    for _ in range(3):
        await _cycle(coord, clock)
    return shadow.read_onset_delay_safe(), shadow.read_preheat_minutes_safe(17.5, 21.0)


class TestOnsetDelayFeedsPreheat:
    async def test_a_longer_learned_onset_lengthens_the_preheat_lead_time_one_to_one(self):
        (short_onset, _), (short_minutes, _) = await _preheat_with_onset(8.0)
        (long_onset, _), (long_minutes, _) = await _preheat_with_onset(20.0)

        assert (short_onset, long_onset) == (8.0, 20.0)
        assert round(long_minutes - short_minutes, 1) == 12.0


class TestBoostLifecycleDerivesNoSupplyDelay:
    def test_a_late_onset_with_a_valid_prediction_raises_no_supply_delay_or_failure_flag(self):
        runtime = object.__new__(LearningRuntime)
        runtime._record_runtime_error = lambda *a, **kw: None
        prediction = SimpleNamespace(values={"onset_delay": 8.0}, fallback_used=False)
        zone = SimpleNamespace(last_predictions={PredictionType.ONSET_DELAY: prediction})
        episode = boost_episode("e1", _LATE_ONSET)

        flags = runtime._runtime_confounders(zone, episode, None, None)

        assert not any("supply_delay" in f or f == "heating_failure" for f in flags)

    def test_the_confounder_machinery_is_gone(self):
        assert not hasattr(boost_reliability, "supply_delay_confounder")
        assert not hasattr(boost_reliability, "SupplyDelayParams")
        assert not hasattr(boost_reliability.BoostConfounder, "SUPPLY_DELAY_ANOMALY")
        assert "supply_delay_anomaly" not in inspect.signature(
            boost_reliability.evaluate_confounders).parameters
