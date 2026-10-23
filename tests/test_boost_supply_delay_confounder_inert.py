"""Characterisation: the boost supply-delay confounder never fires today.

``LearningRuntime._runtime_confounders`` compares the heating onset observed in a
boost episode with the onset the onset-delay model expects. It reads the expected
value from the prediction under the key ``onset_delay_minutes``; the model
publishes ``onset_delay``. The lookup fails inside a ``try`` block that swallows
the error, so the comparison — and with it the supply-delay anomaly / heating
failure classification — has never run.

This pins that behaviour. Repairing the key would change which boost episodes are
learned from (an episode with a late onset would be down-weighted or rejected), so
that has to be a deliberate decision, at which point this test changes with it.
"""
from __future__ import annotations

from types import SimpleNamespace

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.models import supply_delay_confounder
from custom_components.thermosmart.learning.models.baseline_comparison import _onset_and_target
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntime
from tests.helpers_boost import boost_episode

# 50 minutes without any response, then the room warms up
_LATE_ONSET = [18.0] * 10 + [18.6, 19.2, 19.8, 20.4, 21.0]
_EXPECTED_ONSET_MIN = 8.0


def _runtime() -> LearningRuntime:
    runtime = object.__new__(LearningRuntime)
    runtime._record_runtime_error = lambda *a, **kw: None
    return runtime


def _zone_with_expected_onset():
    prediction = SimpleNamespace(values={"onset_delay": _EXPECTED_ONSET_MIN},
                                 fallback_used=False)
    return SimpleNamespace(last_predictions={PredictionType.ONSET_DELAY: prediction})


class TestSupplyDelayConfounderIsInert:
    def test_a_late_onset_raises_no_confounder_today(self):
        episode = boost_episode("e1", _LATE_ONSET)

        flags = _runtime()._runtime_confounders(_zone_with_expected_onset(), episode, None, None)

        assert not any("supply_delay" in f or f == "heating_failure" for f in flags)

    def test_the_comparison_itself_would_flag_that_episode(self):
        """What the classification says once it is actually fed the expected onset."""
        episode = boost_episode("e1", _LATE_ONSET)
        onset_ms, _ = _onset_and_target(
            episode.trajectory.points, episode.target, episode.start_temp,
            episode.comfort_tolerance_at_start)

        verdict = supply_delay_confounder(_EXPECTED_ONSET_MIN, onset_ms / 60000.0)

        assert verdict is not None
