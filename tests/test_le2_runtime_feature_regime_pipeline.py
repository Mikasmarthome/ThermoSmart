"""Phase 17B feature + regime pipeline tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime import RuntimePipeline
from custom_components.thermosmart.learning.runtime.capture import CaptureCoordinator
from tests.helpers_runtime import cycle_input


def _drive(temps, *, heating, zone="lz", target=21.0, window=None):
    pipe = RuntimePipeline(zone)
    cap = CaptureCoordinator(zone)
    results = []
    for i, t in enumerate(temps):
        setp = (target + 3.0) if heating[i] else (target - 4.0)
        inp = cycle_input(f"2025-01-01T06:{i * 5:02d}:00+00:00", zone=zone, target=target,
                          setpoint=setp, indoor=t, window_open=window)
        # honest demand signal
        from dataclasses import replace
        inp = replace(inp, controller_demand=heating[i])
        results.append(pipe.process(cap.capture(inp)))
    return results


class TestFeatureRegime:
    def test_one_regime_per_cycle(self):
        res = _drive([18.0, 18.5, 19.0, 19.6], heating=[True] * 4)
        assert all(r.regime is not None for r in res)

    def test_heating_detected(self):
        res = _drive([18.0, 18.6, 19.2, 19.8, 20.4, 21.0], heating=[True] * 6)
        assert any(r.regime == "active_heating" for r in res)

    def test_disturbed_on_window(self):
        res = _drive([21.0, 20.0, 19.0, 18.0], heating=[False] * 4, window=True)
        assert any(r.disturbed for r in res)

    def test_previous_regime_tracked(self):
        res = _drive([18.0, 18.6, 19.2, 19.8], heating=[True] * 4)
        assert res[-1].previous_regime is not None

    def test_feature_valid_points_grow(self):
        res = _drive([18.0, 18.5, 19.0, 19.5, 20.0], heating=[True] * 5)
        assert res[-1].feature_valid_points >= res[0].feature_valid_points

    def test_zero_temp_valid(self):
        res = _drive([0.0, 0.2, 0.4], heating=[True] * 3, target=1.0)
        assert all(r.regime is not None for r in res)

    def test_missing_temp_handled(self):
        pipe = RuntimePipeline("lz")
        cap = CaptureCoordinator("lz")
        r = pipe.process(cap.capture(cycle_input("2025-01-01T06:00:00+00:00", zone="lz",
                                                 indoor=None)))
        assert r.regime is not None  # no crash
