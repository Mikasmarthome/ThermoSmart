"""Phase 18 control application ledger tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime.control import (
    ControlApplicationLedger, ControlFeature)
from tests.helpers_control import control_decision, strong_preheat, weak_purpose
from custom_components.thermosmart.learning.confidence import ComponentKind, ConfidencePurpose

F = ControlFeature.PREHEAT_START


def _led():
    return ControlApplicationLedger()


class TestControlLedger:
    def test_records_applied(self):
        led = _led()
        d = control_decision(F, 60.0, 66.0, strong_preheat(), unit="minutes")
        led.record("z", "dec1", d, "2025-01-01T00:00:00+00:00")
        assert led.summary()["applied_counts"].get("preheat_start") == 1

    def test_records_rejection(self):
        led = _led()
        d = control_decision(F, 60.0, 66.0,
                             weak_purpose(ConfidencePurpose.PREHEAT, ComponentKind.HEAT_RATE),
                             unit="minutes")
        led.record("z", "dec1", d, "t")
        assert led.summary()["fallback_count"] == 1
        assert led.summary()["rejection_counts"]

    def test_mean_clamp(self):
        led = _led()
        d = control_decision(F, 60.0, 85.0, strong_preheat(), unit="minutes")  # clamped
        led.record("z", "dec1", d, "t")
        assert led.summary()["mean_clamp"] is not None and led.summary()["mean_clamp"] > 0

    def test_bounded(self):
        led = ControlApplicationLedger(max_entries=5)
        d = control_decision(F, 60.0, 66.0, strong_preheat(), unit="minutes")
        for i in range(20):
            led.record("z", f"dec{i}", d, f"2025-01-01T00:{i:02d}:00+00:00")
        assert led.size == 5

    def test_research_samples_no_ids(self):
        led = _led()
        d = control_decision(F, 60.0, 66.0, strong_preheat(), unit="minutes")
        led.record("z", "dec_secret", d, "t")
        samples = led.research_samples()
        import json
        blob = json.dumps(samples)
        assert "dec_secret" not in blob and "z" not in [s.get("feature") for s in samples]
        assert "feature" in samples[0] and "baseline" in samples[0]

    def test_recent(self):
        led = _led()
        d = control_decision(F, 60.0, 66.0, strong_preheat(), unit="minutes")
        led.record("z", "d", d, "t")
        assert len(led.recent()) == 1 and led.recent()[0].feature == "preheat_start"
