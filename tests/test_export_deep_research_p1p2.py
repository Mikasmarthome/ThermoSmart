"""_historical_learning_snapshot_for_research() unit tests.

Covers the frozen-legacy-learning-snapshot reshaping used by ThermoSmart
diagnostics (custom_components/thermosmart/diagnostics.py):
  - keeps fachlich useful technical fields (ts/weekday/hour/minute,
    target/delta/heat_rate/outcome_score/...) while never leaking
    entity_id/device_id/person/location, runs scan_payload() as a final
    defense, caps per category with records_truncated, and never crashes
    on missing/corrupt input.

End-to-end diagnostics-payload tests (formerly in this file, calling the
now-removed async_export_learning_data()/async_export_support_data())
live in tests/test_diagnostics.py instead — see the export -> HA
Diagnostics migration audit for why the delivery layer was removed.
"""
from __future__ import annotations

import json

import pytest

from custom_components.thermosmart.export import (
    _HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY,
    _historical_learning_snapshot_for_research,
)
from custom_components.thermosmart.learning.privacy import scan_payload

_FORBIDDEN_SUBSTRINGS = (
    # "presence" itself is excluded: has_presence is a legitimate boolean
    # feature flag (module docstring: "feature flags, booleans only"), not
    # person/presence data. The real identity-carrying patterns are these:
    "entity_id", "device_id", "unique_id", "person.", "presence_persons",
    "latitude", "longitude", "climate.", "sensor.", "zone.home",
)


def _rich_learning_dict() -> dict:
    return {
        "observation_count": 2,
        "trv_observation_count": 1,
        "window_cooling_obs_count": 1,
        "confidence": 0.42,
        "boost_factor": 1.1,
        "forecast_bias": 0.05,
        "heat_loss_ema": 0.031,
        "observations": [
            {
                "ts": "2026-01-13T06:37:22",
                "hour": 6, "minute": 37, "weekday": 2,
                "target": 21.0, "indoor_temp": 20.6, "delta": 0.4,
                "active_control": True, "window_open": False,
                "control_reason": "schedule", "preheat_active": False,
                "heating_failure": False, "vacation": False, "summer_mode": False,
                "outdoor_temp": 5.0, "outdoor_humidity": 70.0, "wind_speed": 3.0,
                "solar_radiation": 100.0, "indoor_humidity": 45.0,
                "heat_rate": 0.31, "norm_heat_rate": 0.02, "cool_rate": None,
                "schedule_period": "wd_comfort", "forecast_high": 8.0,
            },
            {
                "ts": "2026-01-13T07:00:00",
                "hour": 7, "minute": 0, "weekday": 2,
                "target": 21.0, "indoor_temp": 21.0, "delta": 0.0,
            },
        ],
        "trv_observations": [
            {
                "ts": "2026-01-13T06:40:00", "trv_setpoint": 22.0,
                "indoor_temp": 20.6, "target": 21.0, "delta": 0.4,
                "setpoint_excess": 1.0, "heat_rate": 0.31, "efficiency": 0.31,
                "outdoor_temp": 5.0, "wind_speed": 3.0, "solar_radiation": 100.0,
            },
        ],
        "window_cooling_obs": [
            {
                "ts": "2026-01-13T08:00:00", "duration_min": 12.0,
                "temp_drop": 0.8, "cooling_rate_per_min": 0.0667,
                "indoor_start_temp": 21.0, "outdoor_temp": 4.0, "wind_speed": 5.0,
            },
        ],
        "outcome_log": [
            {
                "ts": "2026-01-13T07:22:00", "start_temp": 19.5, "target": 21.0,
                "peak_temp": 21.1, "end_temp": 21.0, "minutes_taken": 45.0,
                "expected_minutes": 40, "controller": "ts", "reason": "reached",
                "outcome_score": 0.82, "outdoor_temp": 5.0, "outdoor_humidity": 70.0,
                "wind_speed": 3.0, "solar_radiation": 100.0, "rain": 0.0,
            },
        ],
    }


def _blob_has_no_forbidden_substrings(payload: dict) -> list[str]:
    blob = json.dumps(payload, default=str).lower()
    return [s for s in _FORBIDDEN_SUBSTRINGS if s in blob]


class TestHistoricalSnapshotStructure:
    def test_available_and_markers(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        assert snap["available"] is True
        assert snap["frozen"] is True
        assert snap["raw_legacy_dump_included"] is False
        # Neutral compat key alongside the pre-existing one (Export-Naming
        # cleanup) — same value, additive, old key not removed.
        assert snap["historical_learning_dump_included"] is False
        assert snap["privacy_checked"] is True

    def test_event_count_summary_reflects_true_totals(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        summary = snap["event_count_summary"]
        assert summary["observations"] == 2
        assert summary["trv_observations"] == 1
        assert summary["window_cooling_observations"] == 1
        assert summary["outcome_entries"] == 1

    def test_model_state_scalars_preserved(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        ms = snap["model_state"]
        assert ms["confidence"] == pytest.approx(0.42)
        assert ms["boost_factor"] == pytest.approx(1.1)
        assert ms["forecast_bias"] == pytest.approx(0.05)
        assert ms["heat_loss_ema"] == pytest.approx(0.031)

    def test_observation_event_keeps_time_and_technical_fields(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        obs = snap["research_events"]["observations"][0]
        assert obs["ts"] == "2026-01-13T06:37:22"
        # 2026-01-13 is a Tuesday -> Python's datetime.weekday() == 1
        assert obs["weekday"] == 1
        assert obs["hour"] == 6
        assert obs["minute"] == 37
        assert obs["target"] == 21.0
        assert obs["delta"] == 0.4
        assert obs["heat_rate"] == 0.31
        assert obs["source"] == "historical"

    def test_trv_observation_event_keeps_technical_fields(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        trv = snap["research_events"]["trv_observations"][0]
        assert trv["trv_setpoint"] == 22.0
        assert trv["heat_rate"] == 0.31
        assert trv["efficiency"] == 0.31
        assert trv["weekday"] == 1 and trv["hour"] == 6

    def test_window_cooling_event_keeps_technical_fields(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        wc = snap["research_events"]["window_cooling"][0]
        assert wc["duration_min"] == 12.0
        assert wc["cooling_rate_per_min"] == pytest.approx(0.0667)

    def test_outcome_event_keeps_outcome_relevant_fields(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        out = snap["research_events"]["outcomes"][0]
        assert out["target"] == 21.0
        assert out["outcome_score"] == 0.82
        assert out["controller"] == "ts"
        assert out["reason"] == "reached"
        assert out["minutes_taken"] == 45.0

    def test_no_forbidden_identifiers_in_full_snapshot(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        assert scan_payload(snap) == []
        assert _blob_has_no_forbidden_substrings(snap) == []

    def test_unexpected_legacy_key_is_not_leaked(self):
        """A legacy entry with an extra, non-allow-listed key (simulating an
        unexpected/older schema field) must not surface it — allow-list only."""
        learning = _rich_learning_dict()
        learning["observations"][0]["some_unexpected_legacy_field"] = "whatever"
        learning["observations"][0]["entity_id"] = "climate.living_room_trv"
        snap = _historical_learning_snapshot_for_research(learning)
        obs = snap["research_events"]["observations"][0]
        assert "some_unexpected_legacy_field" not in obs
        assert "entity_id" not in obs
        assert scan_payload(snap) == []


class TestHistoricalSnapshotEmptyAndMissing:
    def test_empty_learning_dict_stable_no_crash(self):
        snap = _historical_learning_snapshot_for_research({})
        assert snap["available"] is True
        assert snap["research_events"] == {
            "observations": [], "trv_observations": [], "window_cooling": [], "outcomes": [],
        }
        assert snap["event_count_summary"]["observations"] == 0

    def test_missing_keys_do_not_crash(self):
        snap = _historical_learning_snapshot_for_research({"confidence": 0.1})
        assert snap["available"] is True
        assert snap["model_state"]["confidence"] == pytest.approx(0.1)
        assert snap["model_state"]["boost_factor"] is None


class TestHistoricalSnapshotCorruptInputDoesNotCrash:
    def test_non_list_observations_does_not_crash(self):
        learning = _rich_learning_dict()
        learning["observations"] = "not_a_list"
        snap = _historical_learning_snapshot_for_research(learning)
        assert snap["available"] is True
        assert snap["research_events"]["observations"] == []

    def test_non_dict_entries_in_list_are_skipped(self):
        learning = _rich_learning_dict()
        learning["observations"] = [None, "garbage", 42, {"ts": "2026-01-13T06:00:00", "target": 21.0}]
        snap = _historical_learning_snapshot_for_research(learning)
        assert len(snap["research_events"]["observations"]) == 1
        assert snap["research_events"]["observations"][0]["target"] == 21.0

    def test_malformed_ts_does_not_crash(self):
        learning = _rich_learning_dict()
        learning["observations"][0]["ts"] = "not-a-timestamp"
        snap = _historical_learning_snapshot_for_research(learning)
        obs = snap["research_events"]["observations"][0]
        assert obs["ts"] == "not-a-timestamp"
        assert "weekday" not in obs  # parse failed → time context omitted, not guessed

    def test_entirely_malformed_input_yields_available_false(self):
        class _Weird:
            def get(self, *a, **kw):
                raise RuntimeError("boom")
        snap = _historical_learning_snapshot_for_research(_Weird())
        assert snap["available"] is False
        assert snap["reason"] == "historical_snapshot_unavailable"


class TestHistoricalSnapshotCapping:
    def test_records_truncated_reports_excess(self):
        learning = _rich_learning_dict()
        n = _HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY + 25
        learning["observations"] = [
            {"ts": f"2026-01-01T00:{i:02d}:00", "target": 21.0, "delta": 0.0}
            for i in range(n)
        ]
        snap = _historical_learning_snapshot_for_research(learning)
        assert len(snap["research_events"]["observations"]) == _HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY
        assert snap["records_truncated"]["observations"] == 25

    def test_kept_entries_are_most_recent(self):
        learning = _rich_learning_dict()
        n = _HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY + 5
        learning["observations"] = [
            {"ts": f"2026-01-01T00:{i:02d}:00", "target": float(i), "delta": 0.0}
            for i in range(n)
        ]
        snap = _historical_learning_snapshot_for_research(learning)
        kept_targets = [o["target"] for o in snap["research_events"]["observations"]]
        assert kept_targets[0] == 5.0  # the oldest 5 (indices 0-4) were dropped
        assert kept_targets[-1] == float(n - 1)

    def test_no_truncation_when_under_cap(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        assert snap["records_truncated"] == {
            "observations": 0, "trv_observations": 0, "window_cooling": 0, "outcomes": 0,
        }
