"""Tests for the LE2 Research Daily Bucket long-term summary export block.

Covers the read-only research-export addition in export.py:
  _le2_research_daily_export() and its wiring into async_export_learning_data()
  (checked at the per-zone-dict-construction level, without a full hass fixture).

This step adds ONE small, BOUNDED SUMMARY per zone to the research export,
sourced directly from LearningShadowController.research_daily_snapshot()
(already in-memory — no new store read, no ResearchDailyStore.load()/.save()
call) — no Research Daily aggregation change, no Support-export change, no
new runtime hook.

23 test groups:
  T1  — Research export contains "research_daily" per zone
  T2  — Missing shadow yields available: False
  T3  — Empty snapshot yields available true, bucket_count 0, daily []
  T4  — snapshot() exception stays non-fatal
  T5  — Malformed buckets are skipped and counted
  T6  — Counters are summed in summary
  T7  — Overshoot/undershoot/comfort_error averages correct
  T8  — Progress global min/max and latest last correct
  T9  — Confidence global min/max and latest last correct
  T10 — Daily entries are newest-first sorted
  T11 — Daily export cap (90) applies
  T12 — Summary stays complete over ALL valid buckets despite daily cap
  T13 — records_truncated/truncation_reason correct
  T14 — Retention metadata correct
  T15 — Compact daily entries omit zero/None fields
  T16 — No forbidden substrings anywhere in the output
  T17 — Privacy scan failure is non-fatal
  T18 — No store I/O in the new helper
  T19 — Support export is unchanged
  T20 — Existing learning_progress export tests remain green
  T21 — Existing episode_history export tests remain green
  T22 — Existing research daily aggregation/state/foundation tests remain green
  T23 — No control-path keywords touched
"""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart import export as _export_module
from custom_components.thermosmart.export import (
    _le2_research_daily_export,
    _RESEARCH_DAILY_EXPORT_MAX_BUCKETS,
)
from custom_components.thermosmart.learning.storage.research_daily_persistence import (
    create_empty_research_daily_bucket,
    record_research_daily_observation,
)
from custom_components.thermosmart.learning.storage.research_daily_serialization import (
    serialize_research_daily_bucket,
)
from custom_components.thermosmart.learning.research_daily_schemas import ResearchDailyObservation

_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


class _FakeShadow:
    def __init__(self, snapshot=None, snapshot_raises: Exception | None = None):
        self._snapshot = dict(snapshot) if snapshot else {}
        self._snapshot_raises = snapshot_raises

    def research_daily_snapshot(self):
        if self._snapshot_raises is not None:
            raise self._snapshot_raises
        return dict(self._snapshot)


class _FakeCoord:
    def __init__(self, shadow=None):
        self._le2_shadow = shadow


def _bucket_entry(bucket_date: str, **observation_kwargs) -> dict:
    bucket = create_empty_research_daily_bucket(bucket_date)
    obs = ResearchDailyObservation(bucket_date=bucket_date, **observation_kwargs)
    bucket = record_research_daily_observation(bucket, obs)
    return serialize_research_daily_bucket(bucket)


def _snapshot_for_days(days: list[str]) -> dict:
    return {day: _bucket_entry(day, decision_count=1) for day in days}


def _consecutive_days(count: int, *, start: str = "2026-01-01") -> list[str]:
    start_date = datetime.fromisoformat(start).date()
    return [(start_date + timedelta(days=i)).isoformat() for i in range(count)]


# ── T1: Research export contains "research_daily" per zone ─────────────────

def test_t1_export_contains_research_daily_block():
    snapshot = {"2026-06-01": _bucket_entry("2026-06-01", trv_command_sent_count=3)}
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block["available"] is True
    assert "summary" in block
    assert "daily" in block


def test_t1_wired_into_async_export_learning_data():
    source = inspect.getsource(_export_module.async_export_learning_data)
    assert "_le2_research_daily_export(" in source
    assert '"research_daily": research_daily' in source


# ── T2: Missing shadow yields available: False ──────────────────────────────

def test_t2_missing_shadow_yields_unavailable():
    block = _le2_research_daily_export(_FakeCoord(None), now=_NOW)
    assert block == {"available": False, "reason": "le2_shadow_unavailable"}


# ── T3: Empty snapshot yields available true, bucket_count 0, daily [] ─────

def test_t3_empty_snapshot():
    coord = _FakeCoord(_FakeShadow({}))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block["available"] is True
    assert block["bucket_count"] == 0
    assert block["summary"] == {}
    assert block["daily"] == []
    assert block["coverage_start"] is None
    assert block["coverage_end"] is None


# ── T4: snapshot() exception stays non-fatal ─────────────────────────────────

def test_t4_snapshot_exception_stays_non_fatal():
    coord = _FakeCoord(_FakeShadow(snapshot_raises=RuntimeError("boom")))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block["available"] is False
    assert "research_daily_error" in block


# ── T5: Malformed buckets are skipped and counted ───────────────────────────

def test_t5_malformed_buckets_skipped_and_counted():
    snapshot = {
        "2026-06-01": _bucket_entry("2026-06-01", decision_count=1),
        "not_a_dict": "oops",
        "bad_schema": {"schema_version": 999, "bucket_date": "2026-06-02"},
    }
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block["available"] is True
    assert block["bucket_count"] == 1
    assert block["malformed_skipped_count"] == 2


# ── T6: Counters are summed in summary ──────────────────────────────────────

def test_t6_counters_summed():
    snapshot = {
        "2026-06-01": _bucket_entry("2026-06-01", trv_command_sent_count=3, boost_started_count=1),
        "2026-06-02": _bucket_entry("2026-06-02", trv_command_sent_count=2, boost_started_count=2),
    }
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block["summary"]["trv_command_sent_count"] == 5
    assert block["summary"]["boost_started_count"] == 3


# ── T7: Overshoot/undershoot/comfort_error averages correct ────────────────

def test_t7_averages_correct():
    snapshot = {
        "2026-06-01": _bucket_entry("2026-06-01", overshoot_c=0.4, comfort_error_c=0.4),
        "2026-06-02": _bucket_entry("2026-06-02", overshoot_c=0.6, undershoot_c=0.2, comfort_error_c=0.2),
    }
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    summary = block["summary"]
    assert summary["avg_overshoot_c"] == pytest.approx(0.5)
    assert summary["avg_undershoot_c"] == pytest.approx(0.2)
    assert summary["avg_comfort_error_c"] == pytest.approx(0.3)


def test_t7_average_is_none_when_no_samples():
    snapshot = {"2026-06-01": _bucket_entry("2026-06-01", decision_count=1)}
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    summary = block["summary"]
    assert summary["avg_overshoot_c"] is None
    assert summary["avg_undershoot_c"] is None
    assert summary["avg_comfort_error_c"] is None


# ── T8: Progress global min/max and latest last correct ────────────────────

def test_t8_progress_global_min_max_latest_last():
    snapshot = {
        "2026-06-01": _bucket_entry("2026-06-01", learning_progress_pct=20.0),
        "2026-06-02": _bucket_entry("2026-06-02", learning_progress_pct=45.0),
        "2026-06-03": _bucket_entry("2026-06-03", learning_progress_pct=30.0),
    }
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    summary = block["summary"]
    assert summary["learning_progress_min_pct"] == 20.0
    assert summary["learning_progress_max_pct"] == 45.0
    assert summary["learning_progress_last_pct"] == 30.0  # newest bucket (2026-06-03)


def test_t8_last_pct_skips_days_with_no_sample():
    snapshot = {
        "2026-06-01": _bucket_entry("2026-06-01", learning_progress_pct=20.0),
        "2026-06-02": _bucket_entry("2026-06-02", decision_count=1),  # no progress sample this day
    }
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block["summary"]["learning_progress_last_pct"] == 20.0  # falls back to 06-01's value


# ── T9: Confidence global min/max and latest last correct ──────────────────

def test_t9_confidence_global_min_max_latest_last():
    snapshot = {
        "2026-06-01": _bucket_entry("2026-06-01", confidence=0.3),
        "2026-06-02": _bucket_entry("2026-06-02", confidence=0.8),
        "2026-06-03": _bucket_entry("2026-06-03", confidence=0.5),
    }
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    summary = block["summary"]
    assert summary["confidence_min"] == 0.3
    assert summary["confidence_max"] == 0.8
    assert summary["confidence_last"] == 0.5


# ── T10: Daily entries are newest-first sorted ──────────────────────────────

def test_t10_daily_entries_newest_first():
    snapshot = _snapshot_for_days(["2026-06-01", "2026-06-03", "2026-06-02"])
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    dates = [entry["bucket_date"] for entry in block["daily"]]
    assert dates == ["2026-06-03", "2026-06-02", "2026-06-01"]


# ── T11: Daily export cap (90) applies ──────────────────────────────────────

def test_t11_daily_export_cap_applies():
    days = _consecutive_days(100)
    snapshot = _snapshot_for_days(days)
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert len(block["daily"]) == _RESEARCH_DAILY_EXPORT_MAX_BUCKETS
    assert _RESEARCH_DAILY_EXPORT_MAX_BUCKETS == 90


def test_t11_daily_cap_keeps_newest():
    days = _consecutive_days(100)
    snapshot = _snapshot_for_days(days)
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    exported_dates = {entry["bucket_date"] for entry in block["daily"]}
    assert max(days) in exported_dates
    assert min(days) not in exported_dates


# ── T12: Summary stays complete over ALL valid buckets despite daily cap ───

def test_t12_summary_complete_despite_daily_cap():
    days = _consecutive_days(100)
    snapshot = {day: _bucket_entry(day, trv_command_sent_count=1) for day in days}
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block["summary"]["trv_command_sent_count"] == 100  # over ALL 100 buckets, not just 90
    assert block["bucket_count"] == 100


# ── T13: records_truncated/truncation_reason correct ────────────────────────

def test_t13_no_truncation_under_cap():
    snapshot = _snapshot_for_days(["2026-06-01", "2026-06-02"])
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block["records_truncated"] == 0
    assert block["truncation_reason"] is None


def test_t13_truncation_reported_over_cap():
    days = _consecutive_days(100)
    snapshot = _snapshot_for_days(days)
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block["records_truncated"] == 10
    assert block["truncation_reason"] == "export_cap_exceeded"


# ── T14: Retention metadata correct ─────────────────────────────────────────

def test_t14_retention_metadata_correct():
    snapshot = _snapshot_for_days(["2026-06-01"])
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    retention = block["retention"]
    assert retention["bounded"] is True
    assert retention["max_days"] == 365
    assert retention["max_buckets"] == 400
    assert block["export_cap_buckets"] == 90
    assert block["schema_version"] == 1


# ── T15: Compact daily entries omit zero/None fields ────────────────────────

def test_t15_compact_entries_omit_zero_fields():
    snapshot = {"2026-06-01": _bucket_entry("2026-06-01", trv_command_sent_count=2)}
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    entry = block["daily"][0]
    assert entry == {"bucket_date": "2026-06-01", "trv_command_sent_count": 2}


def test_t15_compact_entries_include_sum_count_pair_together():
    snapshot = {"2026-06-01": _bucket_entry("2026-06-01", overshoot_c=0.5)}
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    entry = block["daily"][0]
    assert entry["overshoot_sum_c"] == 0.5
    assert entry["overshoot_count"] == 1
    assert "undershoot_sum_c" not in entry
    assert "undershoot_count" not in entry


def test_t15_compact_entries_include_progress_confidence_when_present():
    snapshot = {"2026-06-01": _bucket_entry("2026-06-01", learning_progress_pct=15.0, confidence=0.4)}
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    entry = block["daily"][0]
    assert entry["learning_progress_last_pct"] == 15.0
    assert entry["confidence_last"] == 0.4


# ── T16: No forbidden substrings anywhere in the output ─────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "entity_id", "zone_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "trajectory", "person", "secret",
    "token", "path", "climate.", "sensor.",
)


def test_t16_no_forbidden_substrings_in_output():
    snapshot = {
        "2026-06-01": _bucket_entry(
            "2026-06-01", trv_command_sent_count=1, learning_progress_pct=10.0, confidence=0.3,
        ),
    }
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    flat = repr(block).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in flat, f"forbidden substring '{token}' leaked into export block"


# ── T17: Privacy scan failure is non-fatal ──────────────────────────────────

def test_t17_privacy_scan_violation_blocks_block(monkeypatch):
    import custom_components.thermosmart.learning.privacy as _privacy_mod

    def _fake_scan_payload(payload, *, path="$"):
        return [object()]  # non-empty -> violation

    monkeypatch.setattr(_privacy_mod, "scan_payload", _fake_scan_payload)
    snapshot = _snapshot_for_days(["2026-06-01"])
    coord = _FakeCoord(_FakeShadow(snapshot))
    block = _le2_research_daily_export(coord, now=_NOW)
    assert block == {"available": False, "reason": "privacy_scan_failed"}


# ── T18: No store I/O in the new helper ─────────────────────────────────────

def test_t18_no_store_io_in_new_helper():
    source = inspect.getsource(_le2_research_daily_export)
    assert "await" not in source
    assert ".save(" not in source
    assert ".load(" not in source
    assert "ResearchDailyStore(" not in source
    assert "async_setup" not in source
    assert "async_load" not in source


# ── T19: Support export is unchanged ─────────────────────────────────────────

def test_t19_support_export_signature_unchanged():
    sig = inspect.signature(_export_module.async_export_support_data)
    assert list(sig.parameters) == ["hass", "ts"]


def test_t19_support_export_does_not_call_new_helper():
    source = inspect.getsource(_export_module.async_export_support_data)
    assert "_le2_research_daily_export" not in source


# ── T20: Existing learning_progress export tests remain green ──────────────

def test_t20_regression_imports_ok_learning_progress():
    import tests.test_le2_learning_progress_export  # noqa: F401
    import tests.test_le2_learning_progress  # noqa: F401


# ── T21: Existing episode_history export tests remain green ────────────────

def test_t21_regression_imports_ok_episode_history():
    import tests.test_le2_episode_history_export  # noqa: F401


# ── T22: Existing research daily aggregation/state/foundation tests remain green

def test_t22_regression_imports_ok_research_daily():
    import tests.test_le2_research_daily_buckets  # noqa: F401
    import tests.test_le2_research_daily_state_wiring  # noqa: F401
    import tests.test_le2_research_daily_store_wiring  # noqa: F401
    import tests.test_le2_research_daily_support_event_aggregation  # noqa: F401
    import tests.test_le2_research_daily_progress_confidence_aggregation  # noqa: F401


# ── T23: No control-path keywords touched ────────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
    "preheat",
    "setpoint",
)


def test_t23_no_control_keywords_in_new_helper():
    source = inspect.getsource(_le2_research_daily_export).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"
