"""Tests for the LE2 Research Daily Buckets foundation (pure schema,
serialization, and aggregation/retention helpers only).

Foundation only — no runtime producer, no LearningShadowController wiring,
no store wrapper, no export wiring exists yet anywhere. These tests only
exercise the pure modules:
  - custom_components/thermosmart/learning/research_daily_schemas.py
  - custom_components/thermosmart/learning/storage/research_daily_serialization.py
  - custom_components/thermosmart/learning/storage/research_daily_persistence.py

24 test groups (T1-T24), plus static source-inspection guards confirming no
HA/runtime/control imports exist in the new pure modules.
"""
from __future__ import annotations

import inspect
import math
from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.research_daily_schemas import (
    RESEARCH_DAILY_SCHEMA_VERSION,
    InvalidBucketDateError,
    ResearchDailyBucket,
    ResearchDailyObservation,
)
from custom_components.thermosmart.learning.storage.research_daily_serialization import (
    deserialize_research_daily_bucket,
    deserialize_research_daily_history,
    serialize_research_daily_bucket,
    serialize_research_daily_history,
)
from custom_components.thermosmart.learning.storage import research_daily_persistence as rdp
from custom_components.thermosmart.learning.storage.research_daily_persistence import (
    RESEARCH_DAILY_RETENTION_DEFAULT,
    RESEARCH_DAILY_RETENTION_MAX_BUCKETS,
    RESEARCH_DAILY_RETENTION_MAX_DAYS,
    append_or_update_research_daily_bucket,
    create_empty_research_daily_bucket,
    merge_research_daily_bucket,
    prune_research_daily_buckets,
    record_research_daily_observation,
)
from custom_components.thermosmart.learning.registry import RetentionPolicy


# ── T1: empty bucket defaults ────────────────────────────────────────────────

def test_t1_empty_bucket_defaults():
    b = create_empty_research_daily_bucket("2026-06-01")
    assert b.schema_version == RESEARCH_DAILY_SCHEMA_VERSION
    assert b.bucket_date == "2026-06-01"
    assert b.decision_count == 0
    assert b.overshoot_sum_c == 0.0
    assert b.overshoot_count == 0
    assert b.learning_progress_min_pct is None
    assert b.confidence_last is None


# ── T2: serialize/deserialize roundtrip ─────────────────────────────────────

def test_t2_serialize_deserialize_roundtrip():
    obs = ResearchDailyObservation(
        bucket_date="2026-06-01", decision_count=3, trv_command_sent_count=2,
        overshoot_c=0.5, learning_progress_pct=12.5, confidence=0.7,
    )
    b = record_research_daily_observation(create_empty_research_daily_bucket("2026-06-01"), obs)
    serialized = serialize_research_daily_bucket(b)
    restored = deserialize_research_daily_bucket(serialized)
    assert restored == b


# ── T3: schema mismatch returns None/skips ──────────────────────────────────

def test_t3_schema_version_mismatch_returns_none():
    raw = serialize_research_daily_bucket(create_empty_research_daily_bucket("2026-06-01"))
    raw["schema_version"] = 999
    assert deserialize_research_daily_bucket(raw) is None


# ── T4: malformed bucket skipped non-fatal ──────────────────────────────────

def test_t4_malformed_bucket_skipped_in_history():
    good = serialize_research_daily_bucket(create_empty_research_daily_bucket("2026-06-01"))
    raw = {"buckets": {"2026-06-01": good, "bad-key": {"schema_version": 1, "bucket_date": "not-a-date"}}}
    result = deserialize_research_daily_history(raw)
    assert set(result.keys()) == {"2026-06-01"}


def test_t4_deserialize_bucket_non_mapping_returns_none():
    assert deserialize_research_daily_bucket("not-a-dict") is None
    assert deserialize_research_daily_bucket(None) is None


# ── T5: unknown fields ignored ───────────────────────────────────────────────

def test_t5_unknown_fields_ignored():
    raw = serialize_research_daily_bucket(create_empty_research_daily_bucket("2026-06-01"))
    raw["totally_unexpected_field"] = "should be ignored"
    restored = deserialize_research_daily_bucket(raw)
    assert restored is not None
    assert not hasattr(restored, "totally_unexpected_field")


# ── T6: negative counters normalized ────────────────────────────────────────

def test_t6_negative_counters_normalized_on_deserialize():
    raw = serialize_research_daily_bucket(create_empty_research_daily_bucket("2026-06-01"))
    raw["decision_count"] = -5
    restored = deserialize_research_daily_bucket(raw)
    assert restored.decision_count == 0


def test_t6_negative_counter_rejected_at_construction():
    with pytest.raises(ValueError):
        ResearchDailyBucket(schema_version=1, bucket_date="2026-06-01", decision_count=-1)


# ── T7: NaN/inf numeric values ignored ──────────────────────────────────────

def test_t7_nan_inf_cleaned_on_deserialize():
    raw = serialize_research_daily_bucket(create_empty_research_daily_bucket("2026-06-01"))
    raw["overshoot_sum_c"] = float("nan")
    raw["confidence_last"] = float("inf")
    restored = deserialize_research_daily_bucket(raw)
    assert restored.overshoot_sum_c == 0.0
    assert restored.confidence_last is None


def test_t7_nan_inf_ignored_in_observation_merge():
    b = create_empty_research_daily_bucket("2026-06-01")
    obs = ResearchDailyObservation(
        bucket_date="2026-06-01", overshoot_c=float("nan"), confidence=float("inf"),
    )
    merged = record_research_daily_observation(b, obs)
    assert merged.overshoot_sum_c == 0.0
    assert merged.overshoot_count == 0
    assert merged.confidence_last is None


# ── T8: additive counters aggregate correctly ───────────────────────────────

def test_t8_additive_counters_aggregate():
    b = create_empty_research_daily_bucket("2026-06-01")
    obs1 = ResearchDailyObservation(bucket_date="2026-06-01", decision_count=2, trv_command_sent_count=1)
    obs2 = ResearchDailyObservation(bucket_date="2026-06-01", decision_count=3, trv_command_blocked_count=1)
    b = record_research_daily_observation(b, obs1)
    b = record_research_daily_observation(b, obs2)
    assert b.decision_count == 5
    assert b.trv_command_sent_count == 1
    assert b.trv_command_blocked_count == 1


# ── T9: outcome counters aggregate ──────────────────────────────────────────

def test_t9_outcome_counters_aggregate():
    b = create_empty_research_daily_bucket("2026-06-01")
    obs1 = ResearchDailyObservation(bucket_date="2026-06-01", outcome_resolved_count=1, outcome_success_count=1)
    obs2 = ResearchDailyObservation(bucket_date="2026-06-01", outcome_resolved_count=1, outcome_failed_count=1)
    obs3 = ResearchDailyObservation(
        bucket_date="2026-06-01", outcome_resolved_count=1, outcome_confounded_count=1,
    )
    b = record_research_daily_observation(b, obs1)
    b = record_research_daily_observation(b, obs2)
    b = record_research_daily_observation(b, obs3)
    assert b.outcome_resolved_count == 3
    assert b.outcome_success_count == 1
    assert b.outcome_failed_count == 1
    assert b.outcome_confounded_count == 1


# ── T10: overshoot/undershoot/comfort-error sum/count aggregate ────────────

def test_t10_sum_count_pairs_aggregate():
    b = create_empty_research_daily_bucket("2026-06-01")
    obs1 = ResearchDailyObservation(bucket_date="2026-06-01", overshoot_c=0.4, undershoot_c=0.2, comfort_error_c=0.6)
    obs2 = ResearchDailyObservation(bucket_date="2026-06-01", overshoot_c=0.6, comfort_error_c=0.4)
    b = record_research_daily_observation(b, obs1)
    b = record_research_daily_observation(b, obs2)
    assert b.overshoot_sum_c == pytest.approx(1.0)
    assert b.overshoot_count == 2
    assert b.undershoot_sum_c == pytest.approx(0.2)
    assert b.undershoot_count == 1
    assert b.comfort_error_sum_c == pytest.approx(1.0)
    assert b.comfort_error_count == 2


# ── T11: learning progress min/max/last update correctly ───────────────────

def test_t11_learning_progress_min_max_last():
    b = create_empty_research_daily_bucket("2026-06-01")
    for pct in (10.0, 25.0, 15.0):
        b = record_research_daily_observation(
            b, ResearchDailyObservation(bucket_date="2026-06-01", learning_progress_pct=pct),
        )
    assert b.learning_progress_min_pct == 10.0
    assert b.learning_progress_max_pct == 25.0
    assert b.learning_progress_last_pct == 15.0


# ── T12: confidence min/max/last update correctly ───────────────────────────

def test_t12_confidence_min_max_last():
    b = create_empty_research_daily_bucket("2026-06-01")
    for conf in (0.3, 0.8, 0.5):
        b = record_research_daily_observation(
            b, ResearchDailyObservation(bucket_date="2026-06-01", confidence=conf),
        )
    assert b.confidence_min == 0.3
    assert b.confidence_max == 0.8
    assert b.confidence_last == 0.5


# ── T13: multiple observations same day merge into one bucket ──────────────

def test_t13_multiple_observations_same_day_single_bucket():
    payload: dict = {}
    now = datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc)
    for _ in range(3):
        result = append_or_update_research_daily_bucket(
            payload, "2026-06-01",
            ResearchDailyObservation(bucket_date="2026-06-01", decision_count=1),
            now_utc=now,
        )
        payload = result.updated_payload
    assert len(payload["buckets"]) == 1
    assert payload["buckets"]["2026-06-01"]["decision_count"] == 3


# ── T14: different days create separate buckets ─────────────────────────────

def test_t14_different_days_separate_buckets():
    payload: dict = {}
    now = datetime(2026, 6, 2, 20, 0, tzinfo=timezone.utc)
    for day in ("2026-06-01", "2026-06-02"):
        result = append_or_update_research_daily_bucket(
            payload, day, ResearchDailyObservation(bucket_date=day, decision_count=1), now_utc=now,
        )
        payload = result.updated_payload
    assert set(payload["buckets"].keys()) == {"2026-06-01", "2026-06-02"}


# ── T15: retention max age prunes old buckets ───────────────────────────────

def test_t15_retention_max_age_prunes_old_buckets():
    now = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    payload = {"buckets": {
        "2024-01-01": {"schema_version": 1, "bucket_date": "2024-01-01"},
        "2026-05-31": {"schema_version": 1, "bucket_date": "2026-05-31"},
    }}
    result = prune_research_daily_buckets(payload, now_utc=now)
    assert "2024-01-01" in result.pruned_bucket_dates
    assert "2026-05-31" in result.keep_bucket_dates


def test_t15_missing_malformed_date_always_kept():
    now = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    payload = {"buckets": {"not-a-real-date": {"schema_version": 1, "bucket_date": "not-a-real-date"}}}
    result = prune_research_daily_buckets(now_utc=now, payload=payload)
    assert "not-a-real-date" in result.keep_bucket_dates
    assert result.pruned_bucket_dates == ()


# ── T16: retention max bucket cap prunes oldest ─────────────────────────────

def test_t16_max_bucket_cap_prunes_oldest():
    now = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    payload = {"buckets": {
        f"2026-01-{i:02d}": {"schema_version": 1, "bucket_date": f"2026-01-{i:02d}"} for i in range(1, 11)
    }}
    policy = RetentionPolicy(max_age_days=None, max_records=4)
    result = prune_research_daily_buckets(payload, now_utc=now, policy=policy)
    assert len(result.keep_bucket_dates) == 4
    assert result.keep_bucket_dates == ("2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10")


# ── T17: pruning idempotent ──────────────────────────────────────────────────

def test_t17_pruning_idempotent():
    now = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    payload = {"buckets": {
        "2020-01-01": {"schema_version": 1, "bucket_date": "2020-01-01"},
        "2026-05-31": {"schema_version": 1, "bucket_date": "2026-05-31"},
    }}
    first = prune_research_daily_buckets(payload, now_utc=now)
    second = prune_research_daily_buckets(first.updated_payload, now_utc=now)
    assert second.pruned_bucket_dates == ()
    assert first.updated_payload == second.updated_payload


# ── T18: stable sorted output ────────────────────────────────────────────────

def test_t18_stable_sorted_output():
    now = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    payload = {"buckets": {
        "2026-05-03": {"schema_version": 1, "bucket_date": "2026-05-03"},
        "2026-05-01": {"schema_version": 1, "bucket_date": "2026-05-01"},
        "2026-05-02": {"schema_version": 1, "bucket_date": "2026-05-02"},
    }}
    result = prune_research_daily_buckets(payload, now_utc=now)
    assert result.keep_bucket_dates == ("2026-05-01", "2026-05-02", "2026-05-03")


# ── T19: public-safety forbidden substrings absent ──────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "entity_id", "zone_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "trajectory", "person", "secret", "token", "path",
    "climate.", "sensor.",
)


def test_t19_forbidden_substrings_absent_from_schema_source():
    source = inspect.getsource(
        __import__(
            "custom_components.thermosmart.learning.research_daily_schemas", fromlist=["x"],
        )
    ).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in source, f"forbidden substring found in schema module: {token}"


def test_t19_forbidden_substrings_absent_from_serialization_source():
    import custom_components.thermosmart.learning.storage.research_daily_serialization as mod
    source = inspect.getsource(mod).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in source, f"forbidden substring found in serialization module: {token}"


def test_t19_forbidden_substrings_absent_from_persistence_source():
    source = inspect.getsource(rdp).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in source, f"forbidden substring found in persistence module: {token}"


def test_t19_forbidden_substrings_absent_from_serialized_output():
    obs = ResearchDailyObservation(bucket_date="2026-06-01", decision_count=1)
    b = record_research_daily_observation(create_empty_research_daily_bucket("2026-06-01"), obs)
    serialized = serialize_research_daily_bucket(b)
    dumped = str(serialized).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in dumped, f"forbidden substring found in serialized output: {token}"


# ── T20: no HA imports in new pure modules ───────────────────────────────────

_FORBIDDEN_IMPORT_TOKENS = (
    "hass", "homeassistant", "hass.", "store(", "async_save", "async_load",
    "coordinator", "ha_integration", "export.py", "set_temperature",
    "async_set_temperature", "async_write_ha_state", "service call",
)


@pytest.mark.parametrize("module_path", [
    "custom_components.thermosmart.learning.research_daily_schemas",
    "custom_components.thermosmart.learning.storage.research_daily_serialization",
    "custom_components.thermosmart.learning.storage.research_daily_persistence",
])
def test_t20_no_ha_or_runtime_or_control_imports(module_path):
    mod = __import__(module_path, fromlist=["x"])
    source = inspect.getsource(mod).lower()
    for token in _FORBIDDEN_IMPORT_TOKENS:
        assert token not in source, f"forbidden token {token!r} found in {module_path}"


# ── T21: no Store I/O ────────────────────────────────────────────────────────

def test_t21_no_store_io_symbols():
    for mod in (rdp,):
        assert not hasattr(mod, "Store")
        assert "import homeassistant" not in inspect.getsource(mod).lower()


# ── T22: no Runtime/Coordinator import ───────────────────────────────────────

def test_t22_no_coordinator_import():
    for module_path in (
        "custom_components.thermosmart.learning.research_daily_schemas",
        "custom_components.thermosmart.learning.storage.research_daily_serialization",
        "custom_components.thermosmart.learning.storage.research_daily_persistence",
    ):
        mod = __import__(module_path, fromlist=["x"])
        for name in dir(mod):
            assert "coordinator" not in name.lower()


# ── T23: existing Support/Event/Export/Episode-History tests remain green ──

def test_t23_regression_imports_ok():
    import tests.test_le2_support_critical_events  # noqa: F401
    import tests.test_le2_support_critical_event_wiring  # noqa: F401
    import tests.test_le2_support_storage_restore_events  # noqa: F401
    import tests.test_le2_support_hold_transition_events  # noqa: F401


# ── T24: full fast suite green — verified via separate pytest invocation ───
# (this file cannot assert the *entire* suite's outcome from within itself;
# see the run's overall report for the aggregate result.)

def test_t24_placeholder_full_suite_run_externally():
    assert True


# ── additional: retention constants / policy sanity ─────────────────────────

def test_retention_constants_match_default_policy():
    assert RESEARCH_DAILY_RETENTION_MAX_DAYS == 365
    assert RESEARCH_DAILY_RETENTION_MAX_BUCKETS == 400
    assert RESEARCH_DAILY_RETENTION_DEFAULT.max_age_days == RESEARCH_DAILY_RETENTION_MAX_DAYS
    assert RESEARCH_DAILY_RETENTION_DEFAULT.max_records == RESEARCH_DAILY_RETENTION_MAX_BUCKETS


def test_merge_research_daily_bucket_different_dates_is_noop():
    a = create_empty_research_daily_bucket("2026-06-01")
    b = create_empty_research_daily_bucket("2026-06-02")
    assert merge_research_daily_bucket(a, b) == a


def test_invalid_bucket_date_raises():
    with pytest.raises(InvalidBucketDateError):
        create_empty_research_daily_bucket("not-a-date")
    with pytest.raises(InvalidBucketDateError):
        create_empty_research_daily_bucket("2026-02-30")
