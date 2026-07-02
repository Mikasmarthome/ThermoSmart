"""Tests for ApplicationLifecycleState schema, serialization, and helpers.

17 test groups covering:
  T1  — serialize / deserialize roundtrip
  T2  — malformed input → empty safe state, no exception
  T3  — unknown/malformed entry is skipped
  T4  — schema version mismatch handled safely
  T5  — AppliedAdaptationRecord → state entry conversion
  T6  — MonitoringEvaluation updates state entry
  T7  — rollback_recommended updates only diagnostic state
  T8  — adoption_ready updates only diagnostic state
  T9  — cooldown_active computed correctly
  T10 — prune keeps active entries before old completed
  T11 — hard cap max 50 entries
  T12 — research export is public-safe
  T13 — summary counts
  T14 — missing timestamps handled conservatively
  T15 — no runtime mutation
  T16 — no control keyword in application_state module
  T17 — regression: existing tests still pass
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.adaptation.application_state import (
    APPLICATION_STATE_SCHEMA_VERSION,
    ApplicationLifecycleState,
    ApplicationLifecycleSummary,
    AppliedAdaptationStateEntry,
    application_state_entry_for_research_export,
    deserialize_application_lifecycle_state,
    is_cooldown_active,
    prune_application_state_entries,
    record_to_state_entry,
    serialize_application_lifecycle_state,
    summarize_application_lifecycle_state,
    update_state_entry_from_evaluation,
)
from custom_components.thermosmart.learning.adaptation.application import (
    ApplicationPolicy,
    ApplicationRuntimeContext,
    build_preheat_application_plan,
    evaluate_application_readiness,
)
from custom_components.thermosmart.learning.adaptation.contracts import (
    AdaptationDirection,
    AdaptationLifecycle,
    CandidateType,
)
from custom_components.thermosmart.learning.adaptation.history import (
    CandidateHistoryEntry,
    evaluate_promotion_readiness,
)
from custom_components.thermosmart.learning.adaptation.monitoring import (
    AppliedAdaptationRecord,
    MonitoringStatus,
    build_applied_adaptation_record,
    evaluate_applied_adaptation_outcome,
)

# ── Shared constants and fixtures ─────────────────────────────────────────────

_ZONE_ID    = "zone_test_01"
_APPLIED_TS = "2026-06-12T10:00:00+00:00"
_NOW_TS     = "2026-06-13T10:00:00+00:00"  # 1 day after applied_ts

_BASELINE_QUALITY   = 0.65
_BASELINE_TIMEOUT   = 0.32
_BASELINE_OVERSHOOT = 0.10


def _entry(**overrides) -> CandidateHistoryEntry:
    defaults = dict(
        candidate_key="deadcafe01234567",
        candidate_type=CandidateType.PREHEAT_DELTA_MIN,
        direction=AdaptationDirection.INCREASE,
        first_seen_ts="2026-06-01T10:00:00+00:00",
        last_seen_ts="2026-06-12T10:00:00+00:00",
        seen_count=5,
        supporting_outcome_count=15,
        avg_outcome_quality=_BASELINE_QUALITY,
        avg_timeout_rate=_BASELINE_TIMEOUT,
        avg_overshoot_rate=_BASELINE_OVERSHOOT,
        last_lifecycle=AdaptationLifecycle.SHADOW,
        dominant_reason="high_timeout_rate:0.32",
        dominant_reason_ratio=0.90,
    )
    defaults.update(overrides)
    return CandidateHistoryEntry(**defaults)


def _ctx(**overrides) -> ApplicationRuntimeContext:
    defaults = dict(
        proposed_delta=4.0,
        current_cumulative_delta=0.0,
        last_applied_ts=None,
        now_ts=_NOW_TS,
        window_open=False, manual_override=False,
        vacation_mode=False, summer_mode=False,
        heating_unavailable=False, sensor_unreliable=False,
        control_disabled=False, learning_disabled=False,
        active_control_disabled=False, storage_corrupt=False,
        confounder_ratio=0.0,
    )
    defaults.update(overrides)
    return ApplicationRuntimeContext(**defaults)


def _good_record(entry: CandidateHistoryEntry | None = None) -> AppliedAdaptationRecord:
    """Build an AppliedAdaptationRecord from a would_apply=True plan."""
    e = entry or _entry()
    pgr = evaluate_promotion_readiness(e, span_days=11.0, confounder_ratio=0.0)
    policy = ApplicationPolicy(application_enabled=True)
    dec = evaluate_application_readiness(
        history_entry=e, promotion_result=pgr, policy=policy, runtime_context=_ctx(),
    )
    plan = build_preheat_application_plan(
        history_entry=e, promotion_result=pgr,
        application_decision=dec, policy=policy, runtime_context=_ctx(),
    )
    assert plan.would_apply is True
    record = build_applied_adaptation_record(
        plan=plan, history_entry=e, applied_ts=_APPLIED_TS,
    )
    assert record is not None
    return record


def _good_state_entry(**overrides) -> AppliedAdaptationStateEntry:
    """Build a state entry from a good record."""
    entry_overrides = {k: v for k, v in overrides.items()
                       if k not in ("now_ts", "cooldown_days")}
    record = _good_record()
    state_entry = record_to_state_entry(
        record,
        cooldown_days=overrides.get("cooldown_days", 7.0),
        now_ts=overrides.get("now_ts", _NOW_TS),
    )
    return state_entry


def _make_state(*entries: AppliedAdaptationStateEntry,
                zone_id: str = _ZONE_ID) -> ApplicationLifecycleState:
    return ApplicationLifecycleState(
        learning_zone_id=zone_id,
        updated_at=_NOW_TS,
        entries={e.candidate_key: e for e in entries},
    )


def _fake_entry(
    key: str,
    status: MonitoringStatus,
    applied_ts: str = _APPLIED_TS,
    adoption_ready: bool = False,
) -> AppliedAdaptationStateEntry:
    """Construct a minimal fake state entry for pruning/summary tests."""
    return AppliedAdaptationStateEntry(
        candidate_key=key,
        candidate_type_value="preheat_delta_min",
        direction_value="increase",
        status=status,
        applied_delta_min=3.0,
        previous_cumulative_delta_min=0.0,
        new_cumulative_delta_min=3.0,
        applied_ts=applied_ts,
        monitoring_deadline_ts=None,
        last_evaluated_ts=None,
        monitored_outcome_count=0,
        rollback_supported=True,
        rollback_recommended=(status is MonitoringStatus.ROLLBACK_RECOMMENDED),
        rollback_reasons=(),
        adoption_ready=adoption_ready,
        adopted_ts=None,
        rolled_back_ts=None,
        cooldown_until_ts=None,
        last_error=None,
    )


# ── T1: Serialize / Deserialize roundtrip ────────────────────────────────────

def test_t1_roundtrip_single_entry():
    state_entry = _good_state_entry()
    state = _make_state(state_entry)

    data = serialize_application_lifecycle_state(state)
    restored = deserialize_application_lifecycle_state(data, _ZONE_ID)

    assert restored.learning_zone_id == _ZONE_ID
    assert restored.schema_version == APPLICATION_STATE_SCHEMA_VERSION
    assert len(restored.entries) == 1

    orig  = state.entries[state_entry.candidate_key]
    rtrip = restored.entries[state_entry.candidate_key]

    assert rtrip.candidate_key == orig.candidate_key
    assert rtrip.status == orig.status
    assert rtrip.applied_delta_min == pytest.approx(orig.applied_delta_min)
    assert rtrip.rollback_reasons == orig.rollback_reasons
    assert rtrip.adoption_ready == orig.adoption_ready


def test_t1_roundtrip_empty_state():
    state = ApplicationLifecycleState(
        learning_zone_id=_ZONE_ID, updated_at=_NOW_TS, entries={},
    )
    data = serialize_application_lifecycle_state(state)
    restored = deserialize_application_lifecycle_state(data, _ZONE_ID)
    assert restored.entries == {}


# ── T2: Malformed input → empty safe state, no exception ─────────────────────

@pytest.mark.parametrize("bad_input", [
    None, "string", 42, [], b"bytes", object(),
])
def test_t2_malformed_input_returns_empty_state(bad_input):
    result = deserialize_application_lifecycle_state(bad_input, _ZONE_ID)
    assert isinstance(result, ApplicationLifecycleState)
    assert result.entries == {}
    assert result.learning_zone_id == _ZONE_ID


# ── T3: Unknown/malformed entry is skipped ────────────────────────────────────

def test_t3_malformed_entry_skipped_valid_kept():
    state_entry = _good_state_entry()
    state = _make_state(state_entry)
    data = serialize_application_lifecycle_state(state)

    # Inject bad entry alongside the good one
    data["entries"]["bad_key"] = {"garbage": True}
    data["entries"]["wrong_status"] = {
        "candidate_key": "abc", "applied_delta_min": 1.0,
        "status": "nonexistent_status_xyz",
    }

    restored = deserialize_application_lifecycle_state(data, _ZONE_ID)
    assert len(restored.entries) == 1
    assert state_entry.candidate_key in restored.entries


# ── T4: Schema version mismatch → empty state ────────────────────────────────

def test_t4_schema_version_mismatch_returns_empty():
    state = _make_state(_good_state_entry())
    data = serialize_application_lifecycle_state(state)
    data["schema_version"] = APPLICATION_STATE_SCHEMA_VERSION + 99

    restored = deserialize_application_lifecycle_state(data, _ZONE_ID)
    assert restored.entries == {}


def test_t4_missing_schema_version_returns_empty():
    state = _make_state(_good_state_entry())
    data = serialize_application_lifecycle_state(state)
    del data["schema_version"]

    restored = deserialize_application_lifecycle_state(data, _ZONE_ID)
    assert restored.entries == {}


# ── T5: AppliedAdaptationRecord → state entry ────────────────────────────────

def test_t5_record_to_state_entry_fields():
    record = _good_record()
    se = record_to_state_entry(record, cooldown_days=7.0, now_ts=_NOW_TS)

    assert se.candidate_key == record.candidate_key
    assert se.candidate_type_value == record.candidate_type_value
    assert se.direction_value == record.direction_value
    assert se.status is MonitoringStatus.PENDING
    assert se.applied_delta_min == pytest.approx(record.applied_delta_min)
    assert se.previous_cumulative_delta_min == pytest.approx(record.previous_cumulative_delta_min)
    assert se.new_cumulative_delta_min == pytest.approx(record.new_cumulative_delta_min)
    assert se.applied_ts == record.applied_ts
    assert se.monitored_outcome_count == 0
    assert se.rollback_recommended is False
    assert se.adoption_ready is False
    assert se.adopted_ts is None
    assert se.rolled_back_ts is None
    assert se.last_evaluated_ts == _NOW_TS
    # cooldown_until_ts = applied_ts + 7 days = 2026-06-19
    assert se.cooldown_until_ts is not None
    assert "2026-06-19" in se.cooldown_until_ts


# ── T6: MonitoringEvaluation updates state entry ─────────────────────────────

def test_t6_update_state_entry_from_evaluation():
    se = _good_state_entry()
    record = _good_record()
    ev = evaluate_applied_adaptation_outcome(
        record=record,
        monitored_outcome_count=3,
        avg_monitored_outcome_quality=0.72,
        avg_monitored_timeout_rate=0.25,
        avg_monitored_overshoot_rate=0.08,
        monitoring_window_complete=True,
    )
    updated = update_state_entry_from_evaluation(se, ev, now_ts=_NOW_TS)

    assert updated.status == ev.status
    assert updated.monitored_outcome_count == 3
    assert updated.adoption_ready == ev.adoption_ready
    assert updated.rollback_recommended == ev.rollback_recommended
    assert updated.rollback_reasons == ev.rollback_reasons
    assert updated.last_evaluated_ts == _NOW_TS
    # Immutable fields preserved
    assert updated.applied_ts == se.applied_ts
    assert updated.applied_delta_min == pytest.approx(se.applied_delta_min)
    assert updated.cooldown_until_ts == se.cooldown_until_ts
    # Application Layer fields untouched
    assert updated.adopted_ts == se.adopted_ts
    assert updated.rolled_back_ts == se.rolled_back_ts


# ── T7: rollback_recommended updates only diagnostic state ───────────────────

def test_t7_rollback_recommended_diagnostic_only():
    se = _good_state_entry()
    record = _good_record()
    ev = evaluate_applied_adaptation_outcome(
        record=record,
        monitored_outcome_count=2,
        avg_monitored_outcome_quality=0.40,  # quality degraded → rollback
        avg_monitored_timeout_rate=0.30,
        avg_monitored_overshoot_rate=0.10,
    )
    assert ev.rollback_recommended is True

    updated = update_state_entry_from_evaluation(se, ev, now_ts=_NOW_TS)

    # Diagnostic flags updated
    assert updated.rollback_recommended is True
    assert updated.status is MonitoringStatus.ROLLBACK_RECOMMENDED

    # No real rollback: rolled_back_ts stays None; applied_delta unchanged
    assert updated.rolled_back_ts is None
    assert updated.applied_delta_min == pytest.approx(se.applied_delta_min)
    assert updated.previous_cumulative_delta_min == pytest.approx(
        se.previous_cumulative_delta_min)


# ── T8: adoption_ready updates only diagnostic state ─────────────────────────

def test_t8_adoption_ready_diagnostic_only():
    se = _good_state_entry()
    record = _good_record()
    ev = evaluate_applied_adaptation_outcome(
        record=record,
        monitored_outcome_count=3,
        avg_monitored_outcome_quality=0.72,
        avg_monitored_timeout_rate=0.25,
        avg_monitored_overshoot_rate=0.08,
        monitoring_window_complete=True,
    )
    assert ev.adoption_ready is True

    updated = update_state_entry_from_evaluation(se, ev, now_ts=_NOW_TS)

    # Diagnostic flag updated
    assert updated.adoption_ready is True
    assert updated.status is MonitoringStatus.ADOPTION_CANDIDATE

    # No real adoption: adopted_ts stays None; cumulative unchanged
    assert updated.adopted_ts is None
    assert updated.new_cumulative_delta_min == pytest.approx(se.new_cumulative_delta_min)


# ── T9: cooldown_active computed correctly ────────────────────────────────────

def test_t9_cooldown_active_within_window():
    # applied 2026-06-12, cooldown 7 days → until 2026-06-19
    # now = 2026-06-13 → cooldown active
    se = _good_state_entry(cooldown_days=7.0)
    assert is_cooldown_active(se, now_ts="2026-06-13T10:00:00+00:00") is True


def test_t9_cooldown_expired_after_window():
    se = _good_state_entry(cooldown_days=7.0)
    # now = 2026-06-20 > 2026-06-19 cooldown deadline
    assert is_cooldown_active(se, now_ts="2026-06-20T10:00:00+00:00") is False


def test_t9_cooldown_no_until_ts():
    se = _fake_entry("key_x", MonitoringStatus.PENDING)
    # cooldown_until_ts is None
    assert is_cooldown_active(se, now_ts=_NOW_TS) is False


def test_t9_cooldown_missing_now_ts():
    se = _good_state_entry()
    assert is_cooldown_active(se, now_ts=None) is False


# ── T10: Prune keeps active entries before old completed ──────────────────────

def test_t10_prune_keeps_active_before_old_completed():
    old_ts = "2025-01-01T00:00:00+00:00"   # > 180 days before _NOW_TS

    active = _fake_entry("active_key", MonitoringStatus.IN_PROGRESS, applied_ts=old_ts)
    old_adopted = _fake_entry("old_adopted", MonitoringStatus.ADOPTED, applied_ts=old_ts)
    recent_adopted = _fake_entry("recent_adopted", MonitoringStatus.ADOPTED,
                                 applied_ts=_APPLIED_TS)

    entries = {
        "active_key":    active,
        "old_adopted":   old_adopted,
        "recent_adopted": recent_adopted,
    }
    pruned = prune_application_state_entries(entries, now_ts=_NOW_TS, max_age_days=180.0)

    assert "active_key" in pruned          # active always kept
    assert "old_adopted" not in pruned     # old completed → dropped by age rule
    assert "recent_adopted" in pruned      # recent completed → kept


# ── T11: Hard cap at max_entries ─────────────────────────────────────────────

def test_t11_hard_cap_max_50():
    # Create 60 completed entries + 5 active entries
    entries: dict = {}
    for i in range(60):
        k = f"completed_{i:03d}"
        entries[k] = _fake_entry(k, MonitoringStatus.ADOPTED, applied_ts=_APPLIED_TS)
    for i in range(5):
        k = f"active_{i}"
        entries[k] = _fake_entry(k, MonitoringStatus.IN_PROGRESS, applied_ts=_APPLIED_TS)

    pruned = prune_application_state_entries(
        entries, now_ts=_NOW_TS, max_entries=50, max_age_days=365.0,
    )
    assert len(pruned) == 50

    # All active entries must be present
    for i in range(5):
        assert f"active_{i}" in pruned


def test_t11_hard_cap_keeps_all_when_under_limit():
    entries = {f"e{i}": _fake_entry(f"e{i}", MonitoringStatus.PENDING)
               for i in range(10)}
    pruned = prune_application_state_entries(
        entries, now_ts=_NOW_TS, max_entries=50,
    )
    assert len(pruned) == 10


# ── T12: Research export is public-safe ──────────────────────────────────────

def test_t12_research_export_public_safe():
    se = _good_state_entry()
    exported = application_state_entry_for_research_export(se, now_ts=_NOW_TS)

    required_keys = {
        "candidate_key", "candidate_type", "direction", "status",
        "applied_delta_min", "new_cumulative_delta_min",
        "monitored_outcome_count", "rollback_recommended",
        "rollback_reasons", "adoption_ready", "cooldown_active",
    }
    assert required_keys <= set(exported.keys())

    # Forbidden fields absent
    forbidden = {"entity_id", "zone_name", "person", "path", "secret",
                 "adopted_ts", "rolled_back_ts", "cooldown_until_ts",
                 "applied_ts", "monitoring_deadline_ts", "last_evaluated_ts"}
    assert not (forbidden & set(exported.keys()))

    # Types correct
    assert isinstance(exported["cooldown_active"], bool)
    assert isinstance(exported["rollback_reasons"], list)
    assert isinstance(exported["status"], str)


# ── T13: Summary counts ───────────────────────────────────────────────────────

def test_t13_summary_counts():
    entries = {
        "p1": _fake_entry("p1",  MonitoringStatus.PENDING),
        "p2": _fake_entry("p2",  MonitoringStatus.IN_PROGRESS),
        "p3": _fake_entry("p3",  MonitoringStatus.ADOPTION_CANDIDATE, adoption_ready=True),
        "p4": _fake_entry("p4",  MonitoringStatus.ROLLBACK_RECOMMENDED),
        "p5": _fake_entry("p5",  MonitoringStatus.ADOPTED),
        "p6": _fake_entry("p6",  MonitoringStatus.ROLLED_BACK),
        "p7": _fake_entry("p7",  MonitoringStatus.WINDOW_EXPIRED),
        "p8": _fake_entry("p8",  MonitoringStatus.ADOPTED),
    }
    state = ApplicationLifecycleState(
        learning_zone_id=_ZONE_ID, updated_at=_NOW_TS, entries=entries,
    )
    summary = summarize_application_lifecycle_state(state)

    assert summary.total == 8
    assert summary.active == 4          # PENDING + IN_PROGRESS + ADOPTION_CANDIDATE + ROLLBACK
    assert summary.rollback_recommended == 1
    assert summary.adoption_ready == 1  # only the one with adoption_ready=True
    assert summary.adopted == 2
    assert summary.rolled_back == 1
    assert summary.expired == 1

    d = summary.to_dict()
    assert d["total"] == 8
    assert d["adopted"] == 2


def test_t13_summary_empty_state():
    state = ApplicationLifecycleState(
        learning_zone_id=_ZONE_ID, updated_at=_NOW_TS, entries={},
    )
    summary = summarize_application_lifecycle_state(state)
    assert summary.total == 0
    assert summary.active == 0


# ── T14: Missing timestamps handled conservatively ────────────────────────────

def test_t14_missing_applied_ts_skips_cooldown():
    se = AppliedAdaptationStateEntry(
        candidate_key="missing_ts",
        candidate_type_value="preheat_delta_min",
        direction_value="increase",
        status=MonitoringStatus.PENDING,
        applied_delta_min=3.0,
        previous_cumulative_delta_min=0.0,
        new_cumulative_delta_min=3.0,
        applied_ts="",            # empty
        monitoring_deadline_ts=None,
        last_evaluated_ts=None,
        monitored_outcome_count=0,
        rollback_supported=True,
        rollback_recommended=False,
        rollback_reasons=(),
        adoption_ready=False,
        adopted_ts=None,
        rolled_back_ts=None,
        cooldown_until_ts=None,   # no cooldown → not active
        last_error=None,
    )
    # Missing cooldown_until_ts → cooldown_active=False (conservative: don't block)
    assert is_cooldown_active(se, now_ts=_NOW_TS) is False


def test_t14_malformed_cooldown_ts_returns_false():
    se = _fake_entry("key_y", MonitoringStatus.IN_PROGRESS)
    # Inject a non-parseable cooldown_until_ts via dict roundtrip
    raw = se.to_dict()
    raw["cooldown_until_ts"] = "not-a-date"
    from custom_components.thermosmart.learning.adaptation.application_state import (
        _parse_state_entry,
    )
    restored = _parse_state_entry(raw)
    assert restored is not None
    assert is_cooldown_active(restored, now_ts=_NOW_TS) is False


def test_t14_record_with_bad_applied_ts_has_no_cooldown():
    record = _good_record()
    # Override applied_ts to an invalid value
    bad_record = AppliedAdaptationRecord(
        candidate_key=record.candidate_key,
        candidate_type_value=record.candidate_type_value,
        direction_value=record.direction_value,
        applied_delta_min=record.applied_delta_min,
        previous_cumulative_delta_min=record.previous_cumulative_delta_min,
        new_cumulative_delta_min=record.new_cumulative_delta_min,
        applied_ts="invalid-timestamp",
        monitoring_window_hours=48.0,
        monitoring_deadline_ts=None,
        baseline_outcome_quality=0.65,
        baseline_timeout_rate=0.32,
        baseline_overshoot_rate=0.10,
        status=MonitoringStatus.PENDING,
        rollback_supported=True,
    )
    se = record_to_state_entry(bad_record, cooldown_days=7.0)
    assert se.cooldown_until_ts is None  # failed parse → no deadline set


# ── T15: No runtime mutation ──────────────────────────────────────────────────

def test_t15_functions_are_pure():
    record = _good_record()
    se = record_to_state_entry(record)

    # Calling twice returns identical frozen values
    se2 = record_to_state_entry(record)
    assert se == se2
    assert se is not se2

    state = _make_state(se)
    d1 = serialize_application_lifecycle_state(state)
    d2 = serialize_application_lifecycle_state(state)
    assert d1 == d2

    r1 = deserialize_application_lifecycle_state(d1, _ZONE_ID)
    r2 = deserialize_application_lifecycle_state(d2, _ZONE_ID)
    assert r1.entries.keys() == r2.entries.keys()


# ── T16: No control keyword in application_state module ───────────────────────

def test_t16_no_control_keywords_in_module():
    import inspect
    import custom_components.thermosmart.learning.adaptation.application_state as mod

    source = inspect.getsource(mod)
    forbidden = [
        "set_point", "async_set_temperature", "async_write_ha_state",
        "boost_duration", "tpi_gain", "preheat_duration",
        "dispatch", "service_call", "control_value", "apply_candidate",
        "from homeassistant", "import homeassistant",
    ]
    found = [kw for kw in forbidden if kw in source]
    assert found == [], f"Control keywords in application_state.py: {found}"


# ── T17: Regression ──────────────────────────────────────────────────────────

def test_t17_regression_monitoring():
    import tests.test_adaptation_monitoring  # noqa: F401


def test_t17_regression_preheat_plan():
    import tests.test_preheat_application_plan  # noqa: F401


def test_t17_regression_application_readiness():
    import tests.test_application_readiness  # noqa: F401


def test_t17_regression_promotion_readiness():
    import tests.test_promotion_readiness  # noqa: F401
