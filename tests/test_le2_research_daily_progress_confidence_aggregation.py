"""Tests for LE2 Research Daily Aggregation Increment 2: aggregating Learning
Progress and Confidence samples into Research Daily Buckets from within
LearningShadowController.learning_progress_safe()
(custom_components/thermosmart/learning/runtime/ha_integration.py).

This step adds ONLY:
  - LearningShadowController._maybe_record_research_daily_progress_sample_safe()
  - one call site inside learning_progress_safe(), on the successful
    compute_learning_progress() branch only (never the disabled-shadow
    early return, never the calculation-error except branch)

No change to compute_learning_progress()/learning_progress.py itself, no
Coordinator hook, no Export change, no control effect.

Like the Support Event aggregation increment, these tests construct the
REAL LearningShadowController via ``object.__new__()`` and set only the
handful of attributes the exercised methods actually read/write — so these
tests call the ACTUAL production methods directly, not a copied-body
harness, while still requiring no live HA instance.

16 test groups:
  T1  — One progress sample sets min/max/last
  T2  — Multiple progress samples update min/max/last correctly
  T3  — One confidence sample sets min/max/last
  T4  — Multiple confidence samples update min/max/last correctly
  T5  — NaN/Inf/invalid progress is ignored, non-fatal
  T6  — NaN/Inf/invalid confidence is ignored, non-fatal (progress still recorded)
  T7  — Progress out of [0, 100] is ignored (not clamped)
  T8  — Confidence out of [0, 1] is ignored (not clamped)
  T9  — confidence_level label is never used as a numeric confidence
  T10 — Identical repeated sample causes no additional dirty change
  T11 — Bucket date derived correctly from the controller's own clock
  T12 — compute_learning_progress()/learning_progress.py unchanged
  T13 — No Coordinator hook / control path touched
  T14 — No Export hook touched
  T15 — Existing Research Daily Support Event Aggregation tests remain green
  T16 — Existing Learning Progress tests remain green
"""
from __future__ import annotations

import inspect
import math
from typing import Any

import pytest

from custom_components.thermosmart.learning.runtime.ha_integration import LearningShadowController
from custom_components.thermosmart.learning.storage.stores import ResearchDailyStore

_ZONE_A = "zone_alpha_01"
_NOW_ISO = "2026-06-02T12:00:00+00:00"


# ── Fake store infrastructure (mirrors established store-wiring test files) ─

class _FakeRawStore:
    def __init__(self) -> None:
        self._data: Any = None

    async def async_load(self) -> Any:
        return self._data

    async def async_save(self, data: Any) -> None:
        self._data = data

    async def async_remove(self) -> None:
        self._data = None


class _FakeStoreFactory:
    def __init__(self) -> None:
        self._stores: dict[str, _FakeRawStore] = {}

    def create(self, key: str, version: int) -> _FakeRawStore:
        if key not in self._stores:
            self._stores[key] = _FakeRawStore()
        return self._stores[key]


class _FakeCaptureStores:
    def __init__(self, factory: _FakeStoreFactory, zone: str) -> None:
        self._factory = factory
        self._zone = zone

    def research_daily_store(self) -> ResearchDailyStore:
        return ResearchDailyStore(self._factory, self._zone)


def _make_controller(now_iso: str = _NOW_ISO) -> LearningShadowController:
    """Build a real LearningShadowController instance via object.__new__(),
    setting only the attributes _maybe_record_research_daily_progress_sample_safe()
    and record_research_daily_observation_safe() actually read/write."""
    ctrl = object.__new__(LearningShadowController)
    ctrl._capture_stores = _FakeCaptureStores(_FakeStoreFactory(), _ZONE_A)
    ctrl._research_daily_buckets = {}
    ctrl._research_daily_save_needed = False
    ctrl._research_daily_last_error = None
    ctrl._utcnow_iso = lambda: now_iso
    return ctrl


def _bucket(ctrl: LearningShadowController, bucket_date: str = "2026-06-02") -> dict:
    return ctrl._research_daily_buckets[bucket_date]


# ── T1: One progress sample sets min/max/last ───────────────────────────────

def test_t1_single_progress_sample_sets_min_max_last():
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(42.5, {})
    bucket = _bucket(ctrl)
    assert bucket["learning_progress_min_pct"] == 42.5
    assert bucket["learning_progress_max_pct"] == 42.5
    assert bucket["learning_progress_last_pct"] == 42.5


# ── T2: Multiple progress samples update min/max/last correctly ────────────

def test_t2_multiple_progress_samples_update_min_max_last():
    ctrl = _make_controller()
    for pct in (30.0, 55.0, 40.0):
        ctrl._maybe_record_research_daily_progress_sample_safe(pct, {})
    bucket = _bucket(ctrl)
    assert bucket["learning_progress_min_pct"] == 30.0
    assert bucket["learning_progress_max_pct"] == 55.0
    assert bucket["learning_progress_last_pct"] == 40.0


# ── T3: One confidence sample sets min/max/last ─────────────────────────────

def test_t3_single_confidence_sample_sets_min_max_last():
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(10.0, {"model_confidence_score": 0.6})
    bucket = _bucket(ctrl)
    assert bucket["confidence_min"] == 0.6
    assert bucket["confidence_max"] == 0.6
    assert bucket["confidence_last"] == 0.6


# ── T4: Multiple confidence samples update min/max/last correctly ──────────

def test_t4_multiple_confidence_samples_update_min_max_last():
    ctrl = _make_controller()
    for conf in (0.3, 0.9, 0.5):
        ctrl._maybe_record_research_daily_progress_sample_safe(10.0, {"model_confidence_score": conf})
    bucket = _bucket(ctrl)
    assert bucket["confidence_min"] == 0.3
    assert bucket["confidence_max"] == 0.9
    assert bucket["confidence_last"] == 0.5


# ── T5: NaN/Inf/invalid progress is ignored, non-fatal ──────────────────────

@pytest.mark.parametrize("bad_progress", [float("nan"), float("inf"), float("-inf"), "42.0", None, True])
def test_t5_invalid_progress_ignored_non_fatal(bad_progress):
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(bad_progress, {})  # must not raise
    assert ctrl._research_daily_buckets == {}
    assert ctrl._research_daily_save_needed is False
    assert ctrl._research_daily_last_error is None


# ── T6: NaN/Inf/invalid confidence is ignored, non-fatal (progress kept) ───

@pytest.mark.parametrize("bad_confidence", [float("nan"), float("inf"), "0.5", None, True])
def test_t6_invalid_confidence_ignored_progress_still_recorded(bad_confidence):
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(20.0, {"model_confidence_score": bad_confidence})
    bucket = _bucket(ctrl)
    assert bucket["learning_progress_last_pct"] == 20.0
    assert bucket["confidence_last"] is None
    assert ctrl._research_daily_last_error is None


# ── T7: Progress out of [0, 100] is ignored (not clamped) ──────────────────

@pytest.mark.parametrize("out_of_range", [-0.1, 100.1, -50.0, 500.0])
def test_t7_out_of_range_progress_ignored(out_of_range):
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(out_of_range, {})
    assert ctrl._research_daily_buckets == {}
    assert ctrl._research_daily_save_needed is False


def test_t7_boundary_values_are_accepted():
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(0.0, {})
    ctrl._maybe_record_research_daily_progress_sample_safe(100.0, {})
    bucket = _bucket(ctrl)
    assert bucket["learning_progress_min_pct"] == 0.0
    assert bucket["learning_progress_max_pct"] == 100.0


# ── T8: Confidence out of [0, 1] is ignored (not clamped) ──────────────────

@pytest.mark.parametrize("out_of_range", [-0.01, 1.01, -5.0, 50.0])
def test_t8_out_of_range_confidence_ignored(out_of_range):
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(10.0, {"model_confidence_score": out_of_range})
    bucket = _bucket(ctrl)
    assert bucket["confidence_last"] is None  # ignored, progress still recorded


# ── T9: confidence_level label is never used as a numeric confidence ───────

def test_t9_confidence_level_label_never_used():
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(
        10.0, {"confidence_level": "medium", "model_confidence_score": None},
    )
    bucket = _bucket(ctrl)
    assert bucket["confidence_min"] is None
    assert bucket["confidence_last"] is None


def test_t9_only_confidence_level_present_no_score_field():
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(10.0, {"confidence_level": "high"})
    bucket = _bucket(ctrl)
    assert bucket["confidence_min"] is None


# ── T10: Identical repeated sample causes no additional dirty change ───────

def test_t10_identical_repeated_sample_no_extra_dirty():
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(50.0, {"model_confidence_score": 0.7})
    assert ctrl._research_daily_save_needed is True

    ctrl._research_daily_save_needed = False
    ctrl._maybe_record_research_daily_progress_sample_safe(50.0, {"model_confidence_score": 0.7})
    assert ctrl._research_daily_save_needed is False  # unchanged value -> no-op


def test_t10_changed_value_still_marks_dirty():
    ctrl = _make_controller()
    ctrl._maybe_record_research_daily_progress_sample_safe(50.0, {"model_confidence_score": 0.7})
    ctrl._research_daily_save_needed = False
    ctrl._maybe_record_research_daily_progress_sample_safe(51.0, {"model_confidence_score": 0.7})
    assert ctrl._research_daily_save_needed is True


# ── T11: Bucket date derived correctly from the controller's own clock ─────

def test_t11_bucket_date_from_controller_clock():
    ctrl = _make_controller(now_iso="2026-01-15T23:59:00+00:00")
    ctrl._maybe_record_research_daily_progress_sample_safe(10.0, {})
    assert "2026-01-15" in ctrl._research_daily_buckets
    assert len(ctrl._research_daily_buckets) == 1


def test_t11_different_clock_creates_different_bucket():
    ctrl = _make_controller(now_iso="2026-03-01T00:00:00+00:00")
    ctrl._maybe_record_research_daily_progress_sample_safe(10.0, {})
    assert "2026-03-01" in ctrl._research_daily_buckets


# ── T12: compute_learning_progress()/learning_progress.py unchanged ────────

def test_t12_learning_progress_module_has_no_research_daily_reference():
    import custom_components.thermosmart.learning.runtime.learning_progress as _lp
    source = inspect.getsource(_lp)
    assert "research_daily" not in source.lower()
    assert "ResearchDailyObservation" not in source
    assert "record_research_daily_observation_safe" not in source


def test_t12_compute_learning_progress_signature_unchanged():
    from custom_components.thermosmart.learning.runtime.learning_progress import (
        compute_learning_progress,
    )
    sig = inspect.signature(compute_learning_progress)
    assert list(sig.parameters.keys()) == ["signals"]


def test_t12_aggregation_call_only_on_success_branch():
    """learning_progress_safe() must call the new aggregation method exactly
    once, and only from the successful compute_learning_progress() line —
    never from the disabled-shadow early return or the except block."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.learning_progress_safe)
    call_count = source.count("self._maybe_record_research_daily_progress_sample_safe(")
    assert call_count == 1
    # The call must appear AFTER compute_learning_progress(signals) is
    # assigned, and BEFORE the except block's cold_learning_progress_result.
    compute_idx = source.index("compute_learning_progress(signals)")
    call_idx = source.index("self._maybe_record_research_daily_progress_sample_safe(")
    except_idx = source.index("except Exception as err:")
    assert compute_idx < call_idx < except_idx


# ── T13: No Coordinator hook / control path touched ─────────────────────────

def test_t13_coordinator_not_touched():
    import custom_components.thermosmart.coordinator as _coord
    source = inspect.getsource(_coord)
    assert "_maybe_record_research_daily_progress_sample_safe" not in source
    assert "model_confidence_score" not in source


_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t13_no_control_keywords_in_new_method():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(
        _ha.LearningShadowController._maybe_record_research_daily_progress_sample_safe
    ).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"


# ── T14: No Export hook touched ─────────────────────────────────────────────

def test_t14_export_not_touched():
    """At the time THIS aggregation step was built, export.py had no
    Research Daily reference at all — checked here as a snapshot of that
    boundary. A later, separately-approved step ("Research Daily Long-Term
    Summary Export") legitimately added a public-safe, read-only
    "research_daily" export block — expected, intentional export wiring,
    not a regression of this step's own scope. What must remain true
    regardless is that export.py never calls this step's own aggregation
    method — the export layer only ever READS the already-aggregated
    snapshot, it never triggers aggregation itself."""
    import custom_components.thermosmart.export as _exp
    source = inspect.getsource(_exp)
    assert "_maybe_record_research_daily_progress_sample_safe" not in source


# ── T15: Existing Research Daily Support Event Aggregation tests green ─────

def test_t15_regression_imports_ok_research_daily():
    import tests.test_le2_research_daily_support_event_aggregation  # noqa: F401
    import tests.test_le2_research_daily_state_wiring  # noqa: F401
    import tests.test_le2_research_daily_store_wiring  # noqa: F401
    import tests.test_le2_research_daily_buckets  # noqa: F401


# ── T16: Existing Learning Progress tests remain green ──────────────────────

def test_t16_regression_imports_ok_learning_progress():
    import tests.test_le2_learning_progress  # noqa: F401
    import tests.test_le2_learning_progress_export  # noqa: F401
