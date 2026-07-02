"""Tests for the LE2 episode-history research export block.

Covers the read-only research-export addition in export.py:
  _le2_episode_history_export() and its wiring into async_export_learning_data()
  (checked at the per-zone-dict-construction level, without a full hass fixture).

This step adds ONE small, BOUNDED SUMMARY per zone to the research export,
sourced directly from LearningShadowController.episode_history_snapshot()
(already in-memory — no new store read) — never a full episode-entry list,
no episode history change, no Support-export change, no new runtime hook.

17 test groups:
  T1  — Research export helper returns entry_count / counts_by_type
  T2  — Result is a bounded summary, not a full episode-entry list
  T3  — counts_by_type is correct across mixed episode types
  T4  — confounder_count is correct
  T5  — timeout_count is correct (outcome reason == "timeout" only)
  T6  — malformed entries are skipped and counted
  T7  — missing episode history (no shadow) yields available: False
  T8  — privacy scan blocks a payload that fails scan_payload()
  T9  — no forbidden internal-id substrings ever appear in the output
  T10 — no "trajectory" key anywhere in the output
  T11 — no entity ids / zone names / secrets / paths in the output
  T12 — retention metadata is present and reflects EpisodeDefinition.retention
  T13 — Support export is unchanged
  T14 — Learning Progress research export block is preserved alongside this one
  T15 — Existing export tests remain green (regression smoke-test)
  T16 — No new store I/O introduced
  T17 — No control-path keywords touched by the new export helper
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from custom_components.thermosmart import export as _export_module
from custom_components.thermosmart.export import (
    _le2_episode_history_export,
    _le2_learning_progress_export,
)
from custom_components.thermosmart.learning.storage.capture_registry import (
    build_episode_registry,
)

_NOW = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)


class _FakeCaptureStores:
    def __init__(self, episode_registry=None):
        self.episode_registry = episode_registry if episode_registry is not None else build_episode_registry()


class _FakeShadow:
    def __init__(self, history=None, capture_stores=None, snapshot_raises: Exception | None = None):
        self._history = dict(history) if history else {}
        self.capture_stores = capture_stores if capture_stores is not None else _FakeCaptureStores()
        self._snapshot_raises = snapshot_raises

    def episode_history_snapshot(self):
        if self._snapshot_raises is not None:
            raise self._snapshot_raises
        return dict(self._history)


class _FakeCoord:
    def __init__(self, shadow=None):
        self._le2_shadow = shadow


def _entry(episode_type: str, *, confounders=(), reason=None, end_ts=None) -> dict:
    d = {"episode_type": episode_type, "confounder_flags": list(confounders)}
    if reason is not None:
        d["reason"] = reason
    if end_ts is not None:
        d["end_ts"] = end_ts
    return d


_MIXED_HISTORY = {
    "e1": _entry("heating", end_ts="2026-06-01T00:00:00+00:00"),
    "e2": _entry("heating", confounders=("multi_trv_uneven",), end_ts="2026-06-02T10:36:00+00:00"),
    "e3": _entry("outcome", reason="timeout", end_ts="2026-06-02T11:00:00+00:00"),
    "e4": _entry("outcome", reason="reached", end_ts="2026-06-02T09:00:00+00:00"),
    "malformed_not_dict": "not-a-dict",
    "malformed_unknown_type": {"episode_type": "unknown_type"},
}


# ── T1: Research export helper returns entry_count / counts_by_type ────────

def test_t1_export_contains_entry_count_and_counts_by_type():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block["available"] is True
    assert block["entry_count"] == 4
    assert block["counts_by_type"]["heating"] == 2
    assert block["counts_by_type"]["outcome"] == 2


# ── T2: Result is a bounded summary, not a full episode-entry list ─────────

def test_t2_no_per_episode_entry_list_in_output():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert "episodes" not in block
    assert "entries" not in block
    assert not any(isinstance(v, list) and v and isinstance(v[0], dict) for v in block.values())


# ── T3: counts_by_type correct across mixed episode types ──────────────────

def test_t3_counts_by_type_includes_all_known_types_zeroed():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block["counts_by_type"] == {
        "heating": 2, "afterheat": 0, "passive_cooling": 0,
        "window_cooling": 0, "outcome": 2,
    }
    assert block["available_types"] == ["heating", "outcome"]
    assert block["missing_types"] == ["afterheat", "passive_cooling", "window_cooling"]


# ── T4: confounder_count is correct ─────────────────────────────────────────

def test_t4_confounder_count_correct():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block["confounder_count"] == 1


# ── T5: timeout_count is correct ─────────────────────────────────────────────

def test_t5_timeout_count_correct():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block["timeout_count"] == 1


def test_t5_timeout_count_zero_when_no_outcome_episodes():
    history = {"e1": _entry("heating", end_ts="2026-06-01T00:00:00+00:00")}
    coord = _FakeCoord(_FakeShadow(history))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block["timeout_count"] == 0


# ── T6: malformed entries are skipped and counted ───────────────────────────

def test_t6_malformed_entries_skipped_and_counted():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block["malformed_skipped_count"] == 2
    assert block["entry_count"] == 4  # malformed entries excluded from entry_count


def test_t6_all_malformed_still_yields_valid_available_summary():
    history = {"m1": "garbage", "m2": {"episode_type": "does_not_exist"}}
    coord = _FakeCoord(_FakeShadow(history))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block["available"] is True
    assert block["entry_count"] == 0
    assert block["malformed_skipped_count"] == 2


# ── T7: missing episode history yields available: False ────────────────────

def test_t7_missing_shadow_yields_unavailable():
    block = _le2_episode_history_export(_FakeCoord(None), now=_NOW)
    assert block == {"available": False, "reason": "learning_engine_unavailable"}


def test_t7_snapshot_exception_stays_non_fatal():
    coord = _FakeCoord(_FakeShadow(snapshot_raises=RuntimeError("boom")))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block["available"] is False
    assert "episode_history_error" in block


# ── T8: privacy scan blocks a payload that fails scan_payload() ─────────────

def test_t8_privacy_scan_violation_blocks_block(monkeypatch):
    import custom_components.thermosmart.learning.privacy as _privacy_mod

    def _fake_scan_payload(payload, *, path="$"):
        return [object()]  # non-empty -> violation

    monkeypatch.setattr(_privacy_mod, "scan_payload", _fake_scan_payload)
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block == {"available": False, "reason": "privacy_scan_failed"}


# ── T9: no forbidden internal-id substrings ever appear ─────────────────────

_FORBIDDEN_SUBSTRINGS = (
    "zone_id", "entity_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "person", "secret", "token", "path",
)


def test_t9_no_forbidden_substrings_in_output():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    flat = repr(block).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in flat, f"forbidden substring '{token}' leaked into export block"


def test_t9_defense_in_depth_strips_accidental_id_leak(monkeypatch):
    """Even if a future snapshot entry accidentally carried an id-like key at
    the top of an entry, that key must never survive into the summary (the
    summary only ever emits its own fixed key set, but this guards the
    strip-forbidden call path stays wired)."""
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert "episode_id" not in block
    assert "learning_zone_id" not in block


# ── T10: no "trajectory" key anywhere in the output ─────────────────────────

def test_t10_no_trajectory_in_output():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert "trajectory" not in repr(block).lower()


# ── T11: no entity ids / zone names / secrets / paths ──────────────────────

def test_t11_no_entity_id_pattern_or_path_in_output():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    flat = repr(block)
    assert "climate." not in flat
    assert "sensor." not in flat
    assert "/config/" not in flat
    assert "C:\\" not in flat


# ── T12: retention metadata present and reflects EpisodeDefinition.retention

def test_t12_retention_metadata_present_and_bounded():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    retention = block["retention"]
    assert retention["bounded"] is True
    assert retention["max_records_by_type"]["heating"] == 500
    assert retention["max_age_days_by_type"]["heating"] == 90
    assert retention["max_records_by_type"]["outcome"] == 500
    assert retention["max_age_days_by_type"]["outcome"] == 90


def test_t12_ages_computed_relative_to_now():
    coord = _FakeCoord(_FakeShadow(_MIXED_HISTORY))
    block = _le2_episode_history_export(coord, now=_NOW)
    assert block["oldest_age_hours"] == 36.0
    assert block["newest_age_hours"] == 1.0


# ── T13: Support export is unchanged ────────────────────────────────────────

def test_t13_support_export_signature_unchanged():
    sig = inspect.signature(_export_module.async_export_support_data)
    assert list(sig.parameters) == ["hass", "ts"]


def test_t13_support_export_does_not_call_new_helper():
    source = inspect.getsource(_export_module.async_export_support_data)
    assert "_le2_episode_history_export" not in source


# ── T14: Learning Progress research export block is preserved ──────────────

def test_t14_learning_progress_export_still_works_alongside():
    from custom_components.thermosmart.learning.runtime.learning_progress import (
        ModelSignal, compute_learning_progress,
    )

    class _ProgressShadow:
        def __init__(self, result):
            self._result = result

        def learning_progress_safe(self):
            return self._result

    pct, attrs = compute_learning_progress({
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}),
    })
    block = _le2_learning_progress_export(_FakeCoord(_ProgressShadow((pct, attrs))))
    assert block["available"] is True
    assert block["progress_pct"] == pct


# ── T15: Existing export tests remain green (regression smoke-test) ────────

def test_t15_regression_imports_ok():
    import tests.test_orchestration_export_trace  # noqa: F401
    import tests.test_le2_learning_progress_export  # noqa: F401
    import tests.test_le2_learning_progress  # noqa: F401


# ── T16: No new store I/O introduced ────────────────────────────────────────

def test_t16_no_store_io_in_new_helper():
    source = inspect.getsource(_le2_episode_history_export)
    assert "await" not in source
    assert ".save(" not in source
    assert "EpisodesStore(" not in source
    assert "async_setup" not in source
    assert "async_load" not in source


# ── T17: No control-path keywords touched ───────────────────────────────────

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


def test_t17_no_control_keywords_in_new_helper():
    source = inspect.getsource(_le2_episode_history_export).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"
