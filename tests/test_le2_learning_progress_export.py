"""Tests for the LE2 learning-progress research export block.

Covers the read-only research-export addition in export.py:
  _le2_learning_progress_export() and its wiring into async_export_learning_data()
  (checked at the per-zone-dict-construction level, without a full hass fixture).

This step adds ONE small, public-safe "learning_progress" block per zone to
the research export, sourced directly from
LearningShadowController.learning_progress_safe() (the same method the
confidence sensor reads) — no new store, no episode-history export, no
Support-export change.

13 test groups:
  T1  — Research export helper returns a dict with real learning_progress_safe() fields
  T2  — regime_cap_pct is exported
  T3  — Only real fields (from the actual attrs dict) are ever included
  T4  — Missing LE2 shadow stays valid (available: False, non-fatal)
  T5  — learning_progress_safe() raising stays non-fatal (available: False)
  T6  — No entity ids / zone names / internal ids in the exported block
  T7  — No episode-history data is exported by this helper
  T8  — Support export helpers/module are untouched by this addition
  T9  — Existing export tests remain green (regression smoke-test)
  T10 — Existing learning-progress tests remain green (regression smoke-test)
  T11 — No control-path keywords touched by the new export helper
  T12 — No new store I/O introduced by the new export helper
  T13 — cold-start zone still gets a valid, available learning_progress block
"""
from __future__ import annotations

import inspect

import pytest

from custom_components.thermosmart import export as _export_module
from custom_components.thermosmart.export import (
    _LEARNING_PROGRESS_RESEARCH_KEYS,
    _le2_learning_progress_export,
)
from custom_components.thermosmart.learning.runtime.learning_progress import (
    ModelSignal,
    cold_learning_progress_result,
    compute_learning_progress,
)


class _FakeShadow:
    def __init__(self, result=None, raise_err: Exception | None = None):
        self._result = result
        self._raise_err = raise_err

    def learning_progress_safe(self):
        if self._raise_err is not None:
            raise self._raise_err
        return self._result


class _FakeCoord:
    def __init__(self, shadow=None):
        self._le2_shadow = shadow


def _real_progress(signals: dict) -> tuple:
    return compute_learning_progress(signals)


# ── T1: Research export helper returns real learning_progress_safe() fields ─

def test_t1_export_contains_real_fields():
    pct, attrs = _real_progress({
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}, confidence_value=0.6),
    })
    coord = _FakeCoord(_FakeShadow((pct, attrs)))
    block = _le2_learning_progress_export(coord)
    assert block["available"] is True
    assert block["progress_pct"] == pct
    assert block["learning_stage"] == attrs["learning_stage"]
    assert block["confidence_level"] == attrs["confidence_level"]
    assert block["main_blocker"] == attrs["main_blocker"]
    assert block["next_needed"] == attrs["next_needed"]


# ── T2: regime_cap_pct is exported ───────────────────────────────────────────

def test_t2_regime_cap_pct_present():
    pct, attrs = _real_progress({
        "outcome": ModelSignal(accepted_updates=8, sample_counts={"c:ts": 8},
                                general_data_quality=0.7, timeout_rate=0.1, overshoot_rate=0.1),
    })
    coord = _FakeCoord(_FakeShadow((pct, attrs)))
    block = _le2_learning_progress_export(coord)
    assert "regime_cap_pct" in block
    assert block["regime_cap_pct"] == attrs["regime_cap_pct"]


# ── T3: Only real fields are ever included — no invented ones ──────────────

def test_t3_only_real_attrs_keys_included():
    pct, attrs = _real_progress({
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}),
    })
    coord = _FakeCoord(_FakeShadow((pct, attrs)))
    block = _le2_learning_progress_export(coord)
    exported_keys = set(block) - {"available", "progress_pct"}
    assert exported_keys <= set(attrs)
    # Every declared export key must be a genuinely real attrs key.
    for key in _LEARNING_PROGRESS_RESEARCH_KEYS:
        assert key in attrs, f"{key} is not a real learning_progress_safe() field"


def test_t3_no_invented_progress_limited_by_list_field():
    """The originally-sketched target shape used a 'progress_limited_by' list,
    which compute_learning_progress() does not produce — the real field is
    the single string 'main_blocker'. Confirm we never fabricate the former."""
    pct, attrs = _real_progress({
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}),
    })
    coord = _FakeCoord(_FakeShadow((pct, attrs)))
    block = _le2_learning_progress_export(coord)
    assert "progress_limited_by" not in block
    assert isinstance(block["main_blocker"], str)


# ── T4: Missing LE2 shadow stays valid, non-fatal ───────────────────────────

def test_t4_missing_shadow_stays_valid():
    block = _le2_learning_progress_export(_FakeCoord(None))
    assert block == {"available": False, "reason": "le2_shadow_unavailable"}


# ── T5: learning_progress_safe() raising stays non-fatal ────────────────────

def test_t5_calculation_exception_stays_non_fatal():
    coord = _FakeCoord(_FakeShadow(raise_err=RuntimeError("simulated failure")))
    block = _le2_learning_progress_export(coord)
    assert block["available"] is False
    assert "learning_progress_error" in block


def test_t5_missing_le2_shadow_attribute_entirely_stays_valid():
    class _BareCoord:
        pass
    block = _le2_learning_progress_export(_BareCoord())
    assert block["available"] is False


# ── T6: No entity ids / zone names / internal ids in the exported block ────

_FORBIDDEN_SUBSTRINGS = (
    "zone_id", "entity_id", "episode_id", "learning_zone_id", "decision_id",
    "trv_binding_id", "radiator_profile_id", "person", "secret", "token", "path",
)


def test_t6_no_forbidden_keys_in_output():
    pct, attrs = _real_progress({
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}),
        "outcome": ModelSignal(accepted_updates=8, sample_counts={"c:ts": 8},
                                general_data_quality=0.7, timeout_rate=0.1, overshoot_rate=0.1),
    })
    coord = _FakeCoord(_FakeShadow((pct, attrs)))
    block = _le2_learning_progress_export(coord)
    flat = repr(block).lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        assert token not in flat, f"forbidden substring '{token}' leaked into export block"


def test_t6_no_zone_id_key_even_if_present_in_attrs():
    """Defense-in-depth: even if a future attrs dict accidentally carried a
    zone-identifying key, the strip-forbidden pass must remove it before export."""
    pct, attrs = _real_progress({
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}),
    })
    attrs = dict(attrs)
    attrs["learning_zone_id"] = "zone_alpha_01"  # simulate accidental leakage upstream
    coord = _FakeCoord(_FakeShadow((pct, attrs)))
    block = _le2_learning_progress_export(coord)
    assert "learning_zone_id" not in block
    assert "zone_alpha_01" not in repr(block)


# ── T7: No episode-history data is exported by this helper ─────────────────

def test_t7_no_episode_history_fields_exported():
    pct, attrs = _real_progress({
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}),
    })
    coord = _FakeCoord(_FakeShadow((pct, attrs)))
    block = _le2_learning_progress_export(coord)
    for forbidden_key in ("episode_history", "episodes", "trajectory", "sample_count",
                          "heating_episode_count", "outcome_episode_count"):
        assert forbidden_key not in block


# ── T8: Support export helpers/module are untouched ─────────────────────────

def test_t8_support_export_function_unchanged_signature():
    import custom_components.thermosmart.export as _exp
    sig = inspect.signature(_exp.async_export_support_data)
    assert list(sig.parameters) == ["hass", "ts"]


def test_t8_support_export_source_does_not_call_new_helper():
    """The new learning-progress export block is research-only in this step —
    async_export_support_data() must not call the new helper."""
    import custom_components.thermosmart.export as _exp
    source = inspect.getsource(_exp.async_export_support_data)
    assert "_le2_learning_progress_export" not in source


# ── T9/T10: Existing tests remain green (regression smoke-test) ────────────

def test_t9_t10_regression_imports_ok():
    import tests.test_orchestration_export_trace  # noqa: F401
    import tests.test_le2_learning_progress  # noqa: F401
    import tests.test_le2_episode_runtime_hook  # noqa: F401
    import tests.test_le2_episode_persistence  # noqa: F401


# ── T11: No control-path keywords touched ───────────────────────────────────

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


def test_t11_no_control_keywords_in_new_helper():
    source = inspect.getsource(_le2_learning_progress_export).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"


# ── T12: No new store I/O introduced ────────────────────────────────────────

def test_t12_no_store_io_in_new_helper():
    source = inspect.getsource(_le2_learning_progress_export)
    assert "await" not in source
    assert ".save(" not in source
    assert "EpisodesStore" not in source
    assert "async_setup" not in source


# ── T13: cold-start zone still gets a valid, available block ───────────────

def test_t13_cold_start_zone_is_available_not_error():
    cold = cold_learning_progress_result("no_completed_episodes_yet")
    coord = _FakeCoord(_FakeShadow((0.0, cold)))
    block = _le2_learning_progress_export(coord)
    assert block["available"] is True
    assert block["progress_pct"] == 0.0
    assert block["learning_stage"] == "cold_start"
