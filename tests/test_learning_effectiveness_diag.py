"""Diagnostics-only learning-effectiveness surface (Phase 2).

Covers the five read-only blocks added to the HA diagnostics export:
per-model detail, per-builder episode lifecycle, boost effectiveness, open
preheat-comparison stats and persistence detail — plus the assembled
``learning_effectiveness`` export block. Verifies empty-state shape,
learning-disabled handling, that counters surface, privacy cleanliness, and
that collection never mutates model state.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.thermosmart.export import _learning_effectiveness_export
from custom_components.thermosmart.learning.privacy import scan_payload
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
from tests.helpers import make_coordinator
from tests.helpers_ha_runtime import FakeStore, attach_shadow

_MODELS = ("heat_rate", "heat_loss", "afterheat", "outcome", "onset_delay")
_BUILDERS = ("heating", "cooling", "afterheat", "outcome")
_LIFECYCLE_KEYS = {"opened", "completed", "interrupted", "rejected"}


async def _setup(cfg=None):
    coord = make_coordinator(cfg)
    shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore())
    await shadow.async_setup()
    return coord, shadow


# ── empty-state shape ────────────────────────────────────────────────────────

class TestEmptyStateShape:
    async def test_model_detail_lists_all_models_zeroed(self):
        _, shadow = await _setup()
        d = shadow.learning_model_detail_safe()
        assert d["available"] is True
        assert set(d["models"]) == set(_MODELS)
        for name, e in d["models"].items():
            assert e["accepted_updates"] == 0
            assert e["rejected_updates"] == 0
            # sample_counts may seed zeroed buckets (e.g. {"general": 0}); the
            # point is that nothing is populated yet.
            assert all(v == 0 for v in e["sample_counts"].values())
            assert e["populated_buckets"] == 0
            assert e["last_update_age_s"] is None

    async def test_builder_lifecycle_all_builders_zeroed(self):
        _, shadow = await _setup()
        b = shadow.builder_lifecycle_safe()
        assert b["available"] is True
        assert set(b["builders"]) == set(_BUILDERS)
        for counts in b["builders"].values():
            assert set(counts) == _LIFECYCLE_KEYS
            assert all(v == 0 for v in counts.values())

    async def test_boost_effectiveness_shape(self):
        _, shadow = await _setup()
        bo = shadow.boost_effectiveness_safe()
        assert bo["available"] is True
        assert isinstance(bo["bootstrap_credits"], int)
        assert bo["general_samples"] == 0
        assert bo["proposed_attempts"] == 0
        assert bo["applied_count"] == 0
        assert bo["pending_observer"] == {
            "active": False, "opened": 0, "finalized": 0, "interrupted": 0}
        # with no samples yet, remaining == full credits
        assert bo["bootstrap_credits_remaining"] == bo["bootstrap_credits"]

    async def test_preheat_comparison_empty(self):
        _, shadow = await _setup()
        pc = shadow.preheat_comparison_stats_safe()
        assert pc["available"] is True
        assert pc["open_count"] == 0
        assert pc["resolved_count"] == 0
        assert pc["oldest_age_s"] is None
        assert all(v == 0 for v in pc["age_buckets_s"].values())

    async def test_persistence_detail_shape(self):
        _, shadow = await _setup()
        p = shadow.persistence_detail_safe()
        assert p["available"] is True
        assert isinstance(p["save_count"], int)
        assert isinstance(p["dirty"], bool)
        assert p["warnings_count"] == 0


# ── counters surface ─────────────────────────────────────────────────────────

class TestCountersSurface:
    async def test_model_and_boost_and_builder_counters_reflected(self):
        _, shadow = await _setup()
        zr = shadow._runtime._zone(shadow._zone)
        zr.model_reject_counts["heat_loss"] = 2
        zr.model_update_counts["heat_rate"] = 4
        zr.pending_boost_opened = 3
        zr.pending_boost_finalized = 1
        zr.pending_boost_interrupted = 2
        zr.pipeline._builder_lifecycle["cooling"]["interrupted"] = 5
        zr.pipeline._builder_lifecycle["heating"]["completed"] = 7

        md = shadow.learning_model_detail_safe()["models"]
        assert md["heat_loss"]["rejected_updates"] == 2
        assert md["heat_rate"]["accepted_updates"] == 4

        bl = shadow.builder_lifecycle_safe()["builders"]
        assert bl["cooling"]["interrupted"] == 5
        assert bl["heating"]["completed"] == 7

        bo = shadow.boost_effectiveness_safe()["pending_observer"]
        assert bo == {"active": False, "opened": 3, "finalized": 1, "interrupted": 2}

    async def test_open_comparisons_count_and_age_bucketed(self):
        coord, shadow = await _setup()
        zr = shadow._runtime._zone(shadow._zone)
        now = shadow._utcnow_iso()
        from datetime import datetime, timedelta
        old = (datetime.fromisoformat(now) - timedelta(hours=30)).isoformat()
        recent = (datetime.fromisoformat(now) - timedelta(minutes=10)).isoformat()
        zr.open_comparisons.append(
            SimpleNamespace(comparison_type="preheat_start", created_ts=old, actual_outcome=None))
        zr.open_comparisons.append(
            SimpleNamespace(comparison_type="preheat_start", created_ts=recent, actual_outcome=None))
        pc = shadow.preheat_comparison_stats_safe()
        assert pc["open_count"] == 2
        assert pc["resolved_count"] == 0
        assert pc["age_buckets_s"]["gt_24h"] == 1
        assert pc["age_buckets_s"]["lt_1h"] == 1
        assert pc["oldest_age_s"] >= 24 * 3600


# ── learning-disabled ────────────────────────────────────────────────────────

class TestLearningDisabled:
    async def test_gated_blocks_report_disabled(self):
        _, shadow = await _setup()
        shadow._enabled = False
        for getter in (shadow.learning_model_detail_safe,
                       shadow.builder_lifecycle_safe,
                       shadow.boost_effectiveness_safe,
                       shadow.preheat_comparison_stats_safe):
            out = getter()
            assert out["available"] is False
            assert out["reason"] == "learning_disabled"


# ── export assembly + privacy + no-mutation ──────────────────────────────────

class TestExportBlock:
    async def test_export_assembles_five_blocks(self):
        coord, _ = await _setup()
        out = _learning_effectiveness_export(coord)
        assert out["available"] is True
        for key in ("models", "builders", "boost", "preheat_comparisons", "persistence"):
            assert key in out
        assert out["boost"]["prior_configured"] is False

    async def test_export_reports_prior_configured(self):
        coord, _ = await _setup({"boost_bootstrap_prior_c": 1.0})
        out = _learning_effectiveness_export(coord)
        assert out["boost"]["prior_configured"] is True

    def test_export_without_shadow_is_unavailable(self):
        coord = make_coordinator()
        coord._learning_shadow = None
        out = _learning_effectiveness_export(coord)
        assert out["available"] is False

    async def test_export_is_privacy_clean(self):
        coord, _ = await _setup()
        out = _learning_effectiveness_export(coord)
        assert scan_payload(out) == []

    async def test_collection_does_not_mutate_model_state(self):
        _, shadow = await _setup()
        zr = shadow._runtime._zone(shadow._zone)
        before = zr.orchestrator.serialize_models()
        # exercise every read path
        shadow.learning_model_detail_safe()
        shadow.builder_lifecycle_safe()
        shadow.boost_effectiveness_safe()
        shadow.preheat_comparison_stats_safe()
        shadow.persistence_detail_safe()
        after = zr.orchestrator.serialize_models()
        assert before == after
