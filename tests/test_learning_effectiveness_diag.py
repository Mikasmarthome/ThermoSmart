"""Diagnostics-only learning-effectiveness surface (Phase 2 + review fixes).

Covers the five read-only diagnostics blocks and the assembled
``learning_effectiveness`` export: empty-state shape, learning-disabled and
UNINITIALIZED-zone handling (accessors never create a zone), privacy
cleanliness, no model-state mutation, and — via real cycles / real event
paths — that the counters actually increment:

  * builder lifecycle events (opened/completed) from a real heating ramp
  * model-update rejection (apply_update declined)
  * boost observer finalization and interruption
  * more than 300 ledger decisions (proposed_attempts uncapped)
  * multi-zone isolation
  * repeated diagnostics downloads (idempotent)
  * restart/restore: persisted accepted updates vs session-only counters
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.thermosmart.export import _learning_effectiveness_export
from custom_components.thermosmart.learning.models import BoostOutcomeLifecycleState
from custom_components.thermosmart.learning.privacy import scan_payload
from custom_components.thermosmart.learning.runtime import LearningRuntimeConfig, LearningRuntimeMode
from custom_components.thermosmart.learning.runtime.control import ControlFeature
from custom_components.thermosmart.learning.runtime.lifecycle import _ZoneRuntime
from tests.helpers import make_coordinator
from tests.helpers_boost import boost_context, boost_episode
from tests.helpers_ha_runtime import FakeStore, attach_shadow
from tests.helpers_runtime_scenarios import heating_ramp_then_settle
from tests.helpers_runtime_scenarios import runtime as scenario_runtime
from tests.helpers_runtime_scenarios import step

_MODELS = ("heat_rate", "heat_loss", "afterheat", "outcome", "onset_delay")
_BUILDERS = ("heating", "cooling", "afterheat", "outcome")
_LIFECYCLE_KEYS = {"opened", "completed", "interrupted", "rejected"}


async def _setup(cfg=None, *, init_zone=True):
    coord = make_coordinator(cfg)
    shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore())
    await shadow.async_setup()
    if init_zone:
        # Materialise an empty zone runtime, exactly as the first control cycle
        # would — the diagnostics accessors themselves must never create it.
        shadow._runtime._zone(shadow._zone)
    return coord, shadow


def _ledger_decision(*, applied: bool):
    """A minimal duck-typed ControlDecision the ledger.record() can read."""
    return SimpleNamespace(
        feature=ControlFeature.BOOST_OFFSET, baseline_value=0.0, proposed_value=1.0,
        final_value=(1.0 if applied else 0.0), applied=applied,
        rejection=(None if applied else "low_confidence"),
        confidence=None, clamp_applied=0.0, policy_version=1,
        model_version=1, parameter_version=1)


# ── empty-state shape (zone initialized, no activity) ────────────────────────

class TestEmptyStateShape:
    async def test_model_detail_lists_all_models_zeroed(self):
        _, shadow = await _setup()
        d = shadow.learning_model_detail_safe()
        assert d["available"] is True
        assert set(d["models"]) == set(_MODELS)
        assert d["counter_scopes"]["accepted_updates"] == "persisted_cumulative"
        assert d["counter_scopes"]["rejected_updates"] == "session_since_restart"
        for e in d["models"].values():
            assert e["accepted_updates"] == 0
            assert e["rejected_updates"] == 0
            assert all(v == 0 for v in e["sample_counts"].values())
            assert e["populated_buckets"] == 0
            assert e["last_update_age_s"] is None

    async def test_builder_lifecycle_all_builders_zeroed(self):
        _, shadow = await _setup()
        b = shadow.builder_lifecycle_safe()
        assert b["available"] is True
        assert set(b["builders"]) == set(_BUILDERS)
        assert b["counter_scope"] == "session_since_restart"
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
        assert bo["bootstrap_credits_remaining"] == bo["bootstrap_credits"]
        assert bo["counter_scopes"]["proposed_attempts"] == "session_cumulative_uncapped"

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


# ── uninitialized zone: accessors must NOT create it ─────────────────────────

class TestUninitializedRuntime:
    async def test_accessors_report_zone_not_initialized_and_create_nothing(self):
        _, shadow = await _setup(init_zone=False)
        assert shadow._zone not in shadow._runtime._zones  # precondition
        for getter in (shadow.learning_model_detail_safe,
                       shadow.builder_lifecycle_safe,
                       shadow.boost_effectiveness_safe,
                       shadow.preheat_comparison_stats_safe):
            out = getter()
            assert out["available"] is False
            assert out["reason"] == "zone_not_initialized"
        # the reads themselves must not have materialised the zone
        assert shadow._zone not in shadow._runtime._zones

    async def test_export_block_available_but_zone_blocks_unavailable(self):
        coord, shadow = await _setup(init_zone=False)
        out = _learning_effectiveness_export(coord)
        assert out["available"] is True            # shadow attached
        assert out["models"]["available"] is False  # but zone not initialized
        assert out["models"]["reason"] == "zone_not_initialized"
        assert shadow._zone not in shadow._runtime._zones


# ── real counter increments via real cycles / real event paths ───────────────

class TestRealIncrements:
    def test_builder_lifecycle_counts_from_a_real_heating_ramp(self):
        rt = scenario_runtime()
        heating_ramp_then_settle(rt, zone="lz")
        summary = rt._zones["lz"].pipeline.builder_lifecycle_summary()
        # a heating episode opens during the ramp and closes at the settle edge
        assert summary["heating"]["opened"] >= 1
        assert summary["heating"]["completed"] >= 1

    def test_model_accepted_updates_from_a_real_heating_ramp(self):
        rt = scenario_runtime()
        heating_ramp_then_settle(rt, zone="lz")
        counts = rt._zones["lz"].model_update_counts
        # the heating builder feeds heat_rate / onset_delay
        assert sum(counts.values()) >= 1

    def test_model_rejection_increments_reject_counter(self):
        rt = scenario_runtime()
        rt._zone("lz")  # materialise before stubbing
        zr = rt._zones["lz"]
        # Force heat_rate to decline every completed episode (real apply_update
        # path: ok is False -> model_reject_counts increments). No threshold or
        # update-semantics change — only this test instance's model is stubbed.
        zr.orchestrator.models["heat_rate"].update = \
            lambda episode, context=None: SimpleNamespace(accepted=False)
        heating_ramp_then_settle(rt, zone="lz")
        assert zr.model_reject_counts.get("heat_rate", 0) >= 1
        assert zr.model_update_counts.get("heat_rate", 0) == 0

    def test_boost_observer_finalized_and_interrupted_increment(self):
        rt = scenario_runtime()
        zr = rt._zone("lz")
        snap = SimpleNamespace(ts=boost_episode("x", (18.0,)).end_ts.isoformat(),
                               device_states={}, boost_dispatch=None)

        ep1 = boost_episode("e1", (18.0, 19.5, 20.5, 21.0, 21.0), decision_id="d1")
        rt._open_pending_boost(zr, ep1, boost_context(ep1, zone="lz"), snap)
        assert zr.pending_boost_opened == 1
        assert zr.pending_boost_outcome is not None
        rt._finalize_pending_boost(zr, zr.pending_boost_outcome, reason="deadline")
        assert zr.pending_boost_finalized == 1
        assert zr.pending_boost_interrupted == 0

        ep2 = boost_episode("e2", (18.0, 19.5, 20.5, 21.0, 21.0), decision_id="d2")
        rt._open_pending_boost(zr, ep2, boost_context(ep2, zone="lz"), snap)
        assert zr.pending_boost_opened == 2
        p2 = zr.pending_boost_outcome
        p2.state = BoostOutcomeLifecycleState.INTERRUPTED.value
        rt._finalize_pending_boost(zr, p2, reason="window_open")
        assert zr.pending_boost_finalized == 2
        assert zr.pending_boost_interrupted == 1


# ── proposed_attempts not capped by the 300-entry ledger buffer ──────────────

class TestLedgerNotCappedAt300:
    async def test_proposed_attempts_counts_all_decisions_past_the_buffer(self):
        _, shadow = await _setup()
        zr = shadow._runtime._zone(shadow._zone)
        for i in range(200):
            zr.control_ledger.record("z", f"d{i}", _ledger_decision(applied=True), "t")
        for i in range(101):
            zr.control_ledger.record("z", f"f{i}", _ledger_decision(applied=False), "t")
        bo = shadow.boost_effectiveness_safe()
        # 301 decisions total; the recent-entries buffer caps at 300.
        assert zr.control_ledger.size == 300
        assert bo["proposed_attempts"] == 301
        assert bo["applied_count"] == 200
        assert bo["fallback_count"] == 101


# ── multi-zone isolation ─────────────────────────────────────────────────────

class TestMultiZoneIsolation:
    def test_counters_are_per_zone(self):
        rt = scenario_runtime()
        heating_ramp_then_settle(rt, zone="z1")
        # z2 gets a couple of non-heating cycles only
        for i in range(3):
            step(rt, i, 20.0, heating=False, zone="z2")
        z1 = rt._zones["z1"]
        z2 = rt._zones["z2"]
        assert z1.pipeline.builder_lifecycle_summary()["heating"]["opened"] >= 1
        assert z2.pipeline.builder_lifecycle_summary()["heating"]["opened"] == 0
        assert sum(z1.model_update_counts.values()) >= 1
        assert "z1" in rt._zones and "z2" in rt._zones
        assert z1 is not z2


# ── repeated downloads are idempotent (no read-triggered mutation) ───────────

class TestRepeatedDownloads:
    async def test_export_is_stable_across_repeated_reads(self):
        coord, shadow = await _setup()
        heating_ramp_then_settle(shadow._runtime, zone=shadow._zone)
        first = _learning_effectiveness_export(coord)
        second = _learning_effectiveness_export(coord)
        # ages can advance between reads; compare the stable counter sub-blocks.
        assert first["models"]["models"] == second["models"]["models"]
        assert first["builders"] == second["builders"]
        assert first["boost"]["proposed_attempts"] == second["boost"]["proposed_attempts"]
        assert first["boost"]["pending_observer"] == second["boost"]["pending_observer"]


# ── restart/restore: persisted accepted vs session-only counters ─────────────

class TestRestartRestore:
    def test_accepted_updates_persist_but_session_counters_reset(self):
        cfg = LearningRuntimeConfig(mode=LearningRuntimeMode.SHADOW)
        src = _ZoneRuntime("lz", cfg)
        src.model_update_counts["heat_rate"] = 5          # persisted
        src.model_reject_counts["heat_rate"] = 3          # session-only
        src.pending_boost_opened = 4                      # session-only
        src.pending_boost_finalized = 2
        src.pending_boost_interrupted = 1
        data = src.serialize()

        dst = _ZoneRuntime("lz", cfg)
        dst.restore(data)
        assert dst.model_update_counts.get("heat_rate") == 5   # survived restart
        assert dst.model_reject_counts == {}                   # reset
        assert dst.pending_boost_opened == 0
        assert dst.pending_boost_finalized == 0
        assert dst.pending_boost_interrupted == 0


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

    async def test_export_is_privacy_clean_with_real_data(self):
        coord, shadow = await _setup()
        heating_ramp_then_settle(shadow._runtime, zone=shadow._zone)
        out = _learning_effectiveness_export(coord)
        assert scan_payload(out) == []

    async def test_collection_does_not_mutate_model_state(self):
        _, shadow = await _setup()
        heating_ramp_then_settle(shadow._runtime, zone=shadow._zone)
        zr = shadow._runtime._zone(shadow._zone)
        before = zr.orchestrator.serialize_models()
        for _ in range(3):  # repeated downloads
            shadow.learning_model_detail_safe()
            shadow.builder_lifecycle_safe()
            shadow.boost_effectiveness_safe()
            shadow.preheat_comparison_stats_safe()
            shadow.persistence_detail_safe()
        after = zr.orchestrator.serialize_models()
        assert before == after
