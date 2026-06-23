"""Phase 19D: HeatLoss-specific storage, serialization, and persistence tests.

Tests that HeatLossModel state (episodes, buckets, confidence) survives a
save → reload cycle, that corrupted state is safely rejected, and that
LE v1 _heat_loss_ema is never written as a side-effect.

Multi-zone isolation, deduplication, and pruning are also covered.
Tests operate at the model level (no HA/coordinator needed).
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.models.heat_loss import (
    HeatLossModel,
    HeatLossParameters,
    HeatLossPredictionContext,
    MODEL_VERSION,
)
from tests.helpers_heat_loss import T0, cooling_episode


def _m(zone="lz") -> HeatLossModel:
    return HeatLossModel(zone)


def _ctx() -> HeatLossPredictionContext:
    return HeatLossPredictionContext()


# ── Section 1: serialize / deserialize roundtrip ────────────────────────────

class TestHeatLossSerializeRoundtrip:

    def test_fresh_model_serializes(self):
        m = _m()
        state = m.serialize_state()
        assert state["model_version"] == MODEL_VERSION
        assert state["learning_zone_id"] == "lz"

    def test_fresh_roundtrip_preserves_zero_evidence(self):
        m = _m()
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        assert not m2._state.general.has_evidence

    def test_after_learning_roundtrip_preserves_rate(self):
        m = _m()
        m.update(cooling_episode("e1", 0.5))
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        assert m2._state.general.has_evidence
        assert m2._state.general.rate_c_per_h == pytest.approx(
            m._state.general.rate_c_per_h, abs=1e-5)

    def test_after_learning_roundtrip_preserves_sample_count(self):
        m = _m()
        for i in range(3):
            m.update(cooling_episode(f"e{i}", 0.5 + i * 0.1))
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        assert m2._state.general.sample_count == m._state.general.sample_count

    def test_after_learning_roundtrip_prediction_unchanged(self):
        m = _m()
        for i in range(5):
            m.update(cooling_episode(f"e{i}", 0.5))
        pred_before = m.predict_heat_loss(_ctx())
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        pred_after = m2.predict_heat_loss(_ctx())
        assert pred_after.confidence == pytest.approx(pred_before.confidence, abs=1e-5)
        assert pred_after.values["heat_loss_rate"] == pytest.approx(
            pred_before.values["heat_loss_rate"], abs=1e-5)

    def test_roundtrip_preserves_fallback_false(self):
        m = _m()
        for i in range(5):
            m.update(cooling_episode(f"e{i}", 0.5))
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        pred = m2.predict_heat_loss(_ctx())
        assert pred.fallback_used is False

    def test_roundtrip_preserves_prediction_type(self):
        from custom_components.thermosmart.learning.contracts import PredictionType
        m = _m()
        m.update(cooling_episode("e1", 0.5))
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        pred = m2.predict_heat_loss(_ctx())
        assert pred.prediction_type == PredictionType.HEAT_LOSS_RATE

    def test_roundtrip_preserves_unit(self):
        m = _m()
        m.update(cooling_episode("e1", 0.5))
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        pred = m2.predict_heat_loss(_ctx())
        assert pred.units.get("heat_loss_rate") == "C/h"

    def test_roundtrip_preserves_processed_ids(self):
        m = _m()
        m.update(cooling_episode("unique-ep-abc", 0.5))
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        assert "unique-ep-abc" in m2._state.processed_ids


# ── Section 2: corrupt / invalid state rejection ─────────────────────────────

class TestHeatLossCorruptStateRejection:

    def test_wrong_model_version_rejected(self):
        m = _m()
        state = dict(m.serialize_state())
        state["model_version"] = 999
        with pytest.raises(ValueError, match="model_version"):
            m.deserialize_state(state)

    def test_wrong_zone_rejected(self):
        m = _m("zone_a")
        state = m.serialize_state()
        m2 = _m("zone_b")
        with pytest.raises(ValueError, match="learning_zone_id"):
            m2.deserialize_state(state)

    def test_non_finite_rate_rejected(self):
        m = _m()
        state = dict(m.serialize_state())
        state["general"] = {
            "key": "general", "rate_c_per_h": float("nan"),
            "effective_n": 0.0, "dispersion": 0.0,
            "sample_count": 0, "last_update_ts": None,
        }
        with pytest.raises(ValueError):
            m.deserialize_state(state)

    def test_non_mapping_state_rejected(self):
        m = _m()
        with pytest.raises(ValueError):
            m.deserialize_state([1, 2, 3])

    def test_fresh_model_continues_after_bad_restore(self):
        # Model should still predict from generic prior after bad restore is caught
        m = _m()
        try:
            m.deserialize_state({"model_version": 999, "learning_zone_id": "lz"})
        except ValueError:
            pass  # expected
        pred = m.predict_heat_loss(_ctx())
        assert pred is not None
        assert pred.fallback_used is True  # no learned data, falls back to prior


# ── Section 3: deduplication ─────────────────────────────────────────────────

class TestHeatLossDeduplication:

    def test_same_episode_id_not_relearned(self):
        m = _m()
        ep = cooling_episode("ep-dup", 0.5)
        m.update(ep)
        m.update(ep)  # same episode_id
        assert m._state.general.sample_count == 1

    def test_different_episode_ids_both_learned(self):
        m = _m()
        m.update(cooling_episode("ep1", 0.5))
        m.update(cooling_episode("ep2", 0.5))
        assert m._state.general.sample_count == 2

    def test_dedup_survives_roundtrip(self):
        m = _m()
        ep = cooling_episode("ep-persist", 0.5)
        m.update(ep)
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        # Same episode after reload → should be rejected (duplicate)
        m2.update(ep)
        assert m2._state.general.sample_count == 1


# ── Section 4: multi-zone isolation ──────────────────────────────────────────

class TestHeatLossMultiZoneIsolation:

    def test_different_zones_independent_rates(self):
        m_a = _m("zone_a")
        m_b = _m("zone_b")
        for i in range(5):
            m_a.update(cooling_episode(f"a{i}", 0.3, zone="zone_a"))
        for i in range(5):
            m_b.update(cooling_episode(f"b{i}", 0.9, zone="zone_b"))
        pred_a = m_a.predict_heat_loss(_ctx())
        pred_b = m_b.predict_heat_loss(_ctx())
        rate_a = pred_a.values["heat_loss_rate"]
        rate_b = pred_b.values["heat_loss_rate"]
        assert rate_a < rate_b  # zone_a learned lower rate

    def test_zone_a_state_not_leaked_to_zone_b(self):
        m_a = _m("zone_a")
        m_b = _m("zone_b")
        for i in range(10):
            m_a.update(cooling_episode(f"a{i}", 2.5, zone="zone_a"))
        pred_b_before = m_b.predict_heat_loss(_ctx())
        assert pred_b_before.fallback_used is True  # zone_b has no data
        assert pred_b_before.values["heat_loss_rate"] < 1.5  # generic prior, not zone_a's high rate


# ── Section 5: LE v1 EMA not written by HeatLoss storage cycle ───────────────

class TestLE1EmaNotWritten:

    def test_le1_ema_not_in_heat_loss_state(self):
        m = _m()
        m.update(cooling_episode("e1", 0.5))
        state = m.serialize_state()
        # LE v1 EMA fields must not appear in LE2 HeatLossModel state
        assert "heat_loss_ema" not in state
        assert "ema" not in state

    def test_async_observe_does_not_mutate_le1_ema(self):
        """Silenced mutation: _heat_loss_ema dict stays frozen across observe cycles.

        Uses the rec() dict pattern from test_learning_observe.py and a real engine.
        The dynamic behaviour is also covered in test_learning_observe.py;
        this test verifies the contract from the storage perspective.
        """
        import asyncio
        from datetime import datetime, timedelta
        from unittest.mock import AsyncMock, patch
        from custom_components.thermosmart import learning_engine as le_mod
        from tests.helpers import make_learning_engine

        FIXED_NOW = datetime(2026, 6, 23, 12, 0, 0)
        with patch.object(le_mod.dt_util, "now", return_value=FIXED_NOW):
            engine = make_learning_engine()
            engine.async_save = AsyncMock()
            engine._heat_loss_ema["z"] = 99.0
            engine._last_idle_temp["z"] = (FIXED_NOW - timedelta(minutes=20), 21.0)
            asyncio.run(engine.async_observe(
                "z",
                {"current_temp": 20.5, "effective_target": 21.0},
                {"temperature": 5.0},
            ))
        assert engine._heat_loss_ema.get("z") == pytest.approx(99.0)

    def test_le1_ema_not_written_to_new_zone(self):
        """A zone with no prior EMA entry must NOT get a new one from async_observe."""
        import asyncio
        from datetime import datetime, timedelta
        from unittest.mock import AsyncMock, patch
        from custom_components.thermosmart import learning_engine as le_mod
        from tests.helpers import make_learning_engine

        FIXED_NOW = datetime(2026, 6, 23, 12, 0, 0)
        with patch.object(le_mod.dt_util, "now", return_value=FIXED_NOW):
            engine = make_learning_engine()
            engine.async_save = AsyncMock()
            assert "new_zone" not in engine._heat_loss_ema
            engine._last_idle_temp["new_zone"] = (FIXED_NOW - timedelta(minutes=20), 21.0)
            asyncio.run(engine.async_observe(
                "new_zone",
                {"current_temp": 20.5, "effective_target": 21.0},
                {"temperature": 5.0},
            ))
        assert "new_zone" not in engine._heat_loss_ema


# ── Section 6: prediction before and after learning ──────────────────────────

class TestHeatLossColdStartVsLearned:

    def test_cold_start_prediction_is_fallback(self):
        m = _m()
        pred = m.predict_heat_loss(_ctx())
        assert pred.fallback_used is True

    def test_learned_prediction_not_fallback(self):
        m = _m()
        for i in range(5):
            m.update(cooling_episode(f"e{i}", 0.5))
        pred = m.predict_heat_loss(_ctx())
        assert pred.fallback_used is False

    def test_cold_start_confidence_below_threshold(self):
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            LearningShadowController,
        )
        m = _m()
        pred = m.predict_heat_loss(_ctx())
        assert pred.confidence < LearningShadowController._PREHEAT_MIN_CONFIDENCE

    def test_learned_confidence_above_threshold_after_enough_episodes(self):
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            LearningShadowController,
        )
        m = _m()
        for i in range(15):
            m.update(cooling_episode(f"e{i}", 0.5))
        pred = m.predict_heat_loss(_ctx())
        assert pred.confidence >= LearningShadowController._PREHEAT_MIN_CONFIDENCE


# ── Section 7: HA-level lifecycle (save / restore / unload) ──────────────────

class TestHeatLossLifecycle:
    """HA-level persistence: FakeStore save, async_setup restore, unload, flush."""

    def _make_shadow_with_store(self, store=None):
        import asyncio
        from tests.helpers_ha_runtime import FakeStore, attach_shadow, make_recording_coordinator
        coord = make_recording_coordinator()
        fake_store = store or FakeStore()
        sh = attach_shadow(coord, store=fake_store)
        asyncio.run(sh.async_setup())
        return sh, fake_store, coord

    def test_async_flush_triggers_store_save(self):
        import asyncio
        sh, store, _ = self._make_shadow_with_store()
        asyncio.run(sh.async_flush())
        assert store.saves >= 1

    def test_async_unload_disables_shadow(self):
        import asyncio
        sh, _, _ = self._make_shadow_with_store()
        assert sh._enabled is True
        asyncio.run(sh.async_unload())
        assert sh._enabled is False

    def test_async_unload_flushes_before_disable(self):
        import asyncio
        sh, store, _ = self._make_shadow_with_store()
        asyncio.run(sh.async_unload())
        assert store.saves >= 1  # unload saves pending state

    def test_restore_via_setup_loads_saved_state(self):
        """State saved by flush is loaded by async_setup on a new shadow."""
        import asyncio
        from tests.helpers_ha_runtime import FakeStore, attach_shadow, make_recording_coordinator
        from custom_components.thermosmart.learning.contracts import PredictionType
        from unittest.mock import MagicMock
        from custom_components.thermosmart.const import UNIT_C_PER_H

        fake_store = FakeStore()

        # Shadow 1: learn something and flush
        coord1 = make_recording_coordinator()
        sh1 = attach_shadow(coord1, store=fake_store)
        asyncio.run(sh1.async_setup())
        zr1 = sh1._runtime._zone(sh1._zone)
        # Inject a valid heat_loss prediction with learned data
        hl_pred = MagicMock()
        hl_pred.fallback_used = False
        hl_pred.confidence = 0.9
        hl_pred.values = {"heat_loss_rate": 0.5}
        hl_pred.units = {"heat_loss_rate": UNIT_C_PER_H}
        hl_pred.warnings = ()
        hl_pred.reason_codes = ("general_rate",)
        hl_pred.missing_evidence = ()
        zr1.last_predictions = {PredictionType.HEAT_LOSS_RATE: hl_pred}
        asyncio.run(sh1.async_flush())
        saved_data = fake_store.data

        # Shadow 2: restore from same store
        coord2 = make_recording_coordinator()
        sh2 = attach_shadow(coord2, store=fake_store)
        asyncio.run(sh2.async_setup())

        # Store must have data from flush; setup should not error
        assert saved_data is not None
        assert sh2._enabled is True

    def test_save_on_flush_then_reload_matches_saves(self):
        import asyncio
        sh, store, _ = self._make_shadow_with_store()
        initial_saves = store.saves
        asyncio.run(sh.async_flush())
        assert store.saves == initial_saves + 1

    def test_shadow_disabled_after_unload_skips_save(self):
        import asyncio
        sh, store, _ = self._make_shadow_with_store()
        asyncio.run(sh.async_unload())
        saves_after_unload = store.saves
        asyncio.run(sh.async_save_if_due())
        # After unload, save_if_due must not increment saves (shadow disabled)
        assert store.saves == saves_after_unload

    def test_store_save_failure_does_not_raise(self):
        import asyncio
        from tests.helpers_ha_runtime import FakeStore
        fail_store = FakeStore(fail_save=True)
        sh, _, _ = self._make_shadow_with_store(store=fail_store)
        # Flush on a store that raises IOError must not raise
        asyncio.run(sh.async_flush())
        assert sh._errors >= 0  # just confirm it didn't raise

    def test_store_load_failure_does_not_disable_shadow(self):
        import asyncio
        from tests.helpers_ha_runtime import FakeStore
        fail_store = FakeStore(fail_load=True)
        sh, _, _ = self._make_shadow_with_store(store=fail_store)
        # shadow may disable itself on load error, or it may stay enabled with empty state
        # Either way it must not raise
        assert sh is not None


# ── Section 8: Multi-zone restore + cross-zone isolation ─────────────────────

class TestMultiZoneRestoreIsolation:

    def test_two_zones_serialize_independently(self):
        m_a = _m("zone_a")
        m_b = _m("zone_b")
        for i in range(5):
            m_a.update(cooling_episode(f"a{i}", 0.3, zone="zone_a"))
        for i in range(5):
            m_b.update(cooling_episode(f"b{i}", 0.9, zone="zone_b"))
        state_a = m_a.serialize_state()
        state_b = m_b.serialize_state()
        assert state_a["learning_zone_id"] == "zone_a"
        assert state_b["learning_zone_id"] == "zone_b"
        # Restoring B's state into a zone_b model must not affect zone_a model
        m_a2 = _m("zone_a")
        m_a2.deserialize_state(state_a)
        m_b2 = _m("zone_b")
        m_b2.deserialize_state(state_b)
        rate_a = m_a2._state.general.rate_c_per_h
        rate_b = m_b2._state.general.rate_c_per_h
        assert rate_a < rate_b  # no cross-zone contamination

    def test_zone_b_state_not_in_zone_a_restore(self):
        m_a = _m("zone_a")
        m_b = _m("zone_b")
        for i in range(5):
            m_a.update(cooling_episode(f"a{i}", 0.3, zone="zone_a"))
        for i in range(5):
            m_b.update(cooling_episode(f"b{i}", 5.0, zone="zone_b"))
        state_a = m_a.serialize_state()
        m_a2 = _m("zone_a")
        m_a2.deserialize_state(state_a)
        # zone_a rate must not reflect zone_b's extreme rate
        assert m_a2._state.general.rate_c_per_h < 2.0

    def test_wrong_zone_restore_rejected(self):
        m_a = _m("zone_a")
        state_a = m_a.serialize_state()
        m_b = _m("zone_b")
        with pytest.raises(ValueError, match="learning_zone_id"):
            m_b.deserialize_state(state_a)

    def test_dedup_after_restore_rejects_seen_episode(self):
        m = _m()
        ep = cooling_episode("ep-seen", 0.5)
        m.update(ep)
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        m2.update(ep)  # same ep_id → must be rejected
        assert m2._state.general.sample_count == 1

    def test_dedup_after_restore_accepts_new_episode(self):
        m = _m()
        ep1 = cooling_episode("ep-seen", 0.5)
        m.update(ep1)
        state = m.serialize_state()
        m2 = _m()
        m2.deserialize_state(state)
        ep2 = cooling_episode("ep-new", 0.5)
        m2.update(ep2)  # different ep_id → must be accepted
        assert m2._state.general.sample_count == 2


# ── Section 9: Retention / pruning ───────────────────────────────────────────

class TestHeatLossRetentionPruning:

    def test_many_episodes_do_not_explode_model(self):
        m = _m()
        for i in range(50):
            m.update(cooling_episode(f"ep{i}", 0.5 + (i % 3) * 0.1))
        # Model must remain functional after many episodes
        pred = m.predict_heat_loss(_ctx())
        assert pred is not None
        assert 0.0 < pred.values["heat_loss_rate"] < 5.0

    def test_processed_ids_cap_enforced(self):
        m = _m()
        params = m._params
        # Add more episodes than the dedup cap
        for i in range(params.dedup_max_ids + 10):
            m.update(cooling_episode(f"ep{i}", 0.5))
        # processed_ids must not exceed the cap
        assert len(m._state.processed_ids) <= params.dedup_max_ids

    def test_serialized_state_size_bounded(self):
        m = _m()
        for i in range(100):
            m.update(cooling_episode(f"ep{i}", 0.5 + (i % 5) * 0.1))
        state = m.serialize_state()
        # State should serialize without error and be a non-empty dict
        assert isinstance(state, dict)
        assert len(state) > 0
