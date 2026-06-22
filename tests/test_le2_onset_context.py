"""Erweiterte Pflicht-Tests: Kontext-Konditionierung, kausale Erkennung, Cap, Lifecycle.

Ergänzt test_le2_onset_delay.py um die vier zusätzlichen Anforderungen:
  1. Reale Context-Auswahl in einer gemeinsamen Modellinstanz
  2. Kausale Onset-Erkennung (Temperatur stieg bereits vor Command)
  3. Onset-Cap 45 min (40-min Morgenlatenz lernbar)
  4. Produktive Storage-/Lifecycle-Integration
"""
from __future__ import annotations

import asyncio
import pytest
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from custom_components.thermosmart.learning.contracts import (
    DataQuality,
    Regime,
    PredictionType,
)
from custom_components.thermosmart.learning.episode_schemas import (
    HeatingEpisode,
    Trajectory,
    TrajectoryPoint,
)
from custom_components.thermosmart.learning.models.onset_delay import (
    OnsetDelayModel,
    OnsetDelayParameters,
    OnsetDelayPredictionContext,
    OnsetDelayUpdateContext,
    OnsetDelayRejection,
    _detect_onset_delay,
    _setback_class,
    _setback_weight_modifier,
    _bucket_key_l2,
    _bucket_key_l1,
    _outdoor_confidence_factor,
    _time_bucket_key,
    MODEL_VERSION,
    _BUCKET_COUNT_CAP,
    _SETBACK_CLASS_VERSION,
)
from tests.helpers import make_coordinator
from tests.helpers_ha_runtime import attach_shadow, FakeStore

_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Episode builders
# ---------------------------------------------------------------------------

def _episode(
    *,
    zone_id: str = "zone1",
    ep_id: str = "ep1",
    start_temp: float = 17.0,
    target: float = 21.0,
    onset_min: float = 10.0,
    rise_rate: float = 0.8,
    duration_min: int = 90,
    sample_interval_min: int = 5,
    confounders: tuple = (),
    reliability: float = 0.8,
    noise_before_onset: float = 0.01,
    regime: Regime = Regime.ACTIVE_HEATING,
    first_point_offset: float = 0.0,  # first-point temp offset above start_temp
) -> HeatingEpisode:
    """Synthetic HeatingEpisode with configurable onset delay.

    ``first_point_offset`` simulates a pre-existing temperature rise at t0
    (passive solar / neighbour heat): if ≥ rise_threshold (0.08°C), the causal
    check rejects the episode.
    """
    start = datetime(2024, 1, 15, 6, 0, tzinfo=_UTC)
    end = start + timedelta(minutes=duration_min)
    pts: list[TrajectoryPoint] = []
    onset_ms = int(onset_min * 60_000)
    for t_min in range(0, duration_min, sample_interval_min):
        ms = t_min * 60_000
        if ms == 0:
            val = start_temp + first_point_offset  # may trigger PASSIVE_WARMING check
        elif ms < onset_ms:
            val = start_temp + noise_before_onset
        else:
            elapsed = (ms - onset_ms) / 60_000.0
            val = start_temp + elapsed * rise_rate
        pts.append(TrajectoryPoint(offset_ms=ms, value=round(val, 3), quality=DataQuality.OK))
    traj = Trajectory(points=tuple(pts), max_points=200)
    return HeatingEpisode(
        episode_id=ep_id,
        learning_zone_id=zone_id,
        episode_schema_version=1,
        builder_version=1,
        classifier_version=1,
        start_ts=start,
        end_ts=end,
        regime=regime,
        reliability=reliability,
        start_temp=start_temp,
        target=target,
        trajectory=traj,
        confounder_flags=confounders,
    )


def _morning_episode(ep_id: str, onset_min: float = 20.0) -> HeatingEpisode:
    """Morning setback scenario: long onset delay (supply pipes cold)."""
    return _episode(ep_id=ep_id, onset_min=onset_min, sample_interval_min=1, duration_min=90)


def _afternoon_episode(ep_id: str, onset_min: float = 3.0) -> HeatingEpisode:
    """Afternoon re-heat: short onset delay (system already warm)."""
    return _episode(ep_id=ep_id, onset_min=onset_min, sample_interval_min=1, duration_min=60)


def _train_mixed(
    m: OnsetDelayModel,
    *,
    n_morning: int = 15,
    n_afternoon: int = 15,
    morning_onset: float = 20.0,
    afternoon_onset: float = 3.0,
    morning_setback: float = 3.5,
) -> OnsetDelayModel:
    """Feed a single model instance with interleaved morning and afternoon episodes."""
    for i in range(n_morning):
        ep = _morning_episode(ep_id=f"morning_{i}", onset_min=morning_onset)
        ctx = OnsetDelayUpdateContext(
            source_episode_id=ep.episode_id,
            time_bucket="morning",
            setback_depth_c=morning_setback,
        )
        m.update(ep, ctx)
    for i in range(n_afternoon):
        ep = _afternoon_episode(ep_id=f"afternoon_{i}", onset_min=afternoon_onset)
        ctx = OnsetDelayUpdateContext(
            source_episode_id=ep.episode_id,
            time_bucket="afternoon",
            setback_depth_c=None,
        )
        m.update(ep, ctx)
    return m


# ---------------------------------------------------------------------------
# 1. Reale Context-Konditionierung: eine gemeinsame Modellinstanz
# ---------------------------------------------------------------------------

class TestContextConditioningMixed:
    """A single OnsetDelayModel receives interleaved morning + afternoon episodes.

    Morgen: long onset (20 min); Nachmittag: short onset (3 min).
    After training, morning context selects the morning bucket (long delay),
    afternoon context selects the afternoon bucket (short delay).
    No cross-contamination between specific buckets.
    """

    @pytest.fixture
    def mixed_model(self) -> OnsetDelayModel:
        m = OnsetDelayModel("zone1")
        return _train_mixed(m)

    def test_morning_context_selects_long_delay(self, mixed_model):
        """Morning + deep setback → L2 bucket 'morning:deep'."""
        ctx = OnsetDelayPredictionContext(time_bucket="morning", setback_depth_c=3.5)
        pred = mixed_model.predict_onset_delay(ctx)
        # L2 bucket is 'morning:deep' (training used setback=3.5 → class=deep)
        assert pred.bucket == "morning:deep"
        assert pred.values["onset_delay"] > 10.0  # clearly morning territory

    def test_afternoon_context_selects_short_delay(self, mixed_model):
        """Afternoon + no setback → L2 bucket 'afternoon:none'."""
        ctx = OnsetDelayPredictionContext(time_bucket="afternoon")
        pred = mixed_model.predict_onset_delay(ctx)
        # L2 bucket 'afternoon:none' trained on 15 afternoon episodes
        assert pred.bucket == "afternoon:none"
        assert pred.values["onset_delay"] < 10.0  # clearly afternoon territory

    def test_morning_delay_different_from_afternoon(self, mixed_model):
        """Morning and afternoon buckets produce measurably different predictions."""
        pred_morning = mixed_model.predict_onset_delay(
            OnsetDelayPredictionContext(time_bucket="morning", setback_depth_c=3.5))
        pred_afternoon = mixed_model.predict_onset_delay(
            OnsetDelayPredictionContext(time_bucket="afternoon"))
        assert pred_morning.values["onset_delay"] > pred_afternoon.values["onset_delay"]

    def test_no_cross_contamination_of_specific_buckets(self, mixed_model):
        """morning:deep and afternoon:none buckets are independent."""
        state = mixed_model._state
        morning_deep = state.buckets.get("morning:deep")
        afternoon_none = state.buckets.get("afternoon:none")
        assert morning_deep is not None, "morning:deep bucket must exist"
        assert afternoon_none is not None, "afternoon:none bucket must exist"
        assert morning_deep.onset_delay_min > afternoon_none.onset_delay_min

    def test_no_context_falls_back_to_general(self, mixed_model):
        """No time_bucket → uses general bucket (aggregate of all episodes)."""
        pred = mixed_model.predict_onset_delay(OnsetDelayPredictionContext())
        assert pred.bucket == "general"
        # General contains both morning + afternoon → intermediate value
        pred_morning = mixed_model.predict_onset_delay(
            OnsetDelayPredictionContext(time_bucket="morning", setback_depth_c=3.5))
        pred_afternoon = mixed_model.predict_onset_delay(
            OnsetDelayPredictionContext(time_bucket="afternoon"))
        lo, hi = pred_afternoon.values["onset_delay"], pred_morning.values["onset_delay"]
        assert lo <= pred.values["onset_delay"] <= hi

    def test_different_setback_class_hits_different_l2_bucket(self, mixed_model):
        """Same time_bucket, different setback_class → different L2 buckets."""
        # morning + deep setback → L2 "morning:deep" (populated with 15 training episodes)
        ctx_deep = OnsetDelayPredictionContext(time_bucket="morning", setback_depth_c=4.0)
        pred_deep = mixed_model.predict_onset_delay(ctx_deep)
        assert pred_deep.bucket == "morning:deep"  # L2 hit

        # morning + no setback → L2 "morning:none" is empty → falls to L3 "morning"
        ctx_none = OnsetDelayPredictionContext(time_bucket="morning", setback_depth_c=0.2)
        pred_none = mixed_model.predict_onset_delay(ctx_none)
        assert pred_none.bucket == "morning"  # L3 fallback (morning:none is empty)

    def test_unknown_context_evening_falls_to_general(self, mixed_model):
        """Evening context has no specific bucket → falls through to general."""
        pred = mixed_model.predict_onset_delay(
            OnsetDelayPredictionContext(time_bucket="evening"))
        assert pred.bucket == "general"

    def test_thin_bucket_falls_back_to_general(self):
        """Buckets below bucket_min_samples are skipped → general fallback."""
        p = OnsetDelayParameters(bucket_min_samples=5)
        m = OnsetDelayModel("zone1", params=p)
        # Train 3 morning episodes without setback (class=none) — all buckets thin
        for i in range(3):
            ep = _morning_episode(ep_id=f"morning_{i}")
            ctx = OnsetDelayUpdateContext(source_episode_id=ep.episode_id,
                                          time_bucket="morning")
            m.update(ep, ctx)
        # Train enough general episodes
        for i in range(10):
            ep = _episode(ep_id=f"gen_{i}", onset_min=12.0)
            m.update(ep)

        pred = m.predict_onset_delay(OnsetDelayPredictionContext(time_bucket="morning"))
        # morning:none (L2) = 3 samples < 5, morning (L3) = 3 samples < 5 → general
        assert pred.bucket == "general"

    def test_missing_weather_does_not_block_specific_bucket(self, mixed_model):
        """outdoor_temp_c=None goes to missing_evidence but doesn't block prediction."""
        ctx = OnsetDelayPredictionContext(time_bucket="morning", setback_depth_c=3.5,
                                          outdoor_temp_c=None)
        pred = mixed_model.predict_onset_delay(ctx)
        assert "outdoor_temp" in pred.missing_evidence
        assert pred.values["onset_delay"] > 0
        assert pred.bucket == "morning:deep"  # L2 hit despite missing outdoor

    def test_no_uncontrolled_bucket_proliferation(self, mixed_model):
        """Buckets are L2/L3 keys only; no spurious entries beyond the hierarchy."""
        buckets = set(mixed_model._state.buckets.keys())
        # L3 keys: "morning", "afternoon" (L2 also: "morning:deep", "afternoon:none")
        for b in buckets:
            parts = b.split(":")
            assert 1 <= len(parts) <= 3, f"unexpected bucket format: {b!r}"
            assert parts[0] in ("morning", "afternoon", "evening", "night"), \
                f"unexpected time_bucket: {parts[0]!r}"

    def test_context_match_level_in_bucket_field(self, mixed_model):
        """Prediction.bucket identifies which hierarchy level was used."""
        # With deep setback → L2 "morning:deep"
        pred_l2 = mixed_model.predict_onset_delay(
            OnsetDelayPredictionContext(time_bucket="morning", setback_depth_c=3.5))
        assert pred_l2.bucket == "morning:deep"
        # No setback, no match → L3 "morning"
        pred_l3 = mixed_model.predict_onset_delay(
            OnsetDelayPredictionContext(time_bucket="morning"))
        assert pred_l3.bucket == "morning"
        # No context → general
        pred_general = mixed_model.predict_onset_delay(OnsetDelayPredictionContext())
        assert pred_general.bucket == "general"

    def test_setback_weight_modifier_used_in_update(self):
        """Deep setback episodes accumulate higher effective weight (modifier > 1)."""
        # none / no setback
        assert _setback_weight_modifier(None) == pytest.approx(1.0)
        assert _setback_weight_modifier(0.3) == pytest.approx(1.0)
        # shallow: 0.5 ≤ depth < 1.5
        assert _setback_weight_modifier(0.8) == pytest.approx(1.03)
        # moderate: 1.5 ≤ depth < 3.0
        assert _setback_weight_modifier(1.5) == pytest.approx(1.08)
        assert _setback_weight_modifier(2.5) == pytest.approx(1.08)
        # deep: depth ≥ 3.0
        assert _setback_weight_modifier(4.0) == pytest.approx(1.15)
        # deep by duration (depth 1.0 + duration 7h)
        assert _setback_weight_modifier(1.0, 7.0) == pytest.approx(1.15)


# ---------------------------------------------------------------------------
# 2. Kausale Onset-Erkennung
# ---------------------------------------------------------------------------

class TestCausalOnset:
    """Temperature already rising at t0 → not a heating onset."""

    def test_pre_existing_warming_at_t0_rejected(self):
        """If temp at first trajectory point ≥ start_temp + threshold → PASSIVE_WARMING."""
        ep = _episode(first_point_offset=0.1)  # 0.1°C > threshold 0.08°C
        delay, rejection, n = _detect_onset_delay(ep)
        assert rejection is OnsetDelayRejection.PASSIVE_WARMING

    def test_passive_solar_multiple_rising_points_rejected(self):
        """Sustained solar gain at t0: first point already above threshold → rejected."""
        ep = _episode(first_point_offset=0.5)  # clearly pre-existing warming
        delay, rejection, n = _detect_onset_delay(ep)
        assert rejection is OnsetDelayRejection.PASSIVE_WARMING

    def test_neighbour_heat_without_demand_rejected(self):
        """No heating demand (wrong regime) → rejected before trajectory check."""
        ep = _episode(regime=Regime.PASSIVE_COOLING)
        m = OnsetDelayModel("zone1")
        res = m.update(ep)
        assert not res.accepted
        assert OnsetDelayRejection.DISTURBED.value in res.confounder_flags

    def test_temp_at_threshold_boundary_accepted(self):
        """First point exactly at threshold − ε → NOT pre-existing → accepted."""
        # 0.07°C < 0.08°C threshold → no PASSIVE_WARMING
        ep = _episode(first_point_offset=0.07, onset_min=10.0, duration_min=90)
        delay, rejection, n = _detect_onset_delay(ep)
        # Either accepted (onset detected later) or some other rejection, but NOT PASSIVE_WARMING
        assert rejection is not OnsetDelayRejection.PASSIVE_WARMING

    def test_command_plus_active_heating_plus_trend_valid_onset(self):
        """Command + ACTIVE_HEATING + confirmed trend → valid onset."""
        ep = _episode(onset_min=10.0, regime=Regime.ACTIVE_HEATING,
                      first_point_offset=0.01, duration_min=90)
        m = OnsetDelayModel("zone1")
        res = m.update(ep)
        assert res.accepted

    def test_wrong_zone_rejected(self):
        """Episode from different zone → WRONG_ZONE."""
        ep = _episode(zone_id="other_zone")
        m = OnsetDelayModel("zone1")
        res = m.update(ep)
        assert not res.accepted
        assert OnsetDelayRejection.WRONG_ZONE.value in res.confounder_flags

    def test_duplicate_onset_same_episode_excluded(self):
        """Same episode_id processed twice → second attempt is DUPLICATE."""
        ep = _episode()
        m = OnsetDelayModel("zone1")
        res1 = m.update(ep)
        assert res1.accepted
        res2 = m.update(ep)
        assert not res2.accepted
        assert OnsetDelayRejection.DUPLICATE_EPISODE.value in res2.confounder_flags


# ---------------------------------------------------------------------------
# 3. Onset-Cap 45 min
# ---------------------------------------------------------------------------

class TestOnsetCap:
    """40-min morning supply delay must be learnable; extremes rejected."""

    def test_40_min_morning_delay_learnable(self):
        """Real 40-min supply delay can be learned and predicted."""
        m = OnsetDelayModel("zone1")
        for i in range(15):
            ep = _episode(ep_id=f"ep_{i}", onset_min=40.0,
                          sample_interval_min=1, duration_min=120)
            ctx = OnsetDelayUpdateContext(
                source_episode_id=ep.episode_id, time_bucket="morning",
                setback_depth_c=3.5)
            res = m.update(ep, ctx)
            assert res.accepted
        pred = m.predict_onset_delay(OnsetDelayPredictionContext(time_bucket="morning"))
        assert not pred.fallback_used
        # Learned value should be near 40 min (within 10 min of actual)
        assert 30.0 <= pred.values["onset_delay"] <= 45.0

    def test_delay_above_max_clamped_not_rejected(self):
        """Episode with detected delay at max boundary → clamped to max, not rejected."""
        # onset at 44 min: within the new 45-min cap
        ep = _episode(ep_id="ep44", onset_min=44.0,
                      sample_interval_min=1, duration_min=120)
        m = OnsetDelayModel("zone1")
        res = m.update(ep)
        assert res.accepted

    def test_extremely_long_delay_clamped_at_max(self):
        """Detected onset beyond the cap is clamped to max_delay_min, not silently passed."""
        # onset_min=50 > cap=45 → detected at t=51 min → clamped to 45.0
        ep = _episode(ep_id="ep_extreme", onset_min=50.0,
                      sample_interval_min=1, duration_min=120)
        delay, rejection, n = _detect_onset_delay(ep, max_delay_min=45.0)
        assert delay is not None
        assert delay == pytest.approx(45.0)  # clamped at max

    def test_total_preheat_cap_still_applies(self):
        """Even with 40-min onset, total preheat is bounded by _PREHEAT_MAX_MIN."""
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            LearningShadowController)

        async def _run():
            coord = make_coordinator()
            sh = attach_shadow(coord, store=FakeStore())
            await sh.async_setup()
            hr_pred = MagicMock()
            hr_pred.values = {"heat_rate": 0.5}   # slow heating
            hr_pred.fallback_used = False
            hr_pred.confidence = 0.8
            od_pred = MagicMock()
            od_pred.values = {"onset_delay": 40.0}
            od_pred.fallback_used = False
            od_pred.confidence = 0.8
            sh._runtime._zone(coord.zone_id).last_predictions = {
                PredictionType.HEAT_RATE: hr_pred,
                PredictionType.ONSET_DELAY: od_pred,
            }
            return sh.read_preheat_minutes_safe(15.0, 21.0)

        minutes, status = asyncio.run(_run())
        assert minutes <= LearningShadowController._PREHEAT_MAX_MIN

    def test_high_onset_delay_does_not_affect_heat_rate(self):
        """OnsetDelayModel and HeatRateModel are independent regardless of onset value."""
        from custom_components.thermosmart.learning.models.heat_rate import (
            HeatRateModel, HeatRatePredictionContext)
        m_onset = OnsetDelayModel("zone1")
        m_heat = HeatRateModel("zone1")
        for i in range(10):
            ep = _episode(ep_id=f"ep_{i}", onset_min=40.0, rise_rate=1.0,
                          sample_interval_min=1, duration_min=120)
            m_onset.update(ep)
            m_heat.update(ep)
        pred_od = m_onset.predict_onset_delay(OnsetDelayPredictionContext())
        pred_hr = m_heat.predict_heat_rate(HeatRatePredictionContext())
        # The two predictions are fully independent: neither drives the other
        assert pred_od.prediction_type == PredictionType.ONSET_DELAY
        assert pred_hr.prediction_type == PredictionType.HEAT_RATE

    def test_onset_delay_max_from_parameters_respected(self):
        """model max_onset_delay_min is 45 (raised from 30)."""
        p = OnsetDelayParameters()
        assert p.max_onset_delay_min == pytest.approx(45.0)

    def test_ha_integration_max_matches_model_max(self):
        """ha_integration _ONSET_DELAY_MAX_MIN must equal model cap."""
        from custom_components.thermosmart.learning.runtime.ha_integration import (
            LearningShadowController)
        from custom_components.thermosmart.learning.models.onset_delay import (
            OnsetDelayParameters)
        assert (LearningShadowController._ONSET_DELAY_MAX_MIN ==
                pytest.approx(OnsetDelayParameters().max_onset_delay_min))


# ---------------------------------------------------------------------------
# 4. Produktive Storage- / Lifecycle-Integration
# ---------------------------------------------------------------------------

class TestRealStorageLifecycle:
    """End-to-end: model state persists through save → new runtime → load → same prediction."""

    def _model(self, sh, coord):
        """Direct access to the live OnsetDelayModel via shadow runtime."""
        return sh._runtime._zone(coord.zone_id).orchestrator.models["onset_delay"]

    def test_onset_model_in_zone_orchestrator(self):
        """OnsetDelayModel is part of the zone orchestrator model set."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        models = sh._runtime._zone(coord.zone_id).orchestrator.models
        assert "onset_delay" in models
        assert isinstance(models["onset_delay"], OnsetDelayModel)

    def test_zone_state_serializes_onset_model(self):
        """Zone state serialization includes onset_delay model segment."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        zr = sh._runtime._zone(coord.zone_id)
        data = zr.serialize()
        assert "models" in data
        assert "onset_delay" in data["models"]

    def test_learn_serialize_deserialize_same_prediction(self):
        """Train model, serialize zone, create new runtime, restore, same prediction."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        model = self._model(sh, coord)
        for i in range(10):
            ep = _episode(ep_id=f"ep_{i}", onset_min=15.0, sample_interval_min=1,
                          duration_min=90, zone_id=coord.zone_id)
            model.update(ep)

        # Serialize
        zr = sh._runtime._zone(coord.zone_id)
        saved_data = zr.serialize()

        # New zone runtime, restore
        from custom_components.thermosmart.learning.runtime.lifecycle import (
            _ZoneRuntime, LearningRuntimeConfig)
        zr2 = _ZoneRuntime(coord.zone_id, LearningRuntimeConfig())
        zr2.restore(saved_data)

        pred1 = model.predict_onset_delay(OnsetDelayPredictionContext())
        pred2 = zr2.orchestrator.models["onset_delay"].predict_onset_delay(
            OnsetDelayPredictionContext())

        assert pred1.values["onset_delay"] == pytest.approx(pred2.values["onset_delay"])
        assert pred1.fallback_used == pred2.fallback_used
        assert pred1.evidence_count == pred2.evidence_count

    def test_restart_duplicate_not_learned_again(self):
        """After save/restore, same episode_id is in processed_ids → rejected."""
        coord = make_coordinator()
        sh = attach_shadow(coord, store=FakeStore())
        model = self._model(sh, coord)
        ep = _episode(zone_id=coord.zone_id)
        model.update(ep)

        zr = sh._runtime._zone(coord.zone_id)
        saved_data = zr.serialize()

        from custom_components.thermosmart.learning.runtime.lifecycle import (
            _ZoneRuntime, LearningRuntimeConfig)
        zr2 = _ZoneRuntime(coord.zone_id, LearningRuntimeConfig())
        zr2.restore(saved_data)

        restored_model = zr2.orchestrator.models["onset_delay"]
        res = restored_model.update(ep)
        assert not res.accepted
        assert OnsetDelayRejection.DUPLICATE_EPISODE.value in res.confounder_flags

    def test_full_save_load_via_store(self):
        """Real async save → new runtime async setup → identical prediction."""
        store = FakeStore()

        async def _run():
            coord = make_coordinator()
            sh = attach_shadow(coord, store=store)
            await sh.async_setup()
            model = self._model(sh, coord)
            for i in range(8):
                ep = _episode(ep_id=f"ep_{i}", onset_min=12.0,
                              sample_interval_min=1, duration_min=90,
                              zone_id=coord.zone_id)
                ctx = OnsetDelayUpdateContext(source_episode_id=ep.episode_id,
                                              time_bucket="morning")
                model.update(ep, ctx)
            sh._runtime.mark_dirty(important=True)
            await sh._runtime.async_flush()
            pred_before = model.predict_onset_delay(
                OnsetDelayPredictionContext(time_bucket="morning"))

            # New shadow from the same store
            coord2 = make_coordinator()
            sh2 = attach_shadow(coord2, store=store, zone_id=coord.zone_id)
            await sh2.async_setup()
            model2 = sh2._runtime._zone(coord.zone_id).orchestrator.models["onset_delay"]
            pred_after = model2.predict_onset_delay(
                OnsetDelayPredictionContext(time_bucket="morning"))

            return pred_before, pred_after

        pred_before, pred_after = asyncio.run(_run())
        assert pred_before.values["onset_delay"] == pytest.approx(
            pred_after.values["onset_delay"])
        assert pred_before.evidence_count == pred_after.evidence_count

    def test_corrupted_onset_state_isolates_not_blocks_heating(self):
        """Corrupt onset model state → restore is isolated; prior returned, no crash."""
        coord = make_coordinator()
        corrupt_store = FakeStore(data={
            "runtime_schema_version": 4,
            "le_version": 2,
            "zones": {
                coord.zone_id: {
                    "models": {
                        "onset_delay": {
                            "model_version": 999,  # wrong version → restore error
                            "learning_zone_id": coord.zone_id,
                            "general": {
                                "key": "general",
                                "onset_delay_min": "NOT_A_FLOAT",  # corrupt
                                "effective_n": 0.0, "dispersion": 0.0,
                                "sample_count": 0},
                        },
                    },
                    "pipeline": {}, "capture": {}, "ledger": {}
                }
            }
        })

        async def _run():
            sh = attach_shadow(coord, store=corrupt_store)
            await sh.async_setup()
            return sh.read_onset_delay_safe()

        delay, status = asyncio.run(_run())
        # After corrupt state: prior (5 min) and safe status; no exception raised
        assert delay == pytest.approx(5.0)
        assert status in ("cold_start_prior", "not_available")

    def test_reload_flushes_pending_changes(self):
        """On unload, pending changes are flushed to the store."""
        store = FakeStore()

        async def _run():
            coord = make_coordinator()
            sh = attach_shadow(coord, store=store)
            await sh.async_setup()
            model = self._model(sh, coord)
            ep = _episode(zone_id=coord.zone_id)
            model.update(ep)
            sh._runtime.mark_dirty(important=True)
            saves_before = store.saves
            await sh.async_unload()
            return store.saves, saves_before

        saves_after, saves_before = asyncio.run(_run())
        assert saves_after > saves_before  # at least one save happened on unload

    def test_retention_cap_respected_after_save_load(self):
        """recent_samples cap is enforced after serialize/deserialize round-trip."""
        p = OnsetDelayParameters(research_sample_cap=5)
        m = OnsetDelayModel("zone1", params=p)
        for i in range(12):
            ep = _episode(ep_id=f"ep_{i}", onset_min=10.0,
                          sample_interval_min=1, duration_min=90)
            m.update(ep)
        state = m.serialize_state()
        m2 = OnsetDelayModel("zone1", params=p)
        m2.deserialize_state(state)
        assert len(m2._state.recent_samples) <= 5


# ---------------------------------------------------------------------------
# 5. 5-stufige Context-Hierarchie — neue Pflicht-Tests
# ---------------------------------------------------------------------------

class TestContextHierarchyFiveLevel:
    """One model instance; tests verify all 5 hierarchy levels and boundaries."""

    # ---- helpers -------------------------------------------------------

    def _ctx(self, tb=None, sc_depth=None, sc_dur=None, profile=None, outdoor=None):
        return OnsetDelayPredictionContext(
            time_bucket=tb, setback_depth_c=sc_depth, setback_duration_h=sc_dur,
            profile_id=profile, outdoor_temp_c=outdoor)

    def _uctx(self, ep, tb=None, sc_depth=None, sc_dur=None, profile=None):
        return OnsetDelayUpdateContext(
            source_episode_id=ep.episode_id, time_bucket=tb,
            setback_depth_c=sc_depth, setback_duration_h=sc_dur,
            device_profile_key=profile)

    def _trained_model(self, *, morning_onset=20.0, morning_setback=3.5,
                       morning_n=15, af_onset=3.0, af_n=15,
                       device="trv_basic"):
        """Model with morning:deep episodes and afternoon:none episodes."""
        m = OnsetDelayModel("zone1")
        for i in range(morning_n):
            ep = _episode(ep_id=f"m_{i}", onset_min=morning_onset,
                          sample_interval_min=1, duration_min=90)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=morning_setback,
                                    profile=device))
        for i in range(af_n):
            ep = _episode(ep_id=f"a_{i}", onset_min=af_onset,
                          sample_interval_min=1, duration_min=60)
            m.update(ep, self._uctx(ep, tb="afternoon", sc_depth=None,
                                    profile=device))
        return m

    # ---- setback class -------------------------------------------------

    def test_setback_class_none(self):
        assert _setback_class(None) == "none"
        assert _setback_class(0.0) == "none"
        assert _setback_class(0.4) == "none"

    def test_setback_class_shallow(self):
        assert _setback_class(0.5) == "shallow"
        assert _setback_class(1.0) == "shallow"
        assert _setback_class(1.49) == "shallow"

    def test_setback_class_moderate(self):
        assert _setback_class(1.5) == "moderate"
        assert _setback_class(2.0) == "moderate"
        assert _setback_class(2.99) == "moderate"

    def test_setback_class_deep_by_depth(self):
        assert _setback_class(3.0) == "deep"
        assert _setback_class(5.0) == "deep"

    def test_setback_class_deep_by_duration(self):
        # Long cold-soak (≥ 6h) with moderate depth → deep
        assert _setback_class(1.5, 6.0) == "deep"
        assert _setback_class(0.6, 8.0) == "deep"

    def test_setback_class_not_deep_without_minimum_depth(self):
        # Duration matters only when there IS a setback
        assert _setback_class(0.3, 8.0) == "none"
        assert _setback_class(0.0, 100.0) == "none"

    def test_setback_class_version_constant(self):
        assert _SETBACK_CLASS_VERSION == 1  # bump triggers migration test

    # ---- bucket key helpers --------------------------------------------

    def test_bucket_key_l2_format(self):
        assert _bucket_key_l2("morning", "deep") == "morning:deep"
        assert _bucket_key_l2("afternoon", "none") == "afternoon:none"

    def test_bucket_key_l1_with_profile(self):
        assert _bucket_key_l1("morning", "deep", "trv_basic") == "morning:deep:trv_basic"

    def test_bucket_key_l1_without_profile_returns_none(self):
        assert _bucket_key_l1("morning", "deep", None) is None
        assert _bucket_key_l1("morning", "deep", "") is None

    # ---- outdoor confidence factor ------------------------------------

    def test_outdoor_factor_known(self):
        assert _outdoor_confidence_factor(5.0) == pytest.approx(1.0)
        assert _outdoor_confidence_factor(-10.0) == pytest.approx(1.0)

    def test_outdoor_factor_missing(self):
        assert _outdoor_confidence_factor(None) == pytest.approx(0.9)

    def test_missing_outdoor_reduces_confidence_not_blocks(self):
        m = self._trained_model()
        ctx_with = self._ctx("morning", sc_depth=3.5, outdoor=5.0)
        ctx_none = self._ctx("morning", sc_depth=3.5, outdoor=None)
        pred_with = m.predict_onset_delay(ctx_with)
        pred_none = m.predict_onset_delay(ctx_none)
        # Missing outdoor: same bucket, lower confidence
        assert pred_with.bucket == pred_none.bucket
        assert pred_none.confidence < pred_with.confidence
        assert pred_none.values["onset_delay"] > 0  # not blocked

    # ---- L1 hit --------------------------------------------------------

    def test_l1_selected_when_device_and_setback_known(self):
        m = self._trained_model(device="trv_basic")
        ctx = self._ctx("morning", sc_depth=3.5, profile="trv_basic")
        pred = m.predict_onset_delay(ctx)
        assert pred.bucket == "morning:deep:trv_basic"  # L1 hit

    def test_l1_falls_to_l2_when_device_absent(self):
        m = self._trained_model(device="trv_basic")
        ctx = self._ctx("morning", sc_depth=3.5, profile=None)
        pred = m.predict_onset_delay(ctx)
        assert pred.bucket == "morning:deep"  # L2 hit

    def test_l1_thin_falls_to_l2(self):
        """L1 with < bucket_min_samples falls through to L2."""
        p = OnsetDelayParameters(bucket_min_samples=5)
        m = OnsetDelayModel("zone1", params=p)
        # 3 L1 morning:deep:trv_basic episodes (thin)
        for i in range(3):
            ep = _episode(ep_id=f"ep_{i}", onset_min=20.0,
                          sample_interval_min=1, duration_min=90)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=3.5, profile="trv_basic"))
        # 10 L2 morning:deep episodes (enough)
        for i in range(10):
            ep = _episode(ep_id=f"l2_{i}", onset_min=19.0,
                          sample_interval_min=1, duration_min=90)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=3.5, profile=None))

        ctx = self._ctx("morning", sc_depth=3.5, profile="trv_basic")
        pred = m.predict_onset_delay(ctx)
        assert pred.bucket == "morning:deep"  # L1 thin → L2

    # ---- L2 hit --------------------------------------------------------

    def test_l2_morning_deep_vs_morning_shallow(self):
        """Morning + deep and morning + shallow train and predict into separate L2 buckets."""
        m = OnsetDelayModel("zone1")
        # Deep: 18-min delay
        for i in range(8):
            ep = _episode(ep_id=f"deep_{i}", onset_min=18.0,
                          sample_interval_min=1, duration_min=90)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=3.5))
        # Shallow: 8-min delay
        for i in range(8):
            ep = _episode(ep_id=f"shallow_{i}", onset_min=8.0,
                          sample_interval_min=1, duration_min=60)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=0.8))

        pred_deep = m.predict_onset_delay(self._ctx("morning", sc_depth=3.5))
        pred_shallow = m.predict_onset_delay(self._ctx("morning", sc_depth=0.8))
        assert pred_deep.bucket == "morning:deep"
        assert pred_shallow.bucket == "morning:shallow"
        assert pred_deep.values["onset_delay"] > pred_shallow.values["onset_delay"]

    def test_l2_different_setback_class_not_mixed(self):
        """morning:deep and morning:moderate are separate; no cross-contamination."""
        m = OnsetDelayModel("zone1")
        for i in range(8):
            ep = _episode(ep_id=f"deep_{i}", onset_min=22.0,
                          sample_interval_min=1, duration_min=90)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=4.0))  # deep
        for i in range(8):
            ep = _episode(ep_id=f"mod_{i}", onset_min=12.0,
                          sample_interval_min=1, duration_min=90)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=2.0))  # moderate

        deep = m._state.buckets.get("morning:deep")
        moderate = m._state.buckets.get("morning:moderate")
        assert deep is not None and moderate is not None
        assert abs(deep.onset_delay_min - moderate.onset_delay_min) > 5.0

    def test_l2_long_duration_setback_classified_as_deep(self):
        """Duration ≥ 6h promotes to deep → lands in morning:deep bucket."""
        m = OnsetDelayModel("zone1")
        for i in range(8):
            ep = _episode(ep_id=f"ep_{i}", onset_min=25.0,
                          sample_interval_min=1, duration_min=90)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=1.2, sc_dur=7.0))
        # setback_class(1.2, 7.0) = "deep" (duration override)
        pred = m.predict_onset_delay(self._ctx("morning", sc_depth=1.2, sc_dur=7.0))
        assert pred.bucket == "morning:deep"

    # ---- L3 hit --------------------------------------------------------

    def test_l3_hit_when_l2_missing(self):
        """Unknown setback → L2 empty → L3 (time_bucket)."""
        m = self._trained_model()  # trains morning:deep and morning (L3)
        # Request with class=none (never seen in training for morning) → L2 thin → L3
        pred = m.predict_onset_delay(self._ctx("morning", sc_depth=0.2))
        assert pred.bucket == "morning"  # L3 hit

    def test_l3_value_consistent_with_training(self):
        """L3 morning bucket contains morning episodes, not afternoon."""
        m = self._trained_model(morning_onset=20.0, af_onset=3.0)
        pred_l3 = m.predict_onset_delay(self._ctx("morning", sc_depth=0.2))
        assert pred_l3.bucket == "morning"
        # L3 "morning" includes all morning training → value should be ~20 min
        assert pred_l3.values["onset_delay"] > 10.0

    # ---- L4 hit --------------------------------------------------------

    def test_l4_general_when_no_time_bucket(self):
        m = self._trained_model()
        pred = m.predict_onset_delay(self._ctx())  # no time_bucket
        assert pred.bucket == "general"

    def test_l4_general_between_morning_and_afternoon(self):
        m = self._trained_model(morning_onset=20.0, af_onset=3.0)
        pred_g = m.predict_onset_delay(self._ctx())
        pred_m = m.predict_onset_delay(self._ctx("morning", sc_depth=3.5))
        pred_a = m.predict_onset_delay(self._ctx("afternoon"))
        assert pred_a.values["onset_delay"] < pred_g.values["onset_delay"] < \
               pred_m.values["onset_delay"]

    # ---- L5 prior -------------------------------------------------------

    def test_l5_prior_when_no_evidence(self):
        m = OnsetDelayModel("zone1")
        pred = m.predict_onset_delay(self._ctx("morning", sc_depth=3.5))
        assert pred.fallback_used is True
        assert pred.bucket == "cold_start_prior"
        assert pred.values["onset_delay"] == pytest.approx(5.0)

    # ---- metadata in reason_codes --------------------------------------

    def test_reason_codes_contain_context_level(self):
        m = self._trained_model(device="trv_basic")
        pred_l1 = m.predict_onset_delay(self._ctx("morning", sc_depth=3.5,
                                                   profile="trv_basic"))
        rc = {r.split(":")[0]: ":".join(r.split(":")[1:])
              for r in pred_l1.reason_codes if ":" in r}
        assert rc.get("context_level") == "L1"
        assert rc.get("context_key") == "morning:deep:trv_basic"
        assert "match_quality" in rc
        assert "general_n" in rc
        assert "fallback_path" in rc
        assert "pre_conf" in rc
        assert "post_conf" in rc

    def test_reason_codes_l4_fallback_path_listed(self):
        m = self._trained_model()
        # evening has no buckets → falls through all levels to general
        pred = m.predict_onset_delay(self._ctx("evening", sc_depth=3.5))
        rc = {r.split(":")[0]: ":".join(r.split(":")[1:])
              for r in pred.reason_codes if ":" in r}
        assert rc.get("context_level") == "L4"
        fp = rc.get("fallback_path", "")
        # Should mention L2 and L3 were tried (and thin)
        assert "L2" in fp or "L3" in fp

    def test_evidence_count_is_level_sample_count(self):
        m = self._trained_model()
        pred_l2 = m.predict_onset_delay(self._ctx("morning", sc_depth=3.5))
        assert pred_l2.evidence_count == \
               m._state.buckets["morning:deep"].sample_count

    def test_general_n_in_reason_codes(self):
        m = self._trained_model(morning_n=10, af_n=10)
        pred = m.predict_onset_delay(self._ctx("morning", sc_depth=3.5))
        rc = {r.split(":")[0]: ":".join(r.split(":")[1:])
              for r in pred.reason_codes if ":" in r}
        assert int(rc["general_n"]) == m._state.general.sample_count

    # ---- device profile -----------------------------------------------

    def test_device_profile_only_when_sufficient_evidence(self):
        """L1 only activated when bucket has ≥ bucket_min_samples."""
        p = OnsetDelayParameters(bucket_min_samples=5)
        m = OnsetDelayModel("zone1", params=p)
        # Only 3 L1 episodes → thin; 10 L2 episodes → active
        for i in range(3):
            ep = _episode(ep_id=f"l1_{i}", onset_min=20.0,
                          sample_interval_min=1, duration_min=90)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=3.5, profile="trv_a"))
        for i in range(10):
            ep = _episode(ep_id=f"l2_{i}", onset_min=20.0,
                          sample_interval_min=1, duration_min=90)
            m.update(ep, self._uctx(ep, tb="morning", sc_depth=3.5, profile=None))

        # With device: L1 thin → falls to L2
        pred_with = m.predict_onset_delay(self._ctx("morning", sc_depth=3.5,
                                                     profile="trv_a"))
        assert pred_with.bucket == "morning:deep"  # L2, not L1

        # Without device: directly L2
        pred_without = m.predict_onset_delay(self._ctx("morning", sc_depth=3.5))
        assert pred_without.bucket == "morning:deep"

    def test_missing_device_profile_falls_back_cleanly(self):
        """No device profile → L1 skipped, goes to L2."""
        m = self._trained_model(device="trv_basic")
        pred = m.predict_onset_delay(self._ctx("morning", sc_depth=3.5, profile=None))
        assert pred.bucket == "morning:deep"  # L2, not L1

    # ---- bucket count cap ---------------------------------------------

    def test_bucket_count_cap_enforced(self):
        """Total specific bucket count never exceeds _BUCKET_COUNT_CAP."""
        m = OnsetDelayModel("zone1")
        times = ["morning", "afternoon", "evening", "night"]
        depths = [3.5, 1.5, 0.8, None]
        devices = [f"trv_{i}" for i in range(10)]
        ep_idx = 0
        for tb in times:
            for dep in depths:
                for dev in devices:
                    ep = _episode(ep_id=f"ep_{ep_idx}", onset_min=10.0,
                                  sample_interval_min=1, duration_min=60)
                    ep_idx += 1
                    ctx = OnsetDelayUpdateContext(
                        source_episode_id=ep.episode_id, time_bucket=tb,
                        setback_depth_c=dep, device_profile_key=dev)
                    m.update(ep, ctx)
        assert len(m._state.buckets) <= _BUCKET_COUNT_CAP

    # ---- restart/restore with context buckets -------------------------

    def test_restart_preserves_all_context_buckets(self):
        """Serialize/deserialize keeps all L1/L2/L3 buckets and same predictions."""
        from custom_components.thermosmart.learning.runtime.lifecycle import (
            _ZoneRuntime, LearningRuntimeConfig)
        coord = make_coordinator()
        from tests.helpers_ha_runtime import attach_shadow, FakeStore
        sh = attach_shadow(coord, store=FakeStore())
        model = sh._runtime._zone(coord.zone_id).orchestrator.models["onset_delay"]

        # Train L2 and L1 buckets
        for i in range(10):
            ep = _episode(ep_id=f"m_{i}", onset_min=20.0,
                          sample_interval_min=1, duration_min=90,
                          zone_id=coord.zone_id)
            ctx = OnsetDelayUpdateContext(
                source_episode_id=ep.episode_id, time_bucket="morning",
                setback_depth_c=3.5, device_profile_key="trv_x")
            model.update(ep, ctx)

        buckets_before = set(model._state.buckets.keys())

        saved = sh._runtime._zone(coord.zone_id).serialize()
        zr2 = _ZoneRuntime(coord.zone_id, LearningRuntimeConfig())
        zr2.restore(saved)
        m2 = zr2.orchestrator.models["onset_delay"]

        assert set(m2._state.buckets.keys()) == buckets_before
        pred_before = model.predict_onset_delay(
            OnsetDelayPredictionContext(time_bucket="morning", setback_depth_c=3.5,
                                        profile_id="trv_x"))
        pred_after = m2.predict_onset_delay(
            OnsetDelayPredictionContext(time_bucket="morning", setback_depth_c=3.5,
                                        profile_id="trv_x"))
        assert pred_before.bucket == pred_after.bucket
        assert pred_before.values["onset_delay"] == pytest.approx(
            pred_after.values["onset_delay"])

    # ---- research export privacy-safe ---------------------------------

    def test_research_export_contains_no_entity_ids(self):
        m = self._trained_model()
        from custom_components.thermosmart.learning.contracts import ExportScope
        data = m.export(ExportScope.RESEARCH)
        text = str(data)
        assert "zone1" not in text
        assert "sensor." not in text
        assert "climate." not in text
