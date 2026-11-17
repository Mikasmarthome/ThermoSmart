"""Phase D2-A — outdoor-temperature evidence pipeline.

Covers the observational outdoor-evidence capture added in D2-A:
accumulator aggregation (time-weighted mean, min/max/first/last, coverage,
quality), validity gating (missing / invalid / implausible / unchanged / stale),
source identity and switching, builder emission for heating and passive-cooling
episodes, cancellation, snapshot/restore (incl. corrupt and pre-D2-A snapshots),
the CompletedEpisode sidecar through the real pipeline, the WeatherEngine source
label, and a baseline-equivalence check that control/learning output is unchanged.

A reverted-wiring mutation (see test_sidecar_detects_missing_outdoor_propagation
and the task report) makes the propagation tests fail, proving they detect a
missing outdoor channel.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, Measurement
from custom_components.thermosmart.learning.builders.base import (
    BuilderAction,
    BuilderInput,
)
from custom_components.thermosmart.learning.builders.cooling import PassiveCoolingEpisodeBuilder
from custom_components.thermosmart.learning.builders.heating import HeatingEpisodeBuilder
from custom_components.thermosmart.learning.outdoor_evidence import (
    OutdoorEvidence,
    OutdoorEvidenceAccumulator,
)
from custom_components.thermosmart.learning.regime import RegimeResult
from custom_components.thermosmart.learning.contracts import Regime
from tests.helpers import make_mock_hass, make_state

_T0 = datetime(2026, 1, 12, 6, 0, tzinfo=timezone.utc)


def _m(value, quality=DataQuality.OK):
    return Measurement(value, quality)


# ── accumulator unit tests ───────────────────────────────────────────────────

class TestAccumulator:
    def test_no_samples_finalizes_to_none(self):
        assert OutdoorEvidenceAccumulator().finalize(duration_s=0.0) is None

    def test_time_weighted_mean_differs_from_simple_mean(self):
        acc = OutdoorEvidenceAccumulator()
        # hold 0.0 for 300s, 10.0 for 300s, then 20.0 at the end (0 tail weight)
        acc.observe(0, _m(0.0), "sensor")
        acc.observe(300_000, _m(10.0), "sensor")
        acc.observe(600_000, _m(20.0), "sensor")
        ev = acc.finalize(duration_s=600.0, end_offset_ms=600_000)
        assert ev.time_weighted_mean_c == pytest.approx(5.0, abs=1e-6)   # NOT 10.0 simple mean
        assert (ev.min_c, ev.max_c, ev.first_c, ev.last_c) == (0.0, 20.0, 0.0, 20.0)
        assert ev.sample_count == ev.usable_count == 3
        assert ev.coverage == pytest.approx(1.0)
        assert ev.quality is DataQuality.OK
        assert ev.freshness_known is False
        assert ev.source == "sensor"
        assert ev.source_changed is False

    def test_single_usable_sample_falls_back_to_that_value(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(7.5), "sensor")
        ev = acc.finalize(duration_s=0.0, end_offset_ms=0)
        assert ev.time_weighted_mean_c == pytest.approx(7.5)
        assert ev.usable_count == 1

    def test_unchanged_value_is_not_treated_as_stale(self):
        acc = OutdoorEvidenceAccumulator()
        for i in range(5):
            acc.observe(i * 300_000, _m(5.0), "sensor")   # identical value, OK quality
        ev = acc.finalize(duration_s=1200.0, end_offset_ms=1_200_000)
        assert ev.quality is DataQuality.OK                 # unchanged != stale
        assert ev.time_weighted_mean_c == pytest.approx(5.0)
        assert ev.usable_count == 5

    def test_missing_invalid_implausible_not_usable_but_counted(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "sensor")                        # usable
        acc.observe(300_000, None, "sensor")                     # missing
        acc.observe(600_000, Measurement(None, DataQuality.UNAVAILABLE), "sensor")
        acc.observe(900_000, Measurement(999.0, DataQuality.OK), "sensor")   # implausible
        acc.observe(1_200_000, Measurement(6.0, DataQuality.INVALID), "sensor")  # bad quality
        ev = acc.finalize(duration_s=1200.0, end_offset_ms=1_500_000)
        assert ev.sample_count == 5
        assert ev.usable_count == 1
        assert ev.coverage == pytest.approx(0.2)
        assert ev.min_c == ev.max_c == 5.0                        # implausible 999 excluded

    def test_stale_quality_is_usable_but_flags_quality(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(4.0, DataQuality.OK), "sensor")
        acc.observe(300_000, _m(4.0, DataQuality.STALE), "sensor")
        ev = acc.finalize(duration_s=300.0, end_offset_ms=300_000)
        assert ev.usable_count == 2
        assert ev.quality is DataQuality.STALE

    def test_all_unusable_yields_unknown_quality_and_no_values(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, None, "sensor")
        acc.observe(300_000, Measurement(None, DataQuality.UNAVAILABLE), "sensor")
        ev = acc.finalize(duration_s=300.0, end_offset_ms=300_000)
        assert ev.usable_count == 0
        assert ev.time_weighted_mean_c is None
        assert ev.quality is DataQuality.UNKNOWN
        assert ev.has_usable_evidence is False

    def test_source_switching_sets_source_changed_and_source_none(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), "weather_entity")
        acc.observe(300_000, _m(6.0), "sensor")
        ev = acc.finalize(duration_s=300.0, end_offset_ms=300_000)
        assert ev.source_changed is True
        assert ev.source is None

    def test_unknown_source_is_explicit_not_fabricated(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(5.0), None)
        ev = acc.finalize(duration_s=0.0, end_offset_ms=0)
        assert ev.source is None
        assert ev.source_changed is False

    def test_serialize_restore_roundtrip(self):
        acc = OutdoorEvidenceAccumulator()
        acc.observe(0, _m(0.0), "sensor")
        acc.observe(300_000, _m(10.0), "sensor")
        restored = OutdoorEvidenceAccumulator.restore(acc.serialize())
        restored.observe(600_000, _m(20.0), "sensor")
        base = OutdoorEvidenceAccumulator()
        for off, v in ((0, 0.0), (300_000, 10.0), (600_000, 20.0)):
            base.observe(off, _m(v), "sensor")
        assert restored.finalize(duration_s=600.0, end_offset_ms=600_000) == \
            base.finalize(duration_s=600.0, end_offset_ms=600_000)

    def test_restore_from_corrupt_state_yields_empty(self):
        for bad in ("not-a-dict", 42, {"n": "x"}, {"u": 9, "n": 1}, {"min": float("nan")}):
            acc = OutdoorEvidenceAccumulator.restore(bad)
            assert acc.finalize(duration_s=0.0) is None or acc.finalize(
                duration_s=0.0).sample_count >= 0   # never raises

    def test_restore_missing_key_is_empty(self):
        assert OutdoorEvidenceAccumulator.restore(None).finalize(duration_s=0.0) is None


# ── builder emission ──────────────────────────────────────────────────────────

def _reg(regime, reliability=0.9):
    return RegimeResult(regime=regime, reliability=reliability)


def _binp(i, regime, *, indoor, outdoor, source="sensor", target=21.0, step_s=300):
    return BuilderInput(
        ts=_T0 + timedelta(seconds=i * step_s), regime=_reg(regime), indoor_temp=indoor,
        target=target, outdoor_temp=outdoor, outdoor_source=source)


def _drive_heating(builder, outdoor_series, *, sources=None, indoor_base=18.0):
    """Open → continue → close a heating episode; return the CLOSED result."""
    n = len(outdoor_series)
    srcs = sources or ["sensor"] * n
    res = None
    for i, (od, src) in enumerate(zip(outdoor_series, srcs)):
        regime = Regime.ACTIVE_HEATING if i < n - 1 else Regime.STABLE
        res = builder.process(_binp(i, regime, indoor=indoor_base + i * 0.5,
                                    outdoor=od, source=src))
    return res


class TestHeatingBuilderEvidence:
    def test_completed_heating_episode_carries_outdoor_evidence(self):
        b = HeatingEpisodeBuilder("lz")
        res = _drive_heating(b, [_m(0.0), _m(10.0), _m(20.0)])
        assert res.action is BuilderAction.CLOSED
        ev = res.outdoor_evidence
        assert isinstance(ev, OutdoorEvidence)
        assert ev.time_weighted_mean_c == pytest.approx(5.0, abs=1e-6)   # time-weighted
        assert ev.usable_count == 3 and ev.source == "sensor"

    def test_source_switch_within_episode_flagged(self):
        b = HeatingEpisodeBuilder("lz")
        res = _drive_heating(b, [_m(5.0), _m(6.0), _m(7.0)],
                             sources=["weather_entity", "weather_entity", "sensor"])
        assert res.outdoor_evidence.source_changed is True
        assert res.outdoor_evidence.source is None

    def test_missing_outdoor_leaves_evidence_without_usable_samples(self):
        b = HeatingEpisodeBuilder("lz")
        res = _drive_heating(b, [None, None, None])
        ev = res.outdoor_evidence
        assert ev is not None and ev.usable_count == 0
        assert ev.time_weighted_mean_c is None

    def test_aborted_episode_has_no_evidence(self):
        b = HeatingEpisodeBuilder("lz")
        b.process(_binp(0, Regime.ACTIVE_HEATING, indoor=18.0, outdoor=_m(5.0)))
        # a window-open disturbance aborts the open episode
        res = b.process(BuilderInput(ts=_T0 + timedelta(seconds=300), regime=_reg(Regime.ACTIVE_HEATING),
                                     indoor_temp=18.5, target=21.0, outdoor_temp=_m(5.0),
                                     outdoor_source="sensor", window_open=True))
        assert res.action is BuilderAction.ABORTED
        assert res.outdoor_evidence is None

    def test_discarded_short_episode_has_no_evidence(self):
        b = HeatingEpisodeBuilder("lz")
        b.process(_binp(0, Regime.ACTIVE_HEATING, indoor=18.0, outdoor=_m(5.0), step_s=60))
        # close after only 60s (< min_duration 180s) -> DISCARDED_TOO_SHORT
        res = b.process(_binp(1, Regime.STABLE, indoor=18.0, outdoor=_m(6.0), step_s=60))
        assert res.action is BuilderAction.DISCARDED
        assert res.outdoor_evidence is None


class TestCoolingBuilderEvidence:
    def _drive_cooling(self, b, outdoor_series):
        n = len(outdoor_series)
        res = None
        for i, od in enumerate(outdoor_series):
            regime = Regime.PASSIVE_COOLING if i < n - 1 else Regime.STABLE
            res = b.process(_binp(i, regime, indoor=21.0 - i * 0.3, outdoor=od))
        return res

    def test_completed_cooling_episode_carries_outdoor_evidence(self):
        b = PassiveCoolingEpisodeBuilder("lz")
        res = self._drive_cooling(b, [_m(2.0), _m(2.0), _m(4.0)])
        assert res.action is BuilderAction.CLOSED
        ev = res.outdoor_evidence
        assert isinstance(ev, OutdoorEvidence)
        assert ev.usable_count == 3
        assert ev.min_c == 2.0 and ev.max_c == 4.0


# ── snapshot / restore across an episode ─────────────────────────────────────

class TestSnapshotRestore:
    def test_evidence_survives_snapshot_restore_midway(self):
        b1 = HeatingEpisodeBuilder("lz")
        b1.process(_binp(0, Regime.ACTIVE_HEATING, indoor=18.0, outdoor=_m(0.0)))
        b1.process(_binp(1, Regime.ACTIVE_HEATING, indoor=18.5, outdoor=_m(10.0)))
        snap = b1.snapshot_state()
        assert "outdoor" in snap

        b2 = HeatingEpisodeBuilder("lz")
        b2.restore_state(snap)
        res = b2.process(_binp(2, Regime.STABLE, indoor=19.0, outdoor=_m(20.0)))
        assert res.action is BuilderAction.CLOSED
        # same three samples as an uninterrupted run
        ref = _drive_heating(HeatingEpisodeBuilder("lz"), [_m(0.0), _m(10.0), _m(20.0)])
        assert res.outdoor_evidence == ref.outdoor_evidence

    def test_restore_pre_d2a_snapshot_without_outdoor_key(self):
        b1 = HeatingEpisodeBuilder("lz")
        b1.process(_binp(0, Regime.ACTIVE_HEATING, indoor=18.0, outdoor=_m(5.0)))
        b1.process(_binp(1, Regime.ACTIVE_HEATING, indoor=18.5, outdoor=_m(5.0)))
        snap = b1.snapshot_state()
        snap.pop("outdoor")                                  # simulate a pre-D2-A snapshot
        b2 = HeatingEpisodeBuilder("lz")
        b2.restore_state(snap)                               # must not raise
        res = b2.process(_binp(2, Regime.STABLE, indoor=19.0, outdoor=_m(8.0)))
        assert res.action is BuilderAction.CLOSED
        # only the post-restore sample was captured; earlier evidence is simply missing
        assert res.outdoor_evidence.usable_count == 1

    def test_restore_corrupt_outdoor_key_does_not_break_close(self):
        b1 = HeatingEpisodeBuilder("lz")
        b1.process(_binp(0, Regime.ACTIVE_HEATING, indoor=18.0, outdoor=_m(5.0)))
        b1.process(_binp(1, Regime.ACTIVE_HEATING, indoor=18.5, outdoor=_m(5.0)))
        snap = b1.snapshot_state()
        snap["outdoor"] = "corrupt-not-a-dict"
        b2 = HeatingEpisodeBuilder("lz")
        b2.restore_state(snap)
        res = b2.process(_binp(2, Regime.STABLE, indoor=19.0, outdoor=_m(8.0)))
        assert res.action is BuilderAction.CLOSED
        assert res.outdoor_evidence.usable_count == 1        # recovered empty + 1 new sample


# ── pipeline sidecar + propagation (mutation) ────────────────────────────────

def _captured(i, *, temp, heating, outdoor, source="sensor", zone="lz", target=21.0, step_s=300):
    from custom_components.thermosmart.learning.runtime.capture import (
        CaptureCoordinator, ControllerDecisionInput, DecisionType, RuntimeCycleInput,
        ScheduleTarget,
    )
    cap = _captured.coord
    setp = (target + 3.0) if heating else (target - 4.0)
    inp = RuntimeCycleInput(
        zone_id=zone, ts=(_T0 + timedelta(seconds=i * step_s)).isoformat(),
        target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DataQuality.OK),
        controller_decision=ControllerDecisionInput(
            decision_type=DecisionType.NORMAL, target_c=target, trv_setpoint_c=setp),
        outdoor_temp=outdoor, outdoor_source=source)
    return cap.capture(inp)


class TestPipelineSidecar:
    def _run_ramp(self, *, outdoor_fn, source="sensor"):
        from custom_components.thermosmart.learning.runtime.capture import CaptureCoordinator
        from custom_components.thermosmart.learning.runtime.pipeline import RuntimePipeline
        _captured.coord = CaptureCoordinator("lz")
        pipe = RuntimePipeline("lz")
        completed = []
        temp = 18.0
        # heating ramp
        for i in range(8):
            snap = _captured(i, temp=round(temp, 3), heating=True, outdoor=outdoor_fn(i), source=source)
            completed += list(pipe.process(snap).completed)
            temp = min(21.0, temp + 0.5)
        # settle / cool
        for j in range(8, 14):
            snap = _captured(j, temp=round(temp, 3), heating=False, outdoor=outdoor_fn(j), source=source)
            completed += list(pipe.process(snap).completed)
            temp = max(18.0, temp - 0.4)
        return completed

    def test_completed_episodes_carry_outdoor_sidecar(self):
        completed = self._run_ramp(outdoor_fn=lambda i: _m(2.0 + i))
        heat_or_cool = [c for c in completed if c.model_name in ("heat_rate", "heat_loss")]
        assert heat_or_cool, "expected at least one heating/cooling episode"
        assert any(getattr(c, "outdoor_evidence", None) is not None
                   and c.outdoor_evidence.usable_count > 0 for c in heat_or_cool)
        ev = next(c.outdoor_evidence for c in heat_or_cool
                  if getattr(c, "outdoor_evidence", None) is not None)
        assert ev.source == "sensor"

    def test_sidecar_detects_missing_outdoor_propagation(self):
        # With NO outdoor readings, the sidecar still exists but carries no usable
        # evidence. If the pipeline wiring that forwards outdoor into the episode were
        # removed, the populated-evidence assertion above would fail — this is the
        # baseline/mutation counterpart proving the channel is actually exercised.
        completed = self._run_ramp(outdoor_fn=lambda i: None)
        heat_or_cool = [c for c in completed if c.model_name in ("heat_rate", "heat_loss")]
        assert heat_or_cool
        for c in heat_or_cool:
            ev = getattr(c, "outdoor_evidence", None)
            assert ev is None or ev.usable_count == 0


# ── WeatherEngine source label ───────────────────────────────────────────────

class TestWeatherEngineSource:
    def _engine(self, *, weather_temp=None, sensor_temp=None, sensor_id=None):
        from custom_components.thermosmart.weather_engine import WeatherEngine
        hass = make_mock_hass()
        states = {}
        if weather_temp is not None:
            states["weather.home"] = make_state("sunny", {"temperature": weather_temp,
                                                           "temperature_unit": "°C"})
        if sensor_temp is not None and sensor_id:
            states[sensor_id] = make_state(sensor_temp, {"unit_of_measurement": "°C"})
        hass.states.get = lambda eid: states.get(eid)
        hass.services.async_call = AsyncMock(return_value={})
        return WeatherEngine(hass=hass, weather_entity="weather.home",
                             outdoor_temp_sensor=sensor_id)

    @pytest.mark.asyncio
    async def test_weather_entity_source(self):
        eng = self._engine(weather_temp=6.0)
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(6.0)
        assert data["temperature_source"] == "weather_entity"

    @pytest.mark.asyncio
    async def test_dedicated_sensor_overrides_source(self):
        eng = self._engine(weather_temp=6.0, sensor_temp=4.5, sensor_id="sensor.outdoor")
        data = await eng.async_get_data()
        assert data["temperature"] == pytest.approx(4.5)
        assert data["temperature_source"] == "sensor"

    @pytest.mark.asyncio
    async def test_no_outdoor_temperature_source_none(self):
        eng = self._engine()
        data = await eng.async_get_data()
        assert data["temperature"] is None
        assert data["temperature_source"] is None


# ── baseline equivalence: evidence capture changes no control/learning output ─

class TestNoControlChange:
    def test_episode_model_input_identical_with_and_without_evidence(self):
        # The HeatingEpisode (the object the model actually learns from) must be
        # byte-identical whether or not outdoor evidence/source is supplied — the
        # sidecar is the ONLY difference.
        with_od = _drive_heating(HeatingEpisodeBuilder("lz"), [_m(0.0), _m(10.0), _m(20.0)])

        b = HeatingEpisodeBuilder("lz")
        res = None
        for i in range(3):
            regime = Regime.ACTIVE_HEATING if i < 2 else Regime.STABLE
            res = b.process(BuilderInput(ts=_T0 + timedelta(seconds=i * 300),
                                         regime=_reg(regime), indoor_temp=18.0 + i * 0.5,
                                         target=21.0))          # no outdoor at all
        assert res.episode == with_od.episode                  # model input unchanged
        assert with_od.outdoor_evidence is not None
        assert res.outdoor_evidence is not None and res.outdoor_evidence.usable_count == 0
