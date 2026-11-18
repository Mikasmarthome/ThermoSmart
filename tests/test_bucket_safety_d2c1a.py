"""Phase D2-C1a — persisted conditioned-bucket safety & outdoor-freshness observability.

Covers:
  * per-bucket validation + fail-closed quarantine on restore (HeatRate & HeatLoss),
    with valid general/normalized learning and sibling buckets preserved;
  * the shared bucket validator (base.validate_bucket_dict / validate_bucket_key);
  * restore observability — whole-model failure vs individual bucket quarantine;
  * the defensive prediction-rate plausibility guard (no out-of-range/non-finite
    conditioned rate can reach an active prediction);
  * outdoor-freshness observability — known/unknown distinction, HA state timestamp
    vs verified measurement, source switching and unavailable sensors, with no
    effect on control.

All strictly D2-C1a: NO conditioned prediction activation; the general prediction
path and published predictions are unchanged.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.learning.models import base as mb
from custom_components.thermosmart.learning.models.heat_rate import (
    HeatRateModel, HeatRateBucket, HeatRatePredictionContext, HeatRateParameters,
)
from custom_components.thermosmart.learning.models.heat_loss import (
    HeatLossModel, HeatLossBucket, HeatLossPredictionContext, HeatLossParameters,
)
from custom_components.thermosmart.learning.runtime.orchestration import ModelOrchestrator
from custom_components.thermosmart.learning.runtime.outdoor_freshness import (
    OutdoorFreshness, classify_outdoor_freshness,
)
from custom_components.thermosmart.learning.runtime.ha_integration import (
    build_runtime_cycle_input,
)
from custom_components.thermosmart.learning.contracts import DataQuality, Measurement


# ── fixtures / builders ──────────────────────────────────────────────────────

def _bucket(key="outdoor:3", rate=5.0, n=6.0, disp=0.1, sc=6, lut=None):
    return {"key": key, "rate_c_per_h": rate, "effective_n": n,
            "dispersion": disp, "sample_count": sc, "last_update_ts": lut}


def _hr_state(zone="lz", buckets=None, general_rate=4.0, general_sc=8):
    return {
        "model_version": 1, "parameter_version": 1, "learning_zone_id": zone,
        "general": {"key": "general", "rate_c_per_h": general_rate, "effective_n": 10.0,
                    "dispersion": 0.2, "sample_count": general_sc, "last_update_ts": None},
        "buckets": dict(buckets or {}),
        "processed_ids": [], "aggregate_reliability": 0.8, "last_update_ts": None,
        "rejection_counts": {}, "outlier_counts": {}, "dedup_count": 0, "recent_samples": [],
    }


def _hl_state(zone="lz", buckets=None, general_rate=0.5, general_sc=8):
    return {
        "model_version": 1, "parameter_version": 1, "learning_zone_id": zone,
        "general": {"key": "general", "rate_c_per_h": general_rate, "effective_n": 10.0,
                    "dispersion": 0.05, "sample_count": general_sc, "last_update_ts": None},
        "normalized": {"key": "normalized", "rate_c_per_h": 0.01, "effective_n": 3.0,
                       "dispersion": 0.001, "sample_count": 3, "last_update_ts": None},
        "buckets": dict(buckets or {}),
        "processed_ids": [], "aggregate_reliability": 0.8, "last_update_ts": None,
        "rejection_counts": {}, "outlier_counts": {}, "dedup_count": 0, "recent_samples": [],
    }


# ── shared validator (base) ────────────────────────────────────────────────────

class TestSharedValidator:
    def test_valid_bucket_accepts(self):
        assert mb.validate_bucket_dict("outdoor:3", _bucket(), max_band=6, max_rate=12.0) is None

    def test_valid_profile_key_accepts(self):
        b = _bucket(key="outdoor:3|profile:boiler")
        assert mb.validate_bucket_dict("outdoor:3|profile:boiler", b, max_band=6,
                                       max_rate=12.0) is None

    @pytest.mark.parametrize("key", [
        "", "outdoor:", "outdoor:x", "outdoor:-1", "outdoor:1.0", "general",
        "outdoor:7",  # band > max_band (len edges == 6 -> bands 0..6, 7 invalid)
        "outdoor:3|", "outdoor:3|profile:", "outdoor:3|foo:bar", "outdoor:3|a|b",
    ])
    def test_bad_keys_rejected(self, key):
        assert mb.validate_bucket_key(key, max_band=6) is False

    def test_band_zero_and_max_accepted(self):
        assert mb.validate_bucket_key("outdoor:0", max_band=6)
        assert mb.validate_bucket_key("outdoor:6", max_band=6)

    def test_key_mismatch_rejected(self):
        assert mb.validate_bucket_dict("outdoor:3", _bucket(key="outdoor:2"),
                                       max_band=6, max_rate=12.0) == "key_mismatch"

    @pytest.mark.parametrize("rate,reason", [
        (float("nan"), "non_finite_rate"), (float("inf"), "non_finite_rate"),
        (float("-inf"), "non_finite_rate"), (True, "non_finite_rate"),
        (-0.5, "rate_out_of_range"), (99.0, "rate_out_of_range"),
    ])
    def test_bad_rate_rejected(self, rate, reason):
        assert mb.validate_bucket_dict("outdoor:3", _bucket(rate=rate),
                                       max_band=6, max_rate=12.0) == reason

    def test_zero_rate_accepted(self):
        # 0.0 is a legitimate empty/seed value; must not be rejected by the band floor.
        assert mb.validate_bucket_dict("outdoor:3", _bucket(rate=0.0),
                                       max_band=6, max_rate=12.0) is None

    @pytest.mark.parametrize("sc", [-1, 1.5, True, "5", None])
    def test_bad_sample_count_rejected(self, sc):
        assert mb.validate_bucket_dict("outdoor:3", _bucket(sc=sc),
                                       max_band=6, max_rate=12.0) == "bad_sample_count"

    @pytest.mark.parametrize("n", [-1.0, float("nan"), float("inf"), True])
    def test_bad_effective_n_rejected(self, n):
        assert mb.validate_bucket_dict("outdoor:3", _bucket(n=n),
                                       max_band=6, max_rate=12.0) == "bad_effective_n"

    @pytest.mark.parametrize("disp", [-0.1, float("nan"), float("inf")])
    def test_bad_dispersion_rejected(self, disp):
        assert mb.validate_bucket_dict("outdoor:3", _bucket(disp=disp),
                                       max_band=6, max_rate=12.0) == "bad_dispersion"

    @pytest.mark.parametrize("field", ["key", "rate_c_per_h", "effective_n",
                                       "dispersion", "sample_count"])
    def test_missing_required_field_rejected(self, field):
        b = _bucket()
        del b[field]
        assert mb.validate_bucket_dict("outdoor:3", b, max_band=6, max_rate=12.0) \
            == "missing_field"

    def test_non_mapping_rejected(self):
        assert mb.validate_bucket_dict("outdoor:3", ["not", "a", "dict"],
                                       max_band=6, max_rate=12.0) == "not_a_mapping"

    def test_bad_last_update_ts_rejected(self):
        assert mb.validate_bucket_dict("outdoor:3", _bucket(lut=12345),
                                       max_band=6, max_rate=12.0) == "bad_last_update_ts"

    def test_none_last_update_ts_accepted(self):
        assert mb.validate_bucket_dict("outdoor:3", _bucket(lut=None),
                                       max_band=6, max_rate=12.0) is None


# ── N1: overflow-safe quarantine (huge integers) ────────────────────────────────

_HUGE = 10 ** 400   # not representable as a finite float; math.isfinite raises on it


class TestHugeIntegerQuarantine:
    @pytest.mark.parametrize("kw,reason", [
        ("rate", "non_finite_rate"),
        ("n", "bad_effective_n"),
        ("disp", "bad_dispersion"),
    ])
    def test_validator_does_not_raise_on_huge_int(self, kw, reason):
        b = _bucket(**{kw: _HUGE})
        # must return a reason, never raise OverflowError
        assert mb.validate_bucket_dict("outdoor:3", b, max_band=6, max_rate=12.0) == reason

    def test_quarantine_buckets_never_raises(self):
        raw = {"outdoor:3": _bucket(rate=_HUGE), "outdoor:1": _bucket(key="outdoor:1")}
        loaded, q = mb.quarantine_buckets(
            raw, lambda d: d, max_band=6, max_rate=12.0)
        assert set(loaded) == {"outdoor:1"} and q == {"outdoor:3": "non_finite_rate"}

    @pytest.mark.parametrize("kw,reason", [
        ("rate", "non_finite_rate"),
        ("n", "bad_effective_n"),
        ("disp", "bad_dispersion"),
    ])
    def test_heat_rate_huge_int_quarantined_general_preserved(self, kw, reason):
        state = _hr_state(buckets={"outdoor:3": _bucket(**{kw: _HUGE})})
        m = HeatRateModel("lz")
        m.deserialize_state(copy.deepcopy(state))          # must NOT raise OverflowError
        assert m.diagnostics().general_rate_c_per_h == 4.0
        assert not m._state.buckets
        assert m.restore_quarantine_summary() == {reason: 1}

    @pytest.mark.parametrize("kw,reason", [
        ("rate", "non_finite_rate"),
        ("n", "bad_effective_n"),
        ("disp", "bad_dispersion"),
    ])
    def test_heat_loss_huge_int_quarantined_general_and_normalized_preserved(self, kw, reason):
        state = _hl_state(buckets={"outdoor:2": _bucket(key="outdoor:2", **{kw: _HUGE})})
        m = HeatLossModel("lz")
        m.deserialize_state(copy.deepcopy(state))          # must NOT raise OverflowError
        d = m.diagnostics()
        assert d.general_loss_c_per_h == 0.5
        assert d.normalized_coefficient_per_h is not None  # normalized preserved
        assert not m._state.buckets
        assert m.restore_quarantine_summary() == {reason: 1}

    def test_huge_int_bucket_cannot_escape_via_orchestrator(self):
        orch = ModelOrchestrator("lz")
        errs = orch.restore_models({
            "heat_rate": _hr_state(buckets={"outdoor:3": _bucket(rate=_HUGE)}),
            "heat_loss": _hl_state(buckets={"outdoor:2": _bucket(key="outdoor:2",
                                                                 n=_HUGE)}),
        })
        # quarantine entries, NOT whole-model restore failures
        assert any(e.startswith("heat_rate:quarantine:non_finite_rate") for e in errs)
        assert any(e.startswith("heat_loss:quarantine:bad_effective_n") for e in errs)
        assert not any(":restore:" in e for e in errs)
        assert orch.models["heat_rate"].diagnostics().general_rate_c_per_h == 4.0
        assert orch.models["heat_loss"].diagnostics().general_loss_c_per_h == 0.5


# ── HeatRate quarantine ─────────────────────────────────────────────────────────

class TestHeatRateQuarantine:
    def test_corrupt_bucket_dropped_general_preserved(self):
        state = _hr_state(buckets={"outdoor:3": _bucket(rate=float("nan"))})
        m = HeatRateModel("lz")
        m.deserialize_state(copy.deepcopy(state))              # must NOT raise
        d = m.diagnostics()
        assert d.general_rate_c_per_h == 4.0 and d.sample_counts["general"] == 8
        assert not [k for k in d.bucket_rates if k.startswith("outdoor:")]
        assert d.quarantined_bucket_count == 1
        assert m.restore_quarantine_summary() == {"non_finite_rate": 1}

    def test_valid_bucket_survives_corrupt_sibling(self):
        state = _hr_state(buckets={
            "outdoor:3": _bucket(key="outdoor:3", rate=5.0),        # valid
            "outdoor:5": _bucket(key="outdoor:5", rate=float("inf")),  # corrupt
            "outdoor:1": _bucket(key="outdoor:1", sc=-3),           # corrupt
        })
        m = HeatRateModel("lz")
        m.deserialize_state(copy.deepcopy(state))
        d = m.diagnostics()
        assert set(k for k in d.bucket_rates if k.startswith("outdoor:")) == {"outdoor:3"}
        assert d.quarantined_bucket_count == 2
        assert d.quarantined_bucket_reasons == {"non_finite_rate": 1, "bad_sample_count": 1}

    def test_out_of_range_but_finite_rate_quarantined(self):
        state = _hr_state(buckets={"outdoor:3": _bucket(rate=500.0)})   # finite, implausible
        m = HeatRateModel("lz")
        m.deserialize_state(copy.deepcopy(state))
        assert m.restore_quarantine_summary() == {"rate_out_of_range": 1}
        assert not m._state.buckets

    def test_invalid_key_quarantined(self):
        state = _hr_state(buckets={"outdoor:99": _bucket(key="outdoor:99")})
        m = HeatRateModel("lz")
        m.deserialize_state(copy.deepcopy(state))
        assert m.restore_quarantine_summary() == {"bad_key": 1}

    def test_whole_model_failure_on_corrupt_general(self):
        state = _hr_state()
        state["general"]["rate_c_per_h"] = float("nan")            # model-level invariant
        m = HeatRateModel("lz")
        with pytest.raises(ValueError):
            m.deserialize_state(state)

    def test_valid_state_round_trip_no_quarantine(self):
        state = _hr_state(buckets={"outdoor:3": _bucket()})
        m = HeatRateModel("lz")
        m.deserialize_state(copy.deepcopy(state))
        assert m.restore_quarantine_summary() == {}
        # re-serialize equals the input buckets (no learning mutation on restore)
        again = m.serialize_state()
        assert again["buckets"]["outdoor:3"]["rate_c_per_h"] == 5.0

    def test_legacy_state_without_buckets(self):
        state = _hr_state()
        state.pop("buckets")
        m = HeatRateModel("lz")
        m.deserialize_state(state)                                 # must not raise
        assert m.diagnostics().bucket_rates == {} and m.restore_quarantine_summary() == {}


# ── HeatLoss quarantine ──────────────────────────────────────────────────────────

class TestHeatLossQuarantine:
    def test_corrupt_bucket_dropped_general_and_normalized_preserved(self):
        state = _hl_state(buckets={"outdoor:2": _bucket(key="outdoor:2", rate=float("nan"))})
        m = HeatLossModel("lz")
        m.deserialize_state(copy.deepcopy(state))
        d = m.diagnostics()
        assert d.general_loss_c_per_h == 0.5 and d.sample_counts["general"] == 8
        assert d.normalized_coefficient_per_h is not None       # normalized preserved
        assert not [k for k in d.bucket_rates if k.startswith("outdoor:")]
        assert d.quarantined_bucket_count == 1

    def test_negative_loss_rate_quarantined(self):
        state = _hl_state(buckets={"outdoor:2": _bucket(key="outdoor:2", rate=-0.3)})
        m = HeatLossModel("lz")
        m.deserialize_state(copy.deepcopy(state))
        assert m.restore_quarantine_summary() == {"rate_out_of_range": 1}

    def test_heat_loss_max_rate_band(self):
        # 6.5 exceeds heat_loss max_loss_rate (6.0) but would pass heat_rate's 12.0 band —
        # proves the per-model limit is used.
        state = _hl_state(buckets={"outdoor:2": _bucket(key="outdoor:2", rate=6.5)})
        m = HeatLossModel("lz")
        m.deserialize_state(copy.deepcopy(state))
        assert m.restore_quarantine_summary() == {"rate_out_of_range": 1}

    def test_valid_bucket_survives_corrupt_sibling(self):
        state = _hl_state(buckets={
            "outdoor:2": _bucket(key="outdoor:2", rate=0.6),
            "outdoor:4": _bucket(key="outdoor:4", rate=float("inf")),
        })
        m = HeatLossModel("lz")
        m.deserialize_state(copy.deepcopy(state))
        assert set(k for k in m._state.buckets) == {"outdoor:2"}
        assert m.restore_quarantine_summary() == {"non_finite_rate": 1}


# ── restore observability (orchestrator) ────────────────────────────────────────

class TestRestoreObservability:
    def test_bucket_quarantine_distinct_from_model_failure(self):
        orch = ModelOrchestrator("lz")
        errs = orch.restore_models({"heat_rate": _hr_state(
            buckets={"outdoor:3": _bucket(rate=float("nan"))})})
        assert any(e.startswith("heat_rate:quarantine:non_finite_rate:1") for e in errs)
        assert not any(":restore:" in e for e in errs)
        # the model is otherwise healthy — general preserved
        assert orch.models["heat_rate"].diagnostics().general_rate_c_per_h == 4.0

    def test_whole_model_failure_reported_as_restore(self):
        orch = ModelOrchestrator("lz")
        bad = _hr_state()
        bad["general"]["rate_c_per_h"] = float("nan")
        errs = orch.restore_models({"heat_rate": bad})
        assert any(e.startswith("heat_rate:restore:ValueError") for e in errs)
        assert not any(":quarantine:" in e for e in errs)

    def test_clean_restore_no_entries(self):
        orch = ModelOrchestrator("lz")
        errs = orch.restore_models({"heat_rate": _hr_state(buckets={"outdoor:3": _bucket()})})
        assert errs == ()


# ── prediction-rate plausibility guard ─────────────────────────────────────────

class TestPredictionRateGuard:
    def test_heat_rate_out_of_range_falls_back(self):
        m = HeatRateModel("lz")
        m._state = replace(m._state, general=HeatRateBucket(
            key="general", rate_c_per_h=999.0, effective_n=10.0, dispersion=0.1, sample_count=8))
        pred = m.predict(HeatRatePredictionContext())
        assert pred.fallback_used is True
        assert pred.values["heat_rate"] == HeatRateParameters().generic_prior_rate_c_per_h

    def test_heat_rate_negative_falls_back(self):
        m = HeatRateModel("lz")
        m._state = replace(m._state, general=HeatRateBucket(
            key="general", rate_c_per_h=-3.0, effective_n=10.0, dispersion=0.1, sample_count=8))
        pred = m.predict(HeatRatePredictionContext())
        assert pred.fallback_used is True
        assert pred.values["heat_rate"] == HeatRateParameters().generic_prior_rate_c_per_h

    def test_heat_loss_negative_falls_back(self):
        m = HeatLossModel("lz")
        m._state = replace(m._state, general=HeatLossBucket(
            key="general", rate_c_per_h=-1.0, effective_n=10.0, dispersion=0.1, sample_count=8))
        pred = m.predict(HeatLossPredictionContext())
        assert pred.fallback_used is True
        assert pred.values["heat_loss_rate"] == HeatLossParameters().generic_prior_loss_c_per_h

    def test_valid_rate_unchanged(self):
        # a normal learned rate must pass through the guard untouched (no behaviour change)
        m = HeatRateModel("lz")
        m._state = replace(m._state, general=HeatRateBucket(
            key="general", rate_c_per_h=4.2, effective_n=10.0, dispersion=0.1, sample_count=8))
        pred = m.predict(HeatRatePredictionContext())
        assert pred.fallback_used is False and pred.values["heat_rate"] == 4.2


# ── outdoor-freshness observability ─────────────────────────────────────────────

class TestFreshnessClassifier:
    def test_no_source_is_unknown(self):
        r = classify_outdoor_freshness(source=None, ha_state_last_updated="2026-01-01T00:00:00")
        assert r.freshness is OutdoorFreshness.UNKNOWN and not r.is_verified_fresh

    def test_source_without_timestamp_is_unknown(self):
        r = classify_outdoor_freshness(source="sensor", ha_state_last_updated=None)
        assert r.freshness is OutdoorFreshness.UNKNOWN and not r.is_verified_fresh

    def test_ha_state_timestamp_is_not_verified_fresh(self):
        r = classify_outdoor_freshness(source="sensor",
                                       ha_state_last_updated="2026-01-01T00:00:00")
        assert r.freshness is OutdoorFreshness.HA_STATE_TIMESTAMP
        assert r.is_verified_fresh is False            # HA state ts never proves freshness
        assert r.ha_state_last_updated == "2026-01-01T00:00:00"
        assert r.measurement_timestamp is None

    def test_verified_measurement_is_fresh(self):
        r = classify_outdoor_freshness(source="sensor", ha_state_last_updated="x",
                                       measurement_timestamp="2026-01-01T00:00:00")
        assert r.freshness is OutdoorFreshness.VERIFIED_MEASUREMENT and r.is_verified_fresh

    def test_source_switch_reflected_not_inferred(self):
        a = classify_outdoor_freshness("weather_entity", "2026-01-01T00:00:00")
        b = classify_outdoor_freshness("sensor", "2026-01-01T00:05:00")
        assert a.source == "weather_entity" and b.source == "sensor"
        # neither claims verified freshness from an HA state timestamp
        assert not a.is_verified_fresh and not b.is_verified_fresh

    def test_ok_quality_does_not_imply_fresh(self):
        # No measurement timestamp exists even when the reading is DataQuality.OK.
        r = classify_outdoor_freshness(source="sensor", ha_state_last_updated=None)
        assert not r.is_verified_fresh


class TestFreshnessThreading:
    def test_build_cycle_input_carries_last_updated(self):
        inp = build_runtime_cycle_input(
            "lz", {"current_temp": 20.0},
            weather={"temperature": 2.0, "temperature_source": "sensor",
                     "temperature_last_updated": "2026-01-01T00:00:00"},
            ts_iso="2026-01-01T00:00:00+00:00")
        assert inp.outdoor_source == "sensor"
        assert inp.outdoor_last_updated == "2026-01-01T00:00:00"
        # the Measurement stays OK (control/regime path unchanged)
        assert inp.outdoor_temp.quality is DataQuality.OK

    def test_build_cycle_input_missing_timestamp_is_none(self):
        inp = build_runtime_cycle_input(
            "lz", {"current_temp": 20.0},
            weather={"temperature": 2.0, "temperature_source": "sensor"},
            ts_iso="2026-01-01T00:00:00+00:00")
        assert inp.outdoor_last_updated is None

    def test_capture_snapshot_carries_last_updated(self):
        from custom_components.thermosmart.learning.runtime.capture import CaptureCoordinator
        inp = build_runtime_cycle_input(
            "lz", {"current_temp": 20.0},
            weather={"temperature": 2.0, "temperature_source": "sensor",
                     "temperature_last_updated": "2026-01-01T00:00:00"},
            ts_iso="2026-01-01T00:00:00+00:00")
        snap = CaptureCoordinator("lz").capture(inp)
        assert snap.outdoor_last_updated == "2026-01-01T00:00:00"
        assert snap.outdoor_source == "sensor"


class _FakeState:
    def __init__(self, state, attributes=None, last_updated=None):
        self.state = state
        self.attributes = attributes or {}

        class _TS:
            def __init__(self, s): self._s = s
            def isoformat(self): return self._s
        self.last_updated = _TS(last_updated) if last_updated else None


class _FakeStates:
    def __init__(self, mapping): self._m = mapping
    def get(self, eid): return self._m.get(eid)


class _FakeServices:
    async def async_call(self, *a, **k):
        raise RuntimeError("no forecast service")     # caught by async_get_data


class _FakeHass:
    def __init__(self, mapping):
        self.states = _FakeStates(mapping)
        self.services = _FakeServices()
        self.config = MagicMock()


class TestWeatherEngineFreshnessCapture:
    @pytest.mark.asyncio
    async def test_weather_entity_timestamp_captured(self):
        from custom_components.thermosmart.weather_engine import WeatherEngine
        hass = _FakeHass({"weather.home": _FakeState(
            "sunny", {"temperature": 3.0, "temperature_unit": "°C"},
            last_updated="2026-01-01T06:00:00+00:00")})
        eng = WeatherEngine(hass=hass, weather_entity="weather.home")
        data = await eng.async_get_data()
        assert data["temperature"] is not None
        assert data["temperature_source"] == "weather_entity"
        assert data["temperature_last_updated"] == "2026-01-01T06:00:00+00:00"

    @pytest.mark.asyncio
    async def test_dedicated_sensor_timestamp_wins_over_weather_entity(self):
        # N2: valid weather entity AND a valid dedicated sensor with DIFFERENT timestamps.
        # The dedicated sensor takes precedence for both the value and the timestamp.
        from custom_components.thermosmart.weather_engine import WeatherEngine
        weather_ts = "2026-01-01T06:00:00+00:00"
        sensor_ts = "2026-01-01T06:04:30+00:00"
        hass = _FakeHass({
            "weather.home": _FakeState(
                "sunny", {"temperature": 3.0, "temperature_unit": "°C"},
                last_updated=weather_ts),
            "sensor.outdoor": _FakeState(
                "1.5", {"unit_of_measurement": "°C"}, last_updated=sensor_ts),
        })
        eng = WeatherEngine(hass=hass, weather_entity="weather.home",
                            outdoor_temp_sensor="sensor.outdoor")
        data = await eng.async_get_data()
        # existing precedence: the dedicated sensor wins the VALUE (unchanged behaviour)
        assert data["temperature"] == pytest.approx(1.5)
        assert data["temperature_source"] == "sensor"
        # the timestamp belongs to the SELECTED (sensor) source, not the weather entity
        assert data["temperature_last_updated"] == sensor_ts
        assert data["temperature_last_updated"] != weather_ts

    @pytest.mark.asyncio
    async def test_unavailable_sensor_yields_none(self):
        from custom_components.thermosmart.weather_engine import WeatherEngine
        hass = _FakeHass({"weather.home": _FakeState("unavailable")})
        eng = WeatherEngine(hass=hass, weather_entity="weather.home")
        data = await eng.async_get_data()
        assert data["temperature"] is None
        assert data["temperature_source"] is None
        assert data["temperature_last_updated"] is None
        # classifier on this output -> UNKNOWN
        r = classify_outdoor_freshness(data["temperature_source"],
                                       data["temperature_last_updated"])
        assert r.freshness is OutdoorFreshness.UNKNOWN
