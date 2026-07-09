"""Phase 16 Learning Export tests (models populated so research samples exist)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.export import (
    LEARNING_EXPORT_NAME,
    build_learning_export,
    to_canonical_json,
    validate_export_payload,
)
from custom_components.thermosmart.learning.privacy import scan_payload
from tests.helpers_orchestration import export_input, heat_rate_confidence, zone_input

from tests.helpers_outcome import reached_clean
from tests.helpers_forecast import decision, evaluation
from tests.helpers_boost import boost_context, boost_context_with_comparison, boost_episode
from tests.helpers_manual_correction import correction_context, correction_event
from custom_components.thermosmart.learning.models import (
    BoostModel,
    ForecastModel,
    ManualCorrectionModel,
    OutcomeModel,
)


def populated_models(zone="living_room"):
    outcome = OutcomeModel(zone)
    for i in range(4):
        outcome.update(reached_clean(f"o{i}", zone=zone))
    forecast = ForecastModel(zone)
    for i in range(4):
        d = decision(f"d{i}", 10.0, created_offset_h=i * 24, zone=zone)
        forecast.register_decision(d)
        forecast.update(evaluation(f"e{i}", f"d{i}", 10.5, created_offset_h=i * 24, zone=zone))
    boost = BoostModel(zone)
    for i in range(4):
        ep = boost_episode(f"b{i}", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id=f"bd{i}",
                           zone=zone)
        boost.update(ep, boost_context_with_comparison(ep))
    manual = ManualCorrectionModel(zone)
    for d in range(5):
        ev = correction_event(f"m{d}", 21.0, 21.3, day=d, zone=zone)
        manual.update(ev, correction_context(ev))
    return {"outcome": outcome, "forecast": forecast, "boost": boost,
            "manual_correction": manual}


def _zone():
    return zone_input(models=populated_models())


class TestLearningExport:
    def test_metadata_name(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        assert r.payload["metadata"]["scope_name"] == LEARNING_EXPORT_NAME
        assert r.payload["metadata"]["scope"] == "research"

    def test_generated_at_day_bucketed(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        # learning export exposes only the day, not the wall-clock time
        assert r.payload["metadata"]["generated_at"] == "2026-06-22"

    def test_outcome_dimensions(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        samples = r.payload["zones"][0]["models"]["outcome"]["samples"]
        assert samples and {"physical", "comfort", "controller", "data_quality"} <= set(samples[0])

    def test_forecast_error_bias(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        samples = r.payload["zones"][0]["models"]["forecast"]["samples"]
        assert samples and "absolute_error_c" in samples[0] and "horizon_bucket" in samples[0]

    def test_boost_effect(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        samples = r.payload["zones"][0]["models"]["boost"]["samples"]
        assert samples and "overshoot_c" in samples[0] and "attribution_reliability" in samples[0]

    def test_manual_correction_intent(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        samples = r.payload["zones"][0]["models"]["manual_correction"]["samples"]
        assert samples and "intent" in samples[0] and "direction" in samples[0]

    def test_no_ids_in_samples(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        blob = to_canonical_json(r.payload)
        for bad in ("decision_id", "episode_id", "event_id", "source_episode_id",
                    "correction_event_id", "living_room"):
            assert bad not in blob

    def test_no_raw_trajectories(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        assert "trajectory" not in to_canonical_json(r.payload)

    def test_quantization(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        # forecast error quantised to 0.1 grid -> at most 1 decimal place of resolution
        for s in r.payload["zones"][0]["models"]["forecast"]["samples"]:
            err = s["absolute_error_c"]
            assert abs(round(err / 0.1) * 0.1 - err) < 1e-9

    def test_budget_truncates_samples(self):
        from custom_components.thermosmart.learning.export import ExportBudget
        model_outcome = populated_models()
        z = zone_input(models=model_outcome)
        r = build_learning_export(export_input(zones=(z,),
                                               budget=ExportBudget(max_samples_per_model=2)))
        assert all(len(m.get("samples", [])) <= 2
                   for m in r.payload["zones"][0]["models"].values() if "samples" in m)

    def test_no_privacy_violations(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        assert scan_payload(r.payload) == []

    def test_validates(self):
        r = build_learning_export(export_input(zones=(_zone(),)))
        assert validate_export_payload(r.payload) == []

    def test_deterministic(self):
        a = to_canonical_json(build_learning_export(export_input(zones=(_zone(),))).payload)
        b = to_canonical_json(build_learning_export(export_input(zones=(_zone(),))).payload)
        assert a == b
