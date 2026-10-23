"""The unused manual-correction learning model is gone; user overrides are not.

Stored learning snapshots written by earlier versions still carry a
"manual_correction" model entry. Restoring them must keep working (the entry is
ignored and dropped on the next save), and the user's manual temperature
override must keep controlling the zone.
"""
from __future__ import annotations

from custom_components.thermosmart.learning.runtime.orchestration import (
    ModelOrchestrator,
    build_zone_models,
)
from tests.helpers_ha_runtime import make_recording_coordinator


class TestStoredSnapshotsFromEarlierVersions:
    def test_zone_models_no_longer_include_the_manual_correction_model(self):
        assert "manual_correction" not in build_zone_models("lz_1")

    def test_stale_manual_correction_state_is_ignored_on_restore(self):
        orchestrator = ModelOrchestrator("lz_1")
        errors = orchestrator.restore_models({
            "manual_correction": {"schema_version": 1, "samples": [], "anything": "else"},
            "heat_rate": orchestrator.serialize_models()["heat_rate"],
        })
        assert errors == ()

    def test_next_save_drops_the_stale_entry(self):
        orchestrator = ModelOrchestrator("lz_1")
        orchestrator.restore_models({"manual_correction": {"x": 1}})
        assert "manual_correction" not in orchestrator.serialize_models()


class TestManualTemperatureOverrideStillControls:
    async def test_override_sets_the_target_and_the_control_reason(self):
        coord = make_recording_coordinator({"schedule_enabled": False}, indoor="19.0")
        coord.set_active_control(True)

        baseline = (await coord._async_update_data())["zone"]
        assert baseline["effective_target"] == 21.0

        coord.set_override(23.0)
        zone = (await coord._async_update_data())["zone"]

        assert zone["effective_target"] == 23.0
        assert zone["override_active"] is True
        assert coord._control_reason(zone) == "manual_override"
