"""Support-/Research-Export Audit — P1 (historical_learning_snapshot Deep
Research reshaping) and P2 (support export reserved-block cleanup) regression
tests.

Covers:
  - P1: _historical_learning_snapshot_for_research() keeps fachlich
    useful technical fields (ts/weekday/hour/minute, target/delta/heat_rate/
    outcome_score/...) while never leaking entity_id/device_id/person/
    location, runs scan_payload() as a final defense, caps per category with
    records_truncated, and never crashes on missing/corrupt input.
  - P1 integration: async_export_learning_data() end-to-end produces the new
    structure and the full serialized export contains no forbidden identifiers.
  - P2: async_export_support_data() drops adaptation_application/
    orchestration_preview in favor of one reserved_diagnostics placeholder,
    while runtime_pending/adaptation/adaptation_history/critical_events/
    device_profile/runtime_state/runtime_health remain untouched, and
    TRV_COMMAND_FAILED stays visible via critical_events.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart.const import DOMAIN
from custom_components.thermosmart.export import (
    _HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY,
    _historical_learning_snapshot_for_research,
    _learning_reserved_diagnostics_summary,
    async_export_learning_data,
    async_export_support_data,
)
from custom_components.thermosmart.learning.privacy import scan_payload
from custom_components.thermosmart.learning.support_event_schemas import (
    SupportCriticalEvent,
    SupportEventSeverity,
    SupportEventType,
)

_FORBIDDEN_SUBSTRINGS = (
    # "presence" itself is excluded: has_presence is a legitimate boolean
    # feature flag (module docstring: "feature flags, booleans only"), not
    # person/presence data. The real identity-carrying patterns are these:
    "entity_id", "device_id", "unique_id", "person.", "presence_persons",
    "latitude", "longitude", "climate.", "sensor.", "zone.home",
)


def _rich_learning_dict() -> dict:
    return {
        "observation_count": 2,
        "trv_observation_count": 1,
        "window_cooling_obs_count": 1,
        "confidence": 0.42,
        "boost_factor": 1.1,
        "forecast_bias": 0.05,
        "heat_loss_ema": 0.031,
        "observations": [
            {
                "ts": "2026-01-13T06:37:22",
                "hour": 6, "minute": 37, "weekday": 2,
                "target": 21.0, "indoor_temp": 20.6, "delta": 0.4,
                "active_control": True, "window_open": False,
                "control_reason": "schedule", "preheat_active": False,
                "heating_failure": False, "vacation": False, "summer_mode": False,
                "outdoor_temp": 5.0, "outdoor_humidity": 70.0, "wind_speed": 3.0,
                "solar_radiation": 100.0, "indoor_humidity": 45.0,
                "heat_rate": 0.31, "norm_heat_rate": 0.02, "cool_rate": None,
                "schedule_period": "wd_comfort", "forecast_high": 8.0,
            },
            {
                "ts": "2026-01-13T07:00:00",
                "hour": 7, "minute": 0, "weekday": 2,
                "target": 21.0, "indoor_temp": 21.0, "delta": 0.0,
            },
        ],
        "trv_observations": [
            {
                "ts": "2026-01-13T06:40:00", "trv_setpoint": 22.0,
                "indoor_temp": 20.6, "target": 21.0, "delta": 0.4,
                "setpoint_excess": 1.0, "heat_rate": 0.31, "efficiency": 0.31,
                "outdoor_temp": 5.0, "wind_speed": 3.0, "solar_radiation": 100.0,
            },
        ],
        "window_cooling_obs": [
            {
                "ts": "2026-01-13T08:00:00", "duration_min": 12.0,
                "temp_drop": 0.8, "cooling_rate_per_min": 0.0667,
                "indoor_start_temp": 21.0, "outdoor_temp": 4.0, "wind_speed": 5.0,
            },
        ],
        "outcome_log": [
            {
                "ts": "2026-01-13T07:22:00", "start_temp": 19.5, "target": 21.0,
                "peak_temp": 21.1, "end_temp": 21.0, "minutes_taken": 45.0,
                "expected_minutes": 40, "controller": "ts", "reason": "reached",
                "outcome_score": 0.82, "outdoor_temp": 5.0, "outdoor_humidity": 70.0,
                "wind_speed": 3.0, "solar_radiation": 100.0, "rain": 0.0,
            },
        ],
    }


def _blob_has_no_forbidden_substrings(payload: dict) -> list[str]:
    blob = json.dumps(payload, default=str).lower()
    return [s for s in _FORBIDDEN_SUBSTRINGS if s in blob]


# ── P1: _historical_learning_snapshot_for_research() unit tests ─────────

class TestHistoricalSnapshotStructure:
    def test_available_and_markers(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        assert snap["available"] is True
        assert snap["frozen"] is True
        assert snap["raw_legacy_dump_included"] is False
        # Neutral compat key alongside the pre-existing one (Export-Naming
        # cleanup) — same value, additive, old key not removed.
        assert snap["historical_learning_dump_included"] is False
        assert snap["privacy_checked"] is True

    def test_event_count_summary_reflects_true_totals(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        summary = snap["event_count_summary"]
        assert summary["observations"] == 2
        assert summary["trv_observations"] == 1
        assert summary["window_cooling_observations"] == 1
        assert summary["outcome_entries"] == 1

    def test_model_state_scalars_preserved(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        ms = snap["model_state"]
        assert ms["confidence"] == pytest.approx(0.42)
        assert ms["boost_factor"] == pytest.approx(1.1)
        assert ms["forecast_bias"] == pytest.approx(0.05)
        assert ms["heat_loss_ema"] == pytest.approx(0.031)

    def test_observation_event_keeps_time_and_technical_fields(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        obs = snap["research_events"]["observations"][0]
        assert obs["ts"] == "2026-01-13T06:37:22"
        # 2026-01-13 is a Tuesday -> Python's datetime.weekday() == 1
        assert obs["weekday"] == 1
        assert obs["hour"] == 6
        assert obs["minute"] == 37
        assert obs["target"] == 21.0
        assert obs["delta"] == 0.4
        assert obs["heat_rate"] == 0.31
        assert obs["source"] == "historical"

    def test_trv_observation_event_keeps_technical_fields(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        trv = snap["research_events"]["trv_observations"][0]
        assert trv["trv_setpoint"] == 22.0
        assert trv["heat_rate"] == 0.31
        assert trv["efficiency"] == 0.31
        assert trv["weekday"] == 1 and trv["hour"] == 6

    def test_window_cooling_event_keeps_technical_fields(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        wc = snap["research_events"]["window_cooling"][0]
        assert wc["duration_min"] == 12.0
        assert wc["cooling_rate_per_min"] == pytest.approx(0.0667)

    def test_outcome_event_keeps_outcome_relevant_fields(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        out = snap["research_events"]["outcomes"][0]
        assert out["target"] == 21.0
        assert out["outcome_score"] == 0.82
        assert out["controller"] == "ts"
        assert out["reason"] == "reached"
        assert out["minutes_taken"] == 45.0

    def test_no_forbidden_identifiers_in_full_snapshot(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        assert scan_payload(snap) == []
        assert _blob_has_no_forbidden_substrings(snap) == []

    def test_unexpected_legacy_key_is_not_leaked(self):
        """A legacy entry with an extra, non-allow-listed key (simulating an
        unexpected/older schema field) must not surface it — allow-list only."""
        learning = _rich_learning_dict()
        learning["observations"][0]["some_unexpected_legacy_field"] = "whatever"
        learning["observations"][0]["entity_id"] = "climate.living_room_trv"
        snap = _historical_learning_snapshot_for_research(learning)
        obs = snap["research_events"]["observations"][0]
        assert "some_unexpected_legacy_field" not in obs
        assert "entity_id" not in obs
        assert scan_payload(snap) == []


class TestHistoricalSnapshotEmptyAndMissing:
    def test_empty_learning_dict_stable_no_crash(self):
        snap = _historical_learning_snapshot_for_research({})
        assert snap["available"] is True
        assert snap["research_events"] == {
            "observations": [], "trv_observations": [], "window_cooling": [], "outcomes": [],
        }
        assert snap["event_count_summary"]["observations"] == 0

    def test_missing_keys_do_not_crash(self):
        snap = _historical_learning_snapshot_for_research({"confidence": 0.1})
        assert snap["available"] is True
        assert snap["model_state"]["confidence"] == pytest.approx(0.1)
        assert snap["model_state"]["boost_factor"] is None


class TestHistoricalSnapshotCorruptInputDoesNotCrash:
    def test_non_list_observations_does_not_crash(self):
        learning = _rich_learning_dict()
        learning["observations"] = "not_a_list"
        snap = _historical_learning_snapshot_for_research(learning)
        assert snap["available"] is True
        assert snap["research_events"]["observations"] == []

    def test_non_dict_entries_in_list_are_skipped(self):
        learning = _rich_learning_dict()
        learning["observations"] = [None, "garbage", 42, {"ts": "2026-01-13T06:00:00", "target": 21.0}]
        snap = _historical_learning_snapshot_for_research(learning)
        assert len(snap["research_events"]["observations"]) == 1
        assert snap["research_events"]["observations"][0]["target"] == 21.0

    def test_malformed_ts_does_not_crash(self):
        learning = _rich_learning_dict()
        learning["observations"][0]["ts"] = "not-a-timestamp"
        snap = _historical_learning_snapshot_for_research(learning)
        obs = snap["research_events"]["observations"][0]
        assert obs["ts"] == "not-a-timestamp"
        assert "weekday" not in obs  # parse failed → time context omitted, not guessed

    def test_entirely_malformed_input_yields_available_false(self):
        class _Weird:
            def get(self, *a, **kw):
                raise RuntimeError("boom")
        snap = _historical_learning_snapshot_for_research(_Weird())
        assert snap["available"] is False
        assert snap["reason"] == "historical_snapshot_unavailable"


class TestHistoricalSnapshotCapping:
    def test_records_truncated_reports_excess(self):
        learning = _rich_learning_dict()
        n = _HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY + 25
        learning["observations"] = [
            {"ts": f"2026-01-01T00:{i:02d}:00", "target": 21.0, "delta": 0.0}
            for i in range(n)
        ]
        snap = _historical_learning_snapshot_for_research(learning)
        assert len(snap["research_events"]["observations"]) == _HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY
        assert snap["records_truncated"]["observations"] == 25

    def test_kept_entries_are_most_recent(self):
        learning = _rich_learning_dict()
        n = _HISTORICAL_RESEARCH_EVENTS_MAX_PER_CATEGORY + 5
        learning["observations"] = [
            {"ts": f"2026-01-01T00:{i:02d}:00", "target": float(i), "delta": 0.0}
            for i in range(n)
        ]
        snap = _historical_learning_snapshot_for_research(learning)
        kept_targets = [o["target"] for o in snap["research_events"]["observations"]]
        assert kept_targets[0] == 5.0  # the oldest 5 (indices 0-4) were dropped
        assert kept_targets[-1] == float(n - 1)

    def test_no_truncation_when_under_cap(self):
        snap = _historical_learning_snapshot_for_research(_rich_learning_dict())
        assert snap["records_truncated"] == {
            "observations": 0, "trv_observations": 0, "window_cooling": 0, "outcomes": 0,
        }


# ── P1 integration: async_export_learning_data() end-to-end ────────────────

class TestLearningExportIntegration:
    async def _run(self, tmp_path, learning: dict) -> dict:
        entry = MagicMock()
        entry.entry_id = "zone_1"
        entry.data = {}
        entry.options = {}

        le = MagicMock()
        le.get_export_data.return_value = learning

        hass = MagicMock()
        hass.config_entries.async_entries.return_value = [entry]
        hass.data = {DOMAIN: {"learning_engine": le}}
        hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
        hass.async_add_executor_job = AsyncMock(side_effect=lambda fn: fn())

        ts = datetime(2026, 1, 13, 8, 0, tzinfo=timezone.utc)
        filepath = await async_export_learning_data(hass, ts=ts)
        with open(filepath, encoding="utf-8") as fh:
            return json.load(fh)

    async def test_full_export_contains_research_events(self, tmp_path):
        export = await self._run(tmp_path, _rich_learning_dict())
        snap = export["zones"][0]["historical_learning_snapshot"]
        assert snap["available"] is True
        assert snap["research_events"]["observations"][0]["heat_rate"] == 0.31

    async def test_full_export_has_no_forbidden_identifiers(self, tmp_path):
        export = await self._run(tmp_path, _rich_learning_dict())
        assert _blob_has_no_forbidden_substrings(export) == []

    async def test_export_without_historical_data_is_stable(self, tmp_path):
        export = await self._run(tmp_path, {})
        snap = export["zones"][0]["historical_learning_snapshot"]
        assert snap["available"] is True
        assert snap["event_count_summary"]["observations"] == 0


# ── P2: reserved_diagnostics + preserved blocks ──────────────────────────────

class TestReservedDiagnosticsSummary:
    def test_no_shadow_yields_unavailable(self):
        coord = MagicMock()
        coord._learning_shadow = None
        result = _learning_reserved_diagnostics_summary(coord, "zone_1")
        assert result == {"available": False, "reason": "reserved_inactive", "last_error": None}

    def test_never_raises_on_broken_shadow(self):
        coord = MagicMock()
        coord._learning_shadow = MagicMock()
        coord._learning_shadow.application_lifecycle_snapshot.side_effect = RuntimeError("boom")
        coord._learning_shadow.adaptation_history_snapshot.side_effect = RuntimeError("boom")
        result = _learning_reserved_diagnostics_summary(coord, "zone_1")
        assert result["available"] is False
        assert result["reason"] == "reserved_inactive"

    def test_surfaces_application_lifecycle_last_error(self):
        coord = MagicMock()
        coord._learning_shadow.application_lifecycle_snapshot.return_value = None
        coord._learning_shadow._application_lifecycle_last_error = "some internal error"
        coord._learning_shadow.adaptation_history_snapshot.return_value = {}
        coord._learning_shadow.adaptation_last_error.return_value = None
        result = _learning_reserved_diagnostics_summary(coord, "zone_1")
        assert result["last_error"] == "some internal error"
        assert result["available"] is False


class TestSupportExportIntegration:
    async def _run(self, hass) -> dict:
        ts = datetime(2026, 1, 13, 8, 0, tzinfo=timezone.utc)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            hass.config.path.side_effect = lambda *parts: str(__import__("pathlib").Path(tmpdir, *parts))
            filepath = await async_export_support_data(hass, ts=ts)
            with open(filepath, encoding="utf-8") as fh:
                return json.load(fh)

    def _base_hass(self, entry) -> MagicMock:
        hass = MagicMock()
        hass.config_entries.async_entries.return_value = [entry]
        hass.async_add_executor_job = AsyncMock(side_effect=lambda fn: fn())
        return hass

    async def test_no_coordinator_zone_has_reserved_diagnostics_not_removed_keys(self, tmp_path):
        entry = MagicMock()
        entry.entry_id = "zone_1"
        entry.data = {"entry_type": "zone"}
        entry.options = {}
        hass = self._base_hass(entry)
        hass.data = {DOMAIN: {}}

        export = await self._run(hass)
        zone = export["zones"][0]
        assert "reserved_diagnostics" in zone
        assert zone["reserved_diagnostics"]["available"] is False
        assert "adaptation_application" not in zone
        assert "orchestration_preview" not in zone
        # not removed
        assert "runtime_pending" in zone
        assert "adaptation" in zone
        assert "adaptation_history" in zone
        assert "critical_events" in zone

    async def test_with_coordinator_reserved_diagnostics_replaces_old_blocks(self, tmp_path):
        entry = MagicMock()
        entry.entry_id = "zone_1"
        entry.data = {"entry_type": "zone"}
        entry.options = {}
        hass = self._base_hass(entry)

        coord = MagicMock()
        coord._learning_shadow = None  # simplest valid state — every _learning_* helper falls back safely
        coord.data = {}
        hass.data = {DOMAIN: {"zone_1": {"coordinator": coord}}}

        export = await self._run(hass)
        zone = export["zones"][0]
        assert "reserved_diagnostics" in zone
        assert "adaptation_application" not in zone
        assert "orchestration_preview" not in zone
        assert "runtime_pending" in zone
        assert "adaptation" in zone
        assert "adaptation_history" in zone
        assert "device_profile" in zone

    async def test_trv_command_failed_visible_in_critical_events(self, tmp_path):
        entry = MagicMock()
        entry.entry_id = "zone_1"
        entry.data = {"entry_type": "zone"}
        entry.options = {}
        hass = self._base_hass(entry)

        coord = MagicMock()
        coord.data = {}
        shadow = MagicMock()
        event = SupportCriticalEvent(
            schema_version=1, event_id="evt1",
            event_type=SupportEventType.TRV_COMMAND_FAILED, ts=datetime.now(timezone.utc),
            severity=SupportEventSeverity.WARNING, reason="repeated_dispatch_failure",
            summary="TRV setpoint command failed repeatedly",
            details={"consecutive_failures": 5},
        )
        from custom_components.thermosmart.learning.storage.support_event_serialization import (
            serialize_support_event,
        )
        shadow.support_critical_events_snapshot.return_value = {
            "evt1": serialize_support_event(event),
        }
        shadow.capture_stores = None
        coord._learning_shadow = shadow
        hass.data = {DOMAIN: {"zone_1": {"coordinator": coord}}}

        export = await self._run(hass)
        zone = export["zones"][0]
        events = zone["critical_events"].get("events", [])
        assert any(e.get("event_type") == "trv_command_failed" for e in events)

    async def test_full_support_export_no_forbidden_identifiers(self, tmp_path):
        entry = MagicMock()
        entry.entry_id = "zone_1"
        entry.data = {"entry_type": "zone"}
        entry.options = {}
        hass = self._base_hass(entry)
        hass.data = {DOMAIN: {}}

        export = await self._run(hass)
        assert _blob_has_no_forbidden_substrings(export) == []


# ── Export-creation top-level timestamp fields ───────────────────────────────

_LE_BRANDING_SUBSTRINGS = ("le1", "le2", "thermosmart_le2")


def _blob_has_no_le_branding(payload: dict, *, allow: tuple[str, ...] = ()) -> list[str]:
    """Like _blob_has_no_forbidden_substrings, but for LE1/LE2 branding terms.

    ``allow`` lets a caller permit specific sanctioned substrings (e.g. the
    storage_layout note's deliberate thermosmart_le2__ migration-fallback
    mention) without disabling the check entirely.
    """
    blob = json.dumps(payload).lower()
    return [s for s in _LE_BRANDING_SUBSTRINGS if s in blob and s not in allow]


class TestResearchExportTimestampFields:
    async def _run(self, tmp_path, learning: dict | None = None, *, tz: str = "Europe/Berlin") -> dict:
        # dt_util.as_local() (used for exported_at_local) reads the process-wide
        # HA default timezone, not hass.config.time_zone directly — in real HA
        # these are kept in sync by core startup; tests must set it explicitly.
        from homeassistant.util import dt as dt_util
        previous_tz = dt_util.DEFAULT_TIME_ZONE
        dt_util.set_default_time_zone(dt_util.get_time_zone(tz))
        try:
            entry = MagicMock()
            entry.entry_id = "zone_1"
            entry.data = {}
            entry.options = {}

            le = MagicMock()
            le.get_export_data.return_value = learning or {}

            hass = MagicMock()
            hass.config_entries.async_entries.return_value = [entry]
            hass.data = {DOMAIN: {"learning_engine": le}}
            hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
            hass.config.time_zone = tz
            hass.async_add_executor_job = AsyncMock(side_effect=lambda fn: fn())

            ts = datetime(2026, 1, 13, 8, 0, tzinfo=timezone.utc)
            filepath = await async_export_learning_data(hass, ts=ts)
            with open(filepath, encoding="utf-8") as fh:
                return json.load(fh)
        finally:
            dt_util.set_default_time_zone(previous_tz)

    async def test_contains_exported_at_utc(self, tmp_path):
        export = await self._run(tmp_path)
        assert export["exported_at_utc"] == "2026-01-13T08:00:00+00:00"

    async def test_contains_exported_at_local(self, tmp_path):
        export = await self._run(tmp_path, tz="Europe/Berlin")
        # 2026-01-13 08:00 UTC -> 09:00 CET (Europe/Berlin, standard time in January)
        assert export["exported_at_local"] == "2026-01-13T09:00:00+01:00"

    async def test_contains_timezone_name(self, tmp_path):
        export = await self._run(tmp_path, tz="Europe/Berlin")
        assert export["timezone"] == "Europe/Berlin"

    async def test_legacy_export_timestamp_field_still_present(self, tmp_path):
        """Compatibility: the pre-existing export_timestamp field must not be
        removed — exported_at_utc is additive, not a replacement."""
        export = await self._run(tmp_path)
        assert export["export_timestamp"] == export["exported_at_utc"]

    async def test_export_has_no_le1_le2_branding(self, tmp_path):
        export = await self._run(tmp_path, _rich_learning_dict())
        assert _blob_has_no_le_branding(export) == []


class TestSupportExportTimestampFields:
    async def _run(self, hass, tz: str = "Europe/Berlin") -> dict:
        from homeassistant.util import dt as dt_util
        ts = datetime(2026, 1, 13, 8, 0, tzinfo=timezone.utc)
        hass.config.time_zone = tz
        previous_tz = dt_util.DEFAULT_TIME_ZONE
        dt_util.set_default_time_zone(dt_util.get_time_zone(tz))
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                hass.config.path.side_effect = lambda *parts: str(__import__("pathlib").Path(tmpdir, *parts))
                filepath = await async_export_support_data(hass, ts=ts)
                with open(filepath, encoding="utf-8") as fh:
                    return json.load(fh)
        finally:
            dt_util.set_default_time_zone(previous_tz)

    def _base_hass(self, entry) -> MagicMock:
        hass = MagicMock()
        hass.config_entries.async_entries.return_value = [entry]
        hass.async_add_executor_job = AsyncMock(side_effect=lambda fn: fn())
        return hass

    async def test_contains_exported_at_utc(self):
        entry = MagicMock()
        entry.entry_id = "zone_1"
        entry.data = {"entry_type": "zone"}
        entry.options = {}
        hass = self._base_hass(entry)
        hass.data = {DOMAIN: {}}

        export = await self._run(hass)
        assert export["exported_at_utc"] == "2026-01-13T08:00:00+00:00"

    async def test_contains_exported_at_local_and_timezone(self):
        entry = MagicMock()
        entry.entry_id = "zone_1"
        entry.data = {"entry_type": "zone"}
        entry.options = {}
        hass = self._base_hass(entry)
        hass.data = {DOMAIN: {}}

        export = await self._run(hass, tz="Europe/Berlin")
        assert export["exported_at_local"] == "2026-01-13T09:00:00+01:00"
        assert export["timezone"] == "Europe/Berlin"

    async def test_storage_layout_note_only_sanctioned_le2_mention(self):
        """The note may mention thermosmart_le2__ solely as the legacy
        migration-fallback key — never as if it were still the active/only
        naming scheme (no other LE1/LE2 branding anywhere in the export)."""
        entry = MagicMock()
        entry.entry_id = "zone_1"
        entry.data = {"entry_type": "zone"}
        entry.options = {}
        hass = self._base_hass(entry)
        hass.data = {DOMAIN: {}}

        export = await self._run(hass)
        note = export["storage_layout"]["note"]
        assert "thermosmart_learning__" in note
        assert "thermosmart_le2__" in note  # sanctioned: legacy migration-fallback key only

        # Confirm that mention is the ONLY place "le2" appears anywhere in the
        # export — strip the note out, then the broader check must be clean.
        export_without_note = json.loads(json.dumps(export))
        export_without_note["storage_layout"]["note"] = ""
        assert _blob_has_no_le_branding(export_without_note) == []


# ── Storage-Metadata Commit C: Support Export storage_summary ───────────────

class _FakeMetaStore:
    def __init__(self, data=None):
        self.data = data

    async def async_load(self):
        return self.data

    async def async_save(self, data):
        self.data = data

    async def async_remove(self):
        self.data = None


def _patch_metadata_factory(monkeypatch, stores_by_key: dict):
    """Make every StorageMetadataStore constructed anywhere in export.py read
    from ``stores_by_key`` instead of touching a real HA Store — hermetic,
    no file I/O, independent of whatever the test's MagicMock hass supports."""
    from custom_components.thermosmart.learning.storage.stores import HomeAssistantStoreFactory

    def _create(self, key, version):
        return stores_by_key.setdefault(key, _FakeMetaStore())

    monkeypatch.setattr(HomeAssistantStoreFactory, "create", _create)


class TestStorageSummaryExport:
    async def _run(self, hass) -> dict:
        ts = datetime(2026, 1, 13, 8, 0, tzinfo=timezone.utc)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            hass.config.path.side_effect = lambda *parts: str(__import__("pathlib").Path(tmpdir, *parts))
            filepath = await async_export_support_data(hass, ts=ts)
            with open(filepath, encoding="utf-8") as fh:
                return json.load(fh)

    def _base_hass(self, entry) -> MagicMock:
        hass = MagicMock()
        hass.config_entries.async_entries.return_value = [entry]
        hass.async_add_executor_job = AsyncMock(side_effect=lambda fn: fn())
        hass.data = {DOMAIN: {}}
        return hass

    def _entry(self, zone_id="zone_1") -> MagicMock:
        entry = MagicMock()
        entry.entry_id = zone_id
        entry.data = {"entry_type": "zone"}
        entry.options = {}
        return entry

    async def test_support_export_contains_storage_summary(self, monkeypatch):
        entry = self._entry()
        hass = self._base_hass(entry)
        _patch_metadata_factory(monkeypatch, {})

        export = await self._run(hass)
        zone = export["zones"][0]
        assert "storage_summary" in zone
        assert "stores" in zone["storage_summary"]

    async def test_storage_summary_exists_true_for_written_store(self, monkeypatch):
        from custom_components.thermosmart.learning.storage import naming

        entry = self._entry()
        hass = self._base_hass(entry)
        pair = naming.storage_metadata_key_pair("zone_1")
        stores_by_key = {
            pair.current: _FakeMetaStore({
                "store_schema_version": 1,
                "data": {
                    "schema_version": 1,
                    "stores": {
                        "episodes": {
                            "exists": True,
                            "created_at_utc": "2026-01-13T07:00:00Z",
                            "updated_at_utc": "2026-01-13T07:50:00Z",
                            "last_write_reason": "episode_closed",
                            "storage_key_state": "current",
                        },
                        "models": {"exists": False},
                    },
                },
            }),
        }
        _patch_metadata_factory(monkeypatch, stores_by_key)

        export = await self._run(hass)
        stores = export["zones"][0]["storage_summary"]["stores"]
        assert stores["episodes"]["exists"] is True
        assert stores["episodes"]["last_write_reason"] == "episode_closed"
        assert stores["episodes"]["storage_key_state"] == "current"

    async def test_storage_summary_exists_false_for_missing_store(self, monkeypatch):
        from custom_components.thermosmart.learning.storage import naming

        entry = self._entry()
        hass = self._base_hass(entry)
        pair = naming.storage_metadata_key_pair("zone_1")
        stores_by_key = {
            pair.current: _FakeMetaStore({
                "store_schema_version": 1,
                "data": {"schema_version": 1, "stores": {"models": {"exists": False}}},
            }),
        }
        _patch_metadata_factory(monkeypatch, stores_by_key)

        export = await self._run(hass)
        stores = export["zones"][0]["storage_summary"]["stores"]
        assert stores["models"] == {"exists": False}

    async def test_age_minutes_derived_from_exported_at_and_updated_at(self, monkeypatch):
        """ts=08:00 UTC, updated_at_utc=07:50 UTC -> 10 minutes old."""
        from custom_components.thermosmart.learning.storage import naming

        entry = self._entry()
        hass = self._base_hass(entry)
        pair = naming.storage_metadata_key_pair("zone_1")
        stores_by_key = {
            pair.current: _FakeMetaStore({
                "store_schema_version": 1,
                "data": {"schema_version": 1, "stores": {
                    "episodes": {
                        "exists": True,
                        "created_at_utc": "2026-01-13T07:00:00Z",
                        "updated_at_utc": "2026-01-13T07:50:00Z",
                        "last_write_reason": "episode_closed",
                        "storage_key_state": "current",
                    },
                }},
            }),
        }
        _patch_metadata_factory(monkeypatch, stores_by_key)

        export = await self._run(hass)
        assert export["zones"][0]["storage_summary"]["stores"]["episodes"]["age_minutes"] == 10.0

    async def test_missing_metadata_store_does_not_break_support_export(self, monkeypatch):
        """Every StorageMetadataStore construction/read fails -> storage_summary
        is unavailable, but the rest of the support export still completes."""
        from custom_components.thermosmart.learning.storage.stores import HomeAssistantStoreFactory

        def _boom(self, key, version):
            raise RuntimeError("simulated storage-metadata factory failure")

        monkeypatch.setattr(HomeAssistantStoreFactory, "create", _boom)

        entry = self._entry()
        hass = self._base_hass(entry)
        export = await self._run(hass)
        zone = export["zones"][0]
        assert zone["storage_summary"] == {
            "available": False, "reason": "storage_metadata_unavailable",
        }
        # rest of the export still completed
        assert "critical_events" in zone

    async def test_corrupt_metadata_store_does_not_break_support_export(self, monkeypatch):
        from custom_components.thermosmart.learning.storage import naming

        entry = self._entry()
        hass = self._base_hass(entry)
        pair = naming.storage_metadata_key_pair("zone_1")
        stores_by_key = {
            # wrong envelope schema version -> StoreVersionError inside get_store_summary()
            pair.current: _FakeMetaStore({"store_schema_version": 999, "data": {}}),
        }
        _patch_metadata_factory(monkeypatch, stores_by_key)

        export = await self._run(hass)
        zone = export["zones"][0]
        assert zone["storage_summary"] == {
            "available": False, "reason": "storage_metadata_unavailable",
        }
        assert "critical_events" in zone

    async def test_storage_summary_has_no_forbidden_identifiers(self, monkeypatch):
        from custom_components.thermosmart.learning.storage import naming

        entry = self._entry()
        hass = self._base_hass(entry)
        pair = naming.storage_metadata_key_pair("zone_1")
        stores_by_key = {
            pair.current: _FakeMetaStore({
                "store_schema_version": 1,
                "data": {"schema_version": 1, "stores": {
                    "episodes": {
                        "exists": True,
                        "created_at_utc": "2026-01-13T07:00:00Z",
                        "updated_at_utc": "2026-01-13T07:50:00Z",
                        "last_write_reason": "episode_closed",
                        "storage_key_state": "current",
                    },
                }},
            }),
        }
        _patch_metadata_factory(monkeypatch, stores_by_key)

        export = await self._run(hass)
        assert _blob_has_no_forbidden_substrings(export["zones"][0]["storage_summary"]) == []

    async def test_storage_summary_has_no_le1_le2_branding(self, monkeypatch):
        entry = self._entry()
        hass = self._base_hass(entry)
        _patch_metadata_factory(monkeypatch, {})

        export = await self._run(hass)
        assert _blob_has_no_le_branding(export["zones"][0]["storage_summary"]) == []
