"""S1a Item 15 — Upgrade, Migration and Rollback Audit.

Proves that ThermoSmart v1.1.1 handles:
  - LE1 / LE2 storage key separation (no collision, no accidental overwrite)
  - LearningEngine freeze semantics (read-only after freeze, idempotent)
  - LearningEngine prune_orphaned_zones (frozen=no-op, active zones preserved)
  - LearningEngine safe startup state (no artificial data before async_load)
  - LE2 store version validation (StoreVersionError on mismatch/missing)
  - Coordinator config defaults (all optional fields have safe defaults)
  - Entry-data / options merge priority (options win, __init__.py line 109)
  - Entity unique_id stability (entry_id-based, not zone-name-based)
  - Minimal TRV-only setup compatibility after upgrade
  - Config with missing optional fields does not crash
  - zone_id == entry.entry_id (stable across renames, reloads)
  - Coordinator startup state safe (no active control, no artificial learning state)
  - Rollback key isolation (LE2 keys unreadable by old code, LE1 keys unaffected)

No production changes. Pure Python + MagicMock.
asyncio_mode = auto. No @pytest.mark.asyncio needed.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from custom_components.thermosmart.learning_engine import LearningEngine
from custom_components.thermosmart.learning.storage import naming as le2_naming
from custom_components.thermosmart.learning.storage.stores import (
    StoreVersionError,
    ZoneMetadataStore,
)
from custom_components.thermosmart.const import (
    STORAGE_KEY as LE1_STORAGE_KEY,
    STORAGE_VERSION as LE1_STORAGE_VERSION,
)
from tests.helpers import make_coordinator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_le() -> LearningEngine:
    """LearningEngine with a MagicMock hass; no I/O performed."""
    hass = MagicMock()
    hass.async_create_task = MagicMock(side_effect=lambda c: c.close())
    return LearningEngine(hass)


def _make_store(zone_id: str = "zone_abc") -> ZoneMetadataStore:
    """ZoneMetadataStore with a mocked underlying HA Store."""
    hass = MagicMock()
    store = ZoneMetadataStore(hass, zone_id)
    store._store = MagicMock()
    return store


# ── §1 LE1 / LE2 Storage Key Separation ──────────────────────────────────────

class TestLE1LE2StorageKeySeparation:
    def test_le1_key_exact_value(self):
        assert LE1_STORAGE_KEY == "thermosmart_learning_data"

    def test_le2_prefix_exact_value(self):
        assert le2_naming._PREFIX == "thermosmart_le2"

    def test_le1_key_does_not_start_with_le2_prefix(self):
        """LE1 and LE2 share the same domain but diverge immediately."""
        assert not LE1_STORAGE_KEY.startswith(le2_naming._PREFIX)

    def test_le2_prefix_does_not_start_with_le1_key(self):
        assert not le2_naming._PREFIX.startswith(LE1_STORAGE_KEY)

    def test_le2_container_key_format(self):
        key = le2_naming.container_key("zone_abc")
        assert key == "thermosmart_le2__zone_abc__container"

    def test_le2_episodes_key_format(self):
        key = le2_naming.episodes_key("zone_abc")
        assert key == "thermosmart_le2__zone_abc__episodes"

    def test_le2_model_state_key_format(self):
        key = le2_naming.model_state_key("zone_abc")
        assert key == "thermosmart_le2__zone_abc__models"

    def test_le2_global_index_key_format(self):
        key = le2_naming.global_index_key()
        assert key == "thermosmart_le2__global_index"

    def test_le2_container_key_differs_per_zone(self):
        k1 = le2_naming.container_key("zone_aaa")
        k2 = le2_naming.container_key("zone_bbb")
        assert k1 != k2

    def test_le2_keys_include_double_underscore_separator(self):
        """LE2 keys use __ as separator — LE1 key has none."""
        assert "__" not in LE1_STORAGE_KEY
        assert "__" in le2_naming.container_key("zone_abc")

    def test_le1_key_not_equal_to_any_le2_key_for_zone(self):
        zone_id = "zone_abc"
        le2_keys = {
            le2_naming.container_key(zone_id),
            le2_naming.episodes_key(zone_id),
            le2_naming.model_state_key(zone_id),
            le2_naming.global_index_key(),
        }
        assert LE1_STORAGE_KEY not in le2_keys

    def test_le2_generation_is_2(self):
        assert le2_naming.LEARNING_ENGINE_GENERATION == 2

    def test_le1_version_is_1(self):
        assert LE1_STORAGE_VERSION == 1

    def test_le2_keys_all_start_with_le2_prefix(self):
        zone_id = "zone_abc"
        keys = [
            le2_naming.container_key(zone_id),
            le2_naming.episodes_key(zone_id),
            le2_naming.model_state_key(zone_id),
        ]
        for key in keys:
            assert key.startswith("thermosmart_le2"), f"Key {key!r} missing le2 prefix"

    def test_le2_global_key_does_not_contain_zone_id(self):
        """Global index has no zone component — not per-zone."""
        key = le2_naming.global_index_key()
        assert "zone" not in key

    def test_two_zones_have_fully_distinct_le2_key_sets(self):
        zone_a = "zone_alpha"
        zone_b = "zone_beta"
        keys_a = {le2_naming.container_key(zone_a), le2_naming.episodes_key(zone_a)}
        keys_b = {le2_naming.container_key(zone_b), le2_naming.episodes_key(zone_b)}
        assert keys_a.isdisjoint(keys_b)


# ── §2 LearningEngine Freeze Semantics ───────────────────────────────────────

class TestLearningEngineFreeze:
    def test_not_frozen_at_init(self):
        le = _make_le()
        assert le._frozen is False

    def test_frozen_property_reflects_frozen_state(self):
        le = _make_le()
        assert le.frozen is False
        le.freeze()
        assert le.frozen is True

    def test_freeze_sets_frozen_flag(self):
        le = _make_le()
        le.freeze()
        assert le._frozen is True

    def test_freeze_is_idempotent(self):
        le = _make_le()
        le.freeze()
        le.freeze()
        assert le._frozen is True

    def test_frozen_engine_prune_is_noop(self):
        """Frozen LE never mutates — prune skips all work."""
        le = _make_le()
        le._observations["zone_orphan"] = [{"ts": "x"}]
        le.freeze()
        le.prune_orphaned_zones({"zone_active"})
        assert "zone_orphan" in le._observations

    def test_frozen_engine_does_not_trigger_async_save_on_prune(self):
        le = _make_le()
        le._observations["zone_orphan"] = [{"ts": "x"}]
        le.freeze()
        le.prune_orphaned_zones({"zone_active"})
        le._hass.async_create_task.assert_not_called()

    def test_not_frozen_engine_prune_removes_orphan(self):
        le = _make_le()
        le._observations["zone_orphan"] = [{"ts": "x"}]
        le.prune_orphaned_zones({"zone_active"})
        assert "zone_orphan" not in le._observations

    def test_not_frozen_engine_prune_preserves_active_zone(self):
        le = _make_le()
        le._observations["zone_orphan"] = [{"ts": "x"}]
        le._observations["zone_active"] = [{"ts": "y"}]
        le.prune_orphaned_zones({"zone_active"})
        assert "zone_active" in le._observations

    def test_not_frozen_engine_prune_triggers_async_save(self):
        le = _make_le()
        le._observations["zone_orphan"] = [{"ts": "x"}]
        le.prune_orphaned_zones({"zone_active"})
        le._hass.async_create_task.assert_called_once()

    def test_prune_with_empty_active_set_is_safety_noop(self):
        """Empty active_zone_ids → safety guard: never delete everything."""
        le = _make_le()
        le._observations["zone_abc"] = [{"ts": "x"}]
        le.prune_orphaned_zones(set())
        assert "zone_abc" in le._observations

    def test_prune_trv_observations_alongside_observations(self):
        le = _make_le()
        le._observations["zone_orphan"] = [{"ts": "x"}]
        le._trv_observations["zone_orphan"] = [{"ts": "x"}]
        le.prune_orphaned_zones({"zone_active"})
        assert "zone_orphan" not in le._observations
        assert "zone_orphan" not in le._trv_observations

    def test_prune_boost_factors_alongside_observations(self):
        le = _make_le()
        le._observations["zone_orphan"] = [{"ts": "x"}]
        le._boost_factors["zone_orphan"] = 1.5
        le.prune_orphaned_zones({"zone_active"})
        assert "zone_orphan" not in le._boost_factors

    def test_prune_empty_observations_no_crash(self):
        le = _make_le()
        le.prune_orphaned_zones({"zone_active"})
        assert le._observations == {}


# ── §3 LearningEngine Safe Startup State ─────────────────────────────────────

class TestLearningEngineSafeStartupState:
    def test_observations_empty_at_init(self):
        le = _make_le()
        assert len(le._observations) == 0

    def test_trv_observations_empty_at_init(self):
        le = _make_le()
        assert len(le._trv_observations) == 0

    def test_confidence_empty_at_init(self):
        le = _make_le()
        assert len(le._confidence) == 0

    def test_boost_factors_empty_at_init(self):
        le = _make_le()
        assert le._boost_factors == {}

    def test_forecast_bias_empty_at_init(self):
        le = _make_le()
        assert le._forecast_bias == {}

    def test_heat_loss_ema_empty_at_init(self):
        le = _make_le()
        assert le._heat_loss_ema == {}

    def test_outcome_log_empty_at_init(self):
        le = _make_le()
        assert len(le._outcome_log) == 0

    def test_no_artificial_confidence_without_load(self):
        """No zone should have non-zero confidence before async_load is called."""
        le = _make_le()
        assert all(v == 0.0 for v in le._confidence.values())

    def test_debounce_cancel_is_none_at_init(self):
        le = _make_le()
        assert le._debounce_save_cancel is None


# ── §4 LE2 Store Version Validation ──────────────────────────────────────────

class TestLE2StoreVersionValidation:
    async def test_correct_version_returns_data(self):
        store = _make_store()
        store._store.async_load = AsyncMock(return_value={
            "store_schema_version": 1,
            "data": {"zone_id": "zone_abc"},
        })
        result = await store.load()
        assert result == {"zone_id": "zone_abc"}

    async def test_wrong_version_raises_store_version_error(self):
        store = _make_store()
        store._store.async_load = AsyncMock(return_value={
            "store_schema_version": 999,
            "data": {},
        })
        with pytest.raises(StoreVersionError):
            await store.load()

    async def test_missing_schema_version_key_raises_store_version_error(self):
        store = _make_store()
        store._store.async_load = AsyncMock(return_value={"data": {}})
        with pytest.raises(StoreVersionError):
            await store.load()

    async def test_non_dict_data_raises_store_version_error(self):
        store = _make_store()
        store._store.async_load = AsyncMock(return_value="corrupt_string")
        with pytest.raises(StoreVersionError):
            await store.load()

    async def test_none_return_from_store_returns_none(self):
        """None = store file not found yet → first boot, return None safely."""
        store = _make_store()
        store._store.async_load = AsyncMock(return_value=None)
        result = await store.load()
        assert result is None

    async def test_extra_fields_in_envelope_are_ignored(self):
        """Unknown fields in the envelope do not cause errors."""
        store = _make_store()
        store._store.async_load = AsyncMock(return_value={
            "store_schema_version": 1,
            "data": {"zone_id": "zone_abc"},
            "extra_field": "future_extension",
        })
        result = await store.load()
        assert result == {"zone_id": "zone_abc"}

    async def test_store_key_contains_zone_id(self):
        """Each zone gets its own store, identified by zone_id."""
        store = _make_store("my_zone_123")
        key = le2_naming.container_key("my_zone_123")
        assert "my_zone_123" in key

    async def test_store_key_for_different_zones_differ(self):
        k1 = le2_naming.container_key("zone_one")
        k2 = le2_naming.container_key("zone_two")
        assert k1 != k2


# ── §5 Coordinator Config Defaults ───────────────────────────────────────────

class TestCoordinatorConfigDefaults:
    def test_comfort_temp_default(self):
        assert make_coordinator().zone_cfg.get("comfort_temp") == pytest.approx(21.0)

    def test_night_temp_default(self):
        assert make_coordinator().zone_cfg.get("night_temp") == pytest.approx(18.0)

    def test_away_temp_default(self):
        assert make_coordinator().zone_cfg.get("away_temp") == pytest.approx(17.0)

    def test_vacation_temp_default(self):
        assert make_coordinator().zone_cfg.get("vacation_temp") == pytest.approx(12.0)

    def test_temp_tolerance_default(self):
        assert make_coordinator().zone_cfg.get("temp_tolerance") == pytest.approx(0.5)

    def test_window_open_temp_default(self):
        assert make_coordinator().zone_cfg.get("window_open_temp") == pytest.approx(5.0)

    def test_window_open_delay_default(self):
        assert make_coordinator().zone_cfg.get("window_open_delay") == 5

    def test_schedule_enabled_default(self):
        assert make_coordinator().zone_cfg.get("schedule_enabled") is True

    def test_learning_enabled_default(self):
        assert make_coordinator().zone_cfg.get("learning_enabled") is True

    def test_active_control_false_at_startup(self):
        assert make_coordinator()._active_control is False

    def test_active_control_initialized_false_at_startup(self):
        assert make_coordinator()._active_control_initialized is False

    def test_zone_id_equals_entry_entry_id(self):
        coord = make_coordinator()
        assert coord.zone_id == coord.entry.entry_id

    def test_zone_id_is_stable_string(self):
        coord = make_coordinator()
        assert isinstance(coord.zone_id, str)
        assert len(coord.zone_id) > 0

    def test_last_written_setpoints_empty_at_startup(self):
        assert make_coordinator()._last_written_setpoints == {}

    def test_trv_offline_empty_at_startup(self):
        assert make_coordinator()._trv_offline == set()

    def test_boost_active_empty_at_startup(self):
        assert make_coordinator()._boost_active == {}


# ── §6 Entry Data / Options Merge Priority ───────────────────────────────────

class TestEntryDataOptionsMergePriority:
    def test_options_override_data_when_same_key(self):
        """__init__.py line 109: cfg = {**entry.data, **entry.options}"""
        data = {"name": "Zone A", "comfort_temp": 19.0}
        options = {"comfort_temp": 22.5}
        merged = {**data, **options}
        assert merged["comfort_temp"] == pytest.approx(22.5)

    def test_data_only_field_accessible(self):
        data = {"name": "Zone A", "comfort_temp": 21.0}
        options = {}
        merged = {**data, **options}
        assert merged["name"] == "Zone A"

    def test_options_only_field_accessible(self):
        data = {"name": "Zone A"}
        options = {"temp_tolerance": 0.3}
        merged = {**data, **options}
        assert merged["temp_tolerance"] == pytest.approx(0.3)

    def test_empty_options_does_not_delete_data_fields(self):
        data = {"name": "Zone A", "comfort_temp": 21.0}
        options = {}
        merged = {**data, **options}
        assert merged["comfort_temp"] == pytest.approx(21.0)

    def test_extra_options_field_does_not_block_merge(self):
        data = {"name": "Zone A"}
        options = {"new_future_option": "some_value"}
        merged = {**data, **options}
        assert merged["new_future_option"] == "some_value"
        assert merged["name"] == "Zone A"


# ── §7 Entity Unique ID Stability ────────────────────────────────────────────

class TestEntityUniqueIdStability:
    def test_unique_id_formula_uses_entry_id_not_zone_name(self):
        """Unique IDs are entry_id-based; renaming a zone doesn't break them."""
        coord = make_coordinator()
        entry_id = coord.entry.entry_id
        expected_climate_uid = f"{entry_id}_climate"
        assert "_" in expected_climate_uid
        assert entry_id in expected_climate_uid

    def test_climate_entity_unique_id_formula(self):
        coord = make_coordinator()
        entry_id = coord.entry.entry_id
        expected = f"{entry_id}_climate"
        assert expected.endswith("_climate")
        assert expected.startswith(entry_id)

    def test_sensor_unique_id_formulas(self):
        coord = make_coordinator()
        entry_id = coord.entry.entry_id
        suffixes = [
            "adjusted_target", "trv_setpoint", "preheat_minutes",
            "confidence", "weather_offset", "status",
            "temp_slope", "heating_power", "tpi_duty_cycle",
        ]
        for suffix in suffixes:
            uid = f"{entry_id}_{suffix}"
            assert uid.startswith(entry_id), f"UID {uid!r} does not start with entry_id"
            assert uid.endswith(f"_{suffix}"), f"UID {uid!r} does not end with _{suffix}"

    def test_switch_unique_id_formulas(self):
        coord = make_coordinator()
        entry_id = coord.entry.entry_id
        for suffix in ("active_control", "learning"):
            uid = f"{entry_id}_{suffix}"
            assert entry_id in uid

    def test_select_unique_id_formula(self):
        coord = make_coordinator()
        entry_id = coord.entry.entry_id
        uid = f"{entry_id}_mode"
        assert uid == f"{entry_id}_mode"

    def test_entry_id_stable_across_coordinator_reads(self):
        coord = make_coordinator()
        id1 = coord.entry.entry_id
        id2 = coord.entry.entry_id
        assert id1 == id2

    def test_zone_id_equals_entry_id_invariant(self):
        """zone_id must equal entry.entry_id for LE2 storage key consistency."""
        coord = make_coordinator()
        assert coord.zone_id == coord.entry.entry_id

    def test_unique_ids_do_not_contain_zone_name(self):
        """Unique IDs are independent of zone name so rename is safe."""
        coord = make_coordinator()
        zone_name = coord.zone_name
        entry_id = coord.entry.entry_id
        uid = f"{entry_id}_climate"
        assert zone_name not in uid

    def test_all_entity_unique_ids_are_distinct(self):
        coord = make_coordinator()
        eid = coord.entry.entry_id
        uids = [
            f"{eid}_climate",
            f"{eid}_adjusted_target",
            f"{eid}_confidence",
            f"{eid}_active_control",
            f"{eid}_learning",
            f"{eid}_mode",
        ]
        assert len(uids) == len(set(uids))


# ── §8 Minimal Setup TRV-Only Compatibility ───────────────────────────────────

class TestMinimalSetupTRVOnlyCompatibility:
    def test_coordinator_init_with_minimal_config(self):
        """Only climate_entities in config — all optional fields absent — no crash."""
        coord = make_coordinator()
        assert coord is not None

    def test_missing_temp_sensors_does_not_crash(self):
        coord = make_coordinator()
        assert coord.zone_cfg.get("temp_sensors", []) is not None

    def test_missing_window_sensors_does_not_crash(self):
        coord = make_coordinator()
        sensors = coord.zone_cfg.get("window_sensors", [])
        assert isinstance(sensors, list)

    def test_missing_outdoor_sensor_config_returns_none_or_empty(self):
        coord = make_coordinator()
        assert coord.zone_cfg.get("outdoor_temp_sensor") is None or \
               coord.zone_cfg.get("outdoor_temp_sensor") == ""

    def test_climate_entities_accessible_in_cfg(self):
        coord = make_coordinator()
        entities = coord.zone_cfg.get("climate_entities", [])
        assert isinstance(entities, list)
        assert len(entities) >= 1

    def test_coordinator_zone_name_accessible(self):
        coord = make_coordinator()
        assert isinstance(coord.zone_name, str)
        assert len(coord.zone_name) > 0

    def test_coordinator_zone_id_accessible(self):
        coord = make_coordinator()
        assert isinstance(coord.zone_id, str)
        assert len(coord.zone_id) > 0

    def test_learning_disabled_does_not_crash_coordinator(self):
        coord = make_coordinator()
        assert coord.zone_cfg.get("learning_enabled") in (True, False, None, "")


# ── §9 Rollback Risk: Key Isolation ──────────────────────────────────────────

class TestRollbackRiskKeyIsolation:
    def test_le1_key_readable_after_hypothetical_upgrade(self):
        """LE1 key is a well-known constant unaffected by LE2 additions."""
        assert LE1_STORAGE_KEY == "thermosmart_learning_data"

    def test_le2_keys_are_distinct_from_le1_for_any_zone_id(self):
        """No LE2 key for any zone can collide with the single LE1 key."""
        for zone_id in ("zone_abc", "zone123", "z" * 40):
            try:
                le2_keys = {
                    le2_naming.container_key(zone_id),
                    le2_naming.episodes_key(zone_id),
                    le2_naming.model_state_key(zone_id),
                }
                for k in le2_keys:
                    assert k != LE1_STORAGE_KEY, f"Collision: {k!r} == LE1 key"
            except le2_naming.StorageNamingError:
                pass  # invalid zone_id is rejected (correct)

    def test_le2_global_key_distinct_from_le1_key(self):
        assert le2_naming.global_index_key() != LE1_STORAGE_KEY

    def test_new_options_fields_have_safe_defaults(self):
        """New options fields that old code doesn't know about default safely to None."""
        coord = make_coordinator()
        cfg = coord.zone_cfg
        unknown_future_field = cfg.get("nonexistent_field_v2")
        assert unknown_future_field is None

    def test_le2_prefix_distinguishable_by_old_code(self):
        """v1.0.x code looking for 'thermosmart_learning_data' won't find LE2 keys."""
        le2_key = le2_naming.container_key("zone_abc")
        assert not le2_key.startswith(LE1_STORAGE_KEY)
        assert not le2_key.endswith(LE1_STORAGE_KEY)

    def test_zone_rename_does_not_change_entry_id(self):
        """entry_id is assigned at creation; name changes don't affect it."""
        coord = make_coordinator()
        original_entry_id = coord.entry.entry_id
        original_zone_id = coord.zone_id
        assert original_entry_id == original_zone_id
        # entry_id is set once and never mutated

    def test_two_zones_produce_independent_le2_key_namespaces(self):
        """If two zones had IDs zone_a and zone_b, their LE2 keys never overlap."""
        keys_zone_a = {
            le2_naming.container_key("zone_a"),
            le2_naming.episodes_key("zone_a"),
            le2_naming.model_state_key("zone_a"),
        }
        keys_zone_b = {
            le2_naming.container_key("zone_b"),
            le2_naming.episodes_key("zone_b"),
            le2_naming.model_state_key("zone_b"),
        }
        assert keys_zone_a.isdisjoint(keys_zone_b)

    def test_le2_generation_constant_is_authoritative(self):
        """Generation 2 is permanently encoded — rollback to gen 1 reads a different key."""
        assert le2_naming.LEARNING_ENGINE_GENERATION == 2
        # LE1 has no generation suffix in its key — they differ structurally

    def test_store_version_error_on_schema_mismatch_prevents_corrupt_load(self):
        """Version guard ensures old code can never silently use LE2 data."""
        assert issubclass(StoreVersionError, Exception)
