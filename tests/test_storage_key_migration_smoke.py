"""Learning Naming / Storage-Key-Migration audit — Commit 1 smoke tests.

Covers the additive, non-production-affecting groundwork only:
  - the new neutral prefix (LEARNING_STORAGE_PREFIX) vs the legacy prefix
    every existing *_key() function still actually produces
  - LearningStorageKeyPair is computable for every store type
  - thermosmart_learning_data (the v1 legacy store) is untouched — it was
    already neutral and must never be renamed
  - the global-index and runtime-snapshot key pairs specifically

No store's actual read/write key has changed in this commit — every existing
naming.*_key() function and ha_store.store_key() are asserted to still
produce exactly their pre-existing thermosmart_le2__ values.
"""
from __future__ import annotations

from custom_components.thermosmart.const import STORAGE_KEY
from custom_components.thermosmart.learning.raw_schemas import RawTrackName
from custom_components.thermosmart.learning.runtime import ha_store
from custom_components.thermosmart.learning.storage import naming


class TestNeutralPrefixConstants:
    def test_neutral_prefix_has_no_generation_codename(self):
        assert naming.LEARNING_STORAGE_PREFIX == "thermosmart_learning"
        assert "le2" not in naming.LEARNING_STORAGE_PREFIX
        assert "le1" not in naming.LEARNING_STORAGE_PREFIX

    def test_legacy_prefix_constant_matches_current_production_prefix(self):
        assert naming.LEGACY_LEARNING_STORAGE_PREFIX == "thermosmart_le2"
        assert naming.container_key("lz_1").startswith(
            f"{naming.LEGACY_LEARNING_STORAGE_PREFIX}__"
        )


class TestExistingKeyFunctionsUnchanged:
    """Production behavior must be identical to before this commit."""

    def test_container_key_unchanged(self):
        assert naming.container_key("lz_1") == "thermosmart_le2__lz_1__container"

    def test_episodes_key_unchanged(self):
        assert naming.episodes_key("lz_1") == "thermosmart_le2__lz_1__episodes"

    def test_model_state_key_unchanged(self):
        assert naming.model_state_key("lz_1") == "thermosmart_le2__lz_1__models"

    def test_global_index_key_unchanged(self):
        assert naming.global_index_key() == "thermosmart_le2__global_index"

    def test_raw_index_key_unchanged(self):
        assert naming.raw_index_key("lz_1", RawTrackName.ROOM) == \
            "thermosmart_le2__lz_1__raw_room_index"

    def test_raw_segment_key_unchanged(self):
        assert naming.raw_segment_key("lz_1", RawTrackName.ROOM, 3) == \
            "thermosmart_le2__lz_1__raw_room_seg_3"

    def test_adaptation_history_key_unchanged(self):
        assert naming.adaptation_history_key("lz_1") == \
            "thermosmart_le2__lz_1__adaptation_history"

    def test_application_lifecycle_key_unchanged(self):
        assert naming.application_lifecycle_key("lz_1") == \
            "thermosmart_le2__lz_1__application_lifecycle"

    def test_support_critical_events_key_unchanged(self):
        assert naming.support_critical_events_key("lz_1") == \
            "thermosmart_le2__lz_1__support_critical_events"

    def test_research_daily_key_unchanged(self):
        assert naming.research_daily_key("lz_1") == \
            "thermosmart_le2__lz_1__research_daily"


class TestKeyPairsComputable:
    """Every store type must have a computable (current, legacy) pair —
    current always neutral, legacy always the existing production key."""

    def test_container_key_pair(self):
        pair = naming.container_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__container"
        assert pair.legacy == "thermosmart_le2__lz_1__container"

    def test_episodes_key_pair(self):
        pair = naming.episodes_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__episodes"
        assert pair.legacy == naming.episodes_key("lz_1")

    def test_model_state_key_pair(self):
        pair = naming.model_state_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__models"
        assert pair.legacy == naming.model_state_key("lz_1")

    def test_raw_index_key_pair(self):
        pair = naming.raw_index_key_pair("lz_1", RawTrackName.TRV)
        assert pair.current == "thermosmart_learning__lz_1__raw_trv_index"
        assert pair.legacy == naming.raw_index_key("lz_1", RawTrackName.TRV)

    def test_raw_segment_key_pair(self):
        pair = naming.raw_segment_key_pair("lz_1", RawTrackName.TRV, 7)
        assert pair.current == "thermosmart_learning__lz_1__raw_trv_seg_7"
        assert pair.legacy == naming.raw_segment_key("lz_1", RawTrackName.TRV, 7)

    def test_adaptation_history_key_pair(self):
        pair = naming.adaptation_history_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__adaptation_history"
        assert pair.legacy == naming.adaptation_history_key("lz_1")

    def test_application_lifecycle_key_pair(self):
        pair = naming.application_lifecycle_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__application_lifecycle"
        assert pair.legacy == naming.application_lifecycle_key("lz_1")

    def test_support_critical_events_key_pair(self):
        pair = naming.support_critical_events_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__support_critical_events"
        assert pair.legacy == naming.support_critical_events_key("lz_1")

    def test_research_daily_key_pair(self):
        pair = naming.research_daily_key_pair("lz_1")
        assert pair.current == "thermosmart_learning__lz_1__research_daily"
        assert pair.legacy == naming.research_daily_key("lz_1")

    def test_global_index_key_pair(self):
        pair = naming.global_index_key_pair()
        assert pair.current == "thermosmart_learning__global_index"
        assert pair.legacy == "thermosmart_le2__global_index"

    def test_no_current_key_contains_legacy_codename(self):
        pairs = [
            naming.container_key_pair("lz_1"),
            naming.episodes_key_pair("lz_1"),
            naming.model_state_key_pair("lz_1"),
            naming.adaptation_history_key_pair("lz_1"),
            naming.application_lifecycle_key_pair("lz_1"),
            naming.support_critical_events_key_pair("lz_1"),
            naming.research_daily_key_pair("lz_1"),
            naming.global_index_key_pair(),
        ]
        for pair in pairs:
            assert "le2" not in pair.current
            assert pair.legacy is not None and "le2" in pair.legacy


class TestLegacyStoreDataKeyUntouched:
    """thermosmart_learning_data (v1) is already neutral and must never be
    renamed by this migration — no le1/le2/v1/v2 codename in it."""

    def test_storage_key_constant_unchanged(self):
        assert STORAGE_KEY == "thermosmart_learning_data"

    def test_storage_key_has_no_generation_codename(self):
        for token in ("le1", "le2", "v1", "v2", "legacy"):
            assert token not in STORAGE_KEY

    def test_global_index_key_pair_is_distinct_from_legacy_v1_store(self):
        pair = naming.global_index_key_pair()
        assert pair.current != STORAGE_KEY
        assert pair.legacy != STORAGE_KEY


class TestRuntimeSnapshotKeyPair:
    """ha_store.py's runtime-snapshot key (hashed zone segment), mirrored via
    its own store_key_pair() rather than naming.py's suffix-based pairs."""

    def test_store_key_unchanged(self):
        assert ha_store.store_key("abc123") == "thermosmart_le2__abc123"

    def test_new_prefix_constant_has_no_generation_codename(self):
        assert ha_store.NEW_STORE_KEY_PREFIX == "thermosmart_learning__"

    def test_store_key_pair(self):
        pair = ha_store.store_key_pair("abc123")
        assert pair.current == "thermosmart_learning__abc123"
        assert pair.legacy == "thermosmart_le2__abc123"

    def test_store_key_pair_empty_segment_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            ha_store.store_key_pair("")


class TestZoneSegmentConsolidation:
    """naming.zone_segment() is the single source of truth for the hash used
    by both ha_integration.py's _zone_segment() and __init__.py's cleanup
    path — same algorithm, same output for any given input."""

    def test_zone_segment_deterministic(self):
        assert naming.zone_segment("entry_abc") == naming.zone_segment("entry_abc")

    def test_zone_segment_differs_per_zone(self):
        assert naming.zone_segment("entry_abc") != naming.zone_segment("entry_xyz")

    def test_zone_segment_is_16_hex_chars(self):
        seg = naming.zone_segment("entry_abc")
        assert len(seg) == 16
        int(seg, 16)  # raises ValueError if not valid hex

    def test_zone_segment_matches_known_sha256_prefix(self):
        import hashlib
        expected = hashlib.sha256("entry_abc".encode("utf-8")).hexdigest()[:16]
        assert naming.zone_segment("entry_abc") == expected

    def test_store_key_pair_uses_same_hash_as_zone_segment(self):
        seg = naming.zone_segment("entry_abc")
        pair = ha_store.store_key_pair(seg)
        assert pair.legacy == ha_store.store_key(naming.zone_segment("entry_abc"))
