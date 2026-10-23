"""Stored learning state from versions that still carried manual-correction
bookkeeping loads cleanly and is written back without it.

No producer of manual-correction events exists, so the counters, the boost
contamination reference and the episode close reason were always empty. They are
gone from the models; state saved earlier simply has extra keys that are read
past and dropped on the next save.
"""
from __future__ import annotations

from custom_components.thermosmart.learning.builders.base import (
    BuilderInput,
    ConfounderCode,
    OutcomeReason,
)
from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
from custom_components.thermosmart.learning.models import BoostModel, BoostUpdateContext, OutcomeModel
from custom_components.thermosmart.learning.storage.episode_serialization import (
    deserialize_episode,
    serialize_episode,
)
from tests.helpers_boost import boost_context, boost_episode

_VALUES = [18.0, 18.6, 19.2, 19.8, 20.4, 21.0]


class TestModelStateFromEarlierVersions:
    def test_outcome_state_with_the_old_counter_loads_and_resaves_clean(self):
        saved = OutcomeModel("lz").serialize_state()
        saved["manual_correction_count"] = 3                      # as written by an earlier version

        model = OutcomeModel("lz")
        model.deserialize_state(saved)

        assert "manual_correction_count" not in model.serialize_state()
        assert not hasattr(model.diagnostics(), "manual_correction_rate")

    def test_boost_state_with_the_old_counter_loads_and_resaves_clean(self):
        saved = BoostModel("lz").serialize_state()
        saved["manual_correction_count"] = 2

        model = BoostModel("lz")
        model.deserialize_state(saved)

        assert "manual_correction_count" not in model.serialize_state()
        assert not hasattr(model.diagnostics(), "manual_correction_rate")

    def test_boost_context_with_the_old_reference_loads_and_resaves_clean(self):
        episode = boost_episode("e1", _VALUES)
        data = boost_context(episode).to_dict()
        data["manual_correction_ref"] = "mc_1"

        restored = BoostUpdateContext.from_dict(data)

        assert "manual_correction_ref" not in restored.to_dict()
        assert restored.source_episode_id == "e1"


class TestStoredEpisodeWithRetiredReason:
    def test_old_manual_correction_reason_is_read_as_interrupted_and_resaved(self):
        stored = serialize_episode(boost_episode("e1", _VALUES))
        stored["reason"] = "manual_correction"                    # as written by an earlier version

        episode = deserialize_episode(stored)

        assert episode is not None
        assert episode.reason is EpisodeReason.INTERRUPTED
        assert serialize_episode(episode)["reason"] == "interrupted"


class TestRetiredNamesAreGone:
    def test_no_manual_correction_reason_or_confounder_remains(self):
        assert not hasattr(EpisodeReason, "MANUAL_CORRECTION")
        assert not hasattr(OutcomeReason, "MANUAL_CORRECTION")
        assert not hasattr(ConfounderCode, "MANUAL_CORRECTION")
        assert "manual_correction" not in BuilderInput.__dataclass_fields__
