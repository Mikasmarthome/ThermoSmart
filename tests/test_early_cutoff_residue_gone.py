"""The early cutoff is gone everywhere.

The mechanism never ran and was removed; its labels must not linger in the
enums that other code dispatches on. The last leftover lived in the adaptation
candidate type, which has now been removed together with the rest of the
adaptation stack — so no early-cutoff label survives anywhere.
"""
from __future__ import annotations

from custom_components.thermosmart.learning.confidence import ConfidencePurpose
from custom_components.thermosmart.learning.models.boost_reliability import BoostConfounder
from custom_components.thermosmart.learning.runtime.control import ControlFeature
from custom_components.thermosmart.learning.runtime.shadow import ComparisonType


def _labels(enum_cls):
    return {member.value for member in enum_cls}


def test_no_early_cutoff_label_in_the_dispatching_enums():
    for enum_cls in (ControlFeature, ConfidencePurpose, BoostConfounder, ComparisonType):
        assert not [v for v in _labels(enum_cls) if "early_cutoff" in v], enum_cls.__name__
