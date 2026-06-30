"""Phase 18 boost control tests (additive °C; reduce-only; limit guard)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.const import TPI_MAX_BOOST_CELSIUS
from custom_components.thermosmart.learning.confidence import ComponentKind, ConfidencePurpose
from custom_components.thermosmart.learning.models import BoostParameters
from custom_components.thermosmart.learning.runtime.control import (
    ControlContext, ControlFeature, ControlRejection)
from tests.helpers_control import control_decision, strong_boost, weak_purpose


F = ControlFeature.BOOST_OFFSET


def _strong():
    return strong_boost()


class TestBoostControl:
    def test_additive_offset_semantics(self):
        # baseline 2.0 °C offset, proposed 1.7 -> reduced within clamp
        d = control_decision(F, 2.0, 1.7, _strong(), unit="celsius_offset",
                             context=ControlContext(boost_runtime_limit=TPI_MAX_BOOST_CELSIUS))
        assert d.applied and d.final_value == 1.7

    def test_overshoot_reduces_boost(self):
        d = control_decision(F, 3.0, 2.6, _strong(), unit="celsius_offset",
                             context=ControlContext(boost_runtime_limit=TPI_MAX_BOOST_CELSIUS))
        assert d.applied and d.final_value < 3.0

    def test_low_attribution_falls_back(self):
        d = control_decision(F, 2.0, 1.5,
                             weak_purpose(ConfidencePurpose.BOOST, ComponentKind.BOOST),
                             unit="celsius_offset",
                             context=ControlContext(boost_runtime_limit=TPI_MAX_BOOST_CELSIUS))
        assert not d.applied and d.final_value == 2.0

    def test_limit_mismatch_blocks(self):
        d = control_decision(F, 2.0, 1.5, _strong(), unit="celsius_offset",
                             context=ControlContext(boost_runtime_limit=99.0))
        assert d.rejection == ControlRejection.BOOST_LIMIT_MISMATCH.value

    def test_runtime_limit_matches_model(self):
        assert BoostParameters().max_boost_offset_c == TPI_MAX_BOOST_CELSIUS

    def test_duration_clamp(self):
        d = control_decision(ControlFeature.BOOST_DURATION, 1800.0, 3000.0,
                             strong_boost(),
                             unit="seconds")
        # max_step 600 -> clamped to 2400
        assert d.applied and d.final_value == 2400.0

    def test_excessive_offset_deviation_falls_back(self):
        d = control_decision(F, 0.5, 6.0, _strong(), unit="celsius_offset",
                             context=ControlContext(boost_runtime_limit=TPI_MAX_BOOST_CELSIUS))
        assert not d.applied  # excessive deviation

    def test_offset_max_clamp_within_runtime_limit(self):
        d = control_decision(F, 7.8, 8.5, _strong(), unit="celsius_offset",
                             context=ControlContext(boost_runtime_limit=TPI_MAX_BOOST_CELSIUS))
        assert d.final_value <= TPI_MAX_BOOST_CELSIUS + 1e-9
