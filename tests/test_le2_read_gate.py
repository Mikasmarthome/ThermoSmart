"""Phase 19A-B: validated previous-cycle prediction read gate (rule #4)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import PredictionType
from custom_components.thermosmart.learning.decision.runtime_adapter import (
    READ_OK, READ_MISSING, READ_WRONG_ZONE, READ_UNIT_MISMATCH, READ_VERSION_MISMATCH,
    READ_STALE, READ_SUPERSEDED, READ_NONFINITE, validated_prediction_read)


class _Pred:
    def __init__(self, values):
        self.values = values


def _preds(value=1.5):
    return {PredictionType.BOOST_FACTOR: _Pred({"boost_factor": value})}


class TestReadGate:
    def test_ok(self):
        v, r = validated_prediction_read(_preds(), "boost_offset", zone_id="z",
                                         prediction_zone_id="z", expected_unit="celsius_offset")
        assert r == READ_OK and v == 1.5

    def test_missing(self):
        v, r = validated_prediction_read({}, "boost_offset", zone_id="z",
                                         prediction_zone_id="z", expected_unit="celsius_offset")
        assert v is None and r == READ_MISSING

    def test_wrong_zone(self):
        v, r = validated_prediction_read(_preds(), "boost_offset", zone_id="z",
                                         prediction_zone_id="other",
                                         expected_unit="celsius_offset")
        assert v is None and r == READ_WRONG_ZONE

    def test_unit_mismatch(self):
        v, r = validated_prediction_read(_preds(), "boost_offset", zone_id="z",
                                         prediction_zone_id="z", expected_unit="celsius")
        assert v is None and r == READ_UNIT_MISMATCH

    def test_version_mismatch(self):
        v, r = validated_prediction_read(_preds(), "boost_offset", zone_id="z",
                                         prediction_zone_id="z", expected_unit="celsius_offset",
                                         parameter_version=2, accepted_parameter_version=1)
        assert v is None and r == READ_VERSION_MISMATCH

    def test_stale(self):
        v, r = validated_prediction_read(_preds(), "boost_offset", zone_id="z",
                                         prediction_zone_id="z", expected_unit="celsius_offset",
                                         age_seconds=900, max_age_seconds=600)
        assert v is None and r == READ_STALE

    def test_superseded_flag(self):
        v, r = validated_prediction_read(_preds(), "boost_offset", zone_id="z",
                                         prediction_zone_id="z", expected_unit="celsius_offset",
                                         superseded=True)
        assert v is None and r == READ_SUPERSEDED

    def test_superseded_decision_id(self):
        v, r = validated_prediction_read(_preds(), "boost_offset", zone_id="z",
                                         prediction_zone_id="z", expected_unit="celsius_offset",
                                         prediction_decision_id="d1", current_decision_id="d2")
        assert v is None and r == READ_SUPERSEDED

    def test_nonfinite(self):
        v, r = validated_prediction_read(_preds(float("inf")), "boost_offset", zone_id="z",
                                         prediction_zone_id="z", expected_unit="celsius_offset")
        assert v is None and r == READ_NONFINITE

    def test_unknown_zone_id_not_rejected(self):
        # when prediction carries no zone (None), zone check is skipped (still validated otherwise)
        v, r = validated_prediction_read(_preds(), "boost_offset", zone_id="z",
                                         prediction_zone_id=None, expected_unit="celsius_offset")
        assert r == READ_OK and v == 1.5

    def test_fresh_within_age(self):
        v, r = validated_prediction_read(_preds(), "boost_offset", zone_id="z",
                                         prediction_zone_id="z", expected_unit="celsius_offset",
                                         age_seconds=120, max_age_seconds=600)
        assert r == READ_OK and v == 1.5
