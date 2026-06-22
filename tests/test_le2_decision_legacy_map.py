"""Phase 19A: legacy inventory mapping integrity."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import (
    LEGACY_INVENTORY, Disposition, by_disposition, validate_inventory)


class TestLegacyMap:
    def test_validates(self):
        validate_inventory()  # raises on duplicate / bad REPLACE / any REMOVE

    def test_no_removals_in_19a(self):
        assert by_disposition(Disposition.REMOVE) == ()

    def test_replace_rows_name_le2_target(self):
        for e in by_disposition(Disposition.REPLACE):
            assert e.le2_replacement

    def test_keep_covers_baseline_safety(self):
        keep = {e.responsibility for e in by_disposition(Disposition.KEEP)}
        for must in ("frost_protection", "summer_mode", "window_logic", "tpi_duty"):
            assert must in keep

    def test_learning_truths_replaced(self):
        repl = {e.responsibility for e in by_disposition(Disposition.REPLACE)}
        for truth in ("heat_rate", "heat_loss", "afterheat", "outcome_score", "confidence",
                      "boost_factor", "forecast_bias", "preheat_minutes"):
            assert truth in repl

    def test_each_responsibility_unique(self):
        names = [e.responsibility for e in LEGACY_INVENTORY]
        assert len(names) == len(set(names))

    def test_dispatch_is_move_not_remove(self):
        disp = {e.responsibility: e.disposition for e in LEGACY_INVENTORY}
        assert disp["dispatch_path"] is Disposition.MOVE
