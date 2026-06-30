"""Phase 15 ManualCorrectionModel update / weighting / revert / buckets / rebuild tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.raw_schemas import CorrectionSource
from custom_components.thermosmart.learning.models import (
    ManualCorrectionModel,
    ManualCorrectionParameters,
    ManualCorrectionRebuildItem,
    ManualCorrectionRejection,
    ManualCorrectionUpdateContext,
)
from tests.helpers_manual_correction import (
    correction_context,
    correction_event,
    recurring_direct,
)


def m(params=None):
    return ManualCorrectionModel("lz", params)


class TestUpdate:
    def test_first_sample(self):
        model = m()
        ev = correction_event("e", 21.0, 21.5)
        assert model.update(ev, correction_context(ev)).accepted
        assert model._state.general.sample_count == 1
        assert model._state.general.warmer_count == 1

    def test_multiple_consistent(self):
        model = m()
        recurring_direct(model, 6)
        assert model._state.general.sample_count == 6
        assert model._state.general.signed_magnitude.value == pytest.approx(0.3, abs=0.05)

    def test_conflicting_corrections(self):
        model = m()
        for d in range(4):
            ev = correction_event(f"w{d}", 21.0, 21.5, day=d)
            model.update(ev, correction_context(ev))
        for d in range(4):
            ev = correction_event(f"c{d}", 21.0, 20.5, day=d + 4)
            model.update(ev, correction_context(ev))
        # mixed directions -> general signed magnitude near zero
        assert abs(model._state.general.signed_magnitude.value) < 0.3
        assert model._state.general.warmer_count == 4 and model._state.general.cooler_count == 4

    def test_revert_counts(self):
        model = m()
        ev = correction_event("e", 21.0, 21.5)
        # reverted but still learnable direction? a quick revert is rejected as temporary
        r = model.update(ev, correction_context(ev, reverted_within_s=120.0))
        assert not r.accepted

    def test_conditioned_bucket(self):
        model = m()
        for d in range(6):
            ev = correction_event(f"d{d}", 21.0, 21.5, day=d)
            model.update(ev, correction_context(ev, decision_type="preheat", in_preheat=True,
                                                time_since_decision_s=300.0))
        assert any(model._state.buckets)

    def test_direct_weighted_higher_than_automation(self):
        direct = m()
        ev = correction_event("e", 21.0, 21.5)
        rd = direct.update(ev, correction_context(ev))
        auto = m()
        eva = correction_event("e", 21.0, 21.5, source=CorrectionSource.AUTOMATION)
        ra = auto.update(eva, correction_context(eva))
        assert rd.weight > ra.weight

    def test_single_action_does_not_dominate(self):
        model = m(ManualCorrectionParameters(min_samples_for_outlier=999))
        recurring_direct(model, 8, new=21.3)
        before = model._state.general.signed_magnitude.value
        ev = correction_event("spike", 21.0, 25.0, day=20)
        model.update(ev, correction_context(ev))
        assert abs(model._state.general.signed_magnitude.value - before) < 1.0

    def test_two_zones_isolated(self):
        a = ManualCorrectionModel("lz_a")
        b = ManualCorrectionModel("lz_b")
        ev = correction_event("e", 21.0, 21.5, zone="lz_a")
        a.update(ev, correction_context(ev))
        assert a._state.general.sample_count == 1 and b._state.general.sample_count == 0

    def test_automation_does_not_build_preference_alone(self):
        model = m()
        for d in range(6):
            ev = correction_event(f"a{d}", 21.0, 21.5, source=CorrectionSource.AUTOMATION, day=d)
            model.update(ev, correction_context(ev))
        # automation is not a direct user action -> no belastbarer preference bias
        assert model._state.preference_bias_c == 0.0


class TestContext:
    def test_roundtrip(self):
        ctx = ManualCorrectionUpdateContext(correction_event_id="e", decision_id="d",
                                            time_since_decision_s=0.0, current_temperature_c=20.0)
        assert ManualCorrectionUpdateContext.from_dict(ctx.to_dict()) == ctx

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            ManualCorrectionUpdateContext(correction_event_id="e",
                                          current_temperature_c=float("inf"))

    def test_zero_valid(self):
        ctx = ManualCorrectionUpdateContext(correction_event_id="e", time_since_decision_s=0.0)
        assert ctx.time_since_decision_s == 0.0


class TestRebuild:
    def _items(self, n=6):
        items = []
        for d in range(n):
            ev = correction_event(f"d{d}", 21.0, 21.3, day=d)
            items.append(ManualCorrectionRebuildItem(event=ev, context=correction_context(ev)))
        return items

    def test_rebuild_equals_sequential(self):
        items = self._items()
        seq = m()
        for it in items:
            seq.update(it.event, it.context)
        rb = m()
        res = rb.rebuild(None, items)
        assert res.accepted_count == 6
        assert rb._state.preference_bias_c == pytest.approx(seq._state.preference_bias_c)
        assert rb._state.general.signed_magnitude.value == pytest.approx(
            seq._state.general.signed_magnitude.value)

    def test_rebuild_deterministic(self):
        items = self._items()
        a = m(); a.rebuild(None, items)
        b = m(); b.rebuild(None, items)
        assert a.serialize_state() == b.serialize_state()

    def test_rebuild_duplicates(self):
        ev = correction_event("same", 21.0, 21.5)
        items = [ManualCorrectionRebuildItem(event=ev, context=correction_context(ev)),
                 ManualCorrectionRebuildItem(event=ev, context=correction_context(ev))]
        rb = m()
        res = rb.rebuild(None, items)
        assert res.duplicate_count == 1 and rb._state.general.sample_count == 1

    def test_rebuild_reverts_reproduced(self):
        ev = correction_event("e", 21.0, 21.5)
        items = [ManualCorrectionRebuildItem(
            event=ev, context=correction_context(ev, reverted_within_s=120.0))]
        rb = m()
        res = rb.rebuild(None, items)
        assert res.accepted_count == 0  # temporary override not learned
