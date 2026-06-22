"""Shared builders for LE 2.0 decision-architecture tests (Phase 19A)."""
from __future__ import annotations

from custom_components.thermosmart.learning.decision import (
    DecisionMode, DecisionPipeline, LearningPrediction, LearningPredictionSet)
from custom_components.thermosmart.learning.decision.resolver import FinalResolver
from custom_components.thermosmart.learning.decision.dispatch import SingleDispatcher
from tests.helpers_control import strong_boost, strong_preheat, strong_purpose
from custom_components.thermosmart.learning.confidence import ComponentKind, ConfidencePurpose

TS = "2025-01-01T07:00:00+00:00"


def rec(target=21.0, setpoint=23.0, indoor=19.0, **extra):
    d = {"effective_target": target, "trv_setpoint": setpoint, "indoor_temperature": indoor}
    d.update(extra)
    return d


def preds(zone="z", *, boost=None, preheat=None, cutoff=None, with_conf=True):
    p = {}
    c = {}
    if boost is not None:
        p["boost_offset"] = LearningPrediction("boost_offset", boost, "celsius_offset", 0.85,
                                                "boost", False)
        if with_conf:
            c[ConfidencePurpose.BOOST.value] = strong_boost()
    if preheat is not None:
        p["preheat_start"] = LearningPrediction("preheat_start", preheat, "minutes", 0.85,
                                                "preheat", False)
        if with_conf:
            c[ConfidencePurpose.PREHEAT.value] = strong_preheat()
    if cutoff is not None:
        p["early_cutoff"] = LearningPrediction("early_cutoff", cutoff, "minutes", 0.85,
                                               "early_cutoff", False)
        if with_conf:
            c[ConfidencePurpose.EARLY_CUTOFF.value] = strong_purpose(
                ConfidencePurpose.EARLY_CUTOFF, ComponentKind.AFTERHEAT)
    return LearningPredictionSet(zone, "dec1", predictions=p, confidence_results=c)


def pipeline(sink=None, boost_runtime_limit=8.0):
    return DecisionPipeline(
        resolver=FinalResolver(boost_runtime_limit=boost_runtime_limit),
        dispatcher=SingleDispatcher(sink=sink))


def run(mode, *, recommendation=None, predictions=None, sink=None, active_control=True):
    p = pipeline(sink=sink)
    return p.run("z", TS, recommendation or rec(), predictions or preds(boost=1.5),
                 mode=mode, active_control=active_control, decision_id="dec1")
