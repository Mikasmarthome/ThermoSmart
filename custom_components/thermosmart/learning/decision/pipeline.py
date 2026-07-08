"""Decision pipeline entry point (Phase 19A, pure Python).

Ties the typed contracts together end-to-end:

    recommendation dict + Learning predictions
      -> ZoneRuntimeInput + ControllerBaselineDecision (baseline adapter)
      -> FinalResolver (single confidence authority + guards + device)
      -> DecisionTrace + DispatchResult (single dispatch boundary)

In SHADOW the full decision is computed but the command is ``shadow_only`` (never
dispatched). In CONTROL a released, guarded value is dispatched through the one
``SingleDispatcher``. There is no separate shadow/control decision logic.
"""
from __future__ import annotations

from typing import Mapping, Optional

from .baseline import baseline_from_recommendation, zone_input_from_recommendation
from .contracts import DecisionMode, DecisionTrace, DispatchResult, LearningPredictionSet
from .dispatch import SingleDispatcher
from .resolver import FinalResolver


class DecisionPipeline:
    def __init__(self, *, resolver: Optional[FinalResolver] = None,
                 dispatcher: Optional[SingleDispatcher] = None) -> None:
        self.resolver = resolver or FinalResolver()
        self.dispatcher = dispatcher or SingleDispatcher()

    def run(self, zone_id: str, ts: str, recommendation: Mapping,
            predictions: LearningPredictionSet, *, mode: DecisionMode,
            active_control: bool = False,
            decision_id: Optional[str] = None) -> tuple[DecisionTrace, DispatchResult]:
        zin = zone_input_from_recommendation(zone_id, ts, recommendation,
                                             decision_id=decision_id)
        base = baseline_from_recommendation(zone_id, recommendation,
                                            active_control=active_control)
        trace, planned = self.resolver.resolve(zin, base, predictions, mode=mode)
        dispatched = self.dispatcher.dispatch(planned.command)
        return trace, dispatched
