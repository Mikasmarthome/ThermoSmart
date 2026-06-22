"""Model orchestration for LE 2.0 shadow runtime (pure Python).

Owns the per-zone model instances, applies already-materialised typed update
contexts (eligibility-gated, failure-isolated), collects predictions and calls
the ConfidenceAggregator exactly once per purpose. Never mutates the existing
controller and never calls a service. Concrete models are imported here because
this is the runtime shell layer, not a pure model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..confidence import (
    ComponentKind,
    ConfidenceAggregator,
    ConfidenceComponent,
    ConfidenceInput,
    ConfidencePurpose,
    EvidenceDomain,
)
from ..contracts import PredictionType
from ..models import (
    AfterheatModel,
    AfterheatPredictionContext,
    BoostModel,
    BoostPredictionContext,
    ForecastModel,
    ForecastPredictionContext,
    HeatLossModel,
    HeatLossPredictionContext,
    HeatRateModel,
    HeatRatePredictionContext,
    ManualCorrectionModel,
    ManualCorrectionPredictionContext,
    OutcomeModel,
    OutcomePredictionContext,
)


def build_zone_models(zone_id: str) -> dict[str, Any]:
    """Instantiate the per-zone model set (all optional models included)."""
    return {
        "heat_rate": HeatRateModel(zone_id),
        "heat_loss": HeatLossModel(zone_id),
        "afterheat": AfterheatModel(zone_id),
        "outcome": OutcomeModel(zone_id),
        "forecast": ForecastModel(zone_id),
        "boost": BoostModel(zone_id),
        "manual_correction": ManualCorrectionModel(zone_id),
    }


_PREDICT_CONTEXT = {
    "heat_rate": HeatRatePredictionContext,
    "heat_loss": HeatLossPredictionContext,
    "afterheat": AfterheatPredictionContext,
    "outcome": OutcomePredictionContext,
    "forecast": ForecastPredictionContext,
    "boost": BoostPredictionContext,
    "manual_correction": ManualCorrectionPredictionContext,
}

_COMPONENT_KIND = {
    "heat_rate": ComponentKind.HEAT_RATE,
    "heat_loss": ComponentKind.HEAT_LOSS,
    "afterheat": ComponentKind.AFTERHEAT,
    "outcome": ComponentKind.OUTCOME,
    "forecast": ComponentKind.FORECAST,
    "boost": ComponentKind.BOOST,
    "manual_correction": ComponentKind.MANUAL_CORRECTION,
}

_COMPONENT_DOMAINS = {
    "heat_rate": (EvidenceDomain.THERMAL_TRAJECTORY,),
    "heat_loss": (EvidenceDomain.THERMAL_TRAJECTORY,),
    "afterheat": (EvidenceDomain.THERMAL_TRAJECTORY,),
    "outcome": (EvidenceDomain.OUTCOME_HISTORY,),
    "forecast": (EvidenceDomain.FORECAST, EvidenceDomain.WEATHER),
    "boost": (EvidenceDomain.THERMAL_TRAJECTORY, EvidenceDomain.CONTROLLER_EVENTS),
    "manual_correction": (EvidenceDomain.USER_CORRECTION, EvidenceDomain.CONTROLLER_EVENTS),
}


@dataclass(frozen=True)
class OrchestrationResult:
    predictions: Mapping[PredictionType, Any]
    confidence_results: Mapping[str, Any]
    model_errors: tuple[str, ...]


class ModelOrchestrator:
    def __init__(self, zone_id: str, *, models: Optional[Mapping[str, Any]] = None,
                 aggregator: Optional[ConfidenceAggregator] = None) -> None:
        self._zone = zone_id
        self._models = dict(models) if models is not None else build_zone_models(zone_id)
        self._agg = aggregator or ConfidenceAggregator()
        self._errors: list[str] = []

    @property
    def models(self) -> Mapping[str, Any]:
        return self._models

    def apply_update(self, model_name: str, episode: Any, context: Any = None) -> Optional[bool]:
        """Eligibility-gated, isolated update for already-materialised evidence."""
        model = self._models.get(model_name)
        if model is None:
            return None
        try:
            res = model.update(episode, context) if context is not None else model.update(episode)
            return bool(res.accepted)
        except Exception as err:  # learning failure must never propagate
            self._errors.append(f"{model_name}:update:{type(err).__name__}")
            return False

    def collect_predictions(self) -> dict[PredictionType, Any]:
        out: dict[PredictionType, Any] = {}
        for name, model in self._models.items():
            ctx_cls = _PREDICT_CONTEXT.get(name)
            try:
                pred = model.predict(ctx_cls() if ctx_cls else None)
            except Exception as err:
                self._errors.append(f"{name}:predict:{type(err).__name__}")
                continue
            out[pred.prediction_type] = pred
        return out

    def collect_confidence_components(self) -> dict[str, ConfidenceComponent]:
        comps: dict[str, ConfidenceComponent] = {}
        for name, model in self._models.items():
            try:
                contrib = model.confidence()
            except Exception as err:
                self._errors.append(f"{name}:confidence:{type(err).__name__}")
                continue
            comps[name] = ConfidenceComponent(
                kind=_COMPONENT_KIND[name], contribution=contrib,
                evidence_domains=_COMPONENT_DOMAINS[name])
        return comps

    def aggregate(self, purposes: Mapping[ConfidencePurpose, tuple[str, ...]]) -> dict[str, Any]:
        """Run the aggregator once per requested purpose from model components."""
        comps = self.collect_confidence_components()
        results: dict[str, Any] = {}
        for purpose, component_names in purposes.items():
            components = tuple(comps[n] for n in component_names if n in comps)
            try:
                results[purpose.value] = self._agg.aggregate(
                    ConfidenceInput(purpose=purpose, components=components))
            except Exception as err:
                self._errors.append(f"confidence:{purpose.value}:{type(err).__name__}")
        return results

    def run(self, purposes: Optional[Mapping[ConfidencePurpose, tuple[str, ...]]] = None
            ) -> OrchestrationResult:
        self._errors = []
        predictions = self.collect_predictions()
        if purposes is None:
            purposes = {
                ConfidencePurpose.HEAT_RATE: ("heat_rate",),
                ConfidencePurpose.PREHEAT: ("heat_rate", "heat_loss", "afterheat", "outcome"),
                ConfidencePurpose.AFTERHEAT: ("afterheat",),
                ConfidencePurpose.OUTCOME_QUALITY: ("outcome",),
                ConfidencePurpose.FORECAST: ("forecast",),
                ConfidencePurpose.BOOST: ("boost", "outcome"),
                ConfidencePurpose.MANUAL_CORRECTION: ("manual_correction",),
            }
        confidence = self.aggregate(purposes)
        return OrchestrationResult(predictions=predictions, confidence_results=confidence,
                                   model_errors=tuple(self._errors))

    # -- state ----------------------------------------------------------

    def serialize_models(self) -> dict:
        out = {}
        for name, model in self._models.items():
            try:
                out[name] = model.serialize_state()
            except Exception as err:
                self._errors.append(f"{name}:serialize:{type(err).__name__}")
        return out

    def restore_models(self, data: Mapping[str, Any]) -> tuple[str, ...]:
        errors: list[str] = []
        for name, state in data.items():
            model = self._models.get(name)
            if model is None:
                continue
            try:
                model.deserialize_state(state)
            except Exception as err:  # isolate a corrupted segment
                errors.append(f"{name}:restore:{type(err).__name__}")
        return tuple(errors)
