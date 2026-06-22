"""LE 2.0 concrete learning models (pure Python).

Phase 8 contains only the HeatRateModel. Models depend on no Home Assistant,
storage, coordinator or other concrete model; they consume episodes and features
and produce typed predictions.
"""
from .heat_rate import (
    MODEL_VERSION,
    PARAMETER_VERSION,
    HeatRateBucket,
    HeatRateDiagnostics,
    HeatRateModel,
    HeatRateParameters,
    HeatRatePredictionContext,
    HeatRateRebuildItem,
    HeatRateRejection,
    HeatRateSample,
    HeatRateState,
    HeatRateUpdateContext,
    heat_rate_model_definition,
)
from .heat_loss import (
    HeatLossBucket,
    HeatLossDiagnostics,
    HeatLossModel,
    HeatLossParameters,
    HeatLossPredictionContext,
    HeatLossRebuildItem,
    HeatLossRejection,
    HeatLossSample,
    HeatLossState,
    HeatLossUpdateContext,
    heat_loss_model_definition,
)

__all__ = [
    "MODEL_VERSION", "PARAMETER_VERSION",
    "HeatRateBucket", "HeatRateDiagnostics", "HeatRateModel", "HeatRateParameters",
    "HeatRatePredictionContext", "HeatRateRebuildItem", "HeatRateRejection",
    "HeatRateSample", "HeatRateState", "HeatRateUpdateContext",
    "heat_rate_model_definition",
    "HeatLossBucket", "HeatLossDiagnostics", "HeatLossModel", "HeatLossParameters",
    "HeatLossPredictionContext", "HeatLossRebuildItem", "HeatLossRejection",
    "HeatLossSample", "HeatLossState", "HeatLossUpdateContext",
    "heat_loss_model_definition",
]
