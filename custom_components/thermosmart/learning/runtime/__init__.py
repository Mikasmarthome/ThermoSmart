"""LE 2.0 passive shadow runtime foundation (Phase 17).

Pure-core, injected-dependency runtime: capture, evidence materialisation, model
orchestration, shadow comparison and persistence policy. No control effect, no HA
imports in the pure core; the async lifecycle uses an injected store + clock.
"""
from .capture import (
    CaptureCoordinator,
    CapturedDecision,
    CapturedSnapshot,
    CommandLedger,
    CommandStatus,
    CommandType,
    ControllerDecisionInput,
    DecisionType,
    ManualCorrectionObservation,
    RuntimeCycleInput,
    ScheduleTarget,
    SentCommand,
    make_decision_id,
)
from .evidence import EvidenceItem, EvidenceMaterializer, MaterializedEvidence
from .pipeline import CompletedEpisode, PipelineCycleResult, RuntimePipeline
from .prediction_ledger import PredictionSnapshot, PredictionSnapshotLedger
from .orchestration import ModelOrchestrator, OrchestrationResult, build_zone_models
from .persistence import (
    AsyncStore,
    PersistenceOrchestrator,
    SavePolicy,
    SaveTrigger,
)
from .shadow import (
    SHADOW_SCHEMA_VERSION,
    ComparisonType,
    PreheatParameters,
    PreheatPlan,
    ShadowComparison,
    ShadowMetrics,
    ShadowOrchestrator,
    ShadowOutcome,
    ShadowPrediction,
    ShadowReason,
    ShadowStatus,
    compute_preheat_plan,
)
from .lifecycle import (
    RUNTIME_SCHEMA_VERSION,
    CoordinatorBridge,
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
    RuntimeCycleResult,
    RuntimeError,
    RuntimeHealth,
    RuntimeWarning,
)

__all__ = [
    "CaptureCoordinator", "CapturedDecision", "CapturedSnapshot", "CommandLedger",
    "CommandStatus", "CommandType", "ControllerDecisionInput", "DecisionType",
    "ManualCorrectionObservation", "RuntimeCycleInput", "ScheduleTarget", "SentCommand",
    "make_decision_id",
    "EvidenceItem", "EvidenceMaterializer", "MaterializedEvidence",
    "CompletedEpisode", "PipelineCycleResult", "RuntimePipeline",
    "PredictionSnapshot", "PredictionSnapshotLedger",
    "ModelOrchestrator", "OrchestrationResult", "build_zone_models",
    "AsyncStore", "PersistenceOrchestrator", "SavePolicy", "SaveTrigger",
    "SHADOW_SCHEMA_VERSION", "ComparisonType", "PreheatParameters", "PreheatPlan",
    "ShadowComparison", "ShadowMetrics", "ShadowOrchestrator", "ShadowOutcome",
    "ShadowPrediction", "ShadowReason", "ShadowStatus", "compute_preheat_plan",
    "RUNTIME_SCHEMA_VERSION", "CoordinatorBridge", "LearningRuntime",
    "LearningRuntimeConfig", "LearningRuntimeMode", "RuntimeCycleResult", "RuntimeError",
    "RuntimeHealth", "RuntimeWarning",
]
