"""Pure-core Learning runtime foundation.

Injected-dependency runtime: capture, evidence materialisation, model
orchestration, shadow comparison and persistence policy. This core is
prediction/observation-only — it never dispatches Home Assistant service
calls and has no HA imports; the async lifecycle uses an injected store +
clock. The separate Home Assistant integration layer
(``learning/runtime/ha_integration.py``) builds on this core and
additionally exposes bounded adaptive-control adjustments — the coordinator
decides per cycle whether those adjustments are ignored, shadowed, or
applied (see ``ControlAdaptationMode`` in ``coordinator.py``).
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
