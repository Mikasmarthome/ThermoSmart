"""Confidence display adapter (Phase 19A-B, pure Python).

Produces the legacy confidence display values (combined % and the per-track
breakdown attribute shape) purely from Learning ``ConfidenceResult`` objects — so the
existing sensor/attribute semantics are preserved WITHOUT a second confidence
formula. The control/prediction gates already consume Learning ``ConfidenceResult``
directly; this only feeds *display*. When no Learning confidence exists, a neutral,
safe fallback (0.0 / empty) is returned — never the legacy engine.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ..confidence import ConfidencePurpose


def _val(last_confidence: Mapping[str, Any], purpose: ConfidencePurpose) -> Optional[float]:
    cr = last_confidence.get(purpose.value)
    if cr is None:
        return None
    v = getattr(cr, "value", None)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def combined_confidence(last_confidence: Mapping[str, Any]) -> float:
    """Display rollup: mean of the available Learning per-purpose confidences ([0,1]).

    This is a display aggregation over the aggregator's own outputs, not a new gate
    formula. Returns 0.0 when no confidence is available (neutral/safe).
    """
    if not last_confidence:
        return 0.0
    vals = [v for p in ConfidencePurpose
            for v in (_val(last_confidence, p),) if v is not None]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)


def confidence_breakdown(last_confidence: Mapping[str, Any], *,
                         model_update_counts: Optional[Mapping[str, int]] = None) -> dict:
    """Legacy-compatible breakdown attribute shape, sourced from Learning confidences."""
    counts = dict(model_update_counts or {})
    def pct(purpose):
        v = _val(last_confidence, purpose)
        return round((v or 0.0) * 100, 1)
    return {
        "room_patterns_%": pct(ConfidencePurpose.HEAT_RATE),
        "trv_efficiency_%": pct(ConfidencePurpose.OUTCOME_QUALITY),
        "window_cooling_%": pct(ConfidencePurpose.HEAT_LOSS),
        "forecast_confidence_%": pct(ConfidencePurpose.FORECAST),
        "total_observations": int(sum(counts.values())),
        "trv_observations": int(counts.get("outcome", 0)),
        "window_events": int(counts.get("heat_loss", 0)),
    }
