"""Shared builders for LE 2.0 ConfidenceAggregator tests."""
from __future__ import annotations

from typing import Optional, Sequence

from custom_components.thermosmart.learning.confidence import (
    ComponentKind,
    ConfidenceComponent,
    ConfidenceStatus,
    EvidenceDomain,
)
from custom_components.thermosmart.learning.contracts import ConfidenceContribution


def contribution(value: float, *, evidence_count: int = 10, reliability: float = 0.9,
                 domains: Sequence[EvidenceDomain] = (), prior_fraction: float = 0.0,
                 fallback_used: bool = False, reasons: Sequence[str] = (),
                 freshness: Optional[float] = None) -> ConfidenceContribution:
    return ConfidenceContribution(
        value=value, evidence_count=evidence_count, reliability=reliability,
        evidence_domains=tuple(d.value for d in domains), prior_fraction=prior_fraction,
        fallback_used=fallback_used, reasons=tuple(reasons), freshness=freshness)


def component(kind: ComponentKind, value: float, *, status: ConfidenceStatus = ConfidenceStatus.HEALTHY,
              is_required: Optional[bool] = None, domains: Sequence[EvidenceDomain] = (),
              freshness: Optional[float] = None, reliability: Optional[float] = None,
              evidence_count: int = 10, prior_fraction: float = 0.0,
              fallback_used: bool = False, reasons: Sequence[str] = ()) -> ConfidenceComponent:
    return ConfidenceComponent(
        kind=kind,
        contribution=contribution(value, evidence_count=evidence_count,
                                  reliability=reliability if reliability is not None else 0.9,
                                  domains=domains, prior_fraction=prior_fraction,
                                  fallback_used=fallback_used, reasons=reasons),
        status=status, is_required=is_required, evidence_domains=tuple(domains),
        freshness=freshness, reliability=reliability)
