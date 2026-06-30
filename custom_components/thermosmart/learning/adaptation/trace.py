"""Serialization helpers for AdaptationTrace.

Provides export-safe conversion functions. No HA imports; no runtime state.
"""
from __future__ import annotations
from typing import Sequence

from .contracts import AdaptationLifecycle, AdaptationTrace


def trace_to_dict(trace: AdaptationTrace) -> dict:
    """Convert a trace to a deterministic, JSON-serializable dict."""
    return trace.to_dict()


def filter_by_lifecycle(
    traces: Sequence[AdaptationTrace],
    *states: AdaptationLifecycle,
) -> list[AdaptationTrace]:
    """Return only traces whose lifecycle is in *states."""
    return [t for t in traces if t.lifecycle in states]


def traces_for_export(
    traces: Sequence[AdaptationTrace],
    *,
    include_rejected: bool = False,
) -> list[dict]:
    """Build export-safe list of trace dicts for Research/Support Export.

    Args:
        traces: all candidate traces for a zone.
        include_rejected: include REJECTED-lifecycle traces (useful for
            debugging; off by default for research export).
    """
    result = []
    for trace in traces:
        if trace.lifecycle is AdaptationLifecycle.REJECTED and not include_rejected:
            continue
        result.append(trace_to_dict(trace))
    return result
