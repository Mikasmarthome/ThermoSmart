"""Phase D2-C1a — outdoor-temperature freshness observability (pure, no HA, no I/O).

Exposes what is KNOWN about the recency of the live outdoor-temperature reading,
without ever inventing certainty. This is strictly observational: it feeds no
prediction, TPI, Preheat, regime or control path, and it does NOT change D2-B
learning acceptance (which keeps its own ``OutdoorEvidence.freshness_known = False``
policy from D2-A).

Key distinction the policy insists on:
  * A Home Assistant *state* timestamp (``state.last_updated`` / ``last_changed``)
    records when the state object was written — NOT when the sensor physically
    measured. A numeric value can persist unchanged across a dead sensor, and HA
    may re-write a state without a new measurement. So an HA state timestamp
    NEVER establishes physical freshness.
  * A *verified measurement* timestamp would come from the sensor's own device
    clock / measurement metadata. No source available to ThermoSmart provides one
    today, so that verdict is reserved and never produced here.

Therefore ``DataQuality.OK`` is NOT treated as proof of freshness, and when
trustworthy freshness cannot be established the verdict is ``UNKNOWN``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OutdoorFreshness(Enum):
    """What is known about the recency of the current outdoor reading."""

    UNKNOWN = "unknown"                       # no usable timestamp / no reading
    HA_STATE_TIMESTAMP = "ha_state_timestamp"  # only an HA state ts — NOT verified fresh
    VERIFIED_MEASUREMENT = "verified_measurement"  # reserved; never produced today

    @property
    def is_verified_fresh(self) -> bool:
        """The ONLY gate a future phase may use to treat a reading as fresh. False for
        both UNKNOWN and HA_STATE_TIMESTAMP — an HA state timestamp is not a verified
        physical measurement time, so freshness stays unproven."""
        return self is OutdoorFreshness.VERIFIED_MEASUREMENT


@dataclass(frozen=True)
class OutdoorFreshnessReport:
    """Privacy-safe, additive freshness descriptor for the current outdoor reading.

    ``ha_state_last_updated`` is the HA state timestamp when known (provenance, not a
    freshness proof). ``measurement_timestamp`` is a verified physical measurement time
    — always ``None`` today (no source provides one). ``freshness`` is the honest
    verdict; ``is_verified_fresh`` is always False until a verified source exists."""

    source: Optional[str]
    freshness: OutdoorFreshness
    ha_state_last_updated: Optional[str] = None
    measurement_timestamp: Optional[str] = None

    @property
    def is_verified_fresh(self) -> bool:
        return self.freshness.is_verified_fresh


def classify_outdoor_freshness(
    source: Optional[str],
    ha_state_last_updated: Optional[str] = None,
    measurement_timestamp: Optional[str] = None,
) -> OutdoorFreshnessReport:
    """Pure classifier. Never raises; never infers a fresh measurement from an
    unchanged numeric value or from ``DataQuality.OK``.

      * a verified physical ``measurement_timestamp`` -> VERIFIED_MEASUREMENT
        (reserved: no current source supplies one);
      * else an HA ``state.last_updated`` string present -> HA_STATE_TIMESTAMP
        (present but NOT verified fresh — recency stays unproven);
      * else (no reading, missing/unavailable sensor, or no timestamp) -> UNKNOWN.

    A source switch is reflected only by a changed ``source`` between reports; this
    function makes no cross-call inference."""
    src = source if isinstance(source, str) and source else None
    meas = measurement_timestamp if isinstance(measurement_timestamp, str) \
        and measurement_timestamp else None
    ha = ha_state_last_updated if isinstance(ha_state_last_updated, str) \
        and ha_state_last_updated else None
    if meas is not None:
        verdict = OutdoorFreshness.VERIFIED_MEASUREMENT
    elif ha is not None and src is not None:
        verdict = OutdoorFreshness.HA_STATE_TIMESTAMP
    else:
        verdict = OutdoorFreshness.UNKNOWN
    return OutdoorFreshnessReport(
        source=src, freshness=verdict,
        ha_state_last_updated=ha, measurement_timestamp=meas)
