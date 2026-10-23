"""Production adapter: validated read of the runtime's previous-cycle predictions.

The coordinator reads the stored ``Prediction`` objects (keyed by ``PredictionType``)
only through ``validated_prediction_read``, with explicit, non-implicit semantics:

* ``BOOST_FACTOR`` carries an *additive* °C offset (``celsius_offset``), NOT a
  multiplier — see boost.py ``values={"boost_factor": rec.boost_offset_c}``.
* preheat is a *duration in minutes* (``preheat_start`` feature), distinct from the
  schedule comfort *time*.
* early cutoff is expressed as the expected *residual rise* (°C), not a clock time.
* ``FORECAST_BIAS`` carries a signed °C error bias (positive = reality warmer than forecast);
  ``FORECAST_TRUST`` is a dimensionless reliability score in [0, 1] — semantically distinct,
  never mixed.

No HA imports; pure mapping over the runtime's already-computed predictions.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ..confidence import ConfidencePurpose
from ..contracts import PredictionType

# feature key -> (PredictionType, value-key inside Prediction.values, unit, confidence purpose)
_MAP: tuple[tuple[str, PredictionType, str, str, Optional[ConfidencePurpose]], ...] = (
    ("boost_offset", PredictionType.BOOST_FACTOR, "boost_factor", "celsius_offset",
     ConfidencePurpose.BOOST),
    ("preheat_start", PredictionType.PREHEAT_MINUTES, "preheat_minutes", "minutes",
     ConfidencePurpose.PREHEAT),
    ("early_cutoff", PredictionType.RECOMMENDED_EARLY_CUTOFF, "expected_residual_rise",
     "celsius", ConfidencePurpose.EARLY_CUTOFF),
    ("heat_rate", PredictionType.HEAT_RATE, "heat_rate", "C/h", ConfidencePurpose.HEAT_RATE),
    ("heat_loss", PredictionType.HEAT_LOSS_RATE, "heat_loss_rate", "C/h",
     ConfidencePurpose.HEAT_LOSS),
    ("afterheat", PredictionType.EXPECTED_OVERSHOOT, "expected_overshoot", "celsius",
     ConfidencePurpose.AFTERHEAT),
    ("forecast_bias", PredictionType.FORECAST_BIAS, "forecast_bias", "celsius",
     ConfidencePurpose.FORECAST),
    ("forecast_trust", PredictionType.FORECAST_TRUST, "forecast_trust", "score",
     ConfidencePurpose.FORECAST),
    ("manual_preference_bias", PredictionType.MANUAL_PREFERENCE_BIAS, "preference_bias",
     "celsius", ConfidencePurpose.MANUAL_CORRECTION),
)


class PredictionReadReject(str):
    """Reason a previous-cycle prediction was not accepted as authoritative."""


# why an authoritative read was rejected (caller must then use deterministic baseline)
READ_OK = "ok"
READ_MISSING = "missing"
READ_WRONG_ZONE = "wrong_zone"
READ_WRONG_TYPE = "wrong_type"
READ_UNIT_MISMATCH = "unit_mismatch"
READ_VERSION_MISMATCH = "version_mismatch"
READ_STALE = "stale"
READ_SUPERSEDED = "superseded"
READ_NONFINITE = "nonfinite"


def validated_prediction_read(last_predictions: Mapping[Any, Any], feature: str, *,
                              zone_id: str, prediction_zone_id: Optional[str],
                              expected_unit: str,
                              parameter_version: Optional[int] = None,
                              accepted_parameter_version: Optional[int] = None,
                              age_seconds: Optional[float] = None,
                              max_age_seconds: Optional[float] = None,
                              prediction_decision_id: Optional[str] = None,
                              current_decision_id: Optional[str] = None,
                              superseded: bool = False) -> tuple[Optional[float], str]:
    """Rule #4 gate: only accept a previous-cycle prediction as authoritative when zone,
    PredictionType, unit, version, freshness and supersession all match. Any mismatch
    returns (None, reason) so the caller falls back to the deterministic baseline.

    Returns (value, "ok") on success, else (None, <reason>).
    """
    spec = next((m for m in _MAP if m[0] == feature), None)
    if spec is None:
        return None, READ_WRONG_TYPE
    _, ptype, value_key, unit, _purpose = spec
    pred = last_predictions.get(ptype)
    if pred is None:
        return None, READ_MISSING
    if prediction_zone_id is not None and prediction_zone_id != zone_id:
        return None, READ_WRONG_ZONE
    if expected_unit and unit != expected_unit:
        return None, READ_UNIT_MISMATCH
    if (accepted_parameter_version is not None and parameter_version is not None
            and parameter_version != accepted_parameter_version):
        return None, READ_VERSION_MISMATCH
    if (max_age_seconds is not None and age_seconds is not None
            and age_seconds > max_age_seconds):
        return None, READ_STALE
    if superseded:
        return None, READ_SUPERSEDED
    if (prediction_decision_id is not None and current_decision_id is not None
            and prediction_decision_id != current_decision_id):
        return None, READ_SUPERSEDED
    values = getattr(pred, "values", None) or {}
    if value_key not in values:
        return None, READ_MISSING
    raw = values.get(value_key)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, READ_NONFINITE
    import math as _math
    if not _math.isfinite(value):
        return None, READ_NONFINITE
    return value, READ_OK
