"""Privacy primitives for LE 2.0 diagnostics/export orchestration (pure Python).

Provides the field-level privacy classification, a namespaced pseudonymizer, a
versioned privacy policy (quantisation + time strategy) and a defensive privacy
scanner. No HA / storage / coordinator dependency; no file I/O. The salt/secret
is injected and never exported.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import re
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

PRIVACY_POLICY_VERSION = 1


class ExportPrivacyClass(Enum):
    """Field-level export-privacy classification for the two exports.

    Distinct from the registry's raw-track ``PrivacyClass`` (LOW/BEHAVIORAL):
    this enum governs per-field/per-data-class export release only.
    """
    PUBLIC_SAFE = "public_safe"
    SUPPORT_SAFE = "support_safe"
    LEARNING_SAFE = "learning_safe"
    SENSITIVE = "sensitive"
    FORBIDDEN = "forbidden"


class PseudonymNamespace(Enum):
    ZONE = "zone"
    MODEL = "model"
    SOURCE = "source"
    BUCKET = "bucket"


# -- pseudonymizer ------------------------------------------------------------

class Pseudonymizer:
    """Deterministic, namespaced, non-reversible pseudonymiser.

    With an injected ``salt`` the mapping is stable across exports; without one a
    random per-instance salt is generated, so pseudonyms stay deterministic
    *within a single export run* but are not globally recognisable. The salt is
    never exported and the original id cannot be reconstructed from the pseudonym.
    """

    def __init__(self, salt: Optional[bytes] = None, *, length: int = 12) -> None:
        if salt is None:
            self._salt = secrets.token_bytes(32)
            self._stable = False
        else:
            self._salt = salt if isinstance(salt, bytes) else str(salt).encode("utf-8")
            self._stable = True
        if length < 6 or length > 40:
            raise ValueError("pseudonym length must be within [6,40]")
        self._length = length

    @property
    def stable(self) -> bool:
        return self._stable

    def pseudonymize(self, namespace: PseudonymNamespace, value: str) -> str:
        if value is None:
            raise ValueError("cannot pseudonymize None")
        msg = f"{namespace.value}:{value}".encode("utf-8")
        digest = hmac.new(self._salt, msg, hashlib.sha256).hexdigest()
        return f"{namespace.value[:1]}_{digest[:self._length]}"


# -- quantisation -------------------------------------------------------------

@dataclass(frozen=True)
class QuantizationPolicy:
    """Versioned numeric quantisation for the learning export."""
    temperature_step_c: float = 0.1
    delta_step_c: float = 0.1
    rate_step_c_per_h: float = 0.05
    duration_step_s: float = 60.0
    score_step: float = 0.05
    support_round_digits: int = 4

    def quantize(self, value: float, step: float) -> float:
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("cannot quantize a non-finite value")
        if step <= 0:
            return float(value)
        # round half to even on the step grid; 0.0 stays 0.0, sign preserved
        q = round(float(value) / step) * step
        # avoid -0.0 and floating dust
        result = round(q, 9)
        return 0.0 if result == 0.0 else result


# -- time privacy -------------------------------------------------------------

class TimePrivacyStrategy(Enum):
    AGE_COARSE = "age_coarse"          # support: coarse age only
    DAY_BUCKET_SHIFTED = "day_bucket_shifted"  # learning: day bucket + export-wide shift
    RELATIVE_OFFSET = "relative_offset"


@dataclass(frozen=True)
class PrivacyPolicy:
    """Versioned privacy policy bundling quantisation and time strategy."""
    version: int = PRIVACY_POLICY_VERSION
    quantization: QuantizationPolicy = field(default_factory=QuantizationPolicy)
    support_time_strategy: TimePrivacyStrategy = TimePrivacyStrategy.AGE_COARSE
    learning_time_strategy: TimePrivacyStrategy = TimePrivacyStrategy.DAY_BUCKET_SHIFTED
    age_bucket_hours: float = 6.0

    def coarse_age_hours(self, generated_at_iso: str, ts_iso: Optional[str]) -> Optional[float]:
        if not ts_iso:
            return None
        from datetime import datetime
        try:
            gen = datetime.fromisoformat(generated_at_iso)
            ts = datetime.fromisoformat(ts_iso)
        except (TypeError, ValueError):
            return None
        hours = (gen - ts).total_seconds() / 3600.0
        if hours < 0:
            hours = 0.0
        # bucket to age_bucket_hours
        bucket = self.age_bucket_hours
        return round((hours // bucket) * bucket, 3) if bucket > 0 else round(hours, 3)


# -- privacy scanner ----------------------------------------------------------

_FORBIDDEN_KEY_SUBSTRINGS = (
    "entity_id", "entry_id", "device_id", "decision_id", "episode_id", "event_id",
    "evaluation_id", "user_id", "person", "email", "latitude", "longitude", "address",
    "hostname", "ip_address", "ipaddr", "source_episode_id", "correction_event_id",
    "source_decision_id", "trv_binding_id", "learning_zone_id",
)

_ENTITY_ID_RE = re.compile(
    r"\b(climate|sensor|binary_sensor|switch|number|select|button|water_heater|"
    r"input_number|input_boolean|person|device_tracker)\.[a-z0-9_]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_PATH_RE = re.compile(r"(/config/|/home/|/usr/|/etc/|[A-Za-z]:\\\\|[A-Za-z]:/)")
_COORD_RE = re.compile(r"\b-?\d{1,3}\.\d{4,}\b")  # high-precision decimal -> possible coordinate


@dataclass(frozen=True)
class PrivacyViolation:
    path: str
    kind: str
    detail: str


def _key_forbidden(key: str) -> Optional[str]:
    low = key.lower()
    for sub in _FORBIDDEN_KEY_SUBSTRINGS:
        if sub in low:
            return sub
    return None


def _value_violation(value: str) -> Optional[tuple[str, str]]:
    if _ENTITY_ID_RE.search(value):
        return ("entity_id_pattern", value[:40])
    if _EMAIL_RE.search(value):
        return ("email", "<redacted>")
    if _IPV4_RE.search(value):
        return ("ip_address", "<redacted>")
    if _PATH_RE.search(value):
        return ("local_path", "<redacted>")
    return None


def scan_payload(payload: Any, *, path: str = "$") -> list[PrivacyViolation]:
    """Defensive second-barrier scan for forbidden keys and value patterns."""
    out: list[PrivacyViolation] = []
    if isinstance(payload, Mapping):
        for k, v in payload.items():
            kp = f"{path}.{k}"
            if isinstance(k, str):
                sub = _key_forbidden(k)
                if sub is not None:
                    out.append(PrivacyViolation(kp, "forbidden_key", sub))
            out.extend(scan_payload(v, path=kp))
    elif isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            out.extend(scan_payload(v, path=f"{path}[{i}]"))
    elif isinstance(payload, str):
        hit = _value_violation(payload)
        if hit is not None:
            out.append(PrivacyViolation(path, hit[0], hit[1]))
    elif isinstance(payload, float):
        if not math.isfinite(payload):
            out.append(PrivacyViolation(path, "non_finite", repr(payload)))
    return out
