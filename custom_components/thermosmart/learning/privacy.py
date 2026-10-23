"""Defensive privacy scanner for Learning diagnostics payloads (pure Python).

Second barrier for the Home Assistant diagnostics export: flags forbidden keys
(entity / device / person identifiers, coordinates, ...) and value patterns
(entity ids, e-mail addresses, IP addresses, local paths, non-finite numbers).
No HA / storage / coordinator dependency; no file I/O.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional


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
