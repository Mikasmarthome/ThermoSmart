"""Privacy scanner: forbidden keys and value patterns."""
from __future__ import annotations

from custom_components.thermosmart.learning.privacy import scan_payload


class TestScanner:
    def test_entity_id_in_value(self):
        v = scan_payload({"x": "climate.living_room"})
        assert any(s.kind == "entity_id_pattern" for s in v)

    def test_forbidden_key(self):
        v = scan_payload({"entity_id": "abc"})
        assert any(s.kind == "forbidden_key" for s in v)

    def test_decision_id_key(self):
        assert any(s.kind == "forbidden_key" for s in scan_payload({"decision_id": "d1"}))

    def test_email(self):
        assert any(s.kind == "email" for s in scan_payload({"x": "a@b.com"}))

    def test_ip(self):
        assert any(s.kind == "ip_address" for s in scan_payload({"x": "192.168.0.1"}))

    def test_path(self):
        assert any(s.kind == "local_path" for s in scan_payload({"x": "/config/secrets.yaml"}))

    def test_nested_forbidden(self):
        v = scan_payload({"a": {"b": [{"event_id": "e"}]}})
        assert any(s.kind == "forbidden_key" for s in v)

    def test_nan_flagged(self):
        assert any(s.kind == "non_finite" for s in scan_payload({"x": float("nan")}))

    def test_clean_payload_no_violations(self):
        assert scan_payload({"value": 0.5, "level": "high", "count": 3}) == []


