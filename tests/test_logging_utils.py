import json

from hermes_shim_http.logging_utils import format_log_event


def test_format_log_event_as_json():
    payload = format_log_event(event="cache_hit", request_id="abc", log_format="json", extra={"model": "claude-cli"})
    parsed = json.loads(payload)

    assert parsed["event"] == "cache_hit"
    assert parsed["request_id"] == "abc"
    assert parsed["model"] == "claude-cli"


def test_format_log_event_as_text():
    payload = format_log_event(event="stream_end", request_id="xyz", log_format="text", extra={"status": "ok"})

    assert "event=stream_end" in payload
    assert "request_id=xyz" in payload
    assert "status=ok" in payload
