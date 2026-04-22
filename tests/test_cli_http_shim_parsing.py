import json

import pytest

from hermes_shim_http.parsing import (
    ClaudeStreamJsonParser,
    IncrementalToolCallParser,
    parse_claude_stream_json,
    parse_cli_output,
)


def test_parse_plain_text_response():
    parsed = parse_cli_output("Hello from the shim")

    assert parsed.content == "Hello from the shim"
    assert parsed.tool_calls == []


def test_parse_single_tool_call_block():
    parsed = parse_cli_output(
        '<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"/tmp/demo.txt\\"}"}}</tool_call>'
    )

    assert parsed.content == ""
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(parsed.tool_calls[0]["function"]["arguments"]) == {"path": "/tmp/demo.txt"}


def test_parse_tool_call_allows_raw_json_object_arguments():
    parsed = parse_cli_output(
        '<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":{"path":"README.md"}}}</tool_call>'
    )

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(parsed.tool_calls[0]["function"]["arguments"]) == {"path": "README.md"}


def test_parse_tool_call_requires_matching_nonce_when_configured():
    parsed = parse_cli_output(
        '<tool_call nonce="nonce-123">{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{}"}}</tool_call>',
        expected_tool_call_nonce="nonce-123",
    )
    rejected = parse_cli_output(
        '<tool_call nonce="wrong">{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{}"}}</tool_call>',
        expected_tool_call_nonce="nonce-123",
    )

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "read_file"
    assert rejected.tool_calls == []
    assert '<tool_call nonce="wrong">' in rejected.content


def test_parse_multiple_tool_call_blocks_and_visible_text_cleanup():
    parsed = parse_cli_output(
        "Need two actions.\n"
        '<tool_call>{"id":"call_1","type":"function","function":{"name":"search_files","arguments":"{\\"pattern\\":\\"TODO\\"}"}}</tool_call>\n'
        "After first call.\n"
        '<tool_call>{"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}</tool_call>'
    )

    assert parsed.content == "Need two actions.\nAfter first call."
    assert [tc["function"]["name"] for tc in parsed.tool_calls] == ["search_files", "read_file"]


def test_malformed_tool_call_emits_notice_and_does_not_leak_raw_block():
    parsed = parse_cli_output(
        'Before\n<tool_call>{"id":"oops","type":"function","function":{"name":"read_file","arguments":not-json}}</tool_call>\nAfter'
    )

    assert parsed.tool_calls == []
    assert "Before" in parsed.content
    assert "After" in parsed.content
    assert "⚠️ shim: dropped malformed tool_call" in parsed.content
    assert "reason=json_decode_error" in parsed.content
    assert "<tool_call>" not in parsed.content


def test_normalize_rejected_tool_call_emits_notice():
    parsed = parse_cli_output('<tool_call>{"id":"call_1","type":"function","function":{"arguments":"{}"}}</tool_call>')

    assert parsed.tool_calls == []
    assert "⚠️ shim: dropped malformed tool_call" in parsed.content
    assert "reason=normalize_rejected" in parsed.content


def test_malformed_tool_call_can_be_repaired_when_enabled(monkeypatch):
    pytest.importorskip("json_repair")
    monkeypatch.setenv("HERMES_SHIM_JSON_REPAIR_ENABLED", "1")

    parsed = parse_cli_output(
        '<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":{"path":"README.md",}}}</tool_call>'
    )

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "read_file"
    assert "reason=repaired_from_malformed" in parsed.content


def test_bare_tool_call_json_without_wrapper_remains_plain_text():
    raw = '{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'

    parsed = parse_cli_output(raw)

    assert parsed.tool_calls == []
    assert parsed.content == raw


def test_incremental_parser_streams_plain_text_chunks():
    parser = IncrementalToolCallParser()

    first = parser.feed("Hello")
    second = parser.feed(" world")
    final = parser.finalize()

    assert all(event.kind == "text" for event in first + second + final)
    assert "".join(event.text for event in first + second + final if event.text) == "Hello world"


def test_incremental_parser_flushes_long_plain_text_without_tag_hint():
    parser = IncrementalToolCallParser()

    big_chunk = "a" * 100_000
    events = parser.feed(big_chunk)
    text = "".join(event.text for event in events if event.text)

    # With tail-window cap, almost the whole buffer should flush immediately;
    # only a short residual (< len("</tool_call>")) may remain buffered.
    assert len(text) >= len(big_chunk) - len("</tool_call>")


def test_incremental_parser_preserves_partial_open_tag_prefix():
    parser = IncrementalToolCallParser()

    events = parser.feed("some output ending with <tool_cal")
    events_final = parser.finalize()

    streamed = "".join(event.text for event in events if event.text)
    # The "<tool_cal" prefix must NOT be flushed prematurely.
    assert "<tool_cal" not in streamed
    final_text = "".join(event.text for event in events_final if event.text)
    assert "<tool_cal" in (streamed + final_text)


def test_incremental_parser_hides_tool_call_wrapper_until_complete_block():
    parser = IncrementalToolCallParser()

    first = parser.feed("Before <tool")
    second = parser.feed(
        '_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}</tool_call> after'
    )
    final = parser.finalize()

    all_events = first + second + final
    rendered_text = "".join(event.text for event in all_events if event.text)
    tool_calls = [event.tool_call for event in all_events if event.tool_call]

    assert rendered_text == "Before  after"
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "read_file"


def test_incremental_parser_emits_notice_for_malformed_tool_call_without_raw_leak():
    parser = IncrementalToolCallParser()

    events = parser.feed("x <tool_call>{\"id\":\"oops\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":not-json}}</tool_call> y") + parser.finalize()
    rendered_text = "".join(event.text for event in events if event.text)
    tool_calls = [event.tool_call for event in events if event.tool_call]

    assert tool_calls == []
    assert "⚠️ shim: dropped malformed tool_call" in rendered_text
    assert "reason=json_decode_error" in rendered_text
    assert "<tool_call>" not in rendered_text


def test_incremental_parser_requires_matching_nonce_when_configured():
    parser = IncrementalToolCallParser(expected_tool_call_nonce="nonce-123")

    accepted = parser.feed('<tool_call nonce="nonce-123">{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{}"}}</tool_call>')
    rejected = parser.feed('<tool_call nonce="wrong">{"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{}"}}</tool_call>')
    final = parser.finalize()

    all_events = accepted + rejected + final
    tool_calls = [event.tool_call for event in all_events if event.tool_call]
    text = "".join(event.text or "" for event in all_events if event.kind == "text")

    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_1"
    assert '<tool_call nonce="wrong">' in text


def _stream_json(*events: dict) -> str:
    return "".join(json.dumps(ev) + "\n" for ev in events)


def test_claude_stream_parser_emits_text_deltas():
    parser = ClaudeStreamJsonParser()
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m1"}}},
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
    )
    events = parser.feed(blob) + parser.finalize()

    texts = [e.text for e in events if e.kind == "text"]
    assert "".join(texts) == "Hello world"
    assert not [e for e in events if e.kind == "tool_call"]


def test_claude_stream_parser_ignores_thinking_delta():
    parser = ClaudeStreamJsonParser()
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "secret"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
    )
    events = parser.feed(blob) + parser.finalize()

    assert events == []


def test_claude_stream_parser_does_not_emit_progress_text_for_thinking():
    parser = ClaudeStreamJsonParser(synthesize_progress=True)
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read_file"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path":"README.md"}'}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 1}},
    )
    events = parser.feed(blob) + parser.finalize()

    texts = [e.text for e in events if e.kind == "text"]
    tool_events = [e for e in events if e.kind == "tool_call"]
    assert texts == []
    assert len(tool_events) == 1
    assert tool_events[0].tool_call["function"]["name"] == "read_file"


def test_claude_stream_parser_never_emits_thinking_progress_within_a_turn():
    parser = ClaudeStreamJsonParser(synthesize_progress=True)
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "message_start"}},
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 1, "content_block": {"type": "thinking"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 1}},
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 2, "content_block": {"type": "thinking"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 2}},
    )
    events = parser.feed(blob) + parser.finalize()
    texts = [e.text for e in events if e.kind == "text"]
    assert texts == []


def test_claude_stream_parser_never_emits_thinking_progress_across_turns():
    parser = ClaudeStreamJsonParser(synthesize_progress=True)
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "message_start"}},
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 1, "content_block": {"type": "thinking"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 1}},
        {"type": "stream_event", "event": {"type": "message_start"}},
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
    )
    events = parser.feed(blob) + parser.finalize()
    texts = [e.text for e in events if e.kind == "text"]
    assert texts == []


def test_claude_stream_parser_assembles_tool_use():
    parser = ClaudeStreamJsonParser()
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0,
                                           "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read_file"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "input_json_delta", "partial_json": '{"pa'}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "input_json_delta", "partial_json": 'th":"README.md"}'}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
    )
    events = parser.feed(blob) + parser.finalize()

    tool_events = [e for e in events if e.kind == "tool_call"]
    assert len(tool_events) == 1
    tc = tool_events[0].tool_call
    assert tc["id"] == "toolu_1"
    assert tc["function"]["name"] == "read_file"
    assert json.loads(tc["function"]["arguments"]) == {"path": "README.md"}


def test_claude_stream_parser_handles_split_lines_across_feeds():
    parser = ClaudeStreamJsonParser()
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}},
    )
    mid = len(blob) // 2
    events = parser.feed(blob[:mid]) + parser.feed(blob[mid:]) + parser.finalize()
    texts = [e.text for e in events if e.kind == "text"]
    assert "".join(texts) == "hi"


def test_claude_stream_parser_ignores_malformed_json_lines():
    parser = ClaudeStreamJsonParser()
    blob = (
        "not-json\n"
        + json.dumps({"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}}) + "\n"
        + json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}}) + "\n"
    )
    events = parser.feed(blob) + parser.finalize()
    texts = [e.text for e in events if e.kind == "text"]
    assert "".join(texts) == "ok"


def test_claude_stream_parser_aggregate_fallback_when_no_stream_events():
    parser = ClaudeStreamJsonParser()
    blob = json.dumps({
        "type": "assistant",
        "message": {
            "id": "msg_1",
            "content": [
                {"type": "text", "text": "fallback text"},
                {"type": "tool_use", "id": "toolu_x", "name": "read_file", "input": {"path": "README.md"}},
            ],
        },
    }) + "\n"
    events = parser.feed(blob) + parser.finalize()
    texts = [e.text for e in events if e.kind == "text"]
    tool_events = [e for e in events if e.kind == "tool_call"]
    assert "".join(texts) == "fallback text"
    assert len(tool_events) == 1
    assert tool_events[0].tool_call["function"]["name"] == "read_file"


def test_claude_stream_parser_suppresses_aggregate_when_stream_events_present():
    parser = ClaudeStreamJsonParser()
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "streamed"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
        {"type": "assistant", "message": {"id": "m1", "content": [{"type": "text", "text": "streamed"}]}},
    )
    events = parser.feed(blob) + parser.finalize()
    texts = [e.text for e in events if e.kind == "text"]
    assert "".join(texts) == "streamed"


def test_parse_claude_stream_json_whole_blob():
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}},
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
    )
    parsed = parse_claude_stream_json(blob)
    assert parsed.content == "Hello"
    assert parsed.tool_calls == []


def test_parse_claude_stream_json_falls_back_to_plain_text_when_no_json():
    # CLI profile is claude but the CLI happened to emit raw text (e.g. mocked
    # run_cli_prompt, an unrecognised error path, or a legacy build). The shim
    # must still extract content and tool-call tags rather than returning empty.
    parsed = parse_claude_stream_json(
        "Hello plain text\n"
        '<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"/tmp/demo.txt\\"}"}}</tool_call>'
    )
    assert "Hello plain text" in parsed.content
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "read_file"


def test_claude_stream_parser_extracts_tool_call_tags_from_text_deltas():
    # Hermes-style tool invocation: server injects `<tool_call>...</tool_call>`
    # convention into the system prompt. Claude emits those tags inside text
    # blocks via stream-json text_delta events. The parser must surface them
    # as tool_call events, not raw text.
    parser = ClaudeStreamJsonParser()
    tag = (
        '<tool_call>{"id":"call_1","type":"function",'
        '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
        '</tool_call>'
    )
    # Split the tag across multiple text_deltas to exercise the incremental path.
    splits = [tag[:15], tag[15:60], tag[60:]]
    blob = _stream_json(
        {"type": "stream_event", "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}},
        *[
            {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}}}
            for piece in splits
        ],
        {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}},
    )
    events = parser.feed(blob) + parser.finalize()

    tool_events = [e for e in events if e.kind == "tool_call"]
    text_events = [e for e in events if e.kind == "text" and e.text]
    assert len(tool_events) == 1
    assert tool_events[0].tool_call["function"]["name"] == "read_file"
    # The raw <tool_call> tag bytes must not leak through as text.
    rendered = "".join(e.text for e in text_events)
    assert "<tool_call>" not in rendered
    assert "</tool_call>" not in rendered


def test_claude_stream_parser_extracts_tool_call_tags_from_aggregate_text():
    # Same tool-call-tag convention, but Claude emits only the aggregate
    # `assistant` snapshot (no stream_event text_deltas arrived).
    parser = ClaudeStreamJsonParser()
    tag = (
        'prefix '
        '<tool_call>{"id":"call_1","type":"function",'
        '"function":{"name":"search_files","arguments":"{\\"pattern\\":\\"TODO\\"}"}}'
        '</tool_call>'
        ' suffix'
    )
    blob = json.dumps({
        "type": "assistant",
        "message": {"id": "msg_1", "content": [{"type": "text", "text": tag}]},
    }) + "\n"
    events = parser.feed(blob) + parser.finalize()

    tool_events = [e for e in events if e.kind == "tool_call"]
    text_events = [e for e in events if e.kind == "text" and e.text]
    assert len(tool_events) == 1
    assert tool_events[0].tool_call["function"]["name"] == "search_files"
    rendered = "".join(e.text for e in text_events)
    assert "<tool_call>" not in rendered
    assert "prefix" in rendered and "suffix" in rendered


def test_parse_cli_output_tolerates_tool_call_close_tag_in_arguments():
    close = "<" + "/" + "tool_call>"
    open_ = "<tool_call>"
    inner_note = "see " + open_ + "{...}" + close + " protocol"
    args = json.dumps({"note": inner_note})
    block_obj = {"id": "call_1", "type": "function", "function": {"name": "patch", "arguments": args}}
    text = "before " + open_ + json.dumps(block_obj) + close + " after"
    parsed = parse_cli_output(text)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "patch"
    assert parsed.tool_calls[0]["function"]["arguments"] == args
    assert open_ not in parsed.content and close not in parsed.content
    assert "before" in parsed.content and "after" in parsed.content


def test_incremental_parser_tolerates_tool_call_close_tag_in_arguments():
    close = "<" + "/" + "tool_call>"
    open_ = "<tool_call>"
    args = json.dumps({"note": "embedded " + open_ + "{}" + close + " marker"})
    block_obj = {"id": "call_1", "type": "function", "function": {"name": "patch", "arguments": args}}
    payload = "lead " + open_ + json.dumps(block_obj) + close + " tail"
    parser = IncrementalToolCallParser()
    events = parser.feed(payload) + parser.finalize()
    tool_events = [e for e in events if e.kind == "tool_call"]
    text_events = [e for e in events if e.kind == "text" and e.text]
    assert len(tool_events) == 1
    assert tool_events[0].tool_call["function"]["name"] == "patch"
    rendered = "".join(e.text for e in text_events)
    assert open_ not in rendered and close not in rendered
    assert "lead" in rendered and "tail" in rendered
