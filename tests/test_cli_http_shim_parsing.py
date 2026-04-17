import json

from hermes_shim_http.parsing import IncrementalToolCallParser, parse_cli_output


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


def test_parse_multiple_tool_call_blocks_and_visible_text_cleanup():
    parsed = parse_cli_output(
        "Need two actions.\n"
        '<tool_call>{"id":"call_1","type":"function","function":{"name":"search_files","arguments":"{\\"pattern\\":\\"TODO\\"}"}}</tool_call>\n'
        "After first call.\n"
        '<tool_call>{"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}</tool_call>'
    )

    assert parsed.content == "Need two actions.\nAfter first call."
    assert [tc["function"]["name"] for tc in parsed.tool_calls] == ["search_files", "read_file"]


def test_malformed_tool_call_is_ignored_and_text_preserved():
    parsed = parse_cli_output(
        'Before\n<tool_call>{"id":"oops","type":"function","function":{"name":"read_file","arguments":not-json}}</tool_call>\nAfter'
    )

    assert parsed.tool_calls == []
    assert "Before" in parsed.content
    assert "After" in parsed.content


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
