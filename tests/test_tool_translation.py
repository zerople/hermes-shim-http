import json

from hermes_shim_http.tool_translation import (
    CLAUDE_NATIVE_TOOL_NAMES,
    is_claude_native_without_hermes_equivalent,
    translate_tool_call,
)


def _call(name: str, args: dict) -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _args(call: dict) -> dict:
    return json.loads(call["function"]["arguments"])


ALLOWED = {"read_file", "write_file", "patch", "search_files", "terminal", "todo"}


def test_read_maps_to_read_file():
    out = translate_tool_call(
        _call("Read", {"file_path": "/tmp/x.md", "offset": 5, "limit": 100}),
        allowed_names=ALLOWED,
    )
    assert out["function"]["name"] == "read_file"
    assert _args(out) == {"path": "/tmp/x.md", "offset": 5, "limit": 100}


def test_write_maps_to_write_file():
    out = translate_tool_call(
        _call("Write", {"file_path": "/tmp/out.txt", "content": "hi"}),
        allowed_names=ALLOWED,
    )
    assert out["function"]["name"] == "write_file"
    assert _args(out) == {"path": "/tmp/out.txt", "content": "hi"}


def test_edit_maps_to_patch_replace():
    out = translate_tool_call(
        _call("Edit", {
            "file_path": "/a.py",
            "old_string": "foo",
            "new_string": "bar",
            "replace_all": True,
        }),
        allowed_names=ALLOWED,
    )
    assert out["function"]["name"] == "patch"
    assert _args(out) == {
        "mode": "replace",
        "path": "/a.py",
        "old_string": "foo",
        "new_string": "bar",
        "replace_all": True,
    }


def test_glob_maps_to_search_files_files():
    out = translate_tool_call(
        _call("Glob", {"pattern": "**/*.rs", "path": "crates"}),
        allowed_names=ALLOWED,
    )
    assert out["function"]["name"] == "search_files"
    assert _args(out) == {"target": "files", "file_glob": "**/*.rs", "path": "crates"}


def test_grep_maps_to_search_files_content():
    out = translate_tool_call(
        _call("Grep", {
            "pattern": "unwrap",
            "path": "crates",
            "glob": "*.rs",
            "context": 2,
            "head_limit": 50,
        }),
        allowed_names=ALLOWED,
    )
    assert out["function"]["name"] == "search_files"
    assert _args(out) == {
        "target": "content",
        "pattern": "unwrap",
        "path": "crates",
        "file_glob": "*.rs",
        "limit": 50,
        "context": 2,
    }


def test_bash_maps_to_terminal():
    out = translate_tool_call(
        _call("Bash", {"command": "ls -la", "timeout": 5000, "run_in_background": True}),
        allowed_names=ALLOWED,
    )
    assert out["function"]["name"] == "terminal"
    assert _args(out) == {"command": "ls -la", "timeout": 5, "background": True}


def test_todowrite_maps_to_todo():
    out = translate_tool_call(
        _call("TodoWrite", {
            "todos": [
                {"id": "1", "content": "Do X", "status": "in_progress", "activeForm": "Doing X"},
                {"content": "Do Y"},
            ],
        }),
        allowed_names=ALLOWED,
    )
    assert out["function"]["name"] == "todo"
    args = _args(out)
    assert args["todos"][0] == {"id": "1", "content": "Do X", "status": "in_progress"}
    assert args["todos"][1]["content"] == "Do Y"
    assert args["todos"][1]["status"] == "pending"


def test_allowed_name_passthrough_unchanged():
    call = _call("read_file", {"path": "/x"})
    assert translate_tool_call(call, allowed_names=ALLOWED) == call


def test_hermes_mcp_prefixed_tool_maps_back_to_allowed_name():
    call = _call("mcp__hermes__read_file", {"path": "/x"})
    out = translate_tool_call(call, allowed_names=ALLOWED)
    assert out["function"]["name"] == "read_file"
    assert _args(out) == {"path": "/x"}


def test_hermes_mcp_prefixed_tool_stays_unchanged_when_not_allowed():
    call = _call("mcp__hermes__browser_navigate", {"url": "https://example.com"})
    out = translate_tool_call(call, allowed_names=ALLOWED)
    assert out == call


def test_unknown_native_tool_unchanged():
    call = _call("WebSearch", {"query": "hi"})
    assert translate_tool_call(call, allowed_names=ALLOWED) == call


def test_no_allowlist_still_translates_native():
    call = _call("Read", {"file_path": "/x"})
    out = translate_tool_call(call, allowed_names=None)
    assert out["function"]["name"] == "read_file"
    assert _args(out) == {"path": "/x"}


def test_translator_raising_returns_original():
    call = _call("Bash", {"command": "ls", "timeout": "bad"})
    out = translate_tool_call(call, allowed_names=ALLOWED)
    # Bash translator tolerates non-numeric timeout by passing through.
    assert out["function"]["name"] == "terminal"


def test_native_with_no_hermes_equivalent_is_droppable():
    assert is_claude_native_without_hermes_equivalent(
        "WebFetch", allowed_names=ALLOWED
    )
    assert is_claude_native_without_hermes_equivalent(
        "NotebookEdit", allowed_names=ALLOWED
    )
    assert is_claude_native_without_hermes_equivalent(
        "ExitPlanMode", allowed_names=ALLOWED
    )


def test_native_with_hermes_equivalent_is_not_droppable():
    # Read → read_file is present in ALLOWED, so the shim should forward the
    # translated call, not drop it.
    assert not is_claude_native_without_hermes_equivalent(
        "Read", allowed_names=ALLOWED
    )
    assert not is_claude_native_without_hermes_equivalent(
        "Bash", allowed_names=ALLOWED
    )


def test_native_equivalent_missing_from_allowlist_is_droppable():
    # When Hermes did not advertise read_file on this turn, Read cannot be
    # forwarded — drop it silently instead of emitting "unsupported".
    narrow = {"terminal"}
    assert is_claude_native_without_hermes_equivalent(
        "Read", allowed_names=narrow
    )


def test_unknown_tool_name_is_not_native():
    # Model hallucinated a tool name — the shim should still surface this as
    # unsupported so the caller knows something went wrong.
    assert not is_claude_native_without_hermes_equivalent(
        "MadeUpTool", allowed_names=ALLOWED
    )


def test_native_without_allowlist_is_droppable_only_if_no_translator():
    # Without an allowlist we translate whatever we can and drop only the rest.
    assert not is_claude_native_without_hermes_equivalent(
        "Read", allowed_names=None
    )
    assert is_claude_native_without_hermes_equivalent(
        "WebSearch", allowed_names=None
    )


def test_claude_native_names_cover_all_translators():
    # Every name we can translate must also be listed as native so the drop
    # logic and translation logic agree on which calls are Claude-originated.
    from hermes_shim_http.tool_translation import _TRANSLATORS
    assert set(_TRANSLATORS).issubset(CLAUDE_NATIVE_TOOL_NAMES)
