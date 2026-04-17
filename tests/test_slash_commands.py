from hermes_shim_http.slash_commands import dispatch_slash_command


def test_dispatch_clear_command_returns_assistant_response():
    result = dispatch_slash_command("/clear", model="claude-cli", stats={"cache_size": 1})
    assert result.command == "clear"
    assert result.message["role"] == "assistant"
    assert "cleared" in result.message["content"].lower()


def test_dispatch_compact_command_requests_local_compaction_response():
    result = dispatch_slash_command("/compact", model="claude-cli", stats={})
    assert result.command == "compact"
    assert result.force_compaction is True
    assert "compaction" in result.message["content"].lower()


def test_dispatch_model_command_parses_argument():
    result = dispatch_slash_command("/model sonnet", model="claude-cli", stats={})
    assert result.command == "model"
    assert result.model_override == "sonnet"


def test_dispatch_stats_command_formats_summary():
    result = dispatch_slash_command("/stats", model="claude-cli", stats={"cache_size": 2, "active_sessions": 1})
    assert result.command == "stats"
    assert "cache" in result.message["content"].lower()
