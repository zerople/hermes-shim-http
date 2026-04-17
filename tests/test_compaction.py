from hermes_shim_http.prompting import compact_messages


def test_window_compaction_keeps_system_and_recent_turns():
    messages = [
        {"role": "system", "content": "stay"},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
        {"role": "user", "content": "latest question"},
    ]

    compacted, did_compact = compact_messages(
        messages=messages,
        mode="window",
        threshold=0.1,
        context_limit=100,
    )

    assert did_compact is True
    assert compacted[0]["content"] == "stay"
    assert any(msg["content"] == "latest question" for msg in compacted)
    assert not any(msg["content"] == "old user" for msg in compacted)


def test_summarize_compaction_inserts_summary_placeholder():
    messages = [{"role": "user", "content": f"message {idx}"} for idx in range(12)]

    compacted, did_compact = compact_messages(
        messages=messages,
        mode="summarize",
        threshold=0.1,
        context_limit=100,
    )

    assert did_compact is True
    assert compacted[0]["role"] == "system"
    assert "summary" in compacted[0]["content"].lower()
