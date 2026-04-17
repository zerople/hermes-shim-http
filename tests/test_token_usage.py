from hermes_shim_http.token_usage import estimate_token_usage


def test_estimate_token_usage_is_deterministic_and_positive():
    usage = estimate_token_usage(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Explain persistent caches."},
        ],
        response_text="A persistent cache survives restarts.",
    )

    assert usage.context_tokens_used > 0
    assert usage.context_tokens_limit >= usage.context_tokens_used
    assert usage.response_tokens > 0


def test_estimate_token_usage_scales_with_larger_payloads():
    small = estimate_token_usage(messages=[{"role": "user", "content": "short"}], response_text="ok")
    large = estimate_token_usage(messages=[{"role": "user", "content": "long " * 200}], response_text="done " * 50)

    assert large.context_tokens_used > small.context_tokens_used
    assert large.response_tokens > small.response_tokens
