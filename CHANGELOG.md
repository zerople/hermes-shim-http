# Changelog

All notable changes to `@zerople/hermes-shim-http` will be documented in this file.

## [0.1.13] - 2026-04-18

### Fixed
- **Heartbeat-only output no longer masks a genuinely hung child CLI.** `runner._drain_cli_process` and `runner.stream_cli_prompt` previously updated `last_activity` on *any* chunk from stdout/stderr, including the zero-width-space heartbeat bytes emitted by `bin/heartbeat-wrap.py`. A silent inner CLI therefore kept the shim's idle timer pinned forever, and the only thing the HTTP caller ever received was `: ping` SSE comments — leading to multi-hour "waiting for stream response" hangs in Hermes with `iteration 1/N, no chunks yet`. The idle timer is now updated only when the post-strip payload is non-empty, so heartbeats keep the process and HTTP connection alive but real silence still trips the configured `--timeout` cleanly.

### Added
- **`/v1/info` (and `/info`) capability endpoint.** Returns a TGI+LiteLLM-style payload describing the shim: `server`, `version`, `backend`, resolved `cli_profile`, `model_id`, `max_input_length` / `max_total_tokens`, per-model `context_length`, the full list of declared models, api_modes (`chat_completions`, `responses`), and a capabilities block (`streaming`, `tools`, `tool_choice`, `vision`, `prompt_caching`, `session_resume`, `reasoning`). Probes that expected TGI-shaped `/info` or LiteLLM-shaped `/v1/info` no longer 404.
- **Context metadata in `/v1/models` entries.** Each model entry now includes `context_length`, `max_model_len`, and `max_completion_tokens`, so clients that resolve context length via the OpenAI-compat `/models` surface (including Hermes-agent's `fetch_endpoint_model_metadata`) pick up an accurate window instead of falling through to the 128K default.

### Notes
- Values are profile-aware: Claude profile defaults to 200 000 tokens (input+output ceiling); Codex profile to 400 000; OpenCode to 200 000; generic to 128 000. Extended-1M-context Claude deployments can still be reflected by configuring `models` appropriately — the number is a safe floor, not a cap.

## [0.1.12] - 2026-04-18

### Added
- **Layer 1 — Child CLI heartbeat.** New `bin/heartbeat-wrap.py` wrapper, enabled by default via `--heartbeat-wrap`, spawns the child CLI and emits a zero-width-space byte (`U+200B`) to stderr every `--heartbeat-interval` seconds (default 60s) while the child is alive. Legitimate long operations (extended reasoning, network waits, large test runs) no longer trip the shim's stdout/stderr idle timeout. The shim strips the heartbeat bytes from captured output so nothing leaks into the model reply.
- **Layer 2 — HTTP response heartbeat.** Non-streaming `/v1/chat/completions` and `/v1/responses` endpoints now emit a whitespace byte every `--http-heartbeat-interval` seconds (default 30s, `0` disables) while the child CLI runs, then yield the final JSON body. JSON's RFC 8259 leading-whitespace allowance keeps the concatenated stream a valid parse, so Hermes→shim HTTP connections (and intermediate proxies) stay alive through minute-scale runs that used to die on idle TCP close. SSE streaming endpoints were already keepalive-covered via `: ping` comments.

### Changed
- Child CLI stdin is now written as UTF-8 bytes (Popen no longer uses `text=True`), and stdout/stderr pipes are decoded incrementally so partial multi-byte chars across read boundaries no longer corrupt the transcript.

### Notes
- Defaults are conservative: Layer 1 fires every 60s (well under the 300s idle budget), Layer 2 every 30s (well under typical 60s HTTP idle proxies). Either can be tuned or turned off at launch with the new flags.

## [0.1.10] - 2026-04-17

### Fixed
- Buffer streaming response text until the `<silent/>` sentinel is ruled out so silent-ACK turns never leak the literal sentinel through SSE deltas.

## [0.1.9] - 2026-04-17

### Added
- Documented the silence sentinel (`<silent/>`, configurable via `HERMES_SHIM_SILENT_SENTINEL`) in the CLI system prompt so the wrapped reasoning model can intentionally produce a silent ACK turn. Previously the server-side detection existed but the model was never told the protocol, so a model trying to "stay quiet" would emit an empty body and trip the upstream client's empty-response retry loop.

### Notes
- Behavior is opt-in by the model. Empty replies that do not contain the sentinel are still treated as errors by the upstream client, matching the existing semantics covered in `tests/test_cli_http_shim_server.py`.

## [0.1.8] - 2026-04-17

### Fixed
- `stream_cli_prompt` now treats `config.timeout` as an **idle** timeout (time since the last stdout/stderr chunk) instead of a wall-clock total. Long Opus turns that legitimately take more than 120s while streaming tokens no longer get killed mid-response, which prevented `RemoteProtocolError` / `Connection to provider dropped` on the Hermes client.
- `run_cli_prompt` (non-streaming path) was rewritten to use the same `Popen` + drain-thread + idle-timeout pattern instead of `subprocess.run`'s wall-clock timeout, so both request paths behave consistently.

### Changed
- Raised the default `config.timeout` from `120.0` to `300.0` seconds in both the `ShimConfig` model and the `--timeout` CLI flag, giving extended thinking + long Opus responses room to complete.

## [0.1.7] - 2026-04-17

### Added
- Claude session reuse via prefix-matched `--resume --fork-session --session-id` planning.
- Structured request/session telemetry events: `chat_completions_request`, `responses_request`, `cli_dispatch`, `session_plan`, and `session_recorded`.
- Version-alignment regression test covering `package.json`, `pyproject.toml`, and `hermes_shim_http/__init__.py`.
- Maintainer release notes for the `0.1.7` cut.

### Changed
- Claude prompts are now always piped over stdin instead of large argv payloads.
- Large shim system instructions stay in stdin only for fresh Claude sessions.
- Resumed Claude turns now send only the delta transcript on stdin, dramatically reducing repeated prompt size.
- Request/session logging was cleaned up and centralized; verbose per-candidate session-cache logs are now opt-in via `HERMES_SHIM_HTTP_DEBUG_SESSION_CACHE=1`.
- Prompt rendering/normalization was tightened so mixed string/list/dict message content matches reliably across turns.

### Fixed
- Prevented `OSError: [Errno 7] Argument list too long` on large Claude prompts.
- Avoided the Claude CLI `400 invalid_request_error` regression caused by sending the large shim prompt through `--append-system-prompt`.
- Restored real multi-turn session continuity for Hermes-driven conversations.
- Kept model pass-through enabled without forwarding meaningless placeholder values such as `auto` or `cli-http-shim`.

## [0.1.6] - 2026-04-17

### Added
- `/version` endpoint and release metadata alignment checks.
- Startup config visibility and request-summary logging for easier operator debugging.

### Changed
- Refined packaging metadata and publish workflow validation.
