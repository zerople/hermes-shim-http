# Changelog

All notable changes to `@zerople/hermes-shim-http` will be documented in this file.

## [0.1.16] - 2026-04-18

### Fixed
- **Phantom-session child-spawn race (P0).** `InFlightRegistry` (v0.1.14) only blocks concurrent requests that share an `Idempotency-Key` / `X-Request-Id` header. Clients that don't send one — or that send a fresh UUID per request — bypassed the guard entirely, so a slow first turn plus a transport-level retry could spawn a second `claude` / `codex` child alongside the first. Both children then wrote to the same on-disk SSOT (agent state JSON, worktree files, `.lck/` scratch) and produced phantom task IDs, ghost worktrees, and mysterious "file I didn't create" artefacts. Confirmed in the field: two sibling `heartbeat-wrap → claude --fork-session` chains holding inotify watchers on the same cwd.

### Added
- **`single_child_lock_path`: process-level single-instance child-spawn lock.** Before `_drain_cli_process` and `stream_cli_prompt` invoke `subprocess.Popen`, the shim non-blocking `fcntl.flock`s a port-scoped lock file (`/tmp/hermes-shim-http-<port>.child.lock` by default). A concurrent spawn attempt raises `ChildLockBusy` → HTTP **409 `child_lock_busy`** with `Retry-After: 5`. The lock is held for the child's entire lifetime (including the full stream-generator body via `yield from`) and released on return, exception, or client disconnect. Independent of `InFlightRegistry`: covers clients that don't send idempotency keys.
- **CLI flags `--single-child-lock` / `--no-single-child-lock` / `--single-child-lock-path PATH`.** On by default. Explicit path overrides the port-derived default. Echoed in `/v1/info` and the startup config payload for observability.
- New module `hermes_shim_http/single_child.py` (`acquire_single_child_lock` context manager + `ChildLockBusy` exception). Covered by `tests/test_single_child_lock.py` — 5 tests including cross-process contention via `multiprocessing.Process`.

### Notes
- `InFlightRegistry` remains the first line of defence for idempotent clients and is unchanged — the new lock is a backstop that protects on-disk SSOT even when upstream idempotency coverage lapses.
- Recommended opt-out: only for intentional multi-tenant shim deployments where concurrent child CLIs are desired and the state on disk is safely partitioned.

## [0.1.15] - 2026-04-18

### Fixed
- **Silent `-p` text mode triggered `RemoteProtocolError` on long Claude reasoning turns (P0).** The Claude profile invoked the CLI with plain `-p` / `--dangerously-skip-permissions`, which only prints text once the final reply is ready. During multi-minute extended-thinking turns the child emitted nothing to stdout; only the U+200B heartbeat kept the shim alive, and the HTTP caller saw `: ping` comments for 50+ minutes before Hermes tore down the connection with `RemoteProtocolError → Reconnecting (attempt 2/3)`. The retry then re-entered the same turn and risked duplicate agent work (phantom duplicate task IDs, ghost worktrees).
- **Claude profile now uses `--output-format stream-json --verbose --include-partial-messages`** so the child emits incremental NDJSON (`stream_event` → `content_block_delta`) throughout the turn. A new `ClaudeStreamJsonParser` assembles text, thinking, and `tool_use` blocks into the shim's `CliStreamEvent` pipeline; `parse_claude_stream_json` falls back to legacy `<tool_call>` text parsing when the blob contains no JSON lines so existing mocks and non-Claude profiles remain unaffected. `thinking_delta` / `signature_delta` events are consumed silently (raw bytes still reset the runner idle timer).

### Added
- **Idempotency-key retry guard.** New `InFlightRegistry` tracks reservations keyed by the `Idempotency-Key`, `X-Idempotency-Key`, or `X-Request-Id` header. A second request arriving while the first is still streaming for the same key is rejected with `409 duplicate_request` (and `Retry-After: 5`) instead of spawning a second child CLI. Reservations release automatically when the response finishes, errors, or the client disconnects — covering the exact Hermes retry window (`RemoteProtocolError → attempt 2/3`) that previously double-spent tokens and risked duplicate task work. Covered by `tests/test_idempotency.py`.
- **`heartbeat-wrap.py` stderr serialization.** The periodic U+200B heartbeat no longer interleaves mid-line with real stderr bytes — writes now share a lock with the stderr pump so concatenated streams remain cleanly decodable.

## [0.1.14] - 2026-04-18

### Fixed
- **Orphaned child CLI on client disconnect (P0).** When an HTTP client closed a streaming connection mid-stream, Starlette fed `GeneratorExit` into the SSE generator, which the previous `except Exception:` block could not catch — the inner `stream_cli_prompt` generator was abandoned, leaving the child Claude/Codex/OpenCode process and its `heartbeat-wrap` wrapper running until their own idle timers fired. Under unattended autonomous operation (LCK fleet, 24/7 multi-pilot runs) this silently leaked process-table slots, pipe FDs, and memory. Both `run_cli_prompt` and `stream_cli_prompt` now clean up via `finally: _terminate_process(process)`, `_iter_events_with_keepalive` closes the source generator on early exit, and `_stream_live_chat_chunks` / `_stream_live_responses_events` close their `stream_cli_prompt` iterator in `finally`. Covered by a new `test_stream_cli_prompt_finally_kills_child_on_generator_close` test that spawns a long-lived child, calls `gen.close()`, and asserts the PID is gone within 5s.
- **Session-cache fork race (P1).** Two concurrent requests with the same message prefix could both observe the same `best_match` parent session and plan to `--resume --fork-session` against it simultaneously, corrupting stored history. `SessionCache` now tracks `_in_flight_parents` under the existing lock; a second request hitting a parent that is already being forked falls through to a fresh session (and emits a `session_plan_in_flight_skip` event). `record_success` and the new `release_plan` helper clear the in-flight marker on completion or abort.
- **`IncrementalToolCallParser` quadratic buffer scan (P1).** The parser's `_safe_prefix_length` previously scanned the entire accumulated buffer from the start for a potential `<tool_call>` / `</tool_call>` prefix on every feed. A long plain-text chunk without any tool tags (extended reasoning output, cargo test dumps) made this O(n²) and held the whole chunk buffered. The lookahead is now capped to the max tag length, so long plain-text chunks flush immediately.
- **Implicit `subprocess.TimeoutExpired` on final wait (P1).** `process.wait(timeout=...)` after the pipes drained used to propagate `TimeoutExpired` unchanged. It is now explicitly caught, the process force-killed, and a `RuntimeError("CLI process did not exit after pipes closed")` surfaced.

### Added
- **`ShimConfig.hard_deadline_seconds` (default 1800s, `0` disables).** Absolute wall-clock deadline for a single child CLI invocation, enforced by both `run_cli_prompt` and `stream_cli_prompt`. Complements the existing idle `--timeout`: heartbeat activity keeps the idle timer alive indefinitely, but the hard deadline still kicks in and returns a `TimeoutError("CLI process exceeded hard deadline of Xs")` so runaway work cannot pin a worker forever. Intended for autonomous operation where a stuck worker must eventually be reclaimed even when the child keeps streaming output.
- **`ShimConfig.max_output_bytes` (default 32 MiB, `0` disables).** Caps total stdout bytes captured per invocation. `run_cli_prompt` truncates cleanly at the cap; `stream_cli_prompt` raises `RuntimeError("CLI process exceeded max output cap of N bytes")` to guard against runaway `cargo test`-style dumps accidentally ballooning shim memory.

### Changed
- `_terminate_process` helper always waits up to 2s for the child to exit after `kill()`, and pipe-drainer thread joins extended from 0.2s → 0.5s so heartbeat-wrap's daemon pump threads release FDs before the generator returns.
- `session_cache.record_success` now also clears the in-flight parent marker it acquired in `plan_request`.

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
