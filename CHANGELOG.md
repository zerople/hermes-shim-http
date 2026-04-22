# Changelog

All notable changes to `@zerople/hermes-shim-http` will be documented in this file.

## [Unreleased]

## [0.1.31] - 2026-04-22

### Changed
- **Publish workflow now uses npm trusted publishing only.** `.github/workflows/publish-npm.yml` now publishes with `npm publish --provenance --access public` under GitHub OIDC (`id-token: write`) and no longer depends on `NPM_TOKEN`.
- **Maintainer and roadmap docs were realigned to current state (`v0.1.30`).** Updated release guidance and roadmap milestones so docs match actual runtime/packaging behavior.
- **README refresh for current operator reality.** Replaced stale `0.1.7`-era "what's new" text, refreshed CLI option summary, and updated observability/control guidance wording.

## [0.1.30] - 2026-04-21

### Fixed
- **Transcript marker collision hardening in prompt rendering.** `_render_message_body` now escapes literal marker strings inside message content by inserting U+200B (`----- turn​:` and `-----​ end -----`) at render time, so quoted marker text in assistant/user/tool bodies can no longer be misread as real transcript boundaries by the downstream CLI parser.
- **Literal `<tool_call>` strings in ordinary message content are now escaped at render time.** Message-body text now rewrites `<tool_call...` / `</tool_call>` with U+200B so protocol-looking examples inside natural-language text cannot be replayed as live tool-call tags.
- **Request-scoped tool-call nonce enforcement for chat/responses runtime paths.** The server now generates a per-request nonce, injects it into system/transcript prompt guidance (`<tool_call nonce="...">`), and only accepts matching-nonce tool-call blocks during parsing. Mismatched nonce blocks are treated as plain text instead of executable tool invocations.
- **Malformed `<tool_call>` containment now avoids raw block leakage.** Both `IncrementalToolCallParser._drain` (streaming path) and `parse_cli_output` (batch path) now replace malformed tool-call blocks with a compact structured notice (`⚠️ shim: dropped malformed tool_call ...`) instead of surfacing raw malformed XML/JSON back to assistant text. The parser also emits telemetry (`tool_call_malformed`) with reason + preview and writes full raw blocks to disk when raw-log capture is enabled.
- **System prompt now explicitly allows `function.arguments` as either JSON-encoded string or raw JSON object.** This matches existing `_normalize_tool_call` behavior and makes model-side formatting drift recoverable without protocol failure.

### Added
- **Opt-in malformed JSON repair for tool-call blocks.** New env flag `HERMES_SHIM_JSON_REPAIR_ENABLED=1` attempts `json_repair.loads(...)` before dropping malformed tool-call JSON. When repair succeeds, the shim still emits the malformed-block notice with `reason=repaired_from_malformed` for observability.
- **Raw Claude stream logging is now default-on.** If `HERMES_SHIM_CLAUDE_RAW_LOG_DIR` is unset, logs are written by default to `~/.hermes/hermes-shim-http/raw-logs/`. Explicit empty-string (`HERMES_SHIM_CLAUDE_RAW_LOG_DIR=`) disables capture.
- **Raw-log rotation in runner.** On each new log creation, old files are pruned by mtime with lightweight caps (max 200 files and ~500 MB total) to avoid unbounded growth.

### Tests
- Added regression coverage for transcript-marker escaping within message bodies (ensures literal `----- turn:tool -----` in content is escaped and does not appear as a top-level marker), plus message-body `<tool_call>` literal escaping.
- Added parser tests for malformed tool-call notice behavior (`json_decode_error`, `normalize_rejected`), raw JSON object arguments, request-scoped nonce matching/mismatch behavior, incremental malformed-block containment, and opt-in repair path.
- Added server-level nonce pinning fixture + nonce-tag response fixtures to assert chat/responses tool-call extraction under strict nonce enforcement.
- Added runner tests for raw-log directory resolution defaults and explicit opt-out behavior.

## [0.1.29] - 2026-04-21

### Fixed
- **Tool-call parser no longer truncates blocks whose JSON arguments contain a literal </tool_call> substring.** Previously `hermes_shim_http.parsing` extracted the JSON body of a <tool_call>...</tool_call> block with the non-greedy regex `<tool_call>\s*(\{.*?\})\s*</tool_call>`. When the emitted `arguments` JSON string contained the literal substring </tool_call> (for example, when a `patch` or `write_file` tool call targets documentation that itself describes the shim’s tool-call protocol), the regex terminated at the inner close tag and produced a truncated JSON fragment that failed `json.loads`, so the whole block fell through as raw assistant text and leaked into downstream chat surfaces instead of executing. The regex is replaced with a `json.JSONDecoder.raw_decode`-based extractor (`_try_extract_tool_call`) that parses exactly one JSON value starting at the opener and then requires the close tag to follow, so string-escaped occurrences of <tool_call> / </tool_call> inside `arguments` no longer terminate the block early. Both `IncrementalToolCallParser._drain` (streaming path) and `parse_cli_output` (batch path) are updated to use the new extractor; the now-unused `_TOOL_CALL_BLOCK_RE` and its `import re` are removed.

### Tests
- Added regression coverage in `tests/test_cli_http_shim_parsing.py` for both `parse_cli_output` and `IncrementalToolCallParser` handling a tool-call whose `arguments` JSON string embeds a literal <tool_call>{...}</tool_call> sequence.

## [0.1.28] - 2026-04-21

### Changed
- **Transcript turns are now wrapped in `----- turn:ROLE -----` / `----- end -----` markers instead of `<role>...</role>` tags.** The previous `<system>`, `<user>`, `<assistant>`, `<tool>` wrappers shared the same angle-bracket grammar as the shim’s <tool_call>{...}</tool_call> protocol, so a reasoning backend (or a nested tool result echoing the rendered transcript) could emit a role tag that the downstream parser would confuse with a live turn boundary. Rendering turns with `----- turn:ROLE -----` / `----- end -----` moves transcript context out of the tool-call grammar entirely so the two surfaces cannot collide. `build_cli_system_prompt` now tells the CLI explicitly that those markers are transcript-only and must never be emitted by the model. `_render_transcript` in `hermes_shim_http/prompting.py` and the covering tests in `tests/test_cli_http_shim.py` / `tests/test_cli_http_shim_server.py` are updated accordingly.

## [0.1.27] - 2026-04-21

### Fixed
- **Idle TTL now actually applies to pooled Claude children.** `LiveChildPool.stream()` now sweeps stale children before reuse/acquire, so `--live-child-pool-idle-ttl` is no longer a test-only knob and expired children are replaced automatically on the next access.
- **Same pooled conversation can no longer silently queue concurrent turns.** A pooled child now rejects a second turn with `ChildLockBusy` instead of blocking on an internal mutex, preserving the shim's phantom-prevention policy for duplicated/retried requests that target the same live Claude session.
- **Fresh pooled sessions are re-keyed to the actual emitted backend session id.** After the first successful pooled turn, the child is moved from the provisional local session key to the real Claude session lineage so later resume-planned turns actually hit the existing live child instead of silently falling back to a cold spawn.
- **Non-streaming 409 handling now preserves `child_lock_busy` semantics without degrading to HTTP 500.** When pooled-turn rejection happens on the non-streaming chat/responses path (with HTTP heartbeat disabled), the shim now returns the same structured 409 body as the normal exception handler.

### Tests
- Added regression coverage for automatic idle eviction, same-session concurrent-turn rejection, and non-streaming HTTP 409 propagation for pooled-child contention.

## [0.1.26] - 2026-04-20

### Added
- **Optional in-process live child pool for Claude multi-turn reuse.** New `--live-child-pool`, `--live-child-pool-size`, and `--live-child-pool-idle-ttl` flags let the shim keep one long-lived Claude stream-json subprocess per active conversation and reuse it across turns when the same conversation stays on the same shim process. Reuse is keyed by conversation lineage plus spawn-context fingerprint so mismatched model/tool/MCP/system-prompt contexts do not accidentally share a child.

### Changed
- **`run_cli_prompt` and `stream_cli_prompt` can now route Claude turns through the live child pool.** Fresh Claude turns create a pooled child when enabled; later turns reuse the same child if it is still alive, otherwise they cleanly fall back to the existing `--resume ... --fork-session` cold-spawn path. This keeps the prior session-cache/resume mechanism as the durability fallback rather than replacing it.
- **FastAPI app lifecycle now owns pooled-child shutdown.** `create_app()` instantiates the pool from `ShimConfig` when enabled and closes all live children via lifespan shutdown, so test clients and real server exits do not leak Claude subprocesses.

### Fixed
- **Live-child prototype is now fully wired into real request paths.** The previous branch state only had a standalone `live_child_pool.py` + fake CLI tests; none of `runner.py` / `server.py` used it. 0.1.26 wires the pool into chat-completions and responses, streaming and non-streaming paths, with end-to-end server tests proving reuse across actual HTTP requests.

## [0.1.25] - 2026-04-19

### Changed
- **Anthropic-provider parity mode is now the default behavior.** The shim no longer treats Claude built-ins as a fallback execution path when Hermes advertises tools. Claude is launched with built-ins disabled (`--tools ""`) and receives only the request-scoped Hermes tool surface, so tool availability now follows the same explicit-request contract as a pure provider API call instead of diverging into CLI-only fallback behavior.
- **Hermes tools are now exposed to the shim-owned Claude subprocess as request-scoped MCP only.** Rather than relying on prompt-only tool descriptions or any persistent user/global Claude MCP state, the shim creates a per-request temporary `--mcp-config` that points at a shim-owned stdio MCP bridge (`bin/hermes-tools-mcp.py`). This keeps direct local `claude` usage untouched while making shim-launched Claude see only the tools explicitly advertised on that request.

### Fixed
- **Shim-added Discord/UI progress noise has been removed from streaming outputs.** The shim no longer synthesizes `Thinking...`, `Using tool: ...`, or tool-argument preview text into streamed assistant content. Chat/responses streaming now forwards only normalized assistant text and tool-call deltas, which makes shim output much closer to direct provider semantics and removes the most distracting Discord noise.
- **Hermes MCP-prefixed tool names are normalized back into canonical Hermes tool calls.** When Claude selects a shim-owned MCP tool like `mcp__hermes__read_file`, the shim now remaps it back to `read_file` before allowlist enforcement and response shaping, so the upstream Hermes execution path stays identical to the non-MCP tool-call shape.

## [0.1.24] - 2026-04-19

### Fixed
- **HTTP 400 `out of extra usage` regression when Hermes runs on an OAuth Pro/Max plan.** 0.1.23 moved the Hermes tool catalog from the stdin user message into the `--append-system-prompt` flag value (merged with the short turn-discipline preface). With Hermes-scale payloads the catalog is ~22 KB, which alone pushes Anthropic's OAuth billing classifier to route the request to the extra-usage bucket instead of the plan-included bucket — even when the account has no extra-usage credit — and the CLI then fails with `400 ... out of extra usage`. Isolated end-to-end with a real captured request (V1–V5 replay matrix against the live API): big `--append-system-prompt` alone triggers the 400 regardless of stdin size or `--tools ""`. 0.1.24 reverts the catalog back into the stdin user message (0.1.22's channel), so `--append-system-prompt` once again carries only the ~281-byte turn-discipline preface and stays in the plan-included bucket. The `--tools ""` enforcement from 0.1.23 is preserved (it is independently confirmed to be billing-neutral), so Hermes tools still structurally win over Claude built-ins. `build_cli_command` no longer merges the caller-supplied `system_prompt` into the flag value; `_stdin_prompt_text` now prepends it to the Claude stream-json user-message text on new sessions (resumes keep the existing `system_prompt`-off-stdin invariant). Tests in `tests/test_cli_http_shim.py` updated accordingly.

## [0.1.23] - 2026-04-19

### Changed
- **Hermes tools now structurally win over Claude built-ins via `--tools ""`.** When a chat/responses request advertises Hermes tools, the shim now passes `--tools ""` to Claude, which empties the built-in tool set at CLI boot — `init.tools` contains only MCP entries, and Claude has no `Read`/`Bash`/`Edit`/`Glob`/`Grep`/`Write`/`TodoWrite` available at all. Combined with the Hermes tool catalog (moved into `--append-system-prompt`, see below) this forces Claude to emit a `<tool_call name=...>` text block targeting the Hermes equivalent instead of silently invoking a native built-in. Verified E2E against the real `claude` CLI on a Pro/Max OAuth subscription (`apiKeySource: "none"`): with `read_file`, `terminal`, `search_files` advertised, a read request now yields a Hermes `<tool_call name="read_file">{...}</tool_call>` and zero native `tool_use` blocks. An earlier 0.1.22 prototype that used `--disallowed-tools` for the same purpose is superseded: `--disallowed-tools` is a permission-check layer and is bypassed by `--dangerously-skip-permissions` which the shim relies on, so only `--tools ""` gives deterministic enforcement. Plumbed as a new `disable_builtin_tools: bool` parameter through `build_cli_command` / `run_cli_prompt` / `stream_cli_prompt`, defaulted to `False` so non-tool-advertising requests keep Claude's full built-in set. Covered by `tests/test_cli_http_shim.py::test_build_cli_command_disables_builtin_tools_for_claude` and companions.

### Fixed
- **Hermes tool catalog is no longer duplicated into the stdin user message.** Previously the `build_cli_system_prompt(...)` output (tool catalog + turn discipline + silent-ACK sentinel) was concatenated onto the user message body for Claude's stream-json stdin *and* would have duplicated into `--append-system-prompt` once the new steering went in. The catalog is now delivered once, in `--append-system-prompt`, together with the short turn-discipline preface; stdin carries the user transcript only. Applied to both new Claude sessions (merged flag value = turn discipline + catalog) and resumes (both channels skipped, matching prior invariant). Threaded via new `build_cli_user_prompt(...)` + `system_prompt` parameter on `run_cli_prompt` / `stream_cli_prompt` and plumbed into `SessionPlan.system_prompt_text` so the cache keeps them separate.

## [0.1.22] - 2026-04-19

### Changed
- **Tool argument preview is now wrapped in a triple-backtick fenced code block.** The streaming progress text now emits e.g. `Using tool: patch ` followed by `` ```path=a.py mode=replace``` `` instead of the bare `Using tool: patch path=a.py mode=replace`. Chat UIs that render Markdown (notably Discord) now display the args in a monospace code block, which keeps long paths/commands visually separated from the surrounding `Using tool:` prefix and avoids accidental Markdown parsing of characters inside the args (slashes, equals, asterisks, underscores, etc.). The single-scalar fallback for unknown tools is wrapped the same way; empty-arg tools still stay silent; the trailing paragraph break (`\n\n`) introduced in 0.1.21 is preserved. Tests in `tests/test_cli_http_shim_server.py::test_tool_progress_preview_shows_primary_args_inline` and `test_chat_completions_streaming_returns_live_tool_call_chunks` updated to assert on the new fenced-code-block shape.

## [0.1.21] - 2026-04-19

### Fixed
- **Progress text now emits a paragraph break (`\n\n`) instead of a single newline.** Some streaming chat UIs (notably Discord's delta renderer) strip a lone trailing `\n` between successive assistant text deltas, which caused `Thinking...` and the following `Using tool: ...` line to render glued together on one line. Emitting `\n\n` forces a paragraph boundary that survives the delta-merging heuristics, so each progress line lands on its own row in every tested renderer. Applied to both `Thinking...` synthesis in `ClaudeStreamJsonParser._progress_events_for_block_start` and the `Using tool: <name> <args>` prefix in `_stream_live_chat_chunks`. Covered tests updated to assert on the new `\n\n` shape.

## [0.1.20] - 2026-04-19

### Changed
- **Streaming progress text now previews the primary tool arguments inline.** The chat-completions stream emits `Using tool: terminal command=git log -3` / `Using tool: read_file path=README.md` / `Using tool: patch path=a.py mode=replace` etc., instead of a bare `Using tool: terminal`, so chat UIs that only render streamed assistant text (e.g. Discord) can show *what* the agent is doing without having to decode the out-of-band `tool_calls` chunks. Per-tool primary fields are whitelisted (`terminal`→`command`, `read_file`/`write_file`/`patch`→`path`, `search_files`→`pattern`+`path`, `browser_*`→`url`/`ref`/`key`/`direction`/`question`, `skill_view`/`skill_manage`/`skills_list`, `memory`, `send_message`, `cronjob`, `clarify`, `session_search`, `delegate_task`, `vision_analyze`, `text_to_speech`, `process`) and values are whitespace-collapsed + truncated to 80 chars with an ellipsis so the progress line stays compact. Unknown tools with a single scalar arg still get that one value surfaced; everything else stays silent. Covered by new unit tests in `tests/test_cli_http_shim_server.py`.
- **`Thinking...` progress text is now deduplicated within a single assistant turn.** Long extended-thinking responses from Claude's stream-json sometimes open several consecutive `thinking` content blocks before the actual reply. The shim previously emitted one `Thinking...\n` text event per block, which surfaced as 3–5 stacked `Thinking...` lines in chat UIs like Discord with no new information between them. The synthesized progress text is now emitted at most once per `message_start` turn — multiple thinking blocks collapse into a single line, and the counter resets on the next `message_start` so each new assistant turn still signals motion. Covered by new unit tests `test_claude_stream_parser_dedups_thinking_within_a_turn` and `test_claude_stream_parser_emits_thinking_once_per_message_turn` in `tests/test_cli_http_shim_parsing.py`.

## [0.1.18] - 2026-04-19

### Added
- **Claude-native tool translation layer (`hermes_shim_http/tool_translation.py`).** When the wrapped Claude CLI emits its own native tool calls (`Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`, `TodoWrite`, etc.), the shim now translates them into their Hermes equivalents (`read_file`, `write_file`, `patch`, `search_files`, `terminal`, `todo`) so Hermes-provided tools take precedence and no round-trip to Anthropic's API is required. Tools without a Hermes equivalent (`WebSearch`, `WebFetch`, `NotebookEdit`, `ExitPlanMode`, etc.) are silently dropped instead of surfacing a noisy `Wrapped CLI emitted unsupported tool call(s)` warning, letting the wrapped CLI fall back to its own built-in handling transparently. Wired into all three call sites: non-stream `_sanitize_parsed_output`, streaming chat (`_stream_live_chat_chunks`), and streaming responses (`_stream_live_responses_events`). Covered by `tests/test_tool_translation.py` (9 tests) plus new server-level assertions in `tests/test_cli_http_shim_server.py`.
- **`synthesize_progress` mode on `ClaudeStreamJsonParser`.** Live streaming now emits a lightweight `Thinking...\n` text event when Anthropic opens a `thinking` content block but does not stream its body, so upstream chat UIs keep showing motion during long extended-thinking turns instead of stalling behind the U+200B heartbeat. Disabled in `_drain_cli_process` (non-stream path) to keep collected outputs clean; enabled in `_stream_cli_prompt_inner`.
- **`HERMES_SHIM_CLAUDE_RAW_LOG_DIR` debug capture.** Setting this env var causes the runner to tee every raw stdout/stderr chunk (pre- and post-heartbeat-strip) from the child CLI to a timestamped log file under the given directory, along with spawn command, session IDs, and exit code. No-op when the env var is unset, so production paths are unaffected. Intended for diagnosing stream-json regressions and phantom-session races without having to re-instrument the shim.

## [0.1.17] - 2026-04-18

### Changed
- **Claude native I/O now runs fully on stream-json.** The native `claude` path now forces both `--output-format stream-json` and `--input-format stream-json`, and stdin is sent as a structured Claude user event JSON line instead of raw text. This aligns the shim with Claude's event protocol and keeps long-lived runs structurally streamable in both directions.
- **Protocol-critical Claude args are now shielded from custom overrides.** User-supplied Claude args can no longer silently replace `-p/--print`, `--output-format`, `--input-format`, or `--permission-mode`, preventing accidental downgrade back to text mode or other transport breakage.
- **Opus is now treated as a 1M-context model across the shim.** `/v1/models`, `/v1/info`, compaction thresholds, and usage accounting all use a 1,000,000 token context window for Opus instead of the old 200k Claude default.
- **First live user instruction is now pinned through compaction.** The shim system prompt and Claude bootstrap prompt both explicitly state that the VERY FIRST live user message is the highest-priority session instruction even if later context is summarized or compacted, and compaction now injects a pinned synthetic system message so that instruction survives window/summarize trimming.

### Fixed
- **Resume failure no longer poisons stored Claude session IDs.** When a resumed Claude run fails and emits a fresh session ID instead of the requested parent session, the shim now discards that mismatched ID rather than recording it as if resume had succeeded. Successful runs still preserve the emitted Claude session ID and write it back into the session cache.

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
