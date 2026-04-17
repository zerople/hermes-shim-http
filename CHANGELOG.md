# Changelog

All notable changes to `@zerople/hermes-shim-http` will be documented in this file.

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
