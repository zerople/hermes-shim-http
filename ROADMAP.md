# hermes-shim-http Roadmap

> Last updated: 2026-04-22 for `v0.1.31`

## Current State: v0.1.31

`hermes-shim-http` is an OpenAI-compatible local HTTP shim for Claude/Codex/OpenCode with:

- `chat/completions` + `responses` compatibility endpoints
- request-scoped tool-call nonce enforcement (`<tool_call nonce="...">`)
- hardened parser paths for malformed tool-call blocks with containment notices
- optional malformed JSON repair gate (`HERMES_SHIM_JSON_REPAIR_ENABLED=1`)
- default-on raw log capture with retention caps
- Claude session reuse + optional in-process live child pool
- compaction controls, slash commands, and debug stats/quota endpoints
- npm packaging via Node launcher + Python runtime bootstrap

## Next Up: v0.1.32 — Structured tool-call transport tightening

Theme: **"Reduce protocol ambiguity between transcript text and executable tool calls."**

### Goals
- reduce reliance on text wrappers where structured paths already exist
- keep backward compatibility for CLIs that still emit wrapper-based tool calls
- strengthen replay/mismatch telemetry without leaking sensitive content

## v0.1.33 — Multi-CLI parity and operational robustness

Theme: **"Claude-specific maturity applied consistently across profiles."**

### Goals
- close behavior gaps between Claude/Codex/OpenCode profiles
- improve model discovery and profile capability reporting
- make long-running streaming + restart behavior more observable and debuggable

## Design Principles

1. **Hermes remains the real tool executor.**
2. **HTTP compatibility first, CLI-specific optimizations second.**
3. **Observability by default, sensitive payload leakage by default-off.**
4. **Persistence and debug surfaces should be explicit and controllable.**
5. **Release only tested, documented cuts.**
