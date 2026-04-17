# hermes-shim-http Roadmap

> Last updated: 2026-04-17 for `v0.1.7`

## Current State: v0.1.7

`hermes-shim-http` is now a usable Claude-first OpenAI-compatible local shim with:

- `chat/completions` and `responses` compatibility endpoints
- npm launcher + Python runtime bootstrap
- startup preflight (`--doctor`) and startup config visibility
- Claude stdin transport to avoid argv limits
- short bootstrap `--append-system-prompt` only on fresh Claude sessions
- Claude session reuse via prefix-matched `--resume --fork-session --session-id`
- delta-only stdin on resumed Claude turns
- structured request/session telemetry for real Hermes debugging
- version alignment checks across Node + Python packaging metadata

## Next Up: v0.1.8 — Persistent Session Cache + Debug Surface

Theme: **"Make resume survive restarts and make state inspectable."**

### Goals
- optional persistent session cache (`--cache-path ...`) instead of memory-only resume state
- lightweight debug/stats endpoint for cache/session visibility
- explicit retention/TTL controls for cached transcripts
- keep the privacy-sensitive default as in-memory only

## v0.1.9 — Adaptive Timeout + Long-Conversation Resilience

Theme: **"Large conversations should degrade gracefully, not just hang."**

### Goals
- separate first-token timeout from total timeout
- size-aware timeout scaling for long prompts
- transcript trimming / compaction strategy for oversized requests
- better timeout telemetry in logs and debug endpoints

## v0.2.0 — Tool Call Bridging Without Text Wrappers

Theme: **"Move from text conventions to real tool-call semantics."**

### Goals
- prefer structured CLI tool-call output where available
- reduce reliance on `<tool_call>{...}</tool_call>` wrapper text
- tighten tool result round-tripping for Hermes
- preserve compatibility for CLIs that still need wrapper parsing

## Design Principles

1. **Hermes remains the real tool executor.**
2. **Keep the HTTP surface stateless; use sessions as an optimization.**
3. **Prefer observable behavior over opaque cleverness.**
4. **Privacy-sensitive persistence must be opt-in.**
5. **Release only tested, documented cuts.**

## Known Follow-ups

- npm trusted publishing for `@zerople/hermes-shim-http` still needs npm-side setup/verification.
- Claude model discovery is still based on static aliases (`sonnet`, `opus`, `haiku`).
- Session reuse is currently Claude-specific; other CLIs still use the safe non-resume path.
