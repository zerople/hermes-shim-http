# Hermes integration background

This note captures the design context behind `hermes-shim-http` and adapts the original Hermes skill guidance into repository documentation.

## Why this project exists

`hermes-shim-http` exists to expose a local coding CLI through an OpenAI-compatible HTTP surface so Hermes, or any other compatible client, can talk to it like a normal provider.

The key idea is:

- the external CLI does the reasoning and tool selection
- the shim translates that into OpenAI-style responses
- Hermes remains the real tool executor

## Why use an HTTP shim instead of ACP

For this project, a local HTTP shim is usually a better fit than an ACP-style bridge because it:

- reuses normal provider configuration
- is easy to inspect with `curl` and HTTP logs
- is easier to test with request/response fixtures
- keeps the integration on a standard API surface
- avoids many session/subprocess edge cases that appear in tighter protocol bridges

ACP can still make sense in other systems, but this repository is explicitly optimized around the HTTP-shim path.

## Tool authority model

The long-term model is simple:

- the backend CLI reasons
- the backend CLI emits tool intent
- Hermes executes the tools
- tool results go back through the shim

That avoids split tool authority and keeps approval boundaries clear.

## API surface

This standalone repo currently focuses on:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- compatibility probe endpoints for benign client probing

It also supports streamed output and structured tool-call parsing.

## Implementation shape in this repository

The current standalone layout is:

- `hermes_shim_http/models.py`
- `hermes_shim_http/prompting.py`
- `hermes_shim_http/parsing.py`
- `hermes_shim_http/runner.py`
- `hermes_shim_http/server.py`
- `tests/`
- `bin/hermes-shim-http.js`

The Python package implements the shim server, while the npm launcher bootstraps Python and runs the server with a user-friendly `npx` entrypoint.

## Prompting and tool-call contract

The shim expects the wrapped CLI to behave like a backend model rather than a direct tool executor.

In practice that means:

- avoid native filesystem/shell/editor tool execution inside the backend when possible
- have the backend emit structured tool intent
- convert that tool intent into OpenAI-compatible `tool_calls`
- strip tool-call markup from visible assistant text

## Release philosophy

This repo is meant to be publishable as a clean standalone package, but user-facing documentation should stay focused on running the package. Maintainer-only publishing steps belong in `docs/maintainers/releasing.md`, not the front-page README.
