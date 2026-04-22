# `@zerople/hermes-shim-http`

Run a local coding CLI like **Claude**, **Codex**, or **OpenCode** as an **OpenAI-compatible HTTP server**.

This package gives you a local provider endpoint that looks like OpenAI-compatible chat/responses APIs, while the actual reasoning is done by your installed CLI. It was designed for **Hermes Agent**, but it also works with other local clients that speak:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/models`

---

## Highlights

- run locally with `npx @zerople/hermes-shim-http ...`
- supports `claude`, `codex`, and `opencode`
- exposes `chat/completions`, `responses`, `models`, and debug observability endpoints
- bootstraps its Python runtime automatically on first run
- keeps Hermes as the real tool executor when used as a Hermes provider
- reuses Claude sessions across turns and sends only delta prompts on resumed requests
- supports optional persistent session caching when `--cache-path` is set
- reports fallback token/context estimates and supports context compaction controls
- emits structured request/session logs and SSE keepalive pings for long streams

---

## What this does

`@zerople/hermes-shim-http` starts a small local HTTP server and forwards prompts to a local CLI such as:

- `claude`
- `codex`
- `opencode`

The wrapped CLI performs the reasoning. This shim translates the result into an OpenAI-compatible HTTP response, including streamed output and tool-call payloads.

In short:

- your **CLI** thinks and plans
- this **shim** exposes an HTTP API
- your **client app** talks to the shim like it talks to OpenAI-compatible providers

---

## Who this is for

Use this package if you want to:

- connect a local coding CLI to **Hermes Agent**
- expose Claude/Codex/OpenCode through a local OpenAI-compatible endpoint
- test or prototype agent workflows without building a provider integration from scratch
- keep everything running locally on your own machine

---

## Requirements

Before you start, make sure you have:

- **Node.js 18+**
- **Python 3.10+** available in `PATH`
- **Python virtual environment support** available (`python3 -m venv` must work)
  - on Debian/Ubuntu/WSL, this often means installing `python3-venv`
- at least one supported CLI installed and working:
  - `claude`
  - `codex`
  - `opencode`

> On first run, the launcher automatically creates a cached Python virtual environment under `~/.cache/hermes-shim-http/` and installs the pinned Python dependencies for you. If that bootstrap fails, run `npx @zerople/hermes-shim-http --doctor` for a preflight summary.

---

## Quick start

### 1) Run it with `npx`

> **Important:** `--cwd` must be a **real, existing directory**. Do not copy `/path/to/project` literally — it is only a placeholder. If the path does not exist, the wrapped CLI subprocess will fail with `FileNotFoundError: [Errno 2] No such file or directory`.

The safest way is to `cd` into your project first and use `"$(pwd)"`:

```bash
cd /ABSOLUTE/PATH/TO/YOUR/PROJECT    # replace with your real project path
npx @zerople/hermes-shim-http \
  --host 127.0.0.1 \
  --port 8765 \
  --command claude \
  --cwd "$(pwd)" \
  --model opus
```

Or pass an explicit absolute path you know exists. **Replace `/ABSOLUTE/PATH/TO/YOUR/PROJECT` with your own real project directory** — do not copy it literally:

```bash
npx @zerople/hermes-shim-http \
  --host 127.0.0.1 \
  --port 8765 \
  --command claude \
  --cwd /ABSOLUTE/PATH/TO/YOUR/PROJECT \
  --model opus
```

If you prefer Codex or OpenCode, the same rule applies — replace `"$(pwd)"` or the absolute path with your own project location:

```bash
npx @zerople/hermes-shim-http --command codex    --cwd "$(pwd)" --model codex-cli
npx @zerople/hermes-shim-http --command opencode --cwd "$(pwd)" --model opencode-cli
```

### 2) Run a preflight check if you want a quick sanity check

```bash
npx @zerople/hermes-shim-http --doctor --command claude --cwd "$(pwd)"
```

This prints a short preflight summary covering:

- detected Python interpreter
- whether Python virtual environments are supported
- whether your `--cwd` exists
- whether the wrapped CLI command is available
- which CLI args are effectively being used after auto defaults

### 3) Confirm the server is up

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/v1/models
curl http://127.0.0.1:8765/v1/debug/stats
curl http://127.0.0.1:8765/v1/debug/quota
```

### 4) Send a test request

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "opus",
    "messages": [
      {"role": "user", "content": "Say hello in one short sentence."}
    ]
  }'
```

## Observability and control flags

Useful runtime flags (current `v0.1.30`):

```bash
npx @zerople/hermes-shim-http \
  --cache-path ~/.cache/hermes-shim-http/sessions.sqlite \
  --cache-ttl-seconds 3600 \
  --cache-max-entries 256 \
  --compaction window \
  --compaction-threshold 0.9 \
  --log-level info \
  --log-format text
```

Notes:

- Persistent SQLite session caching is **opt-in**; omit `--cache-path` to keep cache state in memory only.

- `GET /v1/debug/stats` exposes cache, latency, uptime, and token/context aggregates.
- `GET /v1/debug/quota` returns `{ "status": "unknown" }` until the wrapped CLI exposes real quota data.
- chat/responses usage objects include fallback `context_tokens_used`, `context_tokens_limit`, and `response_tokens` estimates.
- slash commands `/clear`, `/compact`, `/model <name>`, and `/stats` are handled as normal assistant responses.
- long SSE streams emit `: ping` comments during idle periods to avoid proxy/client timeouts.
- chat/responses runtime now uses request-scoped tool-call nonces (`<tool_call nonce="...">`) and only executes matching-nonce tool-call blocks from CLI output.

Debug / parsing observability env vars:

- `HERMES_SHIM_CLAUDE_RAW_LOG_DIR`
  - default: `~/.hermes/hermes-shim-http/raw-logs/` (enabled even when unset)
  - set to a custom path to override
  - set to empty (`HERMES_SHIM_CLAUDE_RAW_LOG_DIR=`) to disable raw log capture
- `HERMES_SHIM_JSON_REPAIR_ENABLED=1`
  - opt-in malformed `<tool_call>` JSON repair using `json_repair`
  - default is off; when repair succeeds, the shim still emits a malformed-block notice for observability

---

## What's new in 0.1.31

`0.1.31` is a release-hygiene + documentation alignment cut:

- npm publish workflow is now trusted-publishing only (`npm publish --provenance --access public` with GitHub OIDC)
- maintainer release docs and roadmap were refreshed to match current behavior
- README operator guidance/CLI options were cleaned up to remove stale `0.1.7` references

For runtime protocol hardening changes, see `0.1.30` in [`CHANGELOG.md`](./CHANGELOG.md).

---

## Recommended usage with Hermes Agent

Start the shim (use your real project path, not the example below):

```bash
cd /ABSOLUTE/PATH/TO/YOUR/PROJECT    # replace with your real project path
npx @zerople/hermes-shim-http \
  --host 127.0.0.1 \
  --port 8765 \
  --command claude \
  --cwd "$(pwd)" \
  --model opus
```

Then point Hermes to the local provider.

### If you are currently using `openai-codex`

A lot of users already have a `config.yaml` that looks something like this:

```yaml
model:
  base_url: https://chatgpt.com/backend-api/codex
  context_length: 1000000
  default: gpt-5.4
  provider: openai-codex
  api_key: no-key-required

providers: {}
fallback_providers: []
```

You do **not** need to delete that from memory and figure everything out again. The easiest way is:

1. keep a copy of your old config
2. replace the active `model:` block with one of the shim examples below
3. add a `custom_providers:` section

If you want, you can even keep your old provider as a commented backup in the same file.

### Option A: use the shim via `chat_completions`

This is usually the easiest place to start.

```yaml
model:
  default: opus
  provider: custom
  base_url: http://127.0.0.1:8765/v1
  api_key: no-key-required
  api_mode: chat_completions
  context_length: 1000000

providers: {}

custom_providers:
  - name: claude-shim
    base_url: http://127.0.0.1:8765/v1
    api_key: no-key-required
    api_mode: chat_completions
    model: opus

fallback_providers: []
```

### Option B: use the shim via `codex_responses`

Use this if you specifically want Hermes to talk to the shim through `/v1/responses`.

```yaml
model:
  default: opus
  provider: custom
  base_url: http://127.0.0.1:8765/v1
  api_key: no-key-required
  api_mode: codex_responses
  context_length: 1000000

providers: {}

custom_providers:
  - name: claude-shim-responses
    base_url: http://127.0.0.1:8765/v1
    api_key: no-key-required
    api_mode: codex_responses
    model: opus

fallback_providers: []
```

### Example with the old config kept as a backup

This can make editing less stressful for users who want a clear before/after reference:

```yaml
# old setup
# model:
#   base_url: https://chatgpt.com/backend-api/codex
#   context_length: 1000000
#   default: gpt-5.4
#   provider: openai-codex
#   api_key: no-key-required

model:
  default: opus
  provider: custom
  base_url: http://127.0.0.1:8765/v1
  api_key: no-key-required
  api_mode: codex_responses
  context_length: 1000000

providers: {}

custom_providers:
  - name: claude-shim-responses
    base_url: http://127.0.0.1:8765/v1
    api_key: no-key-required
    api_mode: codex_responses
    model: opus

fallback_providers: []
```

### Which one should you use?

- use `chat_completions` if you want the simplest, most familiar setup
- use `codex_responses` if you want Hermes to call the shim's `/v1/responses` endpoint

A practical rule of thumb:

- start with `chat_completions`
- switch to `codex_responses` only if you specifically want the Responses-style flow

---

## CLI options

For the full, up-to-date list always check:

```bash
npx @zerople/hermes-shim-http --help
```

Key options:

```text
--host HOST
--port PORT
--command COMMAND
--cwd CWD
--timeout TIMEOUT
--model MODELS
--fallback-model FALLBACK_MODEL
--profile {auto,claude,codex,opencode,generic}
--cache-path CACHE_PATH
--cache-ttl-seconds CACHE_TTL_SECONDS
--cache-max-entries CACHE_MAX_ENTRIES
--compaction {off,summarize,window}
--compaction-threshold COMPACTION_THRESHOLD
--log-level {info,debug}
--log-format {text,json}
--heartbeat-wrap | --no-heartbeat-wrap
--heartbeat-interval HEARTBEAT_INTERVAL
--single-child-lock | --no-single-child-lock
--single-child-lock-path SINGLE_CHILD_LOCK_PATH
--http-heartbeat-interval HTTP_HEARTBEAT_INTERVAL
--strict-mcp-config | --no-strict-mcp-config
--live-child-pool | --no-live-child-pool
--live-child-pool-size LIVE_CHILD_POOL_SIZE
--live-child-pool-idle-ttl LIVE_CHILD_POOL_IDLE_TTL
```

### Common options

- `--host`  
  Bind address. Default is usually `127.0.0.1`.

- `--port`  
  HTTP port to listen on.

- `--command`  
  Which local CLI executable to run, for example `claude`, `codex`, or `opencode`.

- `--cwd`  
  Working directory passed to the wrapped CLI. Use your project root here.

- `--timeout`  
  Maximum runtime for a single CLI request.

- `--model`  
  Model name reported back to API clients. For Claude-backed usage, the built-in model list is `sonnet`, `opus`, and `haiku`, so using one of those names is recommended.

- `--profile`  
  Command profile. `auto` usually picks a sensible default based on `--command`.
  For Claude with no explicit `-- ...` CLI args, the shim defaults to `-p --dangerously-skip-permissions` so the launcher works out of the box for Hermes-style tool-selection flows.

---

## Supported API behavior

### Main endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

### Streaming

Supports streaming responses for both:

- Chat Completions API
- Responses API

### Tool calls

Tool calls are supported through shim parsing of CLI output blocks like:

```xml
<tool_call>{...}</tool_call>
```

Only tools advertised by the client request are forwarded as callable tool payloads.

### Compatibility probe endpoints

To avoid noisy probe failures from clients, these compatibility routes also return benign responses:

- `GET /api/v1/models`
- `GET /api/tags`
- `GET /v1/props`
- `GET /props`
- `GET /version`

---

## How first-run bootstrap works

When you run:

```bash
npx @zerople/hermes-shim-http ...
```

the Node launcher will:

1. locate `python3` (or another usable Python 3 executable)
2. create a cached virtualenv in `~/.cache/hermes-shim-http/`
3. install pinned Python dependencies from `requirements.txt`
4. launch `python -m hermes_shim_http.server`

This means most users do **not** need to manually create a venv to use the package.

---

## Troubleshooting

### `Python 3 is required but was not found in PATH`

Install Python 3 and make sure one of these works in your shell:

```bash
python3 --version
# or
python --version
```

### `Python virtual environment support is missing` / `ensurepip is not available`

The launcher needs `python3 -m venv` to work on first run. If your system Python was installed without venv support, the launcher now reports a friendlier error and suggests the package to install.

On Debian/Ubuntu/WSL, try:

```bash
sudo apt update
sudo apt install python3-venv
```

If your distro uses versioned packages, install the one that matches your Python version instead, for example:

```bash
sudo apt install python3.12-venv
# or
sudo apt install python3.11-venv
```

You can also run the built-in preflight check:

```bash
npx @zerople/hermes-shim-http --doctor --command claude --cwd "$(pwd)"
```

### My CLI command is not found

Make sure the command you pass in `--command` is already installed and available in your shell:

```bash
claude --help
codex --help
opencode --help
```

### `FileNotFoundError: [Errno 2] No such file or directory: '/path/to/project'`

This means you passed a `--cwd` value that does not exist on your machine. The `/path/to/project` string in the docs is **only a placeholder** — you must replace it with a real, existing directory.

Fix it by either:

- `cd` into your real project and use `--cwd "$(pwd)"`, or
- pass an absolute path you know exists, e.g. `--cwd /ABSOLUTE/PATH/TO/YOUR/PROJECT` (replace with your actual directory)

You can confirm the path exists before launching:

```bash
ls -ld /ABSOLUTE/PATH/TO/YOUR/PROJECT
```

### The shim starts but the CLI behaves oddly

Check the `--cwd` value first. Many tool-use and file-access issues happen because the wrapped CLI is running in the wrong working directory.

Note that `--cwd` only sets the working directory for the **wrapped CLI subprocess**. It does not change the working directory of the client that consumes the shim's responses. If the model emits tool calls with relative paths, the consuming client may resolve them against a different directory. Prefer absolute paths in tool-call arguments when possible.

### First run is slow

That is expected. The launcher creates a cached Python environment and installs dependencies on the first run. Later runs should be faster.

If first-run bootstrap fails, run:

```bash
npx @zerople/hermes-shim-http --doctor --command claude --cwd "$(pwd)"
```

before retrying the normal launch command.

---

## Development

Clone the repository and run tests locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
npm test
```

Useful commands:

```bash
node bin/hermes-shim-http.js --help
node bin/hermes-shim-http.js --doctor --command claude --cwd "$(pwd)"
npm pack
```

---

## Additional documentation

- Maintainers: `docs/maintainers/releasing.md`
- Architecture/background: `docs/architecture/hermes-integration.md`

---

## License

MIT
