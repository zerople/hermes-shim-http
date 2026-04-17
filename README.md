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
- exposes `chat/completions`, `responses`, and `models` endpoints
- bootstraps its Python runtime automatically on first run
- keeps Hermes as the real tool executor when used as a Hermes provider

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
- at least one supported CLI installed and working:
  - `claude`
  - `codex`
  - `opencode`

> On first run, the launcher automatically creates a cached Python virtual environment under `~/.cache/hermes-shim-http/` and installs the pinned Python dependencies for you.

---

## Quick start

### 1) Run it with `npx`

```bash
npx @zerople/hermes-shim-http \
  --host 127.0.0.1 \
  --port 8765 \
  --command claude \
  --cwd /path/to/project \
  --model claude-cli
```

If you prefer Codex or OpenCode:

```bash
npx @zerople/hermes-shim-http --command codex --cwd /path/to/project --model codex-cli
npx @zerople/hermes-shim-http --command opencode --cwd /path/to/project --model opencode-cli
```

### 2) Confirm the server is up

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/v1/models
```

### 3) Send a test request

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-cli",
    "messages": [
      {"role": "user", "content": "Say hello in one short sentence."}
    ]
  }'
```

---

## Recommended usage with Hermes Agent

Start the shim:

```bash
npx @zerople/hermes-shim-http \
  --host 127.0.0.1 \
  --port 8765 \
  --command claude \
  --cwd /tmp/hermes-agent \
  --model claude-cli
```

Then point Hermes to the local provider:

```yaml
model:
  default: claude-cli
  provider: custom
  base_url: http://127.0.0.1:8765/v1
  api_mode: chat_completions

custom_providers:
  - name: claude-shim
    base_url: http://127.0.0.1:8765/v1
    api_key: no-key-required
    api_mode: chat_completions
    model: claude-cli
```

If you want to use the Responses API style instead:

```yaml
custom_providers:
  - name: claude-shim-responses
    base_url: http://127.0.0.1:8765/v1
    api_key: no-key-required
    api_mode: codex_responses
    model: claude-cli
```

---

## CLI options

Current launcher help:

```text
--host HOST
--port PORT
--command COMMAND
--cwd CWD
--timeout TIMEOUT
--model MODELS
--profile {auto,claude,codex,opencode,generic}
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
  Model name reported back to API clients. This does **not** have to be a real remote model ID; it is mainly the model label your client will send in requests.

- `--profile`  
  Command profile. `auto` usually picks a sensible default based on `--command`.

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

### My CLI command is not found

Make sure the command you pass in `--command` is already installed and available in your shell:

```bash
claude --help
codex --help
opencode --help
```

### The shim starts but the CLI behaves oddly

Check the `--cwd` value first. Many tool-use and file-access issues happen because the wrapped CLI is running in the wrong working directory.

### First run is slow

That is expected. The launcher creates a cached Python environment and installs dependencies on the first run. Later runs should be faster.

---

## Development

Clone the repository and run tests locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

Useful commands:

```bash
node bin/hermes-shim-http.js --help
npm pack
```

---

## Additional documentation

- Maintainers: `docs/maintainers/releasing.md`
- Architecture/background: `docs/architecture/hermes-integration.md`

---

## License

MIT
