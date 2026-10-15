# Underdog-Saluki-27B — remote control from your PC

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · part of the **Remote GPU Servers**
collection of remote-GPU inference setups.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

```
   your PC                    Kaggle (GPU)                         internet
 ┌──────────┐   kaggle API   ┌──────────────────────────┐
 │ ctr.sh   │ ─────────────► │ notebook.ipynb           │
 │cluster.py│                │  ├─ download GGUF 7.9 GB │
 └────┬─────┘                │  ├─ llama-server :8080   │
      │  HTTPS               │  └─ cloudflared tunnel   │
      └───────────────────────────────────────────────► trycloudflare.com
```

---

## Quick start

```bash
cd saluki

./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + a sample chat

./ctr.sh chat "Explain gated delta networks in two sentences."
./ctr.sh repl          # interactive multi-turn chat

# or use the agent CLI from the repo root:
../bin/cluster saluki agent
```

Startup takes a few minutes: ~8 GB model download plus model load.

## Commands

| Command | What it does |
| --- | --- |
| `./ctr.sh push` | Push `notebook.ipynb` and start/restart the run |
| `./ctr.sh url` | Scrape the live logs for the tunnel URL → `url.txt` |
| `./ctr.sh status` | Run state (`RUNNING` / `COMPLETE` / `ERROR`) |
| `./ctr.sh logs` | Print the log so far |
| `./ctr.sh follow` | Live-tail the kernel logs (Ctrl-C to detach) |
| `./ctr.sh test` | Health + sample chat through the tunnel |
| `./ctr.sh stop` | Stop the running session from your PC (frees the GPU) |
| `./ctr.sh stop-url` | Print/cache the stop URL for the current run |
| `./ctr.sh output` | Download the run's output files |

### Stopping the session from your PC

Kaggle's public API token **cannot** cancel a session — `kernelSessions.cancel`
is not among the token's scopes (the endpoint returns
`403 Permission 'kernelSessions.cancel' was denied`). So the notebook hosts a
tiny control server of its own, published through a **second** cloudflared
tunnel, and `./ctr.sh stop` calls it:

```bash
./ctr.sh stop        # asks the kernel to exit, then waits for COMPLETE
```

The kernel polls the flag every 5 s and exits cleanly; `stop` then waits (up to
~10 min) for Kaggle to report `COMPLETE`/`ERROR`, so a plain
`./ctr.sh status` right afterwards normally shows the final state.

The stop URL contains a random secret path (`/stop/<hex>`) and is printed in
the logs as `STOP_URL=...`; it is scraped on demand and cached in `stop_url.txt`.
Restarting the run issues a fresh URL, so re-run `./ctr.sh stop-url` if needed.

Each run publishes **two** cloudflared tunnels:

| Tunnel | File | Points at | Purpose |
| --- | --- | --- | --- |
| chat | `url.txt` | `127.0.0.1:8080` (llama-server) | inference, OpenAI API |
| control | `stop_url.txt` | `127.0.0.1:8081` (stdlib server) | `/health` and secret `/stop/<hex>` |

## Client

The client is stdlib-only (`../lib/cluster.py`) — no `pip install` needed on your PC.

```bash
./ctr.sh health                       # {"status":"ok"}
./ctr.sh props                        # model + context info
./ctr.sh models                       # OpenAI-compatible model list
./ctr.sh slots                        # active context slots

./ctr.sh chat "hello"                 # streaming, thinking ON
./ctr.sh chat --no-think "2+2?"       # fast, no reasoning phase
./ctr.sh repl                         # multi-turn; /new /think /nothink /quit
./ctr.sh tools "weather in Paris?"    # OpenAI tool-call demo
./ctr.sh tokenize "some text"         # token count + ids
./ctr.sh raw /props '{}'              # arbitrary POST to any path
```

Useful flags: `--system`, `--temperature`, `--top-p`, `--max-tokens`, `--no-think`.
URL resolution order: `--url` → `$CLUSTER_URL` → `url.txt`.

### Use it from any other tool

It is a plain OpenAI-compatible endpoint, so anything that speaks the OpenAI API
works. Example:

```bash
export OPENAI_BASE_URL="$(cat url.txt)/v1"
export OPENAI_API_KEY="not-needed"
# then point your favourite client at model "saluki"
```

## Measured on Kaggle T4 x2

| Metric | Value |
| --- | --- |
| Throughput | ~10 tok/s (IQ2-mix, 27B) |
| Context | 32768 (native max 262144, raise `__CTX__`) |
| Model | `Underdog-Saluki-27B-1.0-IQ2-mix.gguf`, 7.89 GB |
| Runtime | llama.cpp `b11636`, CUDA 12.8 build |

## Configuration

Edit [`kernel_config.py`](kernel_config.py), then re-run `./ctr.sh push` (which
regenerates the notebook first).

| Key | Default | Notes |
| --- | --- | --- |
| `__CTX__` | `32768` | Raise toward 262144 if you want long context |
| `__NGL__` | `99` | Layers on GPU |
| `__PORT__` | `8080` | llama-server port (chat tunnel target) |
| `__CTL_PORT__` | `8081` | Control server port (stop tunnel target) |
| `__SERVE_MINUTES__` | `660` | Kaggle caps sessions at 12 h |
| `__MMPROJ_FILE__` | `""` | Set to `mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf` for vision |
| `__LLAMA_BUILD__` | `b11636` | Must be ≥ `b10330` for the `qwen35` arch |

GPU type lives in `saluki/kernel-metadata.json` (`machine_shape`).

## Notes and gotchas

- **GGUF only.** This repo ships no safetensors, so `transformers` / vLLM cannot
  load it. llama.cpp is required.
- **`qwen35` architecture.** Older llama.cpp builds fail with
  `unknown model architecture: 'qwen35'`.
- **`--jinja` is required** for the Qwen3.8 chat template (thinking + tool calls).
- **Thinking is on by default.** Use `--no-think` (sets
  `chat_template_kwargs.enable_thinking=false`) for fast tool calls.
- **The tunnel URL is public.** Treat it as a shared secret and stop the session
  when you are done.
- **Stopping:** `./ctr.sh stop` calls the notebook's own control tunnel and the
  kernel exits within a few seconds (it polls the stop flag every 5 s). There is
  no Kaggle-API way to do this (the token lacks `kernelSessions.cancel`). UI
  fallback: open `https://www.kaggle.com/code/zkmohammad/saluki-server` →
  *Stop Session*.
- **Two tunnels.** The chat tunnel points at llama-server; the control tunnel
  points at a tiny stdlib HTTP server that only understands `/health` and the
  secret `/stop/<hex>` path.
- **Quota:** a running GPU session consumes your weekly Kaggle GPU hours.
- **`url.txt` / `stop_url.txt` are per-run.** They are git-ignored and are
  rewritten every push — a fresh run issues fresh tunnel URLs.