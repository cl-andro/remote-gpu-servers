# Qwen3-Coder-30B-A3B — remote control from your PC

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the intended **agent** model of
the collection.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Qwen3-Coder-30B-A3B is a **sparse MoE** model — 30.5B parameters total but only
~3.3B active per token — which is what makes it the fastest thing here on weak
T4s. It is chat-tuned for agentic coding and native tool calling, at 262144
tokens of native context under Apache-2.0.

## Why this one

| Property | Value |
| --- | --- |
| Total / active params | 30.5B total / ~3.3B active (MoE) |
| Quantization | `UD-Q4_K_XL`, 17.67 GB |
| VRAM fit | ~18 GB weights + KV cache, inside 32 GB across both T4s |
| Native context | 262144 (this config sets 65536) |
| Tool calling | native, purpose-built for coding agents |
| License | Apache-2.0 |

Because only ~3B parameters are active per token, decode stays fast even though
the T4 has modest memory bandwidth — expect clearly more tokens/s than the
dense 27B [saluki](../saluki/) baseline.

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + a sample chat

# or use the agent CLI from the repo root:
../bin/cluster qwen3-coder-30b agent
```

Startup takes several minutes: ~18 GB model download plus load.

## Agent mode

This is the project to use with the `cluster` CLI (a wrapper over the
open-source [opencode](https://opencode.ai) agent):

```bash
../bin/cluster qwen3-coder-30b up      # push + wait for the URL
../bin/cluster qwen3-coder-30b agent   # full TUI against the remote model
```

## Commands

| Command | What it does |
| --- | --- |
| `./ctr.sh push` | Regenerate `notebook.ipynb` and start/restart the run |
| `./ctr.sh up` | `push`, then wait until the tunnel URL appears |
| `./ctr.sh url` | Scrape the live logs for the tunnel URL → `url.txt` |
| `./ctr.sh status` | Run state (`RUNNING` / `COMPLETE` / `ERROR`) |
| `./ctr.sh logs` / `follow` | Print recent logs / live-tail them |
| `./ctr.sh test` | Health + sample chat through the tunnel |
| `./ctr.sh chat "…"` | One-shot chat via the stdlib client |
| `./ctr.sh repl` | Interactive multi-turn chat |
| `./ctr.sh stop` | Stop the running session from your PC (frees the GPU) |
| `./ctr.sh output` | Download the run's output files |

### Stopping the session from your PC

Kaggle's public API token cannot cancel a session, so the notebook hosts a tiny
control server on a second tunnel and `./ctr.sh stop` calls it (secret random
path). The kernel polls the flag every 5 s; `stop` then waits for `COMPLETE`.

## Client

The client is stdlib-only (`../lib/cluster.py`) — no `pip install`.

```bash
./ctr.sh chat "Write a Python function to parse a CIDR range."
./ctr.sh chat --no-think "2+2?"     # Qwen3-Coder has no thinking phase anyway
./ctr.sh repl
./ctr.sh tools "weather in Paris?"  # OpenAI tool-call demo
```

Any OpenAI-compatible client also works:

```bash
export OPENAI_BASE_URL="$(cat url.txt)/v1"
export OPENAI_API_KEY="not-needed"   # model name is ignored by llama-server
```

## Configuration

Edit [`kernel_config.py`](kernel_config.py), then re-run `./ctr.sh push` (which
regenerates the notebook first).

| Key | Value | Notes |
| --- | --- | --- |
| `__MODEL_FILE__` | `…-UD-Q4_K_XL.gguf` | 17.67 GB; use `IQ4_XS` (16.4 GB) for more KV headroom |
| `__CTX__` | `65536` | Native max is 262144 — raise in steps and watch VRAM |
| `__NGL__` | `99` | Offload every layer |
| `__SPLIT_MODE__` | `layer` | Shard layers across both T4s |
| `__SERVE_MINUTES__` | `660` | Kaggle caps sessions at 12 h |
| `__LLAMA_BUILD__` | `b11636` | Pin; bump if a release regresses |

## Notes and gotchas

- **MoE speed.** Spreading layers across both T4s gives the best decode rate;
  keep `__SPLIT_MODE__` at `layer`.
- **`-np 1`.** The launcher pins a single parallel slot so one request gets the
  whole context instead of `ctx / slots`.
- **GGUF only.** llama.cpp is required; `transformers` / vLLM cannot load it.
- **The tunnel URL is public.** Treat it as a shared secret and stop the session
  when you are done.
- **Quota:** a running GPU session consumes your weekly Kaggle GPU hours.
- UI fallback to stop: open `https://www.kaggle.com/code/zkmohammad/qwen3-coder-30b-server`
  → *Stop Session*.