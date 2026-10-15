# Devstral-Small-2-24B — remote control from your PC

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **dense coder** model of the
collection.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Devstral-Small-2-24B is a dense 24B parameter code-specialized model from Mistral AI
(quantized by Unsloth) engineered specifically for agentic coding, reasoning, and
multi-turn development workflows. It offers high code precision with a simple,
robust dense architecture that avoids MoE routing overhead.

## Why this one

| Property | Value |
| --- | --- |
| Total / active params | Dense 24B (all 24B active per token) |
| Quantization | `UD-Q4_K_XL`, 14.51 GB |
| VRAM fit | ~15 GB weights + KV cache, comfortable fit across 32 GB (2x T4) |
| Native context | 65536 configured (native max up to 128k) |
| Tool calling | native, purpose-built for coding agents |
| Vision | optional (set `__MMPROJ_FILE__` in `kernel_config.py`) |
| License | Apache-2.0 |

Because it is a dense 24B model, it fits comfortably across the dual T4s with
generous headroom for long contexts while delivering consistent generation quality.

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + a sample chat

# or use the agent CLI from the repo root:
../bin/cluster devstral-small-2 agent
```

Startup takes several minutes: ~14.5 GB model download plus load.

## Agent mode

This project integrates directly with the `cluster` CLI and upstream
[opencode](https://opencode.ai) agent:

```bash
../bin/cluster devstral-small-2 up      # push + wait for the URL
../bin/cluster devstral-small-2 agent   # full TUI against the remote model
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
./ctr.sh chat "Write a Python script to monitor disk usage."
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
| `__MODEL_FILE__` | `…-UD-Q4_K_XL.gguf` | 14.51 GB |
| `__CTX__` | `65536` | Offloads full 64k context |
| `__NGL__` | `99` | Offload every layer to GPU |
| `__SPLIT_MODE__` | `layer` | Shard layers across both T4s |
| `__SERVE_MINUTES__` | `660` | Kaggle caps sessions at 12 h |
| `__MMPROJ_FILE__` | `""` | Set to vision projector to enable multimodal |
| `__LLAMA_BUILD__` | `b11636` | Pinned stable release |

## Notes and gotchas

- **Layer splitting.** Spreading the 24B dense layers across both T4s fits
  weights + 65k context easily.
- **`-np 1`.** The launcher pins a single parallel slot so one request gets the
  whole context instead of `ctx / slots`.
- **GGUF only.** llama.cpp is required; `transformers` / vLLM cannot load it.
- **The tunnel URL is public.** Treat it as a shared secret and stop the session
  when you are done.
- **Quota:** a running GPU session consumes your weekly Kaggle GPU hours.
- UI fallback to stop: open `https://www.kaggle.com/code/zkmohammad/devstral-small-2-server`
  → *Stop Session*.
