# gpt-oss-20b — remote control from your PC

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **lightweight** model of the
collection.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

gpt-oss-20b is OpenAI's open-weight **MoE** model: 21B parameters total, ~3.6B
active per token, shipped natively in **MXFP4** so it needs no quantization
round-trip. At ~12 GB it fits inside a *single* T4, leaving the second GPU free
for context — the roomiest setup in this repo.

## Why this one

| Property | Value |
| --- | --- |
| Total / active params | 21B total / ~3.6B active (MoE) |
| Quantization | native `MXFP4`, 12.11 GB |
| VRAM fit | fits one T4 (16 GB); ~65k context uses little of the second |
| Native context | 131072 (this config sets 65536) |
| Tool calling | yes — harmony format (needs a recent llama.cpp) |
| License | Apache-2.0 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + a sample chat

# or use the agent CLI from the repo root:
../bin/cluster gpt-oss-20b agent
```

Startup takes several minutes: ~12 GB model download plus load.

## Agent mode

```bash
../bin/cluster gpt-oss-20b up      # push + wait for the URL
../bin/cluster gpt-oss-20b agent   # full TUI against the remote model
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
./ctr.sh chat "Summarise the CAP theorem in three sentences."
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
| `__MODEL_FILE__` | `gpt-oss-20b-MXFP4.gguf` | 12.11 GB native MXFP4 |
| `__CTX__` | `65536` | Native max is 131072 |
| `__DRAFT_FILE__` | `""` | Set to `eagle3-gpt-oss-20b-Q8_0.gguf` for speculative decoding |
| `__SERVE_MINUTES__` | `660` | Kaggle caps sessions at 12 h |
| `__LLAMA_BUILD__` | `b11636` | Must be recent enough to parse the harmony format |

## Notes and gotchas

- **Harmony format.** gpt-oss uses the harmony chat format for reasoning and
  tool calls; an old llama.cpp build will not parse it correctly. Keep
  `__LLAMA_BUILD__` current.
- **`-np 1`.** The launcher pins a single parallel slot so one request gets the
  whole context instead of `ctx / slots`.
- **GGUF only.** llama.cpp is required; `transformers` / vLLM cannot load it.
- **The tunnel URL is public.** Treat it as a shared secret and stop the session
  when you are done.
- **Quota:** a running GPU session consumes your weekly Kaggle GPU hours.
- UI fallback to stop: open `https://www.kaggle.com/code/zkmohammad/gpt-oss-20b-server`
  → *Stop Session*.