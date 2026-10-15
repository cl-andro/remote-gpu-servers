# Muse-Glimmer-30B — remote control from your PC

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **multimodal + speculative decoding**
model of the collection.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Muse-Glimmer-30B is an advanced 30B parameter model featuring native multimodal
capabilities (image understanding) combined with speculative decoding. By pairing
the base model with a companion perception projector (`mmproj`) and a lightweight
`dflash` drafter model, it provides both vision inputs and accelerated generation
speeds.

## Why this one

| Property | Value |
| --- | --- |
| Architecture | 30B dense + perception encoder + speculative drafter |
| Base weights | `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` (16.76 GB) |
| Multimodal projector | `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` (1.4 GB) — **enabled** |
| Drafter model | `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` (1.63 GB) — **enabled** |
| Total VRAM footprint | ~19.8 GB weights + KV cache, across 32 GB (2x T4) |
| Native context | 65536 configured |
| Sampling defaults | `--temp 1.0 --top-p 0.95 --top-k 64` |
| Minimum llama.cpp | `b10353`+ (pinned to `b11636`) |
| Vision / image input | Yes — native OpenAI-compatible image URLs / base64 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + a sample chat

# or use the agent CLI from the repo root:
../bin/cluster muse-glimmer-30b agent
```

Startup takes several minutes: ~19.8 GB cumulative download (base + mmproj + drafter) plus load.

## Multimodal and Speculative Decoding

### Vision (Image Input)
With `__MMPROJ_FILE__` enabled, `llama-server` automatically accepts standard OpenAI
multimodal payload formats with image URLs or data URIs:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image in detail."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ]
}
```

### Speculative Decoding (DFlash Drafter)
The `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` draft model generates speculative tokens
that the main model verifies in parallel, dramatically increasing tokens/s on
memory-bandwidth-constrained hardware like the T4.

## Agent mode

```bash
../bin/cluster muse-glimmer-30b up      # push + wait for the URL
../bin/cluster muse-glimmer-30b agent   # full TUI against the remote model
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
./ctr.sh chat "Analyze this problem: ..."
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
| `__MODEL_FILE__` | `…-KQuant-17GB-Q4_K_M.gguf` | 16.76 GB base model |
| `__MMPROJ_FILE__` | `mmproj-…-Q4_K_M.gguf` | 1.4 GB vision encoder (blank to disable) |
| `__DRAFT_FILE__` | `dflash-…-Q4_K_M.gguf` | 1.63 GB drafter for speculative decoding |
| `__CTX__` | `65536` | Full 64k context |
| `__EXTRA_ARGS__` | `--temp 1.0 --top-p 0.95 --top-k 64` | Recommended sampling parameters |
| `__SPLIT_MODE__` | `layer` | Shard layers across both T4s |
| `__SERVE_MINUTES__` | `660` | Kaggle caps sessions at 12 h |
| `__LLAMA_BUILD__` | `b11636` | Must be ≥ `b10353` for Muse-Glimmer architecture |

## Notes and gotchas

- **Architecture requirement.** Older llama.cpp builds do not recognise the
  Muse-Glimmer architecture. Ensure `__LLAMA_BUILD__` is at least `b10353`.
- **Combined footprint.** Base + mmproj + drafter takes ~19.8 GB total weights,
  fitting within the 32 GB dual-T4 VRAM pool with ~12 GB headroom for KV cache.
- **`-np 1`.** The launcher pins a single parallel slot so one request gets the
  whole context instead of `ctx / slots`.
- **GGUF only.** llama.cpp is required; `transformers` / vLLM cannot load it.
- **The tunnel URL is public.** Treat it as a shared secret and stop the session
  when you are done.
- **Quota:** a running GPU session consumes your weekly Kaggle GPU hours.
- UI fallback to stop: open `https://www.kaggle.com/code/zkmohammad/muse-glimmer-30b-server`
  → *Stop Session*.
