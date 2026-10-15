# Qwen2.5-Coder-14B-Instruct — High-Speed Code Intelligence

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **mid-weight coding specialist** of the collection.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Qwen2.5-Coder-14B-Instruct delivers an exceptional balance of speed and coding prowess. Trained on over 5.5 trillion tokens of code, math, and documentation, it punches well above its weight on HumanEval, MultiPL-E, and real-world multi-file refactoring.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Qwen2.5-Coder 14B Dense |
| Quantization | `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` (8.37 GB) |
| Context Window | 65,536 tokens (native 128k supported) |
| VRAM Footprint | ~10 GB total (comfortably inside dual T4s with massive KV cache) |
| Best For | Interactive coding assistant, fast unit tests, API implementation |
| License | Apache-2.0 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Write a high-performance async connection pool in Python with backpressure."

# launch opencode agent against this server:
../bin/cluster qwen2.5-coder-14b agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster qwen2.5-coder-14b stop`.


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
