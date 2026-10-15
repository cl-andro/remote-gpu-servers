# Qwen2.5-Coder-32B-Instruct — Flagship Open-Source Code Model

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **heavyweight coding champion** of the collection.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Qwen2.5-Coder-32B-Instruct is widely considered the strongest open-weights code model available, matching and often outperforming GPT-4o on HumanEval (92.7%) and SWE-bench Verified. Sharded across Kaggle's dual Tesla T4 GPUs (32 GB total VRAM), it fits with ~13 GB VRAM remaining for active KV cache.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Qwen2.5-Coder 32B Dense |
| Quantization | `Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf` (18.49 GB) |
| Context Window | 32,768 tokens |
| VRAM Allocation | Layer-split across 2x T4 (16 GB each = 32 GB total) |
| KV Cache Headroom | ~13.5 GB free for KV cache |
| Best For | Complex repository refactors, architecture design, SWE-bench debugging |
| License | Apache-2.0 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Debug this race condition in a distributed Raft consensus implementation."

# launch opencode agent against this server:
../bin/cluster qwen2.5-coder-32b agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster qwen2.5-coder-32b stop`.


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
