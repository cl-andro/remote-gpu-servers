# DeepSeek-R1-Distill-Llama-8B — Fast Reasoning Engine

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **low-latency reasoning and thinking** specialist.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Distilled from DeepSeek-R1 into Llama-3.1-8B, this model offers rapid `<think>` chain-of-thought generation with minimal VRAM overhead and blistering token generation speeds on Kaggle GPUs.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | DeepSeek-R1 Distill (8B Dense, Llama-3.1 base) |
| Quantization | `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf` (4.58 GB) |
| Context Window | 32,768 tokens |
| VRAM Footprint | ~5.5 GB total |
| Speed | Fastest chain-of-thought generation in the cluster |
| Best For | Real-time math verification, structured reasoning, quick logic queries |
| License | Llama 3.1 Community License |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Explain Gödel's First Incompleteness Theorem step by step."

# launch opencode agent against this server:
../bin/cluster deepseek-r1-8b agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster deepseek-r1-8b stop`.


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
