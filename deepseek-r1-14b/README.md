# DeepSeek-R1-Distill-Qwen-14B — Reasoning & Math Flagship

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **advanced reasoning and chain-of-thought** model.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Distilled directly from DeepSeek-R1 (671B MoE) into Qwen-2.5-14B, this model generates detailed `<think>...</think>` thought chains before answering. It achieves stellar scores on AIME 2024, MATH-500, and competition-level algorithmic challenges.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | DeepSeek-R1 Distill (14B Dense) |
| Quantization | `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf` (8.37 GB) |
| Context Window | 32,768 tokens |
| Reasoning Tokens | Native `<think>` internal monologue generation |
| VRAM Footprint | ~10 GB total (dual T4s provide vast KV buffer) |
| Best For | Complex mathematics, logic puzzles, algorithm derivation, step-by-step proofs |
| License | MIT |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Prove that the sum of the squares of two odd numbers cannot be a square."

# launch opencode agent against this server:
../bin/cluster deepseek-r1-14b agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster deepseek-r1-14b stop`.


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
