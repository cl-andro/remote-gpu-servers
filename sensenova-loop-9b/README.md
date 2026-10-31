# SenseNova-Loop-9B-Preview — Looped Transformer with Dual-Stream mHC & 256K Context

The model runs **on Kaggle** (Dual Tesla T4 GPUs). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

SenseNova-Loop-9B-Preview is a breakthrough multimodal reasoning model developed by SenseTime (`sensenova`). It introduces a **Looped Transformer architecture** combined with **two-stream manifold-constrained hyper-connections (mHC)**:
- While storing only 32 physical layers (~9B parameters in memory), layers 4–27 are executed twice sequentially (`B(B(h))`), achieving an effective computational depth of **56 layer operations** (~14.6B inference compute scale).
- Supports long-context reasoning up to **262,144 tokens (256K)**.
- Features dual-mode reasoning: deep chain-of-thought (`enable_thinking: true` with `<think>...</think>` traces) and fast direct response mode.
- Supports multimodal vision-language inputs (text + images).

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Looped Transformer Decoder + Two-Stream mHC + Hybrid Attention |
| Physical Parameters | ~9B (~18 GB in BF16/FP16) |
| Effective Compute Depth | 56 Layers (Layers 4–27 executed 2x) |
| Context Window | Up to 262,144 tokens (default configured to 32,768) |
| Model Repo | [`sensenova/SenseNova-Loop-9B-Preview`](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview) |
| VRAM Allocation | Sharded across Kaggle's dual Tesla T4 GPUs (~9 GB each = 18 GB / 32 GB total) |
| Reasoning Support | Native Chain-of-Thought (`enable_thinking` toggle & reasoning traces) |
| Modalities | Multimodal: Text & Vision (Images) |
| Web UI | Full-featured streaming chat studio with thinking accordion & image uploads |
| API | OpenAI-compatible `/v1/chat/completions` (streaming SSE & non-streaming) |
| License | Open-weights / Apache-2.0 derivative |

## Quick start

```bash
# Push the notebook to Kaggle and start the dual T4 GPU run
cluster sensenova-loop-9b up

# Check status and fetch the public Cloudflare tunnel URL
cluster sensenova-loop-9b status
cluster sensenova-loop-9b url

# Run health check & test prompt
cluster sensenova-loop-9b test

# Single-shot chat from terminal (streams with thinking trace)
cluster sensenova-loop-9b chat "Explain how looped transformers achieve 56-layer depth with 32 physical layers."

# Single-shot chat with direct answer (bypassing reasoning)
cluster sensenova-loop-9b chat --no-think "Write a Python script to find anagrams in a list."

# Interactive multi-turn REPL in terminal
cluster sensenova-loop-9b repl

# Launch OpenCode agent against SenseNova-Loop-9B
cluster sensenova-loop-9b agent

# Open Interactive Web Studio in your browser
# Visit the URL printed by:
cluster sensenova-loop-9b url

# Stop the server and release Kaggle GPU when finished
cluster sensenova-loop-9b down
```
