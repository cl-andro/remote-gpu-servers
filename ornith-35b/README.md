# Ornith-1.5-35B — High-Performance Reasoning & Coding MoE

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Ornith-1.5-35B is an advanced Mixture-of-Experts (MoE) reasoning and coding model based on Qwen architecture (~35B total parameters with ~3B active per token). It delivers deep reasoning, code synthesis, and agentic task execution while staying fast and memory-efficient.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Ornith-1.5 MoE (~35B total, ~3B active) |
| Quantization | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (20.36 GB) |
| Model Repo | `bartowski/Ornith-1.5-35B-A3B-GGUF` |
| Context Window | 32,768 tokens |
| VRAM Allocation | Layer-split across 2x T4 (16 GB each = 32 GB total) |
| Best For | Advanced coding, reasoning, agentic tool workflows |
| License | Apache-2.0 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Write a Python script to scan network anomalies."

# launch opencode agent against this server:
cluster ornith-35b agent
```
