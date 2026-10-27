# Sage-A1.5-30B — Long-Horizon Agentic & Scientific Reasoning Model

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Sage-A1.5 is an agentic MoE model (~35B/30B class) specialized for deep multi-step problem solving, scientific analysis, and tool-driven workflows.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Sage-A1.5 MoE (~35B total) |
| Quantization | `Sage-A1.5-Q4_K_M.gguf` (20.22 GB) |
| Model Repo | `InternScience/Sage-A1.5-Q4_K_M-GGUF` |
| Context Window | 32,768 tokens |
| VRAM Allocation | Layer-split across 2x T4 (16 GB each = 32 GB total) |
| Best For | Agentic reasoning, long-horizon problem solving, deep code analysis |
| License | Apache-2.0 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Explain multi-agent debate architecture."

# launch opencode agent against this server:
cluster sage-30b agent
```
