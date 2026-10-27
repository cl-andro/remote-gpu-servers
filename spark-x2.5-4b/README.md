# Spark-X2.5-4B — Ultra-Fast Hybrid Attention Dense Model

The model runs **on Kaggle** (GPU T4 x2 or single T4). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Spark-X2.5-4B (by iFLYTEK / SparkLLM) is an ultra-fast, compact 4B general-purpose model with hybrid attention (full attention + sliding window) supporting up to 1M native context and 200+ languages.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Spark-X2.5 (Dense 4B) |
| Quantization | `Spark-X2.5-4B-Q4_K_M.gguf` (2.42 GB) |
| Model Repo | `XHToken/Spark-X2.5-4B-GGUF` |
| Context Window | 32,768 tokens (supports up to 1M) |
| VRAM Allocation | Fits entirely in <4 GB VRAM on Tesla T4 |
| Best For | Ultra-fast agent loops, low-latency coding assistance, quick scripts |
| License | Open weights |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Write a Python script to monitor CPU temperature."

# launch opencode agent against this server:
cluster spark-x2.5-4b agent
```
