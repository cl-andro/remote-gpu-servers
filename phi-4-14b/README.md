# Phi-4 — Microsoft 14B STEM & Analytical Reasoning Champion

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **STEM and academic reasoning** champion.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Phi-4 is Microsoft's 14B state-of-the-art dense model trained on high-quality synthetic textbooks, academic papers, and curated reasoning corpora. It outperforms substantially larger models on GPQA, MATH, MMLU-Pro, and STEM-intensive reasoning benchmarks.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Phi-4 (14B Dense) |
| Quantization | `phi-4-Q4_K_M.gguf` (8.28 GB) |
| Context Window | 16,384 tokens |
| VRAM Footprint | ~9.8 GB total |
| Best For | Academic research questions, physics and chemistry modeling, calculus, STEM Q&A |
| License | MIT |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Derive the Navier-Stokes equations from conservation of momentum."

# launch opencode agent against this server:
../bin/cluster phi-4-14b agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster phi-4-14b stop`.


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
