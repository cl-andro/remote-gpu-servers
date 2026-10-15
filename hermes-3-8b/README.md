# Hermes-3-Llama-3.1-8B — Agentic Tool Calling & Function Execution

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **autonomous agentic & tool-calling** specialist.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Hermes 3 is Nous Research's flagship generalist and agentic model. It is fine-tuned specifically for structured JSON generation, tool calling, multi-step agent loops, roleplay, and strict system-prompt adherence.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Hermes-3 (8B Dense, Llama-3.1 base) |
| Quantization | `Hermes-3-Llama-3.1-8B.Q4_K_M.gguf` (4.58 GB) |
| Context Window | 32,768 tokens (supports up to 128k) |
| Agent Features | Advanced XML/JSON tool calling, scratchpad reasoning |
| VRAM Footprint | ~5.5 GB total |
| Best For | Autonomous agent workflows, function calling, JSON API orchestration |
| License | Llama 3.1 Community License |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Call the search_weather tool for Paris and Tokyo and summarize results."

# launch opencode agent against this server:
../bin/cluster hermes-3-8b agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster hermes-3-8b stop`.


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
