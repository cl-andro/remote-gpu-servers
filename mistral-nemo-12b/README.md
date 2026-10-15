# Mistral-Nemo-Instruct-2407 — 128k Long-Context General Intelligence

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **128k long-context and multilingual** text workhorse.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Developed jointly by Mistral AI and NVIDIA, Mistral-Nemo is a 12.2B parameter dense model featuring the new Tekken tokenizer (128k vocabulary). It is exceptionally strong at long-document synthesis, multi-lingual dialogue, and structured data handling.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Mistral-Nemo 12.2B Dense |
| Quantization | `Mistral-Nemo-Instruct-2407.Q4_K_M.gguf` (6.96 GB) |
| Tokenizer | Tekken (128k vocab, ultra-efficient compression) |
| Context Window | 65,536 tokens (up to 128k native) |
| VRAM Footprint | ~8.5 GB total |
| Best For | Long-document summarization, multilingual translation, general writing |
| License | Apache-2.0 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Synthesize key findings across a 40-page technical specification."

# launch opencode agent against this server:
../bin/cluster mistral-nemo-12b agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster mistral-nemo-12b stop`.


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
