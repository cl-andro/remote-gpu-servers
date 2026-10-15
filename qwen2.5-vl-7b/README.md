# Qwen2.5-VL-7B-Instruct — Vision-Language & Document OCR

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **Vision-Language & OCR** model of the collection.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Qwen2.5-VL-7B-Instruct is Alibaba Cloud's flagship open-weights vision-language model, equipped with native dynamic resolution processing. It excels at document OCR, complex infographic and table extraction, chart reasoning, video analysis, and grounded agentic screen interaction.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Qwen2.5-VL (7B Dense + Vision Transformer) |
| Quantization | `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (4.36 GB) |
| Vision Projector | `mmproj-F16.gguf` (1.26 GB) |
| Context Window | 32k tokens (supports up to 128k native) |
| VRAM Footprint | ~6.0 GB total (fits in a single T4 with headroom) |
| Best For | Document OCR, scanned PDF extraction, chart & diagram analysis |
| License | Apache-2.0 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Analyze this scanned invoice and extract all items in JSON format."

# launch opencode agent against this server:
../bin/cluster qwen2.5-vl-7b agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster qwen2.5-vl-7b stop`.


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
