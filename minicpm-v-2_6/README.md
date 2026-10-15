# MiniCPM-V-2.6 — High-Resolution Multimodal OCR & Dense Extraction

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **dense multimodal OCR and multi-image** specialist.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

MiniCPM-V 2.6 is OpenBMB's leading edge multimodal model (8B parameters). It surpasses proprietary models like GPT-4V and Claude 3.5 Sonnet on single-image and multi-image benchmarks, processing images up to 1.8M pixels with arbitrary aspect ratios. Ideal for dense table extraction, full-page OCR, and multi-page document synthesis.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | MiniCPM-V 2.6 (8B parameters, SigLIP-400M vision encoder) |
| Quantization | `ggml-model-Q4_K_M.gguf` (4.36 GB) |
| Vision Projector | `mmproj-model-f16.gguf` (0.97 GB) |
| Context Window | 32k tokens |
| VRAM Footprint | ~5.8 GB total |
| Best For | High-res table OCR, multi-image comparison, scientific papers |
| License | Apache-2.0 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "Transcribe every table row from this high-resolution document scan."

# launch opencode agent against this server:
../bin/cluster minicpm-v-2_6 agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster minicpm-v-2_6 stop`.


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
