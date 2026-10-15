# SDXL-Turbo — Real-Time 1-Step Text-to-Image Generation

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **ultra-fast 1-step real-time image** generator.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

SDXL-Turbo is Stability AI's adversarial diffusion distillation (ADD) model, synthesizing high-fidelity images in a single inference step (~1.5s on Tesla T4). It runs with a self-contained Web UI and OpenAI-compatible `/v1/images/generations` endpoint.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | SDXL-Turbo (Adversarial Diffusion Distillation) |
| Weights | Single-file FP16 (~6.0 GB) from `stabilityai/sdxl-turbo` |
| Inference Speed | 1 step (~1.5s per 512x512 image on Tesla T4) |
| Resolution | 512x512 (optimal), 768x768, 1024x1024 |
| VRAM Footprint | ~6.5 GB on GPU (runs comfortably on single or dual T4) |
| Interfaces | Built-in HTML5 Web UI + OpenAI `/v1/images/generations` |
| License | Stability AI Community License (Non-Commercial Research / Personal) |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + generates a sample image (512x512)

# generate an image from your terminal:
./ctr.sh gen "A cinematic shot of an astronaut riding a white horse on Mars, photorealistic, 8k" -o test.png

# or open the interactive Web UI in your browser:
cat url.txt
```

## Interactive Web UI & API

Opening the tunnel URL (`cat url.txt`) in your browser provides a clean Web UI:
- Multi-line prompt input
- Resolution selector (`512x512`, `768x768`, `1024x1024`)
- Real-time step control (1–4 steps)
- Live preview and one-click download

It also exposes an OpenAI-compatible API:
```bash
curl -X POST "$(cat url.txt)/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cinematic shot of an astronaut riding a white horse on Mars, photorealistic, 8k", "size": "512x512", "response_format": "b64_json"}'
```


## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
