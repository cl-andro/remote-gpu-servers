# Qwen3.5-4B-3D-Coder — Interactive Three.js 3D Scene & Asset Generator

The model runs **on Kaggle** (GPU Tesla T4). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Qwen3.5-4B-3D-Coder is an open-source fine-tune of `Qwen/Qwen3.5-4B` by developer **CCORE**, specialized in translating natural language prompts and reference images into complete, self-contained, interactive 3D browser scenes using **Three.js** (HTML/JavaScript).

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Qwen3.5 (4B Dense with Gated DeltaNet Hybrid Attention) |
| Quantization | `Qwen3.5-4B-3D-Coder-Q4_K_M.gguf` (2.71 GB) |
| Vision Projector | `mmproj-Qwen3.5-4B-BF16.gguf` (0.68 GB, optional image input support) |
| Model Repo | [`ccore/Qwen3.5-4B-3D-Coder`](https://huggingface.co/ccore/Qwen3.5-4B-3D-Coder) |
| Context Window | 32,768 tokens (supports up to 65,536) |
| VRAM Allocation | Fits entirely in ~3.4 GB VRAM on a single Tesla T4 GPU |
| Primary Specialization | Interactive Three.js 3D browser scenes, animations, procedural meshes, shader materials |
| Modalities | Text & Image-to-3D (via multimodal vision projector) |
| License | Apache-2.0 |

## Quick start

```bash
# Push the notebook to Kaggle and start the GPU run
cluster qwen3.5-4b-3d-coder up

# Check status and fetch the public Cloudflare tunnel URL
cluster qwen3.5-4b-3d-coder status
cluster qwen3.5-4b-3d-coder url

# Health check & test prompt
cluster qwen3.5-4b-3d-coder test

# Single-shot 3D scene generation from terminal:
cluster qwen3.5-4b-3d-coder chat "Create a self-contained HTML file with a rotating wireframe torus knot using Three.js with bloom lighting."

# Launch OpenCode agent against this 3D coding model:
cluster qwen3.5-4b-3d-coder agent

# Stop when done to release GPU
cluster qwen3.5-4b-3d-coder down
```
