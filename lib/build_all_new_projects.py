#!/usr/bin/env python3
"""Automated builder for the 10 new models in remote-gpu-servers."""

import json
import os
import stat
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CTR_SCRIPT = """#!/usr/bin/env bash
# Thin wrapper: all logic lives in ../lib/ctr.sh (shared by every project).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../lib/ctr.sh" --project "$HERE" "$@"
"""

PROJECTS = {
    "qwen2.5-vl-7b": {
        "cfg": {
            "__PROJECT_NAME__": "Qwen2.5-VL-7B-Instruct",
            "__MODEL_REPO__": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            "__MODEL_FILE__": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            "__MODEL_SIZE_GB__": "4.36",
            "__MODEL_ALIAS__": "qwen2.5-vl-7b",
            "__LLAMA_BUILD__": "b11636",
            "__CTX__": "32768",
            "__NGL__": "99",
            "__SPLIT_MODE__": "layer",
            "__EXTRA_ARGS__": "",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__MMPROJ_FILE__": "mmproj-F16.gguf",
            "__MMPROJ_SIZE_GB__": "1.26",
            "__DRAFT_FILE__": "",
            "__DRAFT_SIZE_GB__": "",
        },
        "title": "qwen2.5-vl-7b-server",
        "doc_title": "Qwen2.5-VL-7B-Instruct — Vision-Language & Document OCR",
        "tagline": "the **Vision-Language & OCR** model of the collection.",
        "desc": "Qwen2.5-VL-7B-Instruct is Alibaba Cloud's flagship open-weights vision-language model, equipped with native dynamic resolution processing. It excels at document OCR, complex infographic and table extraction, chart reasoning, video analysis, and grounded agentic screen interaction.",
        "table": [
            ("Architecture", "Qwen2.5-VL (7B Dense + Vision Transformer)"),
            ("Quantization", "`Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (4.36 GB)"),
            ("Vision Projector", "`mmproj-F16.gguf` (1.26 GB)"),
            ("Context Window", "32k tokens (supports up to 128k native)"),
            ("VRAM Footprint", "~6.0 GB total (fits in a single T4 with headroom)"),
            ("Best For", "Document OCR, scanned PDF extraction, chart & diagram analysis"),
            ("License", "Apache-2.0"),
        ],
        "test_cmd": "./ctr.sh test",
        "sample_prompt": "Analyze this scanned invoice and extract all items in JSON format.",
    },
    "minicpm-v-2_6": {
        "cfg": {
            "__PROJECT_NAME__": "MiniCPM-V-2_6",
            "__MODEL_REPO__": "openbmb/MiniCPM-V-2_6-gguf",
            "__MODEL_FILE__": "ggml-model-Q4_K_M.gguf",
            "__MODEL_SIZE_GB__": "4.36",
            "__MODEL_ALIAS__": "minicpm-v-2_6",
            "__LLAMA_BUILD__": "b11636",
            "__CTX__": "32768",
            "__NGL__": "99",
            "__SPLIT_MODE__": "layer",
            "__EXTRA_ARGS__": "",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__MMPROJ_FILE__": "mmproj-model-f16.gguf",
            "__MMPROJ_SIZE_GB__": "0.97",
            "__DRAFT_FILE__": "",
            "__DRAFT_SIZE_GB__": "",
        },
        "title": "minicpm-v-2-6-server",
        "doc_title": "MiniCPM-V-2.6 — High-Resolution Multimodal OCR & Dense Extraction",
        "tagline": "the **dense multimodal OCR and multi-image** specialist.",
        "desc": "MiniCPM-V 2.6 is OpenBMB's leading edge multimodal model (8B parameters). It surpasses proprietary models like GPT-4V and Claude 3.5 Sonnet on single-image and multi-image benchmarks, processing images up to 1.8M pixels with arbitrary aspect ratios. Ideal for dense table extraction, full-page OCR, and multi-page document synthesis.",
        "table": [
            ("Architecture", "MiniCPM-V 2.6 (8B parameters, SigLIP-400M vision encoder)"),
            ("Quantization", "`ggml-model-Q4_K_M.gguf` (4.36 GB)"),
            ("Vision Projector", "`mmproj-model-f16.gguf` (0.97 GB)"),
            ("Context Window", "32k tokens"),
            ("VRAM Footprint", "~5.8 GB total"),
            ("Best For", "High-res table OCR, multi-image comparison, scientific papers"),
            ("License", "Apache-2.0"),
        ],
        "test_cmd": "./ctr.sh test",
        "sample_prompt": "Transcribe every table row from this high-resolution document scan.",
    },
    "qwen2.5-coder-14b": {
        "cfg": {
            "__PROJECT_NAME__": "Qwen2.5-Coder-14B-Instruct",
            "__MODEL_REPO__": "unsloth/Qwen2.5-Coder-14B-Instruct-GGUF",
            "__MODEL_FILE__": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
            "__MODEL_SIZE_GB__": "8.37",
            "__MODEL_ALIAS__": "qwen2.5-coder-14b",
            "__LLAMA_BUILD__": "b11636",
            "__CTX__": "65536",
            "__NGL__": "99",
            "__SPLIT_MODE__": "layer",
            "__EXTRA_ARGS__": "",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__MMPROJ_FILE__": "",
            "__MMPROJ_SIZE_GB__": "",
            "__DRAFT_FILE__": "",
            "__DRAFT_SIZE_GB__": "",
        },
        "title": "qwen2.5-coder-14b-server",
        "doc_title": "Qwen2.5-Coder-14B-Instruct — High-Speed Code Intelligence",
        "tagline": "the **mid-weight coding specialist** of the collection.",
        "desc": "Qwen2.5-Coder-14B-Instruct delivers an exceptional balance of speed and coding prowess. Trained on over 5.5 trillion tokens of code, math, and documentation, it punches well above its weight on HumanEval, MultiPL-E, and real-world multi-file refactoring.",
        "table": [
            ("Architecture", "Qwen2.5-Coder 14B Dense"),
            ("Quantization", "`Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` (8.37 GB)"),
            ("Context Window", "65,536 tokens (native 128k supported)"),
            ("VRAM Footprint", "~10 GB total (comfortably inside dual T4s with massive KV cache)"),
            ("Best For", "Interactive coding assistant, fast unit tests, API implementation"),
            ("License", "Apache-2.0"),
        ],
        "test_cmd": "./ctr.sh test",
        "sample_prompt": "Write a high-performance async connection pool in Python with backpressure.",
    },
    "qwen2.5-coder-32b": {
        "cfg": {
            "__PROJECT_NAME__": "Qwen2.5-Coder-32B-Instruct",
            "__MODEL_REPO__": "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF",
            "__MODEL_FILE__": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
            "__MODEL_SIZE_GB__": "18.49",
            "__MODEL_ALIAS__": "qwen2.5-coder-32b",
            "__LLAMA_BUILD__": "b11636",
            "__CTX__": "32768",
            "__NGL__": "99",
            "__SPLIT_MODE__": "layer",
            "__EXTRA_ARGS__": "",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__MMPROJ_FILE__": "",
            "__MMPROJ_SIZE_GB__": "",
            "__DRAFT_FILE__": "",
            "__DRAFT_SIZE_GB__": "",
        },
        "title": "qwen2.5-coder-32b-server",
        "doc_title": "Qwen2.5-Coder-32B-Instruct — Flagship Open-Source Code Model",
        "tagline": "the **heavyweight coding champion** of the collection.",
        "desc": "Qwen2.5-Coder-32B-Instruct is widely considered the strongest open-weights code model available, matching and often outperforming GPT-4o on HumanEval (92.7%) and SWE-bench Verified. Sharded across Kaggle's dual Tesla T4 GPUs (32 GB total VRAM), it fits with ~13 GB VRAM remaining for active KV cache.",
        "table": [
            ("Architecture", "Qwen2.5-Coder 32B Dense"),
            ("Quantization", "`Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf` (18.49 GB)"),
            ("Context Window", "32,768 tokens"),
            ("VRAM Allocation", "Layer-split across 2x T4 (16 GB each = 32 GB total)"),
            ("KV Cache Headroom", "~13.5 GB free for KV cache"),
            ("Best For", "Complex repository refactors, architecture design, SWE-bench debugging"),
            ("License", "Apache-2.0"),
        ],
        "test_cmd": "./ctr.sh test",
        "sample_prompt": "Debug this race condition in a distributed Raft consensus implementation.",
    },
    "deepseek-r1-14b": {
        "cfg": {
            "__PROJECT_NAME__": "DeepSeek-R1-Distill-Qwen-14B",
            "__MODEL_REPO__": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "__MODEL_FILE__": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            "__MODEL_SIZE_GB__": "8.37",
            "__MODEL_ALIAS__": "deepseek-r1-14b",
            "__LLAMA_BUILD__": "b11636",
            "__CTX__": "32768",
            "__NGL__": "99",
            "__SPLIT_MODE__": "layer",
            "__EXTRA_ARGS__": "",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__MMPROJ_FILE__": "",
            "__MMPROJ_SIZE_GB__": "",
            "__DRAFT_FILE__": "",
            "__DRAFT_SIZE_GB__": "",
        },
        "title": "deepseek-r1-14b-server",
        "doc_title": "DeepSeek-R1-Distill-Qwen-14B — Reasoning & Math Flagship",
        "tagline": "the **advanced reasoning and chain-of-thought** model.",
        "desc": "Distilled directly from DeepSeek-R1 (671B MoE) into Qwen-2.5-14B, this model generates detailed `<think>...</think>` thought chains before answering. It achieves stellar scores on AIME 2024, MATH-500, and competition-level algorithmic challenges.",
        "table": [
            ("Architecture", "DeepSeek-R1 Distill (14B Dense)"),
            ("Quantization", "`DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf` (8.37 GB)"),
            ("Context Window", "32,768 tokens"),
            ("Reasoning Tokens", "Native `<think>` internal monologue generation"),
            ("VRAM Footprint", "~10 GB total (dual T4s provide vast KV buffer)"),
            ("Best For", "Complex mathematics, logic puzzles, algorithm derivation, step-by-step proofs"),
            ("License", "MIT"),
        ],
        "test_cmd": "./ctr.sh test",
        "sample_prompt": "Prove that the sum of the squares of two odd numbers cannot be a square.",
    },
    "deepseek-r1-8b": {
        "cfg": {
            "__PROJECT_NAME__": "DeepSeek-R1-Distill-Llama-8B",
            "__MODEL_REPO__": "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
            "__MODEL_FILE__": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
            "__MODEL_SIZE_GB__": "4.58",
            "__MODEL_ALIAS__": "deepseek-r1-8b",
            "__LLAMA_BUILD__": "b11636",
            "__CTX__": "32768",
            "__NGL__": "99",
            "__SPLIT_MODE__": "layer",
            "__EXTRA_ARGS__": "",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__MMPROJ_FILE__": "",
            "__MMPROJ_SIZE_GB__": "",
            "__DRAFT_FILE__": "",
            "__DRAFT_SIZE_GB__": "",
        },
        "title": "deepseek-r1-8b-server",
        "doc_title": "DeepSeek-R1-Distill-Llama-8B — Fast Reasoning Engine",
        "tagline": "the **low-latency reasoning and thinking** specialist.",
        "desc": "Distilled from DeepSeek-R1 into Llama-3.1-8B, this model offers rapid `<think>` chain-of-thought generation with minimal VRAM overhead and blistering token generation speeds on Kaggle GPUs.",
        "table": [
            ("Architecture", "DeepSeek-R1 Distill (8B Dense, Llama-3.1 base)"),
            ("Quantization", "`DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf` (4.58 GB)"),
            ("Context Window", "32,768 tokens"),
            ("VRAM Footprint", "~5.5 GB total"),
            ("Speed", "Fastest chain-of-thought generation in the cluster"),
            ("Best For", "Real-time math verification, structured reasoning, quick logic queries"),
            ("License", "Llama 3.1 Community License"),
        ],
        "test_cmd": "./ctr.sh test",
        "sample_prompt": "Explain Gödel's First Incompleteness Theorem step by step.",
    },
    "hermes-3-8b": {
        "cfg": {
            "__PROJECT_NAME__": "Hermes-3-Llama-3.1-8B",
            "__MODEL_REPO__": "NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
            "__MODEL_FILE__": "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf",
            "__MODEL_SIZE_GB__": "4.58",
            "__MODEL_ALIAS__": "hermes-3-8b",
            "__LLAMA_BUILD__": "b11636",
            "__CTX__": "32768",
            "__NGL__": "99",
            "__SPLIT_MODE__": "layer",
            "__EXTRA_ARGS__": "",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__MMPROJ_FILE__": "",
            "__MMPROJ_SIZE_GB__": "",
            "__DRAFT_FILE__": "",
            "__DRAFT_SIZE_GB__": "",
        },
        "title": "hermes-3-8b-server",
        "doc_title": "Hermes-3-Llama-3.1-8B — Agentic Tool Calling & Function Execution",
        "tagline": "the **autonomous agentic & tool-calling** specialist.",
        "desc": "Hermes 3 is Nous Research's flagship generalist and agentic model. It is fine-tuned specifically for structured JSON generation, tool calling, multi-step agent loops, roleplay, and strict system-prompt adherence.",
        "table": [
            ("Architecture", "Hermes-3 (8B Dense, Llama-3.1 base)"),
            ("Quantization", "`Hermes-3-Llama-3.1-8B.Q4_K_M.gguf` (4.58 GB)"),
            ("Context Window", "32,768 tokens (supports up to 128k)"),
            ("Agent Features", "Advanced XML/JSON tool calling, scratchpad reasoning"),
            ("VRAM Footprint", "~5.5 GB total"),
            ("Best For", "Autonomous agent workflows, function calling, JSON API orchestration"),
            ("License", "Llama 3.1 Community License"),
        ],
        "test_cmd": "./ctr.sh tools",
        "sample_prompt": "Call the search_weather tool for Paris and Tokyo and summarize results.",
    },
    "mistral-nemo-12b": {
        "cfg": {
            "__PROJECT_NAME__": "Mistral-Nemo-Instruct-2407",
            "__MODEL_REPO__": "unsloth/Mistral-Nemo-Instruct-2407-GGUF",
            "__MODEL_FILE__": "Mistral-Nemo-Instruct-2407.Q4_K_M.gguf",
            "__MODEL_SIZE_GB__": "6.96",
            "__MODEL_ALIAS__": "mistral-nemo-12b",
            "__LLAMA_BUILD__": "b11636",
            "__CTX__": "65536",
            "__NGL__": "99",
            "__SPLIT_MODE__": "layer",
            "__EXTRA_ARGS__": "",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__MMPROJ_FILE__": "",
            "__MMPROJ_SIZE_GB__": "",
            "__DRAFT_FILE__": "",
            "__DRAFT_SIZE_GB__": "",
        },
        "title": "mistral-nemo-12b-server",
        "doc_title": "Mistral-Nemo-Instruct-2407 — 128k Long-Context General Intelligence",
        "tagline": "the **128k long-context and multilingual** text workhorse.",
        "desc": "Developed jointly by Mistral AI and NVIDIA, Mistral-Nemo is a 12.2B parameter dense model featuring the new Tekken tokenizer (128k vocabulary). It is exceptionally strong at long-document synthesis, multi-lingual dialogue, and structured data handling.",
        "table": [
            ("Architecture", "Mistral-Nemo 12.2B Dense"),
            ("Quantization", "`Mistral-Nemo-Instruct-2407.Q4_K_M.gguf` (6.96 GB)"),
            ("Tokenizer", "Tekken (128k vocab, ultra-efficient compression)"),
            ("Context Window", "65,536 tokens (up to 128k native)"),
            ("VRAM Footprint", "~8.5 GB total"),
            ("Best For", "Long-document summarization, multilingual translation, general writing"),
            ("License", "Apache-2.0"),
        ],
        "test_cmd": "./ctr.sh test",
        "sample_prompt": "Synthesize key findings across a 40-page technical specification.",
    },
    "phi-4-14b": {
        "cfg": {
            "__PROJECT_NAME__": "phi-4",
            "__MODEL_REPO__": "unsloth/phi-4-GGUF",
            "__MODEL_FILE__": "phi-4-Q4_K_M.gguf",
            "__MODEL_SIZE_GB__": "8.28",
            "__MODEL_ALIAS__": "phi-4-14b",
            "__LLAMA_BUILD__": "b11636",
            "__CTX__": "16384",
            "__NGL__": "99",
            "__SPLIT_MODE__": "layer",
            "__EXTRA_ARGS__": "",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__MMPROJ_FILE__": "",
            "__MMPROJ_SIZE_GB__": "",
            "__DRAFT_FILE__": "",
            "__DRAFT_SIZE_GB__": "",
        },
        "title": "phi-4-14b-server",
        "doc_title": "Phi-4 — Microsoft 14B STEM & Analytical Reasoning Champion",
        "tagline": "the **STEM and academic reasoning** champion.",
        "desc": "Phi-4 is Microsoft's 14B state-of-the-art dense model trained on high-quality synthetic textbooks, academic papers, and curated reasoning corpora. It outperforms substantially larger models on GPQA, MATH, MMLU-Pro, and STEM-intensive reasoning benchmarks.",
        "table": [
            ("Architecture", "Phi-4 (14B Dense)"),
            ("Quantization", "`phi-4-Q4_K_M.gguf` (8.28 GB)"),
            ("Context Window", "16,384 tokens"),
            ("VRAM Footprint", "~9.8 GB total"),
            ("Best For", "Academic research questions, physics and chemistry modeling, calculus, STEM Q&A"),
            ("License", "MIT"),
        ],
        "test_cmd": "./ctr.sh test",
        "sample_prompt": "Derive the Navier-Stokes equations from conservation of momentum.",
    },
    "sdxl-turbo": {
        "cfg": {
            "__PROJECT_NAME__": "SDXL-Turbo",
            "__MODEL_TYPE__": "sdxl",
            "__MODEL_REPO__": "stabilityai/sdxl-turbo",
            "__MODEL_ALIAS__": "sdxl-turbo",
            "__PORT__": "8080",
            "__CTL_PORT__": "8081",
            "__SERVE_MINUTES__": "660",
            "__DEFAULT_STEPS__": "1",
            "__DEFAULT_WIDTH__": "512",
            "__DEFAULT_HEIGHT__": "512",
        },
        "title": "sdxl-turbo-server",
        "doc_title": "SDXL-Turbo — Real-Time 1-Step Text-to-Image Generation",
        "tagline": "the **ultra-fast 1-step real-time image** generator.",
        "desc": "SDXL-Turbo is Stability AI's adversarial diffusion distillation (ADD) model, synthesizing high-fidelity images in a single inference step (~1.5s on Tesla T4). It runs with a self-contained Web UI and OpenAI-compatible `/v1/images/generations` endpoint.",
        "table": [
            ("Architecture", "SDXL-Turbo (Adversarial Diffusion Distillation)"),
            ("Weights", "Single-file FP16 (~6.0 GB) from `stabilityai/sdxl-turbo`"),
            ("Inference Speed", "1 step (~1.5s per 512x512 image on Tesla T4)"),
            ("Resolution", "512x512 (optimal), 768x768, 1024x1024"),
            ("VRAM Footprint", "~6.5 GB on GPU (runs comfortably on single or dual T4)"),
            ("Interfaces", "Built-in HTML5 Web UI + OpenAI `/v1/images/generations`"),
            ("License", "Stability AI Community License (Non-Commercial Research / Personal)"),
        ],
        "test_cmd": "./ctr.sh test",
        "sample_prompt": "A cinematic shot of an astronaut riding a white horse on Mars, photorealistic, 8k",
        "is_image": True,
    },
}


def make_readme(proj: str, info: dict) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in info["table"])
    is_image = info.get("is_image", False)
    
    if is_image:
        workflow = f"""## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + generates a sample image (512x512)

# generate an image from your terminal:
./ctr.sh gen "{info['sample_prompt']}" -o test.png

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
curl -X POST "$(cat url.txt)/v1/images/generations" \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "{info['sample_prompt']}", "size": "512x512", "response_format": "b64_json"}}'
```
"""
    else:
        workflow = f"""## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# single-shot chat from terminal:
./ctr.sh chat "{info['sample_prompt']}"

# launch opencode agent against this server:
../bin/cluster {proj} agent
```

## Features & Highlights

- **Zero Local Downloads**: The model weights and llama.cpp binary are fetched and executed directly on remote Kaggle dual T4 GPUs.
- **OpenAI-Compatible API**: Works directly with any standard OpenAI client, langchain, llamaindex, or `opencode` by setting the base URL to `$(cat url.txt)/v1`.
- **Remote Control & Auto-Shutdown**: Safely terminate the remote Kaggle session from your local terminal with `./ctr.sh stop` or via `../bin/cluster {proj} stop`.
"""

    return f"""# {info['doc_title']}

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · {info['tagline']}

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

{info['desc']}

## Specifications

| Property | Value |
| --- | --- |
{rows}

{workflow}

## Stopping the Session

Release the Kaggle GPU when done so your quota is preserved:

```bash
./ctr.sh stop
```
"""


def main():
    print(f"Creating {len(PROJECTS)} projects...")
    for proj, info in PROJECTS.items():
        proj_dir = os.path.join(ROOT, proj)
        os.makedirs(proj_dir, exist_ok=True)
        print(f"\nProcessing {proj} -> {proj_dir}")

        # 1. kernel_config.py
        cfg_path = os.path.join(proj_dir, "kernel_config.py")
        cfg_lines = [
            f'"""Project config for {info["cfg"]["__PROJECT_NAME__"]} server.\n',
            'The shared generator (../lib/make_kernel.py) reads `CFG` and writes\n',
            '`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push\n',
            '"""\n\n',
            'CFG = {\n',
        ]
        for k, v in info["cfg"].items():
            cfg_lines.append(f'    "{k}": "{v}",\n')
        cfg_lines.append('}\n')
        with open(cfg_path, "w") as f:
            f.writelines(cfg_lines)
        print(f"  [OK] {cfg_path}")

        # 2. kernel-metadata.json
        meta_path = os.path.join(proj_dir, "kernel-metadata.json")
        meta = {
            "id": f"zkmohammad/{info['title']}",
            "title": info["title"],
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        print(f"  [OK] {meta_path}")

        # 3. ctr.sh
        ctr_path = os.path.join(proj_dir, "ctr.sh")
        with open(ctr_path, "w") as f:
            f.write(CTR_SCRIPT)
        st = os.stat(ctr_path)
        os.chmod(ctr_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  [OK] {ctr_path} (executable)")

        # 4. Compile notebook.ipynb
        subprocess.run([sys.executable, os.path.join(ROOT, "lib/make_kernel.py"), proj_dir], check=True)

        # 5. README.md
        readme_path = os.path.join(proj_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(make_readme(proj, info))
        print(f"  [OK] {readme_path}")

    print("\nAll 10 projects generated successfully!")


if __name__ == "__main__":
    main()
