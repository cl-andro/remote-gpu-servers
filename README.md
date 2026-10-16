# Remote GPU Servers

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

Runnable, self-contained configs that stand up **LLM inference servers on
remote GPUs** (Kaggle, etc.) and let you **drive them from your own PC** — with
no local downloads, no large datasets, and no paid hosting.

Every model lives in its own folder holding just a `kernel_config.py` (the
model, quantization, context, ports), a generated `notebook.ipynb`, a
`kernel-metadata.json`, a `ctr.sh` wrapper, and a README. All of the machinery —
notebook generator, lifecycle driver, HTTP client, and the `cluster` agent CLI —
is shared under [`lib/`](lib/) and [`bin/`](bin/), so adding a new model means
writing one config file and running one command.

## Projects

All of these target **Kaggle T4 x2** (32 GB VRAM total, ~29 GB RAM, 12 h
sessions, ~30 GPU-h/week). Sizes are the quantized GGUF weights.

### 🎨 Image Generation
| Project | Model | Type | Size | Steps / Ctx | Best for |
| --- | --- | --- | --- | --- | --- |
| [`flux1-schnell/`](flux1-schnell/) | FLUX.1-schnell | 12B MMDiT (GGUF Q4_K_S) | 6.3 GB | 4 steps | SOTA photorealistic diffusion transformer |
| [`omnigen/`](omnigen/) | OmniGen-v1 (BAAI) | 3.8B Unified Multimodal | 7.7 GB | 30 steps | 🏆 Multi-subject thumbnails, reference photo composition |
| [`janus-pro-7b/`](janus-pro-7b/) | Janus-Pro-7B (DeepSeek) | 7B Unified Multimodal | 14.0 GB | DeepSeek | Instruction-driven visual generation & multimodal reasoning |
| [`sdxl-turbo/`](sdxl-turbo/) | SDXL-Turbo | 3.5B ADD (FP16) | 6.0 GB | 1 step | Instant real-time text-to-image (~1.5s/img) |

### 💻 Coding Specialists
| Project | Model | Type | Size | Context | Best for |
| --- | --- | --- | --- | --- | --- |
| [`qwen3-coder-30b/`](qwen3-coder-30b/) | Qwen3-Coder-30B-A3B | MoE 30B / 3.3B active | 17.7 GB | 65k | 🏆 Flagship agentic coding — highest throughput |
| [`qwen2.5-coder-32b/`](qwen2.5-coder-32b/) | Qwen2.5-Coder-32B | Dense 32B (Q4_K_M) | 18.5 GB | 32k | 🧠 Heavyweight coding leader (beats GPT-4o on HumanEval) |
| [`qwen2.5-coder-14b/`](qwen2.5-coder-14b/) | Qwen2.5-Coder-14B | Dense 14B (Q4_K_M) | 8.4 GB | 65k | 🚀 Fast refactoring, test gen & API development |
| [`devstral-small-2/`](devstral-small-2/) | Devstral-Small-2-24B | Dense 24B (Q4_K_M) | 14.5 GB | 65k | Coding agent, straightforward dense runtime |

### 🔬 Reasoning & Mathematics
| Project | Model | Type | Size | Context | Best for |
| --- | --- | --- | --- | --- | --- |
| [`deepseek-r1-14b/`](deepseek-r1-14b/) | DeepSeek-R1-Distill-14B | Dense 14B (Q4_K_M) | 8.4 GB | 32k | Advanced math & algorithmic reasoning (`think` CoT) |
| [`deepseek-r1-8b/`](deepseek-r1-8b/) | DeepSeek-R1-Distill-8B | Dense 8B (Q4_K_M) | 4.6 GB | 32k | Ultra-fast low-latency chain-of-thought logic |
| [`phi-4-14b/`](phi-4-14b/) | Phi-4 (Microsoft) | Dense 14B (Q4_K_M) | 8.3 GB | 16k | Academic STEM, physics, calculus & analytical Q&A |
| [`saluki/`](saluki/) | Underdog-Saluki-27B-1.0 | Dense 27B (IQ2-mix) | 7.9 GB | 32k | Proven reasoning baseline with deep domain memory |

### 📄 OCR, Vision & Document AI
| Project | Model | Type | Size | Context | Best for |
| --- | --- | --- | --- | --- | --- |
| [`qwen2.5-vl-7b/`](qwen2.5-vl-7b/) | Qwen2.5-VL-7B-Instruct | 7B + mmproj F16 | 5.6 GB | 32k | Scanned documents, OCR, charts & screen grounding |
| [`minicpm-v-2_6/`](minicpm-v-2_6/) | MiniCPM-V-2.6 | 8B + mmproj F16 | 5.3 GB | 32k | High-res table extraction & multi-image comparison |
| [`muse-glimmer-30b/`](muse-glimmer-30b/) | Muse-Glimmer-30B | Dense 30B + vision + draft | 19.8 GB | 65k | Dense multimodal reasoning + speculative decoding |

### 🛠️ Agentic Tool-Use & Long Context
| Project | Model | Type | Size | Context | Best for |
| --- | --- | --- | --- | --- | --- |
| [`hermes-3-8b/`](hermes-3-8b/) | Hermes-3-Llama-3.1-8B | Dense 8B (Q4_K_M) | 4.6 GB | 32k | Autonomous agents, structured JSON & tool calling |
| [`gpt-oss-20b/`](gpt-oss-20b/) | gpt-oss-20b | MoE 21B / 3.6B active | 12.1 GB | 65k | Lightweight MoE agentic loop, low memory footprint |
| [`mistral-nemo-12b/`](mistral-nemo-12b/) | Mistral-Nemo-Instruct | Dense 12.2B (Q4_K_M) | 7.0 GB | 65k | 128k long-doc synthesis & Tekken multilingual tokenizer |

> Adding another model is a folder with a `kernel_config.py`, a
> `kernel-metadata.json`, and a `ctr.sh` wrapper — copy any project folder and
> edit the config.
>
> **Tip:** You can also override any configuration parameter on the fly without
> editing files (e.g. `__CTX__=32768 ./ctr.sh push` or `__MODEL_FILE__="...IQ4_XS.gguf" ./ctr.sh push`).

## The `cluster` CLI (agent mode)

The raw HTTP client is fine for one-shot prompts, but a terminal chat is
awkward for real work: hard to edit, hard to pipe into a repo, no file edits.
So this repo ships **`bin/cluster`**, which points the open-source
[opencode](https://opencode.ai) agent at a project's tunnel:

```bash
./bin/cluster list                     # projects + live session status
./bin/cluster qwen3-coder-30b up       # push the kernel, wait for the URL
./bin/cluster qwen3-coder-30b agent    # launch the agent against the model
```

`agent` writes a per-project `opencode.json` (an OpenAI-compatible provider
named `cluster` pointing at `…/v1`) and launches `opencode` with it. You get a
full TUI — editable multi-line input, sessions, file reads/writes, and tool
use — backed by the remote GPU.

opencode is **not vendored** here: the launcher uses whatever `opencode` is on
your `PATH`, so it tracks upstream automatically. Install it with:

```bash
npm i -g opencode-ai        # or: brew install sst/tap/opencode
```

Any other OpenAI-compatible client works too — point it at `"$(cat
<project>/url.txt)/v1"`.

## How the pattern works

```
   your PC                    remote GPU host                    internet
 ┌────────────┐  push kernel ┌──────────────────────────┐
 │ bin/cluster│ ───────────► │ hosts the notebook/kernel│
 │ ctr.sh     │              │  ├─ download weights     │
 │ cluster.py │              │  ├─ run llama-server     │
 └─────┬──────┘              │  └─ open public tunnels  │
       │  HTTPS              └────────────┬─────────────┘
       └─────────────────────────────────►│ tunnel URL
                                          ▼
                             chat tunnel  (8080)  → llama-server
                             control tunnel (8081) → stop endpoint
```

1. **Push** the generated notebook to the remote host (from your PC).
2. The host **downloads the weights and starts `llama-server`** there.
3. **Two** Cloudflare quick tunnels publish it: one for the OpenAI-compatible
   API, one for a tiny control server.
4. Your PC talks to the chat tunnel over plain HTTP. To finish, the control
   tunnel ends the session so the GPU is released.

Nothing heavy ever touches your machine.

## Quick start

```bash
git clone git@github-third:zk-mohammad/remote-gpu-servers.git
cd remote-gpu-servers

cp .env.example .env                 # paste your Kaggle token into .env

./bin/cluster list                   # pick a project
./bin/cluster qwen3-coder-30b up     # start the remote GPU run
./bin/cluster qwen3-coder-30b test   # health check + sample chat
./bin/cluster qwen3-coder-30b agent  # use it as an agent (opencode)
./bin/cluster qwen3-coder-30b stop   # end the session, free the GPU
```

Prefer the low-level driver? Every project has its own wrapper:

```bash
cd qwen3-coder-30b
./ctr.sh push     # regenerate + push the notebook, start the run
./ctr.sh url      # fetch the chat tunnel URL -> url.txt
./ctr.sh chat "Explain gated delta networks in two sentences."
./ctr.sh stop     # end the session
```

### `ctr.sh` commands

| Command | What it does |
| --- | --- |
| `push` | regenerate `notebook.ipynb` and push it, starting the run |
| `up` | `push`, then wait until the chat tunnel URL appears |
| `url` | fetch the tunnel URL from the live logs into `url.txt` |
| `status` | kernel run status |
| `logs` / `follow` | print recent logs / live-tail them |
| `test` | health check + a sample chat through the tunnel |
| `chat` `stream` `repl` `tools` `props` `models` `slots` `tokenize` `raw` | passthrough to the stdlib client |
| `stop` | ask the kernel to exit, then wait for `COMPLETE` |
| `stop-url` | print/cache the current stop URL |
| `output` | download the run's output files |

## Requirements

- Python 3.9+ (clients are **stdlib-only** — no `pip install`).
- The `kaggle` CLI, authenticated with your own API token (`.env`).
- `opencode` on your `PATH` — only for `cluster … agent`.
- A remote-GPU account with internet access (e.g. free Kaggle GPU quota).

## Security

- **Never commit `.env`.** It is git-ignored; `.env.example` shows the format.
- Both tunnel URLs are **public** while the session runs — treat them as shared
  secrets and stop the session when you are done.
- The stop channel lives on its own tunnel behind a random per-run secret path,
  because Kaggle's public API token cannot cancel a session.
- Tokens generated at runtime (`url.txt`, `stop_url.txt`) are git-ignored.