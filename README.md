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
| [`z-image-turbo/`](z-image-turbo/) | Z-Image-Turbo (Alibaba) | 6B S3-DiT (BF16) | 12.0 GB | 8 steps | ⚡ Ultra-fast photorealism, world leader thumbnails & bilingual text |
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
| [`nemotron-3.5-bolt-74b/`](nemotron-3.5-bolt-74b/) | Nemotron-3.5-Bolt-74B-A9B (NVIDIA) | Mamba2-Transformer MoE 74B / 9B active + vision | 22.1 GB | 32k | ⚡ Hybrid MoE flagship: agentic tool use, vision, terminal & coding |
| [`qwen3.8-27b/`](qwen3.8-27b/) | Qwen3.8-27B (Alibaba) | Dense 27B + vision (UD-Q4_K_XL) | 18.5 GB | 32k | 👁️ Hybrid linear-attention dense vision-language & reasoning |
| [`qwen2.5-vl-7b/`](qwen2.5-vl-7b/) | Qwen2.5-VL-7B-Instruct | 7B + mmproj F16 | 5.6 GB | 32k | Scanned documents, OCR, charts & screen grounding |
| [`minicpm-v-2_6/`](minicpm-v-2_6/) | MiniCPM-V-2.6 | 8B + mmproj F16 | 5.3 GB | 32k | High-res table extraction & multi-image comparison |
| [`muse-glimmer-30b/`](muse-glimmer-30b/) | Muse-Glimmer-30B | Dense 30B + vision + draft | 19.8 GB | 65k | Dense multimodal reasoning + speculative decoding |

### 🛠️ Agentic Tool-Use & Long Context
| Project | Model | Type | Size | Context | Best for |
| --- | --- | --- | --- | --- | --- |
| [`hermes-3-8b/`](hermes-3-8b/) | Hermes-3-Llama-3.1-8B | Dense 8B (Q4_K_M) | 4.6 GB | 32k | Autonomous agents, structured JSON & tool calling |
| [`gpt-oss-20b/`](gpt-oss-20b/) | gpt-oss-20b | MoE 21B / 3.6B active | 12.1 GB | 65k | Lightweight MoE agentic loop, low memory footprint |
| [`nemotron-3.5-lightning-30b/`](nemotron-3.5-lightning-30b/) | Nemotron-3.5-Lightning-30B-A3B (NVIDIA) | Mamba-2 + MoE 30B / 3B active (Q4_0) | 18.9 GB | 32k (1M native) | ⚡ Fast hybrid MoE: reasoning, tool use & long context |
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
./bin/cluster qwen3-coder-30b agent --with-rea # with REA activate session command
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
| `bench` | measure prefill/decode tok/s, prefix-cache and GPU offload from the server log |
| `chat` `stream` `repl` `tools` `props` `models` `slots` `tokenize` `raw` | passthrough to the stdlib client |
| `stop` | ask the kernel to exit, then wait for `COMPLETE` |
| `stop-url` | print/cache the current stop URL |
| `output` | download the run's output files |

## Performance on 2× T4 (why "32 GB of GPU" still feels slow)

Kaggle gives you **two 16 GB Tesla T4s**, not one 32 GB GPU. They are 2018 cards
with ~320 GB/s of HBM bandwidth each, linked only by PCIe (no NVLink), and with
`--split-mode layer` llama.cpp runs the GPUs **in series**: layer half 1 on GPU 0,
then half 2 on GPU 1, while the idle GPU waits. Decode speed is therefore bounded
by **one** GPU's bandwidth, not two.

Decoding is memory-bandwidth-bound: every generated token must read the weights
of the layers that fire. So the ceiling is roughly
`320 GB/s ÷ bytes read per token`:

| Model class (Q4) | Bytes read / token | Theoretical max | Realistic on 2×T4 |
| --- | --- | --- | --- |
| Dense 24–32B (e.g. qwen2.5-coder-32b) | ~18 GB | ~17 tok/s | **8–12 tok/s** |
| Dense 8–14B | ~5–8 GB | 40–65 tok/s | **20–35 tok/s** |
| MoE 30B / ~3B active (qwen3-coder-30b, nemotron-3.5-lightning) | ~2 GB | ~150 tok/s | **30–60 tok/s** |

If you are below that band, it is almost never "Kaggle being slow" — it is one of
the items in the checklist below. Measure first, guess second:

```bash
./bin/cluster <project> bench                # cold/warm request + long prefill
./bin/cluster <project> bench --long 16000   # what an agent session really costs
```

`bench` prints the server's own `prompt_per_second` / `predicted_per_second`, the
speculative-draft acceptance rate, and greps the llama-server log for offloaded
layers, per-GPU buffer sizes, flash-attention and allocation errors.

### Optimization checklist

1. **Everything must be in VRAM.** `bench` should show `offloaded N/N layers`.
   One layer on CPU can drop you from 40 tok/s to 3. Free VRAM by lowering
   `__CTX__`, using a smaller quant, or dropping `mmproj`.
2. **Prefer MoE with a small active count.** This is the single biggest lever:
   30B-A3B decodes 4–6× faster than a dense 32B of the same file size.
3. **Speculative decoding.** For `nemotron-3.5-lightning-30b` the GGUF repo ships
   an MTP drafter that llama.cpp auto-detects:
   `__DRAFT_FILE__="mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf" __DRAFT_SIZE_GB__=1.16 ./ctr.sh push`
   Expect ~1.3–2× decode. Watch the acceptance rate in `bench`; below ~40% it is
   not paying for itself.
4. **Prefill ≠ decode.** A 20k-token agent prompt at 300 tok/s costs ~70 s before
   the first token. Raise `-ub` (512 → 1024/2048) if VRAM allows — that is a
   prefill-only knob — and keep system prompts/tool lists stable so
   `cache_prompt` can reuse the prefix.
5. **Do not chase parallel requests.** `-np 1` keeps the whole context for one
   stream; two concurrent requests on a T4 pair just split the same bandwidth.
6. **CUDA graphs** are disabled globally in `lib/cells.py`
   (`GGML_CUDA_DISABLE_GRAPHS=1`) to avoid MoE multi-GPU hangs on Turing. If a
   single-GPU project is stable, unsetting it recovers ~5–15% decode.
7. **Flash attention off on sm_75** is deliberate (prefill deadlock on Turing),
   but costs a little on attention-heavy models; test with
   `__FLASH_ATTN__=on ./ctr.sh push`.
8. **Check the hardware itself**: `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,pcie.link.width.current --format=csv`
   inside the kernel. A thermally throttled T4 can halve its clock.
9. **The tunnel is not the bottleneck** for tok/s, but every SSE flush pays an
   RTT through Cloudflare; judge speed with `bench` (server timings), not by
   watching the agent stream.

## Context limits, overflow errors, and compaction

llama-server's `-c` is the **whole** window: prompt **plus** generated tokens. A
client that treats `-c` as prompt space and then asks for 8k more output gets a
hard failure:

```json
{"error":{"code":400,"type":"exceed_context_size_error",
          "message":"the request exceeds the available context size, try increasing it"}}
```

`bin/cluster … agent` now writes a budget-aware OpenCode config, so a 32k server
looks like this instead of 32k-of-prompt:

| Piece | Tokens | Where |
| --- | ---: | --- |
| fixed system prompt + skills + tool schemas | 4–8k | compaction cannot shrink this |
| conversation (older turns → summary) | ≤ 6k kept verbatim | `compaction.keep.tokens` |
| generation | 8k | `limit.output` |
| safety margin | 512 | reserved |
| **prompt budget** | **~24k** | `limit.context` |

Automatic compaction is enabled (`compaction.auto`), triggers with 20% free
(`compaction.buffer`) and, on small windows, prepares the summary in the
background so the session does not stall for a minute of prefill.

> **Check your global config.** A `~/.config/opencode/opencode.jsonc` containing
> the V1 pair `"compaction": {"auto": false, "prune": true}` disables summaries
> everywhere and is a common cause of "context window exceeded" errors, because
> V2 ignores `prune`. The generated project config now sets `compaction.auto`
> explicitly, so it overrides that; removing `prune` and setting `auto: true`
> globally is still the cleaner fix.

Tune it without editing the script:

```bash
CLUSTER_MAX_OUTPUT=4096   ./bin/cluster <project> agent   # smaller replies, more history
CLUSTER_COMPACT_KEEP=8000 ./bin/cluster <project> agent   # keep more recent turns verbatim
CLUSTER_COMPACT_BUFFER=30% ./bin/cluster <project> agent  # compact earlier
CLUSTER_BG_COMPACTION=0   ./bin/cluster <project> agent   # disable experimental background summaries
CLUSTER_COMPACT_MODEL=openai/gpt-5-mini ./bin/cluster <project> agent  # summarize with another provider
```

Server-side, the cheapest fix is a bigger window where the architecture allows
it (`__CTX__=65536 ./ctr.sh push` — nearly free on the hybrid Lightning model,
expensive on dense 32B). As a last-resort safety net (it silently drops the
oldest turns instead of erroring):

```bash
__EXTRA_ARGS__="-b 512 -ub 512 --context-shift --keep 1024" ./ctr.sh push
```

### Open-source projects for context compression

Grouped by what they actually do — mixing these up is the usual mistake:

| Project | Approach | License | Fit for this setup |
| --- | --- | --- | --- |
| [OpenCode compaction](https://opencode.ai/docs/compaction/) | Summarize old turns, keep recent verbatim | MIT | ✅ already built in, now tuned by `bin/cluster` |
| [OpenHands SDK condensers](https://github.com/All-Hands-AI/agent-sdk/tree/main/openhands-sdk/openhands/sdk/context/condenser) | Pluggable condensers: LLM summarizing, token/size, pipeline | MIT | ⭐ best reference implementation to copy |
| [SWE-agent history processors](https://github.com/SWE-agent/SWE-agent/blob/main/sweagent/agent/history_processors.py) | `last_n`, `summarizer`, `tagged` history rewriting | MIT | ⭐ small, readable, battle-tested on code agents |
| [Cline auto-compact](https://github.com/cline/cline/blob/main/docs/features/auto-compact.mdx) | Agent-level compaction + hooks | Apache-2.0 | ⭐ good UX model (status, manual `/compact`) |
| [Roo Code condensing](https://github.com/RooCodeInc/Roo-Code/tree/main/src/core/condense) | Intelligent context condensing | Apache-2.0 | good for IDE-agent patterns |
| [Letta (MemGPT)](https://github.com/letta-ai/letta) | Memory hierarchy, self-editing memory blocks | Apache-2.0 | only if you want a memory-first agent, not a patch |
| [Mem0](https://github.com/mem0ai/mem0) | Long-term memory extraction + retrieval | Apache-2.0 | cross-session recall, not window compression |
| [Graphiti / Zep](https://github.com/getzep/graphiti) | Temporal knowledge graph memory | Apache-2.0 | cross-session recall with relationships |
| [Cognee](https://github.com/topoteretes/cognee) | Memory graph + ingestion pipelines | Apache-2.0 | same category as Mem0/Graphiti |
| [LLMLingua / LLMLingua-2](https://github.com/microsoft/LLMLingua) | Token-level prompt compression (2–20×) | MIT | ⚠️ fine for tool output/docs, risky for code diffs |
| [Selective Context](https://github.com/liyucheng09/Selective_Context) | Self-information based pruning | research code | experimental, same caveat as LLMLingua |
| [RAPTOR](https://github.com/parthsarthi03/raptor) | Recursive summary tree over documents | MIT | good for large doc/PDF corpora, not chat history |
| [LangMem](https://github.com/langchain-ai/langmem) | Memory tools for LangGraph agents | MIT | if you already use LangGraph |

**Recommended combination for a 32k coding agent on Kaggle**

1. OpenCode compaction (already on) with the tuned `keep`/`buffer` above.
2. Keep tool output short at the source — the proxy/agent should truncate
   `cat`/`grep`/test logs to the relevant window; OpenHands' *observation
   masking* condenser and SWE-agent's `last_n` processor are the patterns to copy.
3. Prefer retrieval over retention: let the agent re-read files it needs instead
   of keeping whole files in history.
4. Only add Mem0/Graphiti if you need memory across *sessions*; they do not solve
   a single long session hitting the window.
5. Avoid LLMLingua on code; it drops identifiers and silently changes semantics.

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