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

| Project | Model | Type | Size | Context | Best for |
| --- | --- | --- | --- | --- | --- |
| [`qwen3-coder-30b/`](qwen3-coder-30b/) | Qwen3-Coder-30B-A3B | MoE 30B / 3.3B active | 17.7 GB | 65k | 🏆 agentic coding — fastest |
| [`gpt-oss-20b/`](gpt-oss-20b/) | gpt-oss-20b | MoE 21B / 3.6B active | 12.1 GB | 65k | light, single-GPU footprint |
| [`devstral-small-2/`](devstral-small-2/) | Devstral-Small-2-24B | dense 24B | 14.5 GB | 65k | coding agent, simple runtime |
| [`muse-glimmer-30b/`](muse-glimmer-30b/) | Muse-Glimmer-30B | dense 30B + vision + drafter | 19.8 GB | 65k | agentic + image input |
| [`saluki/`](saluki/) | Underdog-Saluki-27B-1.0 | dense 27B (IQ2-mix) | 7.9 GB | 32k | reasoning, proven baseline |
| [`flux1-schnell/`](flux1-schnell/) | FLUX.1-schnell | 12B MMDiT (GGUF) | 6.3 GB | 4 steps | 🎨 text-to-image generation |

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