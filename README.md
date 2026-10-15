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

Each model lives in its own folder and follows the same shape: a generator that
builds the Kaggle notebook, a `ctr.sh` lifecycle driver, a stdlib-only client,
and a README with measured numbers.

## Projects

| Project | Model | Runtime | Remote | Status |
| --- | --- | --- | --- | --- |
| [`saluki/`](saluki/) | Underdog-Saluki-27B-1.0 (IQ2-mix) | llama.cpp on Kaggle T4 x2 | Kaggle + Cloudflare tunnel | ✅ working, ~10 tok/s |

> More model projects (other quantizations, other runtimes such as vLLM, other
> hosts) are planned to be added here over time. Each one is a subfolder with
> its own README.

## How the pattern works

```
   your PC                    remote GPU host                    internet
 ┌────────────┐   vendor API ┌──────────────────────────┐
 │ ctr.sh     │ ───────────► │ hosts the notebook/kernel│
 │ client.py  │              │  ├─ download weights     │
 └─────┬──────┘              │  ├─ run inference server │
       │  HTTPS              │  └─ open public tunnel   │
       └──────────────────────────────────────────────► tunnel URL
```

1. **Push** the generated notebook to the remote host (from your PC).
2. The host **downloads the weights and starts the server** there.
3. A **tunnel** publishes it on a public URL.
4. Your PC talks to it over plain HTTP (OpenAI-compatible API).

Nothing heavy ever touches your machine. Finished? Stop the session from your
PC so the GPU is released.

## Quick start (this repo)

```bash
git clone git@github-third:zk-mohammad/remote-gpu-servers.git
cd remote-gpu-servers

cp .env.example .env      # then paste your Kaggle token into .env
cd saluki
./ctr.sh push             # start the remote GPU run
./ctr.sh url              # fetch the public tunnel URL
./ctr.sh test             # health check + sample chat
./ctr.sh stop             # end the session, free the GPU
```

See each project's README for the full command set, endpoints, measured
performance, and gotchas.

## Requirements

- Python 3.9+ (clients are **stdlib-only** — no `pip install`).
- The `kaggle` CLI, authenticated with your own API token (`.env`).
- A remote-GPU account with internet access (e.g. free Kaggle GPU quota).

## Security

- **Never commit `.env`.** It is git-ignored; `.env.example` shows the format.
- Tunnel URLs are **public** while the session runs — treat them as shared
  secrets and stop the session when you are done.
- The stop/control channel uses a random per-run secret path.