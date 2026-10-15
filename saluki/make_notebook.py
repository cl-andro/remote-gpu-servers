#!/usr/bin/env python3
"""Generates the Kaggle kernel notebook that serves Underdog-Saluki-27B over
llama.cpp and publishes it through a public Cloudflare quick tunnel.

Everything heavy (model download, llama.cpp, inference) happens ON KAGGLE.
Nothing is downloaded to the local machine.

Run:  python3 make_notebook.py      -> writes notebook.ipynb
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Configuration (edit here, then re-run `python3 make_notebook.py`)
# --------------------------------------------------------------------------
CFG = {
    "__MODEL_REPO__": "ConwayResearch/Underdog-Saluki-27B-1.0",
    "__MODEL_FILE__": "Underdog-Saluki-27B-1.0-IQ2-mix.gguf",   # 7.89 GB
    "__MMPROJ_FILE__": "",                                      # "mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf" for vision
    "__LLAMA_BUILD__": "b11636",                                # needs >= b10330 for qwen35
    "__CTX__": "32768",                                         # native max is 262144
    "__NGL__": "99",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",                                 # Kaggle caps sessions at 12h
}
# --------------------------------------------------------------------------

CELL_1_ENV = r'''
# --- Cell 1: environment + scratch space discovery -------------------------
import os, sys, platform, subprocess

print("python:", sys.version.split()[0])
print("platform:", platform.platform())

for p in ("/kaggle/temp", "/kaggle/working"):
    os.makedirs(p, exist_ok=True)
    st = os.statvfs(p)
    free = st.f_bavail * st.f_frsize / 1e9
    print(f"{p}: free={free:.1f} GB writable={os.access(p, os.W_OK)}")

print("--- GPU ---")
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "no nvidia-smi")

print("--- internet ---")
for u in ("https://huggingface.co", "https://github.com"):
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                        "--max-time", "30", u], capture_output=True, text=True)
    print(u, "->", r.stdout)
'''

CELL_2_MODEL = r'''
# --- Cell 2: download the GGUF (ON KAGGLE, ~7.9 GB) ------------------------
import os, subprocess, sys

MODEL_REPO = "__MODEL_REPO__"
MODEL_FILE = "__MODEL_FILE__"
MMPROJ_FILE = "__MMPROJ_FILE__"

MODEL_DIR = "/kaggle/temp/saluki-models"
os.makedirs(MODEL_DIR, exist_ok=True)


def curl_download(repo, filename, dest_dir):
    """Resume-capable curl download from the HF resolve endpoint."""
    out = os.path.join(dest_dir, filename)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    print(f"downloading {filename} -> {out}", flush=True)
    subprocess.run(["curl", "-L", "-C", "-", "--retry", "10", "--retry-delay", "5",
                    "--progress-bar", "-o", out, url], check=True)
    return out


def hf_download(repo, filename, dest_dir):
    """Preferred path: huggingface_hub (+ hf_transfer when installable)."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer"],
                       check=False)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir)
    except Exception as e:
        print("huggingface_hub path failed -> falling back to curl:", e, flush=True)
        return curl_download(repo, filename, dest_dir)


model_path = os.path.join(MODEL_DIR, MODEL_FILE)
if not os.path.exists(model_path) or os.path.getsize(model_path) < 7_000_000_000:
    model_path = hf_download(MODEL_REPO, MODEL_FILE, MODEL_DIR)

mmproj_path = ""
if MMPROJ_FILE:
    mmproj_path = os.path.join(MODEL_DIR, MMPROJ_FILE)
    if not os.path.exists(mmproj_path):
        mmproj_path = hf_download(MODEL_REPO, MMPROJ_FILE, MODEL_DIR)

print("model:", model_path, round(os.path.getsize(model_path) / 1e9, 2), "GB")
print("mmproj:", mmproj_path or "(none)")
'''

CELL_3_LLAMA = r'''
# --- Cell 3: install a prebuilt llama.cpp (qwen35-capable) -----------------
import os, subprocess, glob

LLAMA_BUILD = "__LLAMA_BUILD__"
LLAMA_DIR = "/kaggle/temp/llama"
BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}"
os.makedirs(LLAMA_DIR, exist_ok=True)


def llama_server_path():
    hits = glob.glob(os.path.join(LLAMA_DIR, "**", "llama-server"), recursive=True)
    return hits[0] if hits else None


if not llama_server_path():
    # CUDA build first; the cudart archive carries the matching CUDA runtime libs.
    for asset in (f"llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                  f"cudart-llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz"):
        subprocess.run(["curl", "-sL", "-o", f"{LLAMA_DIR}/{asset}", f"{BASE}/{asset}"],
                       check=True)
        subprocess.run(["tar", "-xzf", f"{LLAMA_DIR}/{asset}", "-C", LLAMA_DIR], check=True)

LLAMA_SERVER = llama_server_path()
if not LLAMA_SERVER:
    raise RuntimeError("llama-server not found after extraction")

os.chmod(LLAMA_SERVER, 0o755)
LLAMA_BIN_DIR = os.path.dirname(LLAMA_SERVER)
os.environ["LD_LIBRARY_PATH"] = LLAMA_BIN_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")

print("llama-server:", LLAMA_SERVER)
print(subprocess.run([LLAMA_SERVER, "--version"], capture_output=True, text=True).stdout)

# Confirm the binary knows the qwen35 architecture this GGUF declares.
libs = (glob.glob(os.path.join(LLAMA_BIN_DIR, "libllama.so"))
        or glob.glob(os.path.join(LLAMA_DIR, "**", "libllama.so"), recursive=True))
if libs:
    out = subprocess.run(["strings", libs[0]], capture_output=True, text=True).stdout
    print("qwen35 architecture registered:", out.count("\nqwen35") > 0)
'''

CELL_4_SERVE = r'''
# --- Cell 4: start llama-server + publish public tunnels -------------------
import os, re, subprocess, time, urllib.request, json, threading, http.server

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
CTX = int("__CTX__")
NGL = int("__NGL__")

# Set by the control server's /stop route; cell 5 watches it and exits cleanly.
STOP_REQUESTED = threading.Event()

WORK = "/kaggle/temp/saluki-run"
os.makedirs(WORK, exist_ok=True)
LLM_LOG = os.path.join(WORK, "llama-server.log")
CF_LOG = os.path.join(WORK, "cloudflared.log")
CTL_LOG = os.path.join(WORK, "cloudflared-ctl.log")

# ---- cloudflared (public URL) ----
CF = "/kaggle/temp/cloudflared"
if not os.path.exists(CF):
    subprocess.run(["curl", "-sL", "-o", CF,
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"],
                   check=True)
    os.chmod(CF, 0o755)

# ---- llama-server (OpenAI-compatible) ----
cmd = [LLAMA_SERVER, "-m", model_path, "--jinja", "-ngl", str(NGL), "-fa", "on",
       "-c", str(CTX), "--host", "127.0.0.1", "--port", str(PORT)]
if mmproj_path:
    cmd += ["--mmproj", mmproj_path]
print("launching:", " ".join(cmd), flush=True)

llm_log_fh = open(LLM_LOG, "w")
server = subprocess.Popen(cmd, stdout=llm_log_fh, stderr=subprocess.STDOUT,
                          env=os.environ.copy(), cwd=LLAMA_BIN_DIR)

# ---- wait for readiness ----
ready = False
for i in range(240):                      # up to 20 minutes
    time.sleep(5)
    if server.poll() is not None:
        print("llama-server exited early; log tail:", flush=True)
        print(open(LLM_LOG).read()[-4000:], flush=True)
        raise RuntimeError("llama-server died on startup")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=4) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:
        pass
    if i % 6 == 0:
        tail = open(LLM_LOG).read().strip().splitlines()[-1:] or [""]
        print(f"[{i*5}s] loading... {tail[0][:120]}", flush=True)

if not ready:
    raise RuntimeError("llama-server did not become healthy in time")

print("llama-server is UP", flush=True)

# ---- local control server: lets your PC end the session on demand --------
# Kaggle's public API has no session-stop permission, so this tiny server is
# what `./ctr.sh stop` calls. It only listens on 127.0.0.1 and is exposed
# through its own tunnel; the path is secret-ish (random token).
STOP_TOKEN = os.environ.get("SALUKI_STOP_TOKEN") or os.urandom(8).hex()

class CtlHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, {"ok": True, "status": "up", "stopping": STOP_REQUESTED.is_set()})
        elif self.path.startswith("/stop/" + STOP_TOKEN):
            STOP_REQUESTED.set()
            self._send(200, {"ok": True, "message": "shutdown requested"})
            print("STOP_REQUESTED received via control tunnel", flush=True)
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        self.do_GET()

ctl = http.server.ThreadingHTTPServer(("127.0.0.1", CTL_PORT), CtlHandler)
threading.Thread(target=ctl.serve_forever, daemon=True).start()

# ---- publish both tunnels ----
def start_tunnel(url, log_path):
    fh = open(log_path, "w")
    subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", url],
                     stdout=fh, stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(2)
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
        if m:
            return m.group(0)
    return None

public_url = start_tunnel(f"http://127.0.0.1:{PORT}", CF_LOG)
control_url = start_tunnel(f"http://127.0.0.1:{CTL_PORT}", CTL_LOG)

if not public_url:
    print("cloudflared log:", open(CF_LOG).read()[-2000:], flush=True)
    raise RuntimeError("could not obtain a public tunnel URL")
if not control_url:
    print("control cloudflared log:", open(CTL_LOG).read()[-2000:], flush=True)
    raise RuntimeError("could not obtain a control tunnel URL")

with open("/kaggle/working/CONTROL_URL.txt", "w") as f:
    f.write(public_url + "\n")
with open("/kaggle/working/STOP_URL.txt", "w") as f:
    f.write(control_url + "/stop/" + STOP_TOKEN + "\n")

print("=" * 72)
print("CONTROL_URL=" + public_url)
print("STOP_URL=" + control_url + "/stop/" + STOP_TOKEN)
print("=" * 72)
print(f"Chat endpoint : {public_url}/v1/chat/completions")
print(f"Health        : {public_url}/health")
print("Model name    : saluki  (any name works; llama-server ignores it)")
print("Stop from PC  : ./ctr.sh stop   (uses the control tunnel above)")
'''

CELL_5_KEEPALIVE = r'''
# --- Cell 5: keep the tunnel alive (this is what you keep running) ---------
import time, urllib.request

PORT = int("__PORT__")
SERVE_MINUTES = int("__SERVE_MINUTES__")

url = open("/kaggle/working/CONTROL_URL.txt").read().strip()
print(f"serving for ~{SERVE_MINUTES} minutes at {url}", flush=True)

for minute in range(SERVE_MINUTES):
    # Wait up to 60s, but poll the stop flag every 5s so shutdown is prompt.
    for _ in range(12):
        if STOP_REQUESTED.is_set():
            break
        time.sleep(5)
    if STOP_REQUESTED.is_set():
        print("STOP received from control tunnel -> ending session", flush=True)
        break
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            status = str(r.status)
    except Exception as e:
        status = f"err {e}"
    if minute % 5 == 0:
        # Re-print periodically so the URL can always be recovered from logs.
        print(f"[{minute}m] health={status} CONTROL_URL={url}", flush=True)

print("SERVE_WINDOW_COMPLETE", flush=True)
'''

CELLS = [CELL_1_ENV, CELL_2_MODEL, CELL_3_LLAMA, CELL_4_SERVE, CELL_5_KEEPALIVE]


def render(src: str) -> str:
    for key, val in CFG.items():
        src = src.replace(key, val)
    return src


def main():
    cells = [render(c) for c in CELLS]

    # Fail fast if any placeholder is left unsubstituted.
    for i, c in enumerate(cells):
        if "__" in c and any(k in c for k in CFG):
            raise SystemExit(f"unsubstituted placeholder in cell {i}")
        compile(c, f"<cell{i}>", "exec")

    nb = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": [(ln + "\n") for ln in c.strip("\n").split("\n")]}
            for c in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = os.path.join(HERE, "notebook.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()