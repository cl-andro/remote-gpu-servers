"""Notebook cell templates shared by every project.

Each template is a plain string containing ``__PLACEHOLDER__`` tokens that a
project's ``CFG`` dict substitutes. Editing a template here updates every
project that uses the shared generator.

Everything heavy (model download, llama.cpp, inference) happens ON KAGGLE.
Nothing is downloaded to the local machine.
"""

CELL_1_ENV = r'''
# --- Cell 1: environment + scratch space discovery -------------------------
import os, sys, platform, subprocess

print("project:", "__PROJECT_NAME__")
print("python:", sys.version.split()[0])
print("platform:", platform.platform())

for p in ("/kaggle/temp", "/kaggle/working"):
    os.makedirs(p, exist_ok=True)
    st = os.statvfs(p)
    free = st.f_bavail * st.f_frsize / 1e9
    print(f"{p}: free={free:.1f} GB writable={os.access(p, os.W_OK)}")

print("--- GPU ---")
print(subprocess.run(["nvidia-smi",
                      "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout or "no nvidia-smi")

print("--- internet ---")
for u in ("https://huggingface.co", "https://github.com"):
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                        "--max-time", "30", u], capture_output=True, text=True)
    print(u, "->", r.stdout)
'''

CELL_2_MODEL = r'''
# --- Cell 2: download the GGUF (ON KAGGLE) ---------------------------------
# Optional companions (vision projector, speculative drafter) are pulled too.
import os, subprocess, sys

MODEL_REPO = "__MODEL_REPO__"
MODEL_FILE = "__MODEL_FILE__"
MODEL_SIZE_GB = float("__MODEL_SIZE_GB__")
MMPROJ_FILE = "__MMPROJ_FILE__"
MMPROJ_SIZE_GB = float("__MMPROJ_SIZE_GB__" or "0")
DRAFT_FILE = "__DRAFT_FILE__"
DRAFT_SIZE_GB = float("__DRAFT_SIZE_GB__" or "0")

MODEL_DIR = "/kaggle/temp/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def curl_download(repo, filename, dest_dir):
    """Resume-capable curl download from the HF resolve endpoint."""
    out = os.path.join(dest_dir, os.path.basename(filename))
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


def ensure(repo, filename, size_gb, label):
    """Download `filename` unless a complete copy is already present."""
    if not filename:
        return ""
    path = os.path.join(MODEL_DIR, os.path.basename(filename))
    floor = int(size_gb * 1e9 * 0.98) if size_gb else 0
    if os.path.exists(path) and (not floor or os.path.getsize(path) >= floor):
        print(f"{label}: cached {path}", flush=True)
        return path
    path = hf_download(repo, filename, MODEL_DIR)
    actual = os.path.getsize(path) / 1e9
    print(f"{label}: {path} ({actual:.2f} GB)", flush=True)
    if floor and os.path.getsize(path) < floor:
        raise RuntimeError(f"{label} looks truncated: {actual:.2f} GB expected ~{size_gb} GB")
    return path


model_path = ensure(MODEL_REPO, MODEL_FILE, MODEL_SIZE_GB, "model")
mmproj_path = ensure(MODEL_REPO, MMPROJ_FILE, MMPROJ_SIZE_GB, "mmproj")
draft_path = ensure(MODEL_REPO, DRAFT_FILE, DRAFT_SIZE_GB, "draft")
'''

CELL_3_LLAMA = r'''
# --- Cell 3: install a prebuilt llama.cpp ----------------------------------
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
'''

CELL_4_SERVE = r'''
# --- Cell 4: start llama-server + publish public tunnels -------------------
import os, re, subprocess, time, urllib.request, json, threading, http.server

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
CTX = int("__CTX__")
NGL = int("__NGL__")
SPLIT_MODE = "__SPLIT_MODE__"
EXTRA_ARGS = "__EXTRA_ARGS__".split()

# Set by the control server's /stop route; cell 5 watches it and exits cleanly.
STOP_REQUESTED = threading.Event()

WORK = "/kaggle/temp/run"
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
cmd = [LLAMA_SERVER, "-m", model_path, "--jinja", "-ngl", str(NGL),
       "-fa", "on", "-c", str(CTX), "-np", "1",
       "--host", "127.0.0.1", "--port", str(PORT)]
# -np 1 keeps the whole context for a single request (llama-server otherwise
# divides -c across parallel slots, silently shrinking what one generation sees).
if SPLIT_MODE:
    cmd += ["--split-mode", SPLIT_MODE]
if mmproj_path:
    cmd += ["--mmproj", mmproj_path]
if draft_path:
    cmd += ["--model-draft", draft_path, "--gpu-layers-draft", str(NGL)]
cmd += EXTRA_ARGS
print("launching:", " ".join(cmd), flush=True)

llm_log_fh = open(LLM_LOG, "w")
server = subprocess.Popen(cmd, stdout=llm_log_fh, stderr=subprocess.STDOUT,
                          env=os.environ.copy(), cwd=LLAMA_BIN_DIR)

# ---- wait for readiness ----
ready = False
for i in range(300):                      # up to 25 minutes
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
    print(open(LLM_LOG).read()[-4000:], flush=True)
    raise RuntimeError("llama-server did not become healthy in time")

print("llama-server is UP", flush=True)

# ---- local control server: lets your PC end the session on demand --------
# Kaggle's public API has no session-stop permission, so this tiny server is
# what `ctr.sh stop` calls. It only listens on 127.0.0.1 and is exposed
# through its own tunnel; the path is secret-ish (random token).
STOP_TOKEN = os.environ.get("CLUSTER_STOP_TOKEN") or os.urandom(8).hex()

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
            self._send(200, {"ok": True, "status": "up",
                             "stopping": STOP_REQUESTED.is_set(),
                             "model": "__MODEL_FILE__"})
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
print("MODEL=" + "__MODEL_FILE__")
print("=" * 72)
print(f"Chat endpoint : {public_url}/v1/chat/completions")
print(f"Health        : {public_url}/health")
print("Model name    : __MODEL_ALIAS__  (any name works; llama-server ignores it)")
print("Stop from PC  : ctr.sh stop   (uses the control tunnel above)")
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

CELL_2_FLUX_DEPS = r'''
# --- Cell 2: install dependencies for FLUX GGUF inference ------------------
import subprocess, sys

pkgs = [
    "diffusers>=0.30.0",
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "sentencepiece",
    "gguf>=0.10.0",
    "protobuf"
]
print("installing:", " ".join(pkgs), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("packages installed.", flush=True)
'''

CELL_3_FLUX_FETCH = r'''
# --- Cell 3: download FLUX.1-schnell GGUF and pipeline components ----------
import os, subprocess, sys
from huggingface_hub import hf_hub_download, snapshot_download

MODEL_REPO = "__MODEL_REPO__"
MODEL_FILE = "__MODEL_FILE__"
MODEL_SIZE_GB = float("__MODEL_SIZE_GB__")
BASE_REPO = "__BASE_REPO__"

MODEL_DIR = "/kaggle/temp/models"
BASE_DIR = "/kaggle/temp/base_pipe"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

# 1. Download the quantized transformer GGUF from city96/FLUX.1-schnell-gguf
dest_model = os.path.join(MODEL_DIR, MODEL_FILE)
if not os.path.exists(dest_model) or os.path.getsize(dest_model) < 1000:
    print(f"Downloading {MODEL_FILE} from {MODEL_REPO} ...", flush=True)
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
actual_size_gb = os.path.getsize(dest_model) / (1024**3)
print(f"Transformer weights: {dest_model} ({actual_size_gb:.2f} GB)", flush=True)

# 2. Download text encoders, tokenizer, VAE from ungated mirror (skip 24GB transformer)
print(f"Ensuring pipeline components from {BASE_REPO} ...", flush=True)
snapshot_download(
    repo_id=BASE_REPO,
    local_dir=BASE_DIR,
    ignore_patterns=["transformer/*.safetensors*", "*.sft", "*.jpeg", "*.jpg"],
    local_dir_use_symlinks=False
)
print("All model assets ready.", flush=True)
'''

CELL_4_FLUX_SERVE = r'''
# --- Cell 4: load FLUX pipeline + start server + open tunnels --------------
import os, re, subprocess, time, urllib.request, json, threading, http.server
import base64, io, gc
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
DEFAULT_STEPS = int("__DEFAULT_STEPS__" or "4")
DEFAULT_WIDTH = int("__DEFAULT_WIDTH__" or "1024")
DEFAULT_HEIGHT = int("__DEFAULT_HEIGHT__" or "1024")

STOP_REQUESTED = threading.Event()
GEN_LOCK = threading.Lock()

WORK = "/kaggle/temp/run"
os.makedirs(WORK, exist_ok=True)
CF_LOG = os.path.join(WORK, "cloudflared.log")
CTL_LOG = os.path.join(WORK, "cloudflared-ctl.log")

# ---- cloudflared binary ----
CF = "/kaggle/temp/cloudflared"
if not os.path.exists(CF):
    subprocess.run(["curl", "-sL", "-o", CF,
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"],
                   check=True)
    os.chmod(CF, 0o755)

# ---- load pipeline ----
model_path = os.path.join("/kaggle/temp/models", "__MODEL_FILE__")
base_dir = "/kaggle/temp/base_pipe"

# Ensure transformer/config.json exists locally so from_single_file never hits gated repo
trans_cfg_dir = os.path.join(base_dir, "transformer")
os.makedirs(trans_cfg_dir, exist_ok=True)
trans_cfg_file = os.path.join(trans_cfg_dir, "config.json")
if not os.path.exists(trans_cfg_file):
    print("Fetching transformer config.json from ungated mirror...", flush=True)
    try:
        url = "https://huggingface.co/__BASE_REPO__/raw/main/transformer/config.json"
        with urllib.request.urlopen(url) as r:
            with open(trans_cfg_file, "wb") as f:
                f.write(r.read())
    except Exception as e:
        print("Warning: could not fetch config.json directly:", e, flush=True)

# Ensure scheduler_config.json exists (diffusers requires scheduler_config.json)
sched_dir = os.path.join(base_dir, "scheduler")
sched_cfg_file = os.path.join(sched_dir, "config.json")
sched_target = os.path.join(sched_dir, "scheduler_config.json")
if os.path.exists(sched_cfg_file) and not os.path.exists(sched_target):
    import shutil
    shutil.copyfile(sched_cfg_file, sched_target)

print("Loading GGUF transformer with float16 compute...", flush=True)
transformer = FluxTransformer2DModel.from_single_file(
    model_path,
    config=trans_cfg_dir,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
    torch_dtype=torch.float16,
)

print("Loading FluxPipeline...", flush=True)
from diffusers import FlowMatchEulerDiscreteScheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sched_dir)
pipe = FluxPipeline.from_pretrained(
    base_dir,
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
print("FLUX pipeline loaded and ready on GPU!", flush=True)

# ---- Control Server on CTL_PORT (port 8081) ----
STOP_TOKEN = os.environ.get("CLUSTER_STOP_TOKEN") or os.urandom(8).hex()

class CtlHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, {"ok": True, "status": "up",
                             "stopping": STOP_REQUESTED.is_set(),
                             "model": "__MODEL_FILE__"})
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

# ---- Image Generation Server on PORT (port 8080) ----
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FLUX.1-schnell Remote GPU Server</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  body { background: #0f1117; color: #e1e4ea; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 24px 16px; }
  .card { background: #1a1d27; border: 1px solid #2d3139; border-radius: 12px; width: 100%; max-width: 840px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); margin-bottom: 24px; }
  h1 { font-size: 1.6rem; color: #fff; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
  .badge { font-size: 0.75rem; background: #059669; color: #fff; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
  p.subtitle { color: #9ca3af; font-size: 0.9rem; margin-bottom: 20px; }
  label { display: block; font-size: 0.85rem; font-weight: 600; color: #d1d5db; margin-bottom: 6px; }
  textarea { width: 100%; background: #0f1117; border: 1px solid #374151; border-radius: 8px; color: #fff; padding: 12px; font-size: 0.95rem; min-height: 90px; resize: vertical; margin-bottom: 16px; }
  textarea:focus { outline: none; border-color: #3b82f6; }
  .row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  select, input { width: 100%; background: #0f1117; border: 1px solid #374151; border-radius: 8px; color: #fff; padding: 10px; font-size: 0.9rem; }
  button { width: 100%; background: #2563eb; color: #fff; border: none; padding: 12px 20px; font-size: 1rem; font-weight: 600; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #4b5563; cursor: not-allowed; }
  #status { margin-top: 16px; font-size: 0.9rem; color: #93c5fd; text-align: center; }
  #preview { margin-top: 24px; text-align: center; }
  #preview img { max-width: 100%; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.5); }
  .download-btn { margin-top: 12px; display: inline-block; background: #10b981; color: #fff; text-decoration: none; padding: 8px 16px; border-radius: 6px; font-size: 0.9rem; font-weight: 600; }
  .api-info { background: #11131a; border: 1px solid #232733; border-radius: 8px; padding: 16px; margin-top: 24px; font-size: 0.85rem; }
  code { color: #60a5fa; font-family: ui-monospace, SFMono-Regular, monospace; }
  pre { background: #090a0f; padding: 10px; border-radius: 6px; overflow-x: auto; margin-top: 8px; color: #e5e7eb; }
</style>
</head>
<body>
<div class="card">
  <h1>FLUX.1-schnell <span class="badge">ONLINE</span></h1>
  <p class="subtitle">Fast 4-step diffusion transformer with GGUF Q4_K_S running on Kaggle GPU</p>
  
  <label for="prompt">Prompt</label>
  <textarea id="prompt" placeholder="e.g. A futuristic cyberpunk city at midnight, neon reflections on wet asphalt, volumetric mist, hyperrealistic, 8k"></textarea>
  
  <div class="row">
    <div>
      <label for="size">Resolution</label>
      <select id="size">
        <option value="1024x1024">1024 × 1024 (Square)</option>
        <option value="768x768">768 × 768</option>
        <option value="512x512">512 × 512 (Fastest)</option>
        <option value="1024x768">1024 × 768 (Landscape)</option>
        <option value="768x1024">768 × 1024 (Portrait)</option>
      </select>
    </div>
    <div>
      <label for="steps">Inference Steps (1-4)</label>
      <select id="steps">
        <option value="4" selected>4 (Recommended)</option>
        <option value="3">3</option>
        <option value="2">2</option>
        <option value="1">1 (Draft)</option>
      </select>
    </div>
    <div>
      <label for="seed">Seed (optional)</label>
      <input id="seed" type="number" placeholder="Random">
    </div>
  </div>
  
  <button id="genBtn" onclick="generate()">Generate Image</button>
  <div id="status"></div>
  <div id="preview"></div>

  <div class="api-info">
    <strong>OpenAI-compatible Image Generation Endpoint:</strong>
    <pre>POST /v1/images/generations
Content-Type: application/json

{"prompt": "A cybernetic cat on Mars", "size": "1024x1024", "response_format": "b64_json"}</pre>
  </div>
</div>

<script>
async function generate() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return alert('Please enter a prompt');
  const size = document.getElementById('size').value;
  const steps = parseInt(document.getElementById('steps').value);
  const seedVal = document.getElementById('seed').value;
  const btn = document.getElementById('genBtn');
  const status = document.getElementById('status');
  const preview = document.getElementById('preview');
  
  btn.disabled = true;
  preview.innerHTML = '';
  const startTime = Date.now();
  status.textContent = 'Generating on GPU... (~15-30s)';
  
  try {
    const payload = { prompt, size, steps, response_format: 'b64_json' };
    if (seedVal) payload.seed = parseInt(seedVal);
    const resp = await fetch('/v1/images/generations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`Server returned HTTP ${resp.status}: ${err}`);
    }
    const data = await resp.json();
    const b64 = data.data[0].b64_json;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    status.textContent = `Completed in ${elapsed}s!`;
    const imgSrc = 'data:image/png;base64,' + b64;
    preview.innerHTML = `
      <img src="${imgSrc}" alt="Generated Image"><br>
      <a class="download-btn" href="${imgSrc}" download="flux-${Date.now()}.png">Download Full Resolution</a>
    `;
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

class ServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/health", "/v1/health"):
            self._send_json(200, {
                "status": "ok",
                "model": "__MODEL_ALIAS__",
                "architecture": "flux.1-schnell",
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            })
        elif path in ("/models", "/v1/models"):
            self._send_json(200, {
                "data": [{
                    "id": "__MODEL_ALIAS__",
                    "object": "model",
                    "type": "image-generation"
                }]
            })
        else:
            self._send_json(404, {"error": f"Path {path} not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            req = json.loads(post_data.decode("utf-8"))
        except Exception:
            return self._send_json(400, {"error": "Invalid JSON payload"})

        if path in ("/v1/images/generations", "/generate"):
            prompt = req.get("prompt", "")
            if not prompt:
                return self._send_json(400, {"error": "Missing prompt"})

            size = req.get("size", "1024x1024")
            try:
                width, height = map(int, size.lower().split("x"))
            except Exception:
                width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT

            steps = int(req.get("steps") or DEFAULT_STEPS)
            steps = max(1, min(4, steps))
            seed = req.get("seed")

            gen = None
            if seed is not None:
                try:
                    gen = torch.Generator("cpu").manual_seed(int(seed))
                except Exception:
                    pass

            t0 = time.time()
            print(f"[GEN] Prompt: '{prompt[:80]}...' size={width}x{height} steps={steps}", flush=True)

            with GEN_LOCK:
                with torch.inference_mode():
                    img = pipe(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=0.0,
                        max_sequence_length=256,
                        generator=gen,
                    ).images[0]
                torch.cuda.empty_cache()
                gc.collect()

            elapsed = round(time.time() - t0, 2)
            print(f"[GEN] Done in {elapsed}s", flush=True)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64_bytes = base64.b64encode(buf.getvalue()).decode("utf-8")

            if path == "/generate":
                return self._send_json(200, {
                    "image_b64": b64_bytes,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "elapsed_seconds": elapsed
                })
            else:
                return self._send_json(200, {
                    "created": int(time.time()),
                    "data": [
                        {
                            "b64_json": b64_bytes,
                            "revised_prompt": prompt
                        }
                    ]
                })
        else:
            self._send_json(404, {"error": f"POST {path} not supported"})

server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), ServerHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print("Image server running on port", PORT, flush=True)

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
print("MODEL=" + "__MODEL_FILE__")
print("=" * 72)
print(f"Web UI         : {public_url}")
print(f"API endpoint   : {public_url}/v1/images/generations")
print(f"Health         : {public_url}/health")
print("Stop from PC   : ctr.sh stop   (uses the control tunnel above)")
'''

FLUX_CELLS = [CELL_1_ENV, CELL_2_FLUX_DEPS, CELL_3_FLUX_FETCH, CELL_4_FLUX_SERVE, CELL_5_KEEPALIVE]

CELL_2_SDXL_DEPS = r'''
# --- Cell 2: install dependencies for SDXL Turbo inference ----------------
import subprocess, sys

pkgs = [
    "diffusers>=0.30.0",
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "invisible-watermark>=0.2.0",
    "safetensors",
]
print("installing:", " ".join(pkgs), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("packages installed.", flush=True)
'''

CELL_3_SDXL_SERVE = r'''
# --- Cell 3: load SDXL-Turbo pipeline + start server + open tunnels -------
import os, re, subprocess, time, urllib.request, json, threading, http.server
import base64, io, gc
import torch
from diffusers import AutoPipelineForText2Image

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
DEFAULT_STEPS = int("__DEFAULT_STEPS__" or "1")
DEFAULT_WIDTH = int("__DEFAULT_WIDTH__" or "512")
DEFAULT_HEIGHT = int("__DEFAULT_HEIGHT__" or "512")
MODEL_REPO = "__MODEL_REPO__"

STOP_REQUESTED = threading.Event()
GEN_LOCK = threading.Lock()

WORK = "/kaggle/temp/run"
os.makedirs(WORK, exist_ok=True)
CF_LOG = os.path.join(WORK, "cloudflared.log")
CTL_LOG = os.path.join(WORK, "cloudflared-ctl.log")

# ---- cloudflared binary ----
CF = "/kaggle/temp/cloudflared"
if not os.path.exists(CF):
    subprocess.run(["curl", "-sL", "-o", CF,
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"],
                   check=True)
    os.chmod(CF, 0o755)

# ---- load pipeline ----
print(f"Loading SDXL-Turbo from {MODEL_REPO} (fp16)...", flush=True)
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_REPO,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")
print("SDXL-Turbo pipeline loaded and ready on GPU!", flush=True)

# ---- Control Server on CTL_PORT (port 8081) ----
STOP_TOKEN = os.environ.get("CLUSTER_STOP_TOKEN") or os.urandom(8).hex()

class CtlHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, {"ok": True, "status": "up",
                             "stopping": STOP_REQUESTED.is_set(),
                             "model": "__MODEL_FILE__"})
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

# ---- Image Generation Server on PORT (port 8080) ----
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDXL-Turbo Server</title>
<style>
  :root { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0c10; color: #c5c6c7; }
  body { max-width: 900px; margin: 40px auto; padding: 0 20px; }
  .card { background: #1f2833; border: 1px solid #45a29e; border-radius: 12px; padding: 28px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
  h1 { margin-top: 0; color: #66fcf1; font-size: 1.8rem; display: flex; align-items: center; gap: 10px; }
  .badge { background: #45a29e; color: #0b0c10; font-size: 0.75rem; padding: 4px 10px; border-radius: 999px; font-weight: 700; text-transform: uppercase; }
  p.subtitle { color: #c5c6c7; font-size: 0.95rem; margin-top: -6px; margin-bottom: 24px; }
  label { display: block; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #66fcf1; margin-bottom: 6px; }
  textarea { width: 100%; box-sizing: border-box; background: #0b0c10; border: 1px solid #45a29e; border-radius: 8px; color: #fff; padding: 12px; font-size: 1rem; resize: vertical; min-height: 90px; }
  .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 16px; }
  select, input { width: 100%; box-sizing: border-box; background: #0b0c10; border: 1px solid #45a29e; border-radius: 8px; color: #fff; padding: 10px; font-size: 0.95rem; }
  button { width: 100%; margin-top: 24px; padding: 14px; background: #66fcf1; color: #0b0c10; border: none; border-radius: 8px; font-size: 1.1rem; font-weight: 700; cursor: pointer; transition: all 0.2s; }
  button:hover:not(:disabled) { background: #45a29e; color: #fff; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  #status { margin-top: 16px; font-size: 0.95rem; text-align: center; color: #66fcf1; min-height: 24px; }
  #preview { margin-top: 24px; text-align: center; }
  #preview img { max-width: 100%; border-radius: 8px; border: 1px solid #45a29e; box-shadow: 0 4px 16px rgba(0,0,0,0.6); }
  .download-btn { margin-top: 12px; display: inline-block; background: #66fcf1; color: #0b0c10; text-decoration: none; padding: 8px 16px; border-radius: 6px; font-size: 0.9rem; font-weight: 700; }
  .api-info { background: #0b0c10; border: 1px solid #45a29e; border-radius: 8px; padding: 16px; margin-top: 24px; font-size: 0.85rem; }
  code { color: #66fcf1; font-family: ui-monospace, SFMono-Regular, monospace; }
  pre { background: #1f2833; padding: 10px; border-radius: 6px; overflow-x: auto; margin-top: 8px; color: #e5e7eb; }
</style>
</head>
<body>
<div class="card">
  <h1>SDXL-Turbo <span class="badge">ONLINE</span></h1>
  <p class="subtitle">Real-time 1-step text-to-image synthesis running on Kaggle GPU</p>
  
  <label for="prompt">Prompt</label>
  <textarea id="prompt" placeholder="e.g. A cinematic shot of an astronaut riding a white horse on Mars, photorealistic, 8k resolution"></textarea>
  
  <div class="row">
    <div>
      <label for="size">Resolution</label>
      <select id="size">
        <option value="512x512" selected>512 × 512 (Instant ~1.5s)</option>
        <option value="768x768">768 × 768</option>
        <option value="1024x1024">1024 × 1024</option>
      </select>
    </div>
    <div>
      <label for="steps">Inference Steps</label>
      <select id="steps">
        <option value="1" selected>1 (Real-time)</option>
        <option value="2">2</option>
        <option value="4">4</option>
      </select>
    </div>
    <div>
      <label for="seed">Seed (optional)</label>
      <input id="seed" type="number" placeholder="Random">
    </div>
  </div>
  
  <button id="genBtn" onclick="generate()">Generate Image</button>
  <div id="status"></div>
  <div id="preview"></div>

  <div class="api-info">
    <strong>OpenAI-compatible Image Generation Endpoint:</strong>
    <pre>POST /v1/images/generations
Content-Type: application/json

{"prompt": "An astronaut on Mars", "size": "512x512", "response_format": "b64_json"}</pre>
  </div>
</div>

<script>
async function generate() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return alert('Please enter a prompt');
  const size = document.getElementById('size').value;
  const steps = parseInt(document.getElementById('steps').value);
  const seedVal = document.getElementById('seed').value;
  const btn = document.getElementById('genBtn');
  const status = document.getElementById('status');
  const preview = document.getElementById('preview');
  
  btn.disabled = true;
  preview.innerHTML = '';
  const startTime = Date.now();
  status.textContent = 'Generating on GPU... (~1-3s)';
  
  try {
    const payload = { prompt, size, steps, response_format: 'b64_json' };
    if (seedVal) payload.seed = parseInt(seedVal);
    const resp = await fetch('/v1/images/generations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`Server returned HTTP ${resp.status}: ${err}`);
    }
    const data = await resp.json();
    const b64 = data.data[0].b64_json;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    status.textContent = `Completed in ${elapsed}s!`;
    const imgSrc = 'data:image/png;base64,' + b64;
    preview.innerHTML = `
      <img src="${imgSrc}" alt="Generated Image"><br>
      <a class="download-btn" href="${imgSrc}" download="sdxl-${Date.now()}.png">Download Full Resolution</a>
    `;
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

class ServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/health", "/v1/health"):
            self._send_json(200, {
                "status": "ok",
                "model": "__MODEL_ALIAS__",
                "architecture": "sdxl-turbo",
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            })
        elif path in ("/models", "/v1/models"):
            self._send_json(200, {
                "data": [{
                    "id": "__MODEL_ALIAS__",
                    "object": "model",
                    "type": "image-generation"
                }]
            })
        else:
            self._send_json(404, {"error": f"Path {path} not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            req = json.loads(post_data.decode("utf-8"))
        except Exception:
            return self._send_json(400, {"error": "Invalid JSON payload"})

        if path in ("/v1/images/generations", "/generate"):
            prompt = req.get("prompt", "")
            if not prompt:
                return self._send_json(400, {"error": "Missing prompt"})

            size = req.get("size", "512x512")
            try:
                width, height = map(int, size.lower().split("x"))
            except Exception:
                width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT

            steps = int(req.get("steps") or DEFAULT_STEPS)
            steps = max(1, min(10, steps))
            seed = req.get("seed")

            gen = None
            if seed is not None:
                try:
                    gen = torch.Generator("cuda").manual_seed(int(seed))
                except Exception:
                    pass

            t0 = time.time()
            print(f"[GEN] Prompt: '{prompt[:80]}...' size={width}x{height} steps={steps}", flush=True)

            with GEN_LOCK:
                with torch.inference_mode():
                    img = pipe(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=0.0,
                        generator=gen,
                    ).images[0]
                torch.cuda.empty_cache()
                gc.collect()

            elapsed = round(time.time() - t0, 2)
            print(f"[GEN] Done in {elapsed}s", flush=True)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64_bytes = base64.b64encode(buf.getvalue()).decode("utf-8")

            if path == "/generate":
                return self._send_json(200, {
                    "image_b64": b64_bytes,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "elapsed_seconds": elapsed
                })
            else:
                return self._send_json(200, {
                    "created": int(time.time()),
                    "data": [
                        {
                            "b64_json": b64_bytes,
                            "revised_prompt": prompt
                        }
                    ]
                })
        else:
            self._send_json(404, {"error": f"POST {path} not supported"})

server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), ServerHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print("Image server running on port", PORT, flush=True)

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
print("MODEL=" + "__MODEL_FILE__")
print("=" * 72)
print(f"Web UI         : {public_url}")
print(f"API endpoint   : {public_url}/v1/images/generations")
print(f"Health         : {public_url}/health")
print("Stop from PC   : ctr.sh stop   (uses the control tunnel above)")
'''

SDXL_CELLS = [CELL_1_ENV, CELL_2_SDXL_DEPS, CELL_3_SDXL_SERVE, CELL_5_KEEPALIVE]