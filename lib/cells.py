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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

AUTH_USER = "__AUTH_USER__" or os.environ.get("CLUSTER_AUTH_USER", "")
AUTH_PASS = "__AUTH_PASS__" or os.environ.get("CLUSTER_AUTH_PASS", "")
AUTH_KEY = "__AUTH_KEY__" or os.environ.get("CLUSTER_API_KEY", "")
AUTH_ENABLED = bool((AUTH_USER and AUTH_PASS) or AUTH_KEY)
LLAMA_TARGET_PORT = PORT if not AUTH_ENABLED else (PORT + 2)

# ---- llama-server (OpenAI-compatible) ----
cmd = [LLAMA_SERVER, "-m", model_path, "--jinja", "-ngl", str(NGL),
       "-fa", "on", "-c", str(CTX), "-np", "1",
       "--host", "127.0.0.1", "--port", str(LLAMA_TARGET_PORT)]
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
        with urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_TARGET_PORT}/health", timeout=4) as r:
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

print("llama-server is UP on port", LLAMA_TARGET_PORT, flush=True)

# ---- authenticating reverse proxy (if auth enabled) ----
if AUTH_ENABLED:
    import base64
    class AuthProxy(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _check_auth(self):
            path = self.path.split("?")[0].rstrip("/")
            if path in ("/health", "/healthz"):
                return True
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Basic "):
                try:
                    creds = base64.b64decode(auth[6:].strip()).decode("utf-8")
                    u, p = creds.split(":", 1)
                    if u == AUTH_USER and p == AUTH_PASS:
                        return True
                except Exception:
                    pass
            elif auth.startswith("Bearer "):
                token = auth[7:].strip()
                if (AUTH_KEY and token == AUTH_KEY) or (AUTH_PASS and token == AUTH_PASS):
                    return True
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="AI Cluster Protected"')
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"error": "Unauthorized: valid username and password required"}\n')
            return False

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def do_HEAD(self):
            if self._check_auth(): self._forward("HEAD")
        def do_GET(self):
            if self._check_auth(): self._forward("GET")
        def do_POST(self):
            if self._check_auth(): self._forward("POST")

        def _forward(self, method):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len > 0 else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "authorization")}
            req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_TARGET_PORT}{self.path}",
                                         data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    self.send_response(r.status)
                    for k, v in r.getheaders():
                        if k.lower() not in ("transfer-encoding", "content-length"):
                            self.send_header(k, v)
                    self.end_headers()
                    while True:
                        chunk = r.read(4096)
                        if not chunk: break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                for k, v in e.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(e.read())

    proxy = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), AuthProxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    print("Auth Proxy listening on port", PORT, flush=True)

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
CF_TOKEN = "__CLOUDFLARE_TUNNEL_TOKEN__" or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
STATIC_URL = "__STATIC_URL__" or os.environ.get("CLUSTER_URL", "")

def start_public_tunnel(url, log_path):
    fh = open(log_path, "w")
    if CF_TOKEN:
        print("Starting named Cloudflare tunnel with token...", flush=True)
        proc = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "run", "--token", CF_TOKEN, "--url", url],
                                stdout=fh, stderr=subprocess.STDOUT)
        time.sleep(5)
        if proc.poll() is not None:
            fh.flush()
            print("cloudflared failed to start! Log tail:", flush=True)
            print(open(log_path).read()[-2000:], flush=True)
            raise RuntimeError("named cloudflared tunnel exited immediately")
        return STATIC_URL or "https://ai.883742.xyz"
    else:
        subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", url],
                         stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(2)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
            if m:
                return m.group(0)
        return None

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

public_url = start_public_tunnel(f"http://127.0.0.1:{PORT}", CF_LOG)
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
AUTH_USER = "__AUTH_USER__"
AUTH_PASS = "__AUTH_PASS__"

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
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        if AUTH_USER and AUTH_PASS:
            import base64
            auth_str = base64.b64encode(f"{AUTH_USER}:{AUTH_PASS}".encode()).decode()
            req.add_header("Authorization", f"Basic {auth_str}")
        with urllib.request.urlopen(req, timeout=5) as r:
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

AUTH_USER = "__AUTH_USER__" or os.environ.get("CLUSTER_AUTH_USER", "")
AUTH_PASS = "__AUTH_PASS__" or os.environ.get("CLUSTER_AUTH_PASS", "")
AUTH_KEY = "__AUTH_KEY__" or os.environ.get("CLUSTER_API_KEY", "")
AUTH_ENABLED = bool((AUTH_USER and AUTH_PASS) or AUTH_KEY)

class ServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _check_auth(self):
        if not AUTH_ENABLED:
            return True
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health", "/healthz"):
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                creds = base64.b64decode(auth[6:].strip()).decode("utf-8")
                u, p = creds.split(":", 1)
                if u == AUTH_USER and p == AUTH_PASS:
                    return True
            except Exception:
                pass
        elif auth.startswith("Bearer "):
            token = auth[7:].strip()
            if (AUTH_KEY and token == AUTH_KEY) or (AUTH_PASS and token == AUTH_PASS):
                return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="AI Cluster Protected"')
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b'{"error": "Unauthorized: valid username and password required"}\n')
        return False

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
        if not self._check_auth(): return
        self.do_GET()

    def do_GET(self):
        if not self._check_auth(): return
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
        if not self._check_auth(): return
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
CF_TOKEN = "__CLOUDFLARE_TUNNEL_TOKEN__" or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
STATIC_URL = "__STATIC_URL__" or os.environ.get("CLUSTER_URL", "")

def start_public_tunnel(url, log_path):
    fh = open(log_path, "w")
    if CF_TOKEN:
        print("Starting named Cloudflare tunnel with token...", flush=True)
        proc = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "run", "--token", CF_TOKEN, "--url", url],
                                stdout=fh, stderr=subprocess.STDOUT)
        time.sleep(5)
        if proc.poll() is not None:
            fh.flush()
            print("cloudflared failed to start! Log tail:", flush=True)
            print(open(log_path).read()[-2000:], flush=True)
            raise RuntimeError("named cloudflared tunnel exited immediately")
        return STATIC_URL or "https://ai.883742.xyz"
    else:
        subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", url],
                         stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(2)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
            if m:
                return m.group(0)
        return None

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

public_url = start_public_tunnel(f"http://127.0.0.1:{PORT}", CF_LOG)
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

AUTH_USER = "__AUTH_USER__" or os.environ.get("CLUSTER_AUTH_USER", "")
AUTH_PASS = "__AUTH_PASS__" or os.environ.get("CLUSTER_AUTH_PASS", "")
AUTH_KEY = "__AUTH_KEY__" or os.environ.get("CLUSTER_API_KEY", "")
AUTH_ENABLED = bool((AUTH_USER and AUTH_PASS) or AUTH_KEY)

class ServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _check_auth(self):
        if not AUTH_ENABLED:
            return True
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health", "/healthz"):
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                creds = base64.b64decode(auth[6:].strip()).decode("utf-8")
                u, p = creds.split(":", 1)
                if u == AUTH_USER and p == AUTH_PASS:
                    return True
            except Exception:
                pass
        elif auth.startswith("Bearer "):
            token = auth[7:].strip()
            if (AUTH_KEY and token == AUTH_KEY) or (AUTH_PASS and token == AUTH_PASS):
                return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="AI Cluster Protected"')
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b'{"error": "Unauthorized: valid username and password required"}\n')
        return False

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
        if not self._check_auth(): return
        self.do_GET()

    def do_GET(self):
        if not self._check_auth(): return
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
        if not self._check_auth(): return
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
CF_TOKEN = "__CLOUDFLARE_TUNNEL_TOKEN__" or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
STATIC_URL = "__STATIC_URL__" or os.environ.get("CLUSTER_URL", "")

def start_public_tunnel(url, log_path):
    fh = open(log_path, "w")
    if CF_TOKEN:
        print("Starting named Cloudflare tunnel with token...", flush=True)
        proc = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "run", "--token", CF_TOKEN, "--url", url],
                                stdout=fh, stderr=subprocess.STDOUT)
        time.sleep(5)
        if proc.poll() is not None:
            fh.flush()
            print("cloudflared failed to start! Log tail:", flush=True)
            print(open(log_path).read()[-2000:], flush=True)
            raise RuntimeError("named cloudflared tunnel exited immediately")
        return STATIC_URL or "https://ai.883742.xyz"
    else:
        subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", url],
                         stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(2)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
            if m:
                return m.group(0)
        return None

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

public_url = start_public_tunnel(f"http://127.0.0.1:{PORT}", CF_LOG)
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

CELL_2_OMNIGEN_DEPS = r'''
# --- Cell 2: install dependencies for OmniGen inference -------------------
import subprocess, sys, os

pkgs = [
    "accelerate>=0.33.0",
    "diffusers>=0.30.0",
    "transformers>=4.48.0",
    "sentencepiece",
    "timm",
    "peft>=0.9.0",
    "datasets",
    "safetensors",
    "torchvision",
]
print("installing dependencies:", " ".join(pkgs), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)

OMNIGEN_DIR = "/kaggle/temp/OmniGen"
if not os.path.exists(OMNIGEN_DIR):
    print("cloning OmniGen repository...", flush=True)
    subprocess.run(["git", "clone", "--depth=1", "https://github.com/VectorSpaceLab/OmniGen.git", OMNIGEN_DIR], check=True)

# Patch scheduler.py to fix unused OffloadedCache import removed in newer transformers
sched_path = os.path.join(OMNIGEN_DIR, "OmniGen", "scheduler.py")
if os.path.exists(sched_path):
    print("patching OmniGen scheduler for transformers compatibility...", flush=True)
    with open(sched_path, "r") as f:
        code = f.read()
    code = code.replace(
        "from transformers.cache_utils import Cache, DynamicCache, OffloadedCache",
        "from transformers.cache_utils import Cache, DynamicCache\nOffloadedCache = DynamicCache"
    )
    code = code.replace(
        "super().__init__()\n        self.original_device = []",
        "super().__init__()\n        self.key_cache = []\n        self.value_cache = []\n        self._seen_tokens = 0\n        self.original_device = []"
    )
    with open(sched_path, "w") as f:
        f.write(code)

trans_path = os.path.join(OMNIGEN_DIR, "OmniGen", "transformer.py")
if os.path.exists(trans_path):
    print("patching OmniGen transformer for transformers compatibility...", flush=True)
    with open(trans_path, "r") as f:
        tcode = f.read()
    tcode = tcode.replace(
        "layer_outputs = decoder_layer(\n                    hidden_states,\n                    attention_mask=attention_mask,\n                    position_ids=position_ids,\n                    past_key_value=past_key_values,\n                    output_attentions=output_attentions,\n                    use_cache=use_cache,\n                    cache_position=cache_position,\n                )",
        "pos_emb = self.rotary_emb(hidden_states, position_ids=position_ids) if hasattr(self, 'rotary_emb') else None\n                layer_outputs = decoder_layer(\n                    hidden_states,\n                    attention_mask=attention_mask,\n                    position_ids=position_ids,\n                    past_key_values=past_key_values,\n                    past_key_value=past_key_values,\n                    output_attentions=output_attentions,\n                    use_cache=use_cache,\n                    cache_position=cache_position,\n                    position_embeddings=pos_emb,\n                )"
    )
    tcode = tcode.replace(
        "next_decoder_cache = layer_outputs[2 if output_attentions else 1]",
        "next_decoder_cache = layer_outputs[2 if output_attentions else 1] if (isinstance(layer_outputs, tuple) and len(layer_outputs) > (2 if output_attentions else 1)) else past_key_values"
    )
    tcode = tcode.replace(
        "hidden_states = layer_outputs[0]",
        "hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs"
    )
    with open(trans_path, "w") as f:
        f.write(tcode)

subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q", "-e", OMNIGEN_DIR], check=False)
print("OmniGen setup completed successfully.", flush=True)
'''

CELL_3_OMNIGEN_SERVE = r'''
# --- Cell 3: load OmniGen pipeline + start server + open tunnels ----------
import os, re, subprocess, time, urllib.request, json, threading, http.server, sys
import base64, io, gc
import torch
from PIL import Image

# Ensure OffloadedCache alias exists in transformers before any imports
try:
    import transformers.cache_utils
    if not hasattr(transformers.cache_utils, "OffloadedCache"):
        transformers.cache_utils.OffloadedCache = getattr(transformers.cache_utils, "DynamicCache", object)
except Exception:
    pass

# Patch longrope parameter validation in transformers
try:
    import transformers.modeling_rope_utils as rope_utils
    if hasattr(rope_utils, "RotaryEmbeddingConfigMixin"):
        rope_utils.RotaryEmbeddingConfigMixin.validate_rope = lambda self: None
        orig_val = getattr(rope_utils.RotaryEmbeddingConfigMixin, "_validate_longrope_rope_parameters", None)
        if orig_val:
            def patched_val(self, rope_parameters, ignore_keys=None):
                if "original_max_position_embeddings" not in rope_parameters:
                    rope_parameters["original_max_position_embeddings"] = getattr(self, "max_position_embeddings", 4096)
                return orig_val(self, rope_parameters, ignore_keys)
            rope_utils.RotaryEmbeddingConfigMixin._validate_longrope_rope_parameters = patched_val
    if hasattr(rope_utils, "RoPEScalingValidator"):
        rope_utils.RoPEScalingValidator.validate_rope = lambda self: None
except Exception:
    pass

# Patch Phi3Attention.forward and Phi3DecoderLayer.forward for transformers 4.48+
try:
    import transformers.models.phi3.modeling_phi3 as phi3_mod
    orig_attn_forward = phi3_mod.Phi3Attention.forward
    def patched_attn_forward(self, hidden_states, position_embeddings=None, *args, **kwargs):
        if position_embeddings is None:
            if not hasattr(self, "rotary_emb"):
                self.rotary_emb = phi3_mod.Phi3RotaryEmbedding(self.config).to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                seq_len = hidden_states.shape[1]
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        return orig_attn_forward(self, hidden_states, position_embeddings, *args, **kwargs)
    phi3_mod.Phi3Attention.forward = patched_attn_forward

    orig_decoder_forward = phi3_mod.Phi3DecoderLayer.forward
    def patched_decoder_forward(self, hidden_states, *args, **kwargs):
        if "past_key_value" in kwargs and "past_key_values" not in kwargs:
            kwargs["past_key_values"] = kwargs.pop("past_key_value")
        pkv = kwargs.get("past_key_values")
        use_cache = kwargs.get("use_cache", False)
        output_attentions = kwargs.get("output_attentions", False)

        out = orig_decoder_forward(self, hidden_states, *args, **kwargs)
        if not isinstance(out, tuple):
            if output_attentions and use_cache:
                return (out, None, pkv)
            elif use_cache:
                return (out, pkv)
            elif output_attentions:
                return (out, None)
            return (out,)
        return out
    phi3_mod.Phi3DecoderLayer.forward = patched_decoder_forward
except Exception as e:
    print("phi3 patch note:", e, flush=True)

OMNIGEN_DIR = "/kaggle/temp/OmniGen"
if os.path.exists(OMNIGEN_DIR) and OMNIGEN_DIR not in sys.path:
    sys.path.insert(0, OMNIGEN_DIR)

from OmniGen import OmniGenPipeline

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
DEFAULT_STEPS = int("__DEFAULT_STEPS__" or "30")
DEFAULT_WIDTH = int("__DEFAULT_WIDTH__" or "1024")
DEFAULT_HEIGHT = int("__DEFAULT_HEIGHT__" or "1024")
MODEL_REPO = "__MODEL_REPO__"

STOP_REQUESTED = threading.Event()
GEN_LOCK = threading.Lock()

WORK = "/kaggle/temp/run"
UPLOAD_DIR = os.path.join(WORK, "uploads")
os.makedirs(WORK, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
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
print(f"Loading OmniGen from {MODEL_REPO}...", flush=True)
pipe = OmniGenPipeline.from_pretrained(MODEL_REPO)
if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()
if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()
print("OmniGen pipeline loaded and ready on GPU!", flush=True)

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
                             "model": "__MODEL_ALIAS__"})
        elif self.path.startswith("/stop/" + STOP_TOKEN):
            STOP_REQUESTED.set()
            self._send(200, {"ok": True, "message": "shutdown requested"})
            print("STOP_REQUESTED received via control tunnel -> exiting immediately", flush=True)
            threading.Thread(target=lambda: (time.sleep(1), os._exit(0)), daemon=True).start()
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        self.do_GET()

ctl = http.server.ThreadingHTTPServer(("127.0.0.1", CTL_PORT), CtlHandler)
threading.Thread(target=ctl.serve_forever, daemon=True).start()

# ---- OmniGen Image Server on PORT (port 8080) ----
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniGen - Multi-Image & Thumbnail Studio</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  body { background: #0b0f19; color: #e1e4ea; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 24px 16px; }
  .card { background: #151b28; border: 1px solid #232d42; border-radius: 14px; width: 100%; max-width: 900px; padding: 28px; box-shadow: 0 12px 32px rgba(0,0,0,0.5); margin-bottom: 24px; }
  h1 { font-size: 1.7rem; color: #fff; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }
  .badge { font-size: 0.75rem; background: #2563eb; color: #fff; padding: 3px 10px; border-radius: 6px; font-weight: 700; }
  p.subtitle { color: #94a3b8; font-size: 0.92rem; margin-bottom: 20px; line-height: 1.4; }
  .notice { background: #1e293b; border-left: 4px solid #38bdf8; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; font-size: 0.88rem; color: #cbd5e1; }
  .notice code { color: #38bdf8; font-weight: 600; }
  label { display: block; font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 6px; }
  textarea { width: 100%; background: #0b0f19; border: 1px solid #334155; border-radius: 8px; color: #fff; padding: 12px; font-size: 0.95rem; min-height: 100px; resize: vertical; margin-bottom: 20px; }
  textarea:focus { outline: none; border-color: #38bdf8; }
  .upload-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-bottom: 20px; }
  .upload-box { background: #0f172a; border: 2px dashed #334155; border-radius: 10px; padding: 12px; text-align: center; cursor: pointer; transition: border-color 0.2s; position: relative; }
  .upload-box:hover { border-color: #38bdf8; }
  .upload-box input { display: none; }
  .upload-box img { max-width: 100%; max-height: 120px; border-radius: 6px; display: none; margin-top: 8px; }
  .upload-tag { font-size: 0.8rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px; }
  .upload-label { font-size: 0.75rem; color: #64748b; }
  .row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; margin-bottom: 20px; }
  select, input { width: 100%; background: #0b0f19; border: 1px solid #334155; border-radius: 8px; color: #fff; padding: 10px; font-size: 0.9rem; }
  button { width: 100%; background: #0284c7; color: #fff; border: none; padding: 14px 20px; font-size: 1.05rem; font-weight: 700; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #0369a1; }
  button:disabled { background: #475569; cursor: not-allowed; }
  #status { margin-top: 16px; font-size: 0.95rem; color: #38bdf8; text-align: center; }
  #preview { margin-top: 24px; text-align: center; }
  #preview img { max-width: 100%; border-radius: 10px; box-shadow: 0 6px 20px rgba(0,0,0,0.6); }
  .download-btn { margin-top: 14px; display: inline-block; background: #10b981; color: #fff; text-decoration: none; padding: 10px 20px; border-radius: 6px; font-size: 0.95rem; font-weight: 700; }
</style>
</head>
<body>
<div class="card">
  <h1>OmniGen <span class="badge">MULTI-IMAGE STUDIO</span></h1>
  <p class="subtitle">Unified Multi-Image Composition & Thumbnail Generator running on Kaggle GPU</p>

  <div class="notice">
    <strong>Thumbnail Multi-Subject Tip:</strong> Upload your reference photos below. In your prompt, reference them with <code>&lt;img&gt;&lt;|image_1|&gt;&lt;/img&gt;</code>, <code>&lt;img&gt;&lt;|image_2|&gt;&lt;/img&gt;</code>, etc. You can also leave the upload boxes empty for pure text-to-image generation!
  </div>

  <div class="upload-grid">
    <div class="upload-box" onclick="document.getElementById('file1').click()">
      <div class="upload-tag">&lt;|image_1|&gt;</div>
      <div class="upload-label" id="lbl1">Click to upload Subject 1</div>
      <input type="file" id="file1" accept="image/*" onchange="previewImg(1)">
      <img id="prev1" alt="Image 1">
    </div>
    <div class="upload-box" onclick="document.getElementById('file2').click()">
      <div class="upload-tag">&lt;|image_2|&gt;</div>
      <div class="upload-label" id="lbl2">Click to upload Subject 2</div>
      <input type="file" id="file2" accept="image/*" onchange="previewImg(2)">
      <img id="prev2" alt="Image 2">
    </div>
    <div class="upload-box" onclick="document.getElementById('file3').click()">
      <div class="upload-tag">&lt;|image_3|&gt;</div>
      <div class="upload-label" id="lbl3">Click to upload Subject 3</div>
      <input type="file" id="file3" accept="image/*" onchange="previewImg(3)">
      <img id="prev3" alt="Image 3">
    </div>
  </div>

  <label for="prompt">Prompt Instruction</label>
  <textarea id="prompt" placeholder="e.g. A dramatic YouTube thumbnail of <img><|image_1|></img> and <img><|image_2|></img> facing each other across a geopolitical map, intense cinematic lighting, photorealistic 8k"></textarea>

  <div class="row">
    <div>
      <label for="size">Resolution</label>
      <select id="size">
        <option value="1024x1024" selected>1024 × 1024</option>
        <option value="1024x768">1024 × 768 (Landscape)</option>
        <option value="768x1024">768 × 1024 (Portrait)</option>
        <option value="768x768">768 × 768</option>
        <option value="512x512">512 × 512 (Fastest)</option>
      </select>
    </div>
    <div>
      <label for="steps">Steps</label>
      <select id="steps">
        <option value="25">25 (Fast)</option>
        <option value="30" selected>30 (Balanced)</option>
        <option value="40">40 (Quality)</option>
        <option value="50">50 (Max Detail)</option>
      </select>
    </div>
    <div>
      <label for="cfg">Guidance Scale</label>
      <input id="cfg" type="number" step="0.5" value="3.0">
    </div>
    <div>
      <label for="seed">Seed (optional)</label>
      <input id="seed" type="number" placeholder="Random">
    </div>
  </div>

  <button id="genBtn" onclick="generate()">Generate Image</button>
  <div id="status"></div>
  <div id="preview"></div>
</div>

<script>
const imgs = [null, null, null];

function previewImg(idx) {
  const fileInput = document.getElementById('file' + idx);
  if (!fileInput.files || !fileInput.files[0]) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    imgs[idx - 1] = e.target.result;
    const preview = document.getElementById('prev' + idx);
    preview.src = e.target.result;
    preview.style.display = 'block';
    document.getElementById('lbl' + idx).textContent = fileInput.files[0].name.slice(0, 18);
  };
  reader.readAsDataURL(fileInput.files[0]);
}

async function generate() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return alert('Please enter a prompt');
  const size = document.getElementById('size').value;
  const steps = parseInt(document.getElementById('steps').value);
  const guidance_scale = parseFloat(document.getElementById('cfg').value) || 3.0;
  const seedVal = document.getElementById('seed').value;
  const btn = document.getElementById('genBtn');
  const status = document.getElementById('status');
  const preview = document.getElementById('preview');

  btn.disabled = true;
  preview.innerHTML = '';
  const startTime = Date.now();
  status.textContent = 'Generating on Kaggle GPU... (~20-40s)';

  try {
    const input_images = imgs.filter(Boolean);
    const payload = { prompt, size, steps, guidance_scale, input_images, response_format: 'b64_json' };
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
      <a class="download-btn" href="${imgSrc}" download="omnigen-${Date.now()}.png">Download Full Resolution</a>
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

AUTH_USER = "__AUTH_USER__" or os.environ.get("CLUSTER_AUTH_USER", "")
AUTH_PASS = "__AUTH_PASS__" or os.environ.get("CLUSTER_AUTH_PASS", "")
AUTH_KEY = "__AUTH_KEY__" or os.environ.get("CLUSTER_API_KEY", "")
AUTH_ENABLED = bool((AUTH_USER and AUTH_PASS) or AUTH_KEY)

class ServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _check_auth(self):
        if not AUTH_ENABLED:
            return True
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health", "/healthz"):
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                creds = base64.b64decode(auth[6:].strip()).decode("utf-8")
                u, p = creds.split(":", 1)
                if u == AUTH_USER and p == AUTH_PASS:
                    return True
            except Exception:
                pass
        elif auth.startswith("Bearer "):
            token = auth[7:].strip()
            if (AUTH_KEY and token == AUTH_KEY) or (AUTH_PASS and token == AUTH_PASS):
                return True
            try:
                creds = base64.b64decode(token).decode("utf-8")
                u, p = creds.split(":", 1)
                if u == AUTH_USER and p == AUTH_PASS:
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="AI Cluster Protected"')
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b'{"error": "Unauthorized: valid username and password required"}\n')
        return False

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
        if not self._check_auth(): return
        self.do_GET()

    def do_GET(self):
        if not self._check_auth(): return
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
                "architecture": "omnigen",
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
        elif path in ("/last_image", "/v1/images/last"):
            found = False
            for p in ("/kaggle/working/last_image.png", os.path.join(WORK, "last_image.png")):
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(data)
                    found = True
                    break
            if not found:
                self._send_json(404, {"error": "No image generated yet"})
        else:
            self._send_json(404, {"error": f"Path {path} not found"})

    def do_POST(self):
        if not self._check_auth(): return
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
                width = int(req.get("width", DEFAULT_WIDTH))
                height = int(req.get("height", DEFAULT_HEIGHT))

            steps = int(req.get("steps", DEFAULT_STEPS))
            guidance_scale = float(req.get("guidance_scale", 3.0))
            img_guidance_scale = float(req.get("img_guidance_scale", 1.6))
            seed = int(req.get("seed", 0)) if req.get("seed") is not None else None
            max_input_image_size = int(req.get("max_input_image_size", 512))
            separate_cfg_infer = bool(req.get("separate_cfg_infer", True))
            use_kv_cache = bool(req.get("use_kv_cache", True))
            offload_kv_cache = bool(req.get("offload_kv_cache", True))

            # Process input images if provided
            raw_input_images = req.get("input_images", [])
            local_paths = []
            for i, item in enumerate(raw_input_images):
                if not item: continue
                dest_path = os.path.join(UPLOAD_DIR, f"input_{i}.png")
                if item.startswith("data:"):
                    # base64 data URI
                    header, b64 = item.split(",", 1)
                    with open(dest_path, "wb") as f:
                        f.write(base64.b64decode(b64))
                    local_paths.append(dest_path)
                elif item.startswith("http://") or item.startswith("https://"):
                    req_img = urllib.request.Request(item, headers={"User-Agent": "curl/8.5.0"})
                    with urllib.request.urlopen(req_img) as r, open(dest_path, "wb") as f:
                        f.write(r.read())
                    local_paths.append(dest_path)
                elif os.path.exists(item):
                    local_paths.append(item)

            user_offload = req.get("offload_model")
            offload_model = bool(user_offload) if user_offload is not None else False

            print(f"Generating OmniGen ({width}x{height}, {steps} steps, {len(local_paths)} ref images, offload={offload_model}, max_input={max_input_image_size})...", flush=True)
            t0 = time.time()
            try:
                with GEN_LOCK:
                    out = pipe(
                        prompt=prompt,
                        input_images=local_paths if local_paths else None,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        img_guidance_scale=img_guidance_scale,
                        max_input_image_size=max_input_image_size,
                        separate_cfg_infer=separate_cfg_infer,
                        offload_model=offload_model,
                        use_kv_cache=use_kv_cache,
                        offload_kv_cache=offload_kv_cache,
                        seed=seed,
                    )
                    img = out[0]
                    try:
                        img.save("/kaggle/working/last_image.png")
                        img.save(os.path.join(WORK, "last_image.png"))
                    except Exception as e:
                        print("save image warning:", e, flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._send_json(500, {"error": str(e)})
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elapsed = round(time.time() - t0, 2)
            print(f"Generation done in {elapsed}s", flush=True)

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
print("OmniGen server running on port", PORT, flush=True)

# ---- publish both tunnels ----
CF_TOKEN = "__CLOUDFLARE_TUNNEL_TOKEN__" or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
STATIC_URL = "__STATIC_URL__" or os.environ.get("CLUSTER_URL", "")

def start_public_tunnel(url, log_path):
    fh = open(log_path, "w")
    if CF_TOKEN:
        print("Starting named Cloudflare tunnel with token...", flush=True)
        proc = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "run", "--token", CF_TOKEN, "--url", url],
                                stdout=fh, stderr=subprocess.STDOUT)
        time.sleep(5)
        if proc.poll() is not None:
            fh.flush()
            print("cloudflared failed to start! Log tail:", flush=True)
            print(open(log_path).read()[-2000:], flush=True)
            raise RuntimeError("named cloudflared tunnel exited immediately")
        return STATIC_URL or "https://ai.883742.xyz"
    else:
        subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", url],
                         stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(2)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
            if m:
                return m.group(0)
        return None

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

public_url = start_public_tunnel(f"http://127.0.0.1:{PORT}", CF_LOG)
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
print("MODEL=" + "__MODEL_REPO__")
print("=" * 72)
print(f"Web UI         : {public_url}")
print(f"API endpoint   : {public_url}/v1/images/generations")
print(f"Health         : {public_url}/health")
print("Stop from PC   : ctr.sh stop   (uses the control tunnel above)")
'''

OMNIGEN_CELLS = [CELL_1_ENV, CELL_2_OMNIGEN_DEPS, CELL_3_OMNIGEN_SERVE, CELL_5_KEEPALIVE]

CELL_2_JANUS_DEPS = r'''
# --- Cell 2: install dependencies for Janus-Pro inference -----------------
import subprocess, sys

pkgs = [
    "git+https://github.com/deepseek-ai/Janus.git",
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "timm>=0.9.16",
    "sentencepiece",
    "attrdict",
    "einops",
]
print("installing:", " ".join(pkgs), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("Janus packages installed.", flush=True)
'''

CELL_3_JANUS_SERVE = r'''
# --- Cell 3: load Janus-Pro model + start server + open tunnels -----------
import os, re, subprocess, time, urllib.request, json, threading, http.server
import base64, io, gc
import torch
import numpy as np
import PIL.Image
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
DEFAULT_CFG = float("__DEFAULT_CFG__" or "5.0")
DEFAULT_TEMP = float("__DEFAULT_TEMP__" or "1.0")
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

# ---- load model ----
print(f"Loading Janus-Pro from {MODEL_REPO}...", flush=True)
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_REPO)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()
print("Janus-Pro model loaded and ready across GPU(s)!", flush=True)

@torch.inference_mode()
def janus_generate(prompt: str, temperature: float = 1.0, cfg_weight: float = 5.0, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    parallel_size = 1
    image_token_num_per_image = 576
    img_size = 384
    patch_size = 16

    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    full_prompt = sft_format + vl_chat_processor.image_start_tag
    input_ids = vl_chat_processor.tokenizer.encode(full_prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    outputs = None

    for i in range(image_token_num_per_image):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / max(temperature, 0.01), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token, next_token], dim=0)
        inputs_embeds = vl_gpt.prepare_gen_img_embeds(next_token)

    dec = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(dec[0])

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
                             "model": "__MODEL_ALIAS__"})
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

# ---- Janus Image Server on PORT (port 8080) ----
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Janus-Pro-7B - DeepSeek Multimodal & Image Generation</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  body { background: #0c1017; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 24px 16px; }
  .card { background: #161e2e; border: 1px solid #2d3748; border-radius: 14px; width: 100%; max-width: 840px; padding: 28px; box-shadow: 0 12px 32px rgba(0,0,0,0.5); margin-bottom: 24px; }
  h1 { font-size: 1.7rem; color: #fff; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }
  .badge { font-size: 0.75rem; background: #6366f1; color: #fff; padding: 3px 10px; border-radius: 6px; font-weight: 700; }
  p.subtitle { color: #94a3b8; font-size: 0.92rem; margin-bottom: 20px; line-height: 1.4; }
  label { display: block; font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 6px; }
  textarea { width: 100%; background: #0c1017; border: 1px solid #334155; border-radius: 8px; color: #fff; padding: 12px; font-size: 0.95rem; min-height: 100px; resize: vertical; margin-bottom: 20px; }
  textarea:focus { outline: none; border-color: #6366f1; }
  .row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 20px; }
  input { width: 100%; background: #0c1017; border: 1px solid #334155; border-radius: 8px; color: #fff; padding: 10px; font-size: 0.9rem; }
  button { width: 100%; background: #4f46e5; color: #fff; border: none; padding: 14px 20px; font-size: 1.05rem; font-weight: 700; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #4338ca; }
  button:disabled { background: #475569; cursor: not-allowed; }
  #status { margin-top: 16px; font-size: 0.95rem; color: #818cf8; text-align: center; }
  #preview { margin-top: 24px; text-align: center; }
  #preview img { max-width: 100%; border-radius: 10px; box-shadow: 0 6px 20px rgba(0,0,0,0.6); }
  .download-btn { margin-top: 14px; display: inline-block; background: #10b981; color: #fff; text-decoration: none; padding: 10px 20px; border-radius: 6px; font-size: 0.95rem; font-weight: 700; }
</style>
</head>
<body>
<div class="card">
  <h1>Janus-Pro-7B <span class="badge">DEEPSEEK AI</span></h1>
  <p class="subtitle">DeepSeek Unified Multimodal Understanding & Visual Generation running on Kaggle GPU</p>

  <label for="prompt">Prompt Instruction</label>
  <textarea id="prompt" placeholder="e.g. A stunning cinematic YouTube thumbnail featuring a high-stakes meeting in a grand palace, vibrant lighting, highly detailed, photorealistic"></textarea>

  <div class="row">
    <div>
      <label for="cfg">CFG Guidance Scale</label>
      <input id="cfg" type="number" step="0.5" value="5.0">
    </div>
    <div>
      <label for="temp">Temperature</label>
      <input id="temp" type="number" step="0.1" value="1.0">
    </div>
    <div>
      <label for="seed">Seed (optional)</label>
      <input id="seed" type="number" placeholder="Random">
    </div>
  </div>

  <button id="genBtn" onclick="generate()">Generate Image</button>
  <div id="status"></div>
  <div id="preview"></div>
</div>

<script>
async function generate() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return alert('Please enter a prompt');
  const cfg_weight = parseFloat(document.getElementById('cfg').value) || 5.0;
  const temperature = parseFloat(document.getElementById('temp').value) || 1.0;
  const seedVal = document.getElementById('seed').value;
  const btn = document.getElementById('genBtn');
  const status = document.getElementById('status');
  const preview = document.getElementById('preview');

  btn.disabled = true;
  preview.innerHTML = '';
  const startTime = Date.now();
  status.textContent = 'Generating on Kaggle GPU with DeepSeek Janus-Pro... (~15-30s)';

  try {
    const payload = { prompt, guidance_scale: cfg_weight, temperature, response_format: 'b64_json' };
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
      <a class="download-btn" href="${imgSrc}" download="janus-${Date.now()}.png">Download Full Resolution</a>
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

AUTH_USER = "__AUTH_USER__" or os.environ.get("CLUSTER_AUTH_USER", "")
AUTH_PASS = "__AUTH_PASS__" or os.environ.get("CLUSTER_AUTH_PASS", "")
AUTH_KEY = "__AUTH_KEY__" or os.environ.get("CLUSTER_API_KEY", "")
AUTH_ENABLED = bool((AUTH_USER and AUTH_PASS) or AUTH_KEY)

class ServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _check_auth(self):
        if not AUTH_ENABLED:
            return True
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health", "/healthz"):
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                creds = base64.b64decode(auth[6:].strip()).decode("utf-8")
                u, p = creds.split(":", 1)
                if u == AUTH_USER and p == AUTH_PASS:
                    return True
            except Exception:
                pass
        elif auth.startswith("Bearer "):
            token = auth[7:].strip()
            if (AUTH_KEY and token == AUTH_KEY) or (AUTH_PASS and token == AUTH_PASS):
                return True
            try:
                creds = base64.b64decode(token).decode("utf-8")
                u, p = creds.split(":", 1)
                if u == AUTH_USER and p == AUTH_PASS:
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="AI Cluster Protected"')
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b'{"error": "Unauthorized: valid username and password required"}\n')
        return False

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
        if not self._check_auth(): return
        self.do_GET()

    def do_GET(self):
        if not self._check_auth(): return
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
                "architecture": "janus-pro",
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
        if not self._check_auth(): return
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

            cfg_weight = float(req.get("guidance_scale", DEFAULT_CFG))
            temperature = float(req.get("temperature", DEFAULT_TEMP))
            seed = int(req.get("seed", 0)) if req.get("seed") is not None else None

            print(f"Generating Janus-Pro image for: {prompt[:80]}...", flush=True)
            t0 = time.time()
            with GEN_LOCK:
                img = janus_generate(prompt=prompt, temperature=temperature, cfg_weight=cfg_weight, seed=seed)
            elapsed = round(time.time() - t0, 2)
            print(f"Janus generation done in {elapsed}s", flush=True)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64_bytes = base64.b64encode(buf.getvalue()).decode("utf-8")

            if path == "/generate":
                return self._send_json(200, {
                    "image_b64": b64_bytes,
                    "prompt": prompt,
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
print("Janus server running on port", PORT, flush=True)

# ---- publish both tunnels ----
CF_TOKEN = "__CLOUDFLARE_TUNNEL_TOKEN__" or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
STATIC_URL = "__STATIC_URL__" or os.environ.get("CLUSTER_URL", "")

def start_public_tunnel(url, log_path):
    fh = open(log_path, "w")
    if CF_TOKEN:
        print("Starting named Cloudflare tunnel with token...", flush=True)
        proc = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "run", "--token", CF_TOKEN, "--url", url],
                                stdout=fh, stderr=subprocess.STDOUT)
        time.sleep(5)
        if proc.poll() is not None:
            fh.flush()
            print("cloudflared failed to start! Log tail:", flush=True)
            print(open(log_path).read()[-2000:], flush=True)
            raise RuntimeError("named cloudflared tunnel exited immediately")
        return STATIC_URL or "https://ai.883742.xyz"
    else:
        subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", url],
                         stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(2)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
            if m:
                return m.group(0)
        return None

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

public_url = start_public_tunnel(f"http://127.0.0.1:{PORT}", CF_LOG)
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
print("MODEL=" + "__MODEL_REPO__")
print("=" * 72)
print(f"Web UI         : {public_url}")
print(f"API endpoint   : {public_url}/v1/images/generations")
print(f"Health         : {public_url}/health")
print("Stop from PC   : ctr.sh stop   (uses the control tunnel above)")
'''

JANUS_CELLS = [CELL_1_ENV, CELL_2_JANUS_DEPS, CELL_3_JANUS_SERVE, CELL_5_KEEPALIVE]

CELL_2_ZIMAGE_DEPS = r'''
# --- Cell 2: install dependencies for Z-Image-Turbo inference ------------
import subprocess, sys

pkgs = [
    "git+https://github.com/huggingface/diffusers",
    "transformers>=4.45.0",
    "accelerate>=0.33.0",
    "sentencepiece",
    "safetensors",
]
print("installing:", " ".join(pkgs), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("Z-Image packages installed.", flush=True)
'''

CELL_3_ZIMAGE_SERVE = r'''
# --- Cell 3: load Z-Image-Turbo pipeline + start server + open tunnels ----
import os, re, subprocess, time, urllib.request, json, threading, http.server
import base64, io, gc
import torch
from diffusers import ZImagePipeline

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
DEFAULT_STEPS = int("__DEFAULT_STEPS__" or "9")
DEFAULT_WIDTH = int("__DEFAULT_WIDTH__" or "1024")
DEFAULT_HEIGHT = int("__DEFAULT_HEIGHT__" or "1024")
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
print(f"Loading Z-Image-Turbo from {MODEL_REPO}...", flush=True)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
pipe = ZImagePipeline.from_pretrained(
    MODEL_REPO,
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
)
if torch.cuda.get_device_properties(0).total_memory < 20e9:
    print("Enabling model CPU offload for optimal VRAM management...", flush=True)
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")
print("Z-Image-Turbo pipeline loaded and ready on GPU!", flush=True)

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
                             "model": "__MODEL_ALIAS__"})
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

# ---- Z-Image Web UI and Server on PORT (port 8080) ----
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Z-Image-Turbo - 8-Step Ultra-Fast Generator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  body { background: #080b12; color: #e1e7f0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 24px 16px; }
  .card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; width: 100%; max-width: 860px; padding: 28px; box-shadow: 0 12px 32px rgba(0,0,0,0.5); margin-bottom: 24px; }
  h1 { font-size: 1.7rem; color: #fff; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }
  .badge { font-size: 0.75rem; background: #ea580c; color: #fff; padding: 3px 10px; border-radius: 6px; font-weight: 700; }
  p.subtitle { color: #9ca3af; font-size: 0.92rem; margin-bottom: 20px; line-height: 1.4; }
  .speed-tag { background: #1e1b4b; border-left: 4px solid #f97316; padding: 10px 14px; border-radius: 6px; margin-bottom: 20px; font-size: 0.88rem; color: #cbd5e1; }
  label { display: block; font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 6px; }
  textarea { width: 100%; background: #080b12; border: 1px solid #374151; border-radius: 8px; color: #fff; padding: 12px; font-size: 0.95rem; min-height: 100px; resize: vertical; margin-bottom: 20px; }
  textarea:focus { outline: none; border-color: #f97316; }
  .row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-bottom: 20px; }
  select, input { width: 100%; background: #080b12; border: 1px solid #374151; border-radius: 8px; color: #fff; padding: 10px; font-size: 0.9rem; }
  button { width: 100%; background: #ea580c; color: #fff; border: none; padding: 14px 20px; font-size: 1.05rem; font-weight: 700; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #c2410c; }
  button:disabled { background: #4b5563; cursor: not-allowed; }
  #status { margin-top: 16px; font-size: 0.95rem; color: #fb923c; text-align: center; }
  #preview { margin-top: 24px; text-align: center; }
  #preview img { max-width: 100%; border-radius: 10px; box-shadow: 0 6px 20px rgba(0,0,0,0.6); }
  .download-btn { margin-top: 14px; display: inline-block; background: #10b981; color: #fff; text-decoration: none; padding: 10px 20px; border-radius: 6px; font-size: 0.95rem; font-weight: 700; }
</style>
</head>
<body>
<div class="card">
  <h1>⚡ Z-Image-Turbo <span class="badge">8-STEP TURBO</span></h1>
  <p class="subtitle">Alibaba Tongyi Scalable Single-Stream Diffusion Transformer (6B S3-DiT) running on Kaggle GPU</p>

  <div class="speed-tag">
    ⚡ <strong>Ultra-Fast 8-Step Distillation:</strong> Generates crisp 1024x1024 photorealistic visuals and thumbnail artwork in ~3–5 seconds on Kaggle GPU!
  </div>

  <label for="prompt">Prompt Instruction</label>
  <textarea id="prompt" placeholder="e.g. A dramatic YouTube thumbnail of Donald Trump, Vladimir Putin, and Volodymyr Zelenskyy seated around an intense negotiation table in a modern war room, dramatic side lighting, high contrast, 8k, photorealistic"></textarea>

  <div class="row">
    <div>
      <label for="size">Resolution</label>
      <select id="size">
        <option value="1024x1024" selected>1024 × 1024</option>
        <option value="1024x768">1024 × 768 (Landscape)</option>
        <option value="768x1024">768 × 1024 (Portrait)</option>
        <option value="768x768">768 × 768</option>
        <option value="512x512">512 × 512 (Fastest)</option>
      </select>
    </div>
    <div>
      <label for="steps">Inference Steps</label>
      <select id="steps">
        <option value="9" selected>8 DiT Steps (Optimal)</option>
        <option value="12">12 Steps (Detailed)</option>
        <option value="6">6 Steps (Fast)</option>
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
  status.textContent = 'Generating on Kaggle GPU with Z-Image-Turbo... (~3-8s)';

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
      <a class="download-btn" href="${imgSrc}" download="zimage-${Date.now()}.png">Download Full Resolution</a>
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

AUTH_USER = "__AUTH_USER__" or os.environ.get("CLUSTER_AUTH_USER", "")
AUTH_PASS = "__AUTH_PASS__" or os.environ.get("CLUSTER_AUTH_PASS", "")
AUTH_KEY = "__AUTH_KEY__" or os.environ.get("CLUSTER_API_KEY", "")
AUTH_ENABLED = bool((AUTH_USER and AUTH_PASS) or AUTH_KEY)

class ServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _check_auth(self):
        if not AUTH_ENABLED:
            return True
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health", "/healthz"):
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                creds = base64.b64decode(auth[6:].strip()).decode("utf-8")
                u, p = creds.split(":", 1)
                if u == AUTH_USER and p == AUTH_PASS:
                    return True
            except Exception:
                pass
        elif auth.startswith("Bearer "):
            token = auth[7:].strip()
            if (AUTH_KEY and token == AUTH_KEY) or (AUTH_PASS and token == AUTH_PASS):
                return True
            try:
                creds = base64.b64decode(token).decode("utf-8")
                u, p = creds.split(":", 1)
                if u == AUTH_USER and p == AUTH_PASS:
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="AI Cluster Protected"')
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b'{"error": "Unauthorized: valid username and password required"}\n')
        return False

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
        if not self._check_auth(): return
        self.do_GET()

    def do_GET(self):
        if not self._check_auth(): return
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
                "architecture": "z-image-turbo",
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
        if not self._check_auth(): return
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
                width = int(req.get("width", DEFAULT_WIDTH))
                height = int(req.get("height", DEFAULT_HEIGHT))

            steps = int(req.get("steps", DEFAULT_STEPS))
            seed = int(req.get("seed", 0)) if req.get("seed") is not None else None
            gen = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

            print(f"Generating Z-Image-Turbo ({width}x{height}, {steps} steps) for: {prompt[:80]}...", flush=True)
            t0 = time.time()
            with GEN_LOCK:
                out = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    generator=gen,
                )
                img = out.images[0]
            elapsed = round(time.time() - t0, 2)
            print(f"Z-Image generation done in {elapsed}s", flush=True)

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
print("Z-Image server running on port", PORT, flush=True)

# ---- publish both tunnels ----
CF_TOKEN = "__CLOUDFLARE_TUNNEL_TOKEN__" or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
STATIC_URL = "__STATIC_URL__" or os.environ.get("CLUSTER_URL", "")

def start_public_tunnel(url, log_path):
    fh = open(log_path, "w")
    if CF_TOKEN:
        print("Starting named Cloudflare tunnel with token...", flush=True)
        proc = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "run", "--token", CF_TOKEN, "--url", url],
                                stdout=fh, stderr=subprocess.STDOUT)
        time.sleep(5)
        if proc.poll() is not None:
            fh.flush()
            print("cloudflared failed to start! Log tail:", flush=True)
            print(open(log_path).read()[-2000:], flush=True)
            raise RuntimeError("named cloudflared tunnel exited immediately")
        return STATIC_URL or "https://ai.883742.xyz"
    else:
        subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", url],
                         stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(2)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
            if m:
                return m.group(0)
        return None

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

public_url = start_public_tunnel(f"http://127.0.0.1:{PORT}", CF_LOG)
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
print("MODEL=" + "__MODEL_REPO__")
print("=" * 72)
print(f"Web UI         : {public_url}")
print(f"API endpoint   : {public_url}/v1/images/generations")
print(f"Health         : {public_url}/health")
print("Stop from PC   : ctr.sh stop   (uses the control tunnel above)")
'''

Z_IMAGE_CELLS = [CELL_1_ENV, CELL_2_ZIMAGE_DEPS, CELL_3_ZIMAGE_SERVE, CELL_5_KEEPALIVE]


CELL_2_FLUX_FACE_DEPS = r'''
# --- Cell 2: install dependencies for FLUX GGUF + InsightFace Face Swapper ---
import subprocess, sys, os

pkgs = [
    "diffusers>=0.30.0",
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "sentencepiece",
    "gguf>=0.10.0",
    "protobuf",
    "opencv-python-headless",
    "insightface",
    "onnxruntime-gpu",
]
print("installing packages:", " ".join(pkgs), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("packages installed successfully.", flush=True)

# Pre-download inswapper_128.onnx into /kaggle/temp/models
MODEL_DIR = "/kaggle/temp/models"
os.makedirs(MODEL_DIR, exist_ok=True)
inswapper_path = os.path.join(MODEL_DIR, "inswapper_128.onnx")
if not os.path.exists(inswapper_path) or os.path.getsize(inswapper_path) < 1000000:
    print("Downloading inswapper_128.onnx model...", flush=True)
    import urllib.request
    url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    urllib.request.urlretrieve(url, inswapper_path)
    print(f"inswapper_128.onnx ready ({os.path.getsize(inswapper_path) / (1024*1024):.1f} MB)", flush=True)
'''

CELL_4_FLUX_FACE_SERVE = r'''
# --- Cell 4: load FLUX + InsightFace + start server + open tunnels ----------
import os, re, subprocess, time, urllib.request, json, threading, http.server
import base64, io, gc
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
import insightface
from insightface.app import FaceAnalysis

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
DEFAULT_STEPS = int("__DEFAULT_STEPS__" or "4")
DEFAULT_WIDTH = int("__DEFAULT_WIDTH__" or "1024")
DEFAULT_HEIGHT = int("__DEFAULT_HEIGHT__" or "576")

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

# ---- load FLUX pipeline ----
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

# ---- load InsightFace Face Swapper ----
print("Loading InsightFace buffalo_l...", flush=True)
face_app = FaceAnalysis(name="buffalo_l", root="/kaggle/temp/insightface", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

inswapper_path = os.path.join("/kaggle/temp/models", "inswapper_128.onnx")
face_swapper = insightface.model_zoo.get_model(inswapper_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("InsightFace Face Swapper loaded and ready!", flush=True)

# ---- Helper functions for face processing ----
def parse_image_input(item):
    if not item: return None
    if isinstance(item, str):
        try:
            if item.startswith("data:"):
                header, b64 = item.split(",", 1)
                img_bytes = base64.b64decode(b64)
            elif item.startswith("http://") or item.startswith("https://"):
                req = urllib.request.Request(item, headers={"User-Agent": "curl/8.5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    img_bytes = r.read()
            elif os.path.exists(item):
                with open(item, "rb") as f:
                    img_bytes = f.read()
            else:
                return None
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print("Error parsing image input:", e, flush=True)
            return None
    return None

def perform_face_swaps(target_pil_img, face_left=None, face_right=None, face_images=None):
    try:
        target_cv = cv2.cvtColor(np.array(target_pil_img), cv2.COLOR_RGB2BGR)
        target_faces = face_app.get(target_cv)
        if not target_faces:
            print("[FACE] No faces detected in generated target image to swap.", flush=True)
            return target_pil_img
        
        target_faces = sorted(target_faces, key=lambda f: f.bbox[0])
        print(f"[FACE] Detected {len(target_faces)} faces in target image.", flush=True)
        result_cv = target_cv.copy()
        
        # 1. Left face swap
        left_cv = parse_image_input(face_left)
        if left_cv is not None and len(target_faces) >= 1:
            src_faces = face_app.get(left_cv)
            if src_faces:
                src_face = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                result_cv = face_swapper.get(result_cv, target_faces[0], src_face, paste_back=True)
                print("[FACE] Swapped face_left onto target face 0.", flush=True)
            else:
                print("[FACE] Warning: No face detected in face_left source image.", flush=True)

        # 2. Right face swap
        right_cv = parse_image_input(face_right)
        if right_cv is not None and len(target_faces) >= 2:
            src_faces = face_app.get(right_cv)
            if src_faces:
                src_face = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                result_cv = face_swapper.get(result_cv, target_faces[-1], src_face, paste_back=True)
                print(f"[FACE] Swapped face_right onto target face {len(target_faces)-1}.", flush=True)
            else:
                print("[FACE] Warning: No face detected in face_right source image.", flush=True)

        # 3. Arbitrary ordered face_images list
        if face_images and isinstance(face_images, list):
            for idx, item in enumerate(face_images):
                if idx >= len(target_faces): break
                src_cv = parse_image_input(item)
                if src_cv is not None:
                    src_faces = face_app.get(src_cv)
                    if src_faces:
                        src_face = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                        result_cv = face_swapper.get(result_cv, target_faces[idx], src_face, paste_back=True)
                        print(f"[FACE] Swapped face_images[{idx}] onto target face {idx}.", flush=True)

        return Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print("[FACE] Face swap error:", e, flush=True)
        return target_pil_img

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
            import threading
            threading.Thread(target=lambda: (time.sleep(1), os._exit(0))).start()
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
<title>FLUX.1-schnell + Face Swapper Studio</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  body { background: #0f1117; color: #e1e4ea; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 24px 16px; }
  .card { background: #1a1d27; border: 1px solid #2d3139; border-radius: 12px; width: 100%; max-width: 900px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); margin-bottom: 24px; }
  h1 { font-size: 1.6rem; color: #fff; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
  .badge { font-size: 0.75rem; background: #059669; color: #fff; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
  p.subtitle { color: #9ca3af; font-size: 0.9rem; margin-bottom: 20px; }
  label { display: block; font-size: 0.85rem; font-weight: 600; color: #d1d5db; margin-bottom: 6px; }
  textarea { width: 100%; background: #0f1117; border: 1px solid #374151; border-radius: 8px; color: #fff; padding: 12px; font-size: 0.95rem; min-height: 90px; resize: vertical; margin-bottom: 16px; }
  textarea:focus { outline: none; border-color: #3b82f6; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .row-faces { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .face-box { background: #11131a; border: 1px dashed #374151; border-radius: 8px; padding: 12px; }
  select, input[type=file], input[type=text] { width: 100%; background: #0f1117; border: 1px solid #374151; border-radius: 8px; color: #fff; padding: 10px; font-size: 0.9rem; }
  button { width: 100%; background: #2563eb; color: #fff; border: none; padding: 12px 20px; font-size: 1rem; font-weight: 600; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #4b5563; cursor: not-allowed; }
  #status { margin-top: 16px; font-size: 0.9rem; color: #93c5fd; text-align: center; }
  #preview { margin-top: 24px; text-align: center; }
  #preview img { max-width: 100%; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.5); }
  .download-btn { margin-top: 12px; display: inline-block; background: #10b981; color: #fff; text-decoration: none; padding: 8px 16px; border-radius: 6px; font-size: 0.9rem; font-weight: 600; }
</style>
</head>
<body>
<div class="card">
  <h1>FLUX.1-schnell + Face Swapper <span class="badge">ONLINE</span></h1>
  <p class="subtitle">Generate realistic thumbnail scenes in 4 steps (~8s) and automatically swap reference faces with 100% fidelity</p>
  
  <label for="prompt">Scene Prompt</label>
  <textarea id="prompt" placeholder="A dramatic YouTube documentary thumbnail about Bangladesh military ambition, high-tech war room, tactical radar screen, stealth fighter jets, cinematic news lighting..."></textarea>
  
  <div class="row-faces">
    <div class="face-box">
      <label>Face 1 (Left / Foreground Person)</label>
      <input type="file" id="face_left" accept="image/*">
    </div>
    <div class="face-box">
      <label>Face 2 (Right / Background Person)</label>
      <input type="file" id="face_right" accept="image/*">
    </div>
  </div>

  <div class="row">
    <div>
      <label for="size">Resolution</label>
      <select id="size">
        <option value="1024x576" selected>1024 × 576 (YouTube Thumbnail 16:9)</option>
        <option value="1024x1024">1024 × 1024 (Square)</option>
        <option value="768x768">768 × 768</option>
      </select>
    </div>
    <div>
      <label for="steps">Steps</label>
      <select id="steps">
        <option value="4" selected>4 steps (~8s - Best)</option>
        <option value="3">3 steps</option>
        <option value="2">2 steps</option>
      </select>
    </div>
  </div>

  <button id="gen-btn" onclick="generate()">Generate Thumbnail</button>
  <div id="status"></div>
  <div id="preview"></div>
</div>

<script>
async function fileToB64(file) {
  if (!file) return null;
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function generate() {
  const btn = document.getElementById('gen-btn');
  const status = document.getElementById('status');
  const preview = document.getElementById('preview');
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return alert('Please enter a prompt');

  btn.disabled = true;
  status.textContent = 'Generating scene with FLUX and swapping faces on GPU...';
  preview.innerHTML = '';

  const f1 = document.getElementById('face_left').files[0];
  const f2 = document.getElementById('face_right').files[0];

  const payload = {
    prompt: prompt,
    size: document.getElementById('size').value,
    steps: parseInt(document.getElementById('steps').value),
    face_left: await fileToB64(f1),
    face_right: await fileToB64(f2)
  };

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.image_b64) {
      status.textContent = 'Completed in ' + (data.elapsed_seconds || '') + 's!';
      preview.innerHTML = '<img src="data:image/png;base64,' + data.image_b64 + '" /><br><a class="download-btn" href="data:image/png;base64,' + data.image_b64 + '" download="thumbnail.png">Download Thumbnail</a>';
    } else {
      status.textContent = 'Error: ' + (data.error || 'Unknown error');
    }
  } catch (e) {
    status.textContent = 'Request failed: ' + e;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>"""

class ServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _check_auth(self):
        user = os.environ.get("CLUSTER_AUTH_USER")
        passwd = os.environ.get("CLUSTER_AUTH_PASS")
        api_key = os.environ.get("CLUSTER_API_KEY")
        if not user and not passwd and not api_key:
            return True
        auth_header = self.headers.get("Authorization", "")
        if api_key and auth_header.startswith("Bearer ") and auth_header[7:].strip() == api_key:
            return True
        if user and passwd and auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:].strip()).decode("utf-8")
                u, p = decoded.split(":", 1)
                if u == user and p == passwd:
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="cluster"')
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error": "Unauthorized"}\n')
        return False

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

    def do_GET(self):
        if not self._check_auth(): return
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/health":
            self._send_json(200, {
                "status": "ok",
                "model": "__MODEL_ALIAS__",
                "architecture": "flux-face",
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
        elif path in ("/last_image", "/v1/images/last"):
            found = False
            for p in ("/kaggle/working/last_image.png", os.path.join(WORK, "last_image.png")):
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(data)
                    found = True
                    break
            if not found:
                self._send_json(404, {"error": "No image generated yet"})
        else:
            self._send_json(404, {"error": f"Path {path} not found"})

    def do_POST(self):
        if not self._check_auth(): return
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

            size = req.get("size", "1024x576")
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

            face_left = req.get("face_left")
            face_right = req.get("face_right")
            face_images = req.get("face_images")

            t0 = time.time()
            print(f"[GEN] FLUX Prompt: '{prompt[:80]}...' size={width}x{height} steps={steps}", flush=True)

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

                # Perform face swap if requested
                if face_left or face_right or face_images:
                    print("[FACE] Running InsightFace face replacement...", flush=True)
                    t_swap = time.time()
                    img = perform_face_swaps(img, face_left=face_left, face_right=face_right, face_images=face_images)
                    print(f"[FACE] Swap finished in {round(time.time() - t_swap, 2)}s", flush=True)

                try:
                    img.save("/kaggle/working/last_image.png")
                    img.save(os.path.join(WORK, "last_image.png"))
                except Exception as e:
                    print("save image warning:", e, flush=True)

            elapsed = round(time.time() - t0, 2)
            print(f"[GEN] Pipeline complete in {elapsed}s", flush=True)

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
print("FLUX Face Swapper server running on port", PORT, flush=True)

# ---- publish both tunnels ----
CF_TOKEN = "__CLOUDFLARE_TUNNEL_TOKEN__" or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
STATIC_URL = "__STATIC_URL__" or os.environ.get("CLUSTER_URL", "")

def start_public_tunnel(url, log_path):
    fh = open(log_path, "w")
    if CF_TOKEN:
        print("Starting named Cloudflare tunnel with token...", flush=True)
        proc = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "run", "--token", CF_TOKEN, "--url", url],
                                stdout=fh, stderr=subprocess.STDOUT)
        time.sleep(5)
        if proc.poll() is not None:
            fh.flush()
            print("cloudflared failed to start! Log tail:", flush=True)
            print(open(log_path).read()[-2000:], flush=True)
            raise RuntimeError("named cloudflared tunnel exited immediately")
        return STATIC_URL or "https://ai.883742.xyz"
    else:
        subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", url],
                         stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(2)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
            if m:
                return m.group(0)
        return None

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

public_url = start_public_tunnel(f"http://127.0.0.1:{PORT}", CF_LOG)
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

FLUX_FACE_CELLS = [CELL_1_ENV, CELL_2_FLUX_FACE_DEPS, CELL_3_FLUX_FETCH, CELL_4_FLUX_FACE_SERVE, CELL_5_KEEPALIVE]


# =============================================================================
# Nemotron-3.5-Parse (OCR & Document Parsing VLM)
# =============================================================================

CELL_2_PARSE_DEPS = r'''
# --- Cell 2: install dependencies for Nemotron-3.5-Parse inference ---------
import subprocess, sys

pkgs = [
    "transformers>=4.48.0",
    "accelerate>=0.33.0",
    "torchvision",
    "timm>=1.0.0",
    "einops",
    "pillow",
]
print("installing:", " ".join(pkgs), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
print("Nemotron Parse packages installed successfully.", flush=True)
'''

CELL_3_PARSE_SERVE = r'''
# --- Cell 3: load Nemotron-3.5-Parse model + start server + open tunnels ---
import os, re, subprocess, time, urllib.request, json, threading, http.server
import base64, io, gc
import torch
import PIL.Image
from transformers import AutoProcessor, AutoModelForCausalLM

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
MODEL_REPO = "__MODEL_REPO__"
MODEL_ALIAS = "__MODEL_ALIAS__"
DEFAULT_MAX_TOKENS = int("__DEFAULT_MAX_TOKENS__" or "4096")

STOP_REQUESTED = threading.Event()
INFER_LOCK = threading.Lock()

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

# ---- load model ----
print(f"Loading {MODEL_REPO} onto GPU...", flush=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

processor = AutoProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True
).eval()
print(f"Model {MODEL_REPO} loaded successfully onto {device}!", flush=True)

@torch.inference_mode()
def parse_image(img: PIL.Image.Image, prompt_str: str = "", max_tokens: int = DEFAULT_MAX_TOKENS):
    if not prompt_str:
        prompt_str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
    inputs = processor(text=prompt_str, images=img.convert("RGB"), return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    raw_result = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    
    # Strip prompt prefix if echoed back
    res = raw_result
    if prompt_str in res:
        res = res.split(prompt_str, 1)[-1]
    
    # Clean text: remove bounding box tokens like <loc_123> or <bbox_...>
    clean_text = re.sub(r'<loc_\d+>', '', res)
    clean_text = re.sub(r'<box_[^>]+>', '', clean_text)
    clean_text = re.sub(r'</?(?:predict_[^>]+|class_[^>]+|s)>', '', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
    return clean_text, raw_result

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
                             "model": MODEL_ALIAS})
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

# ---- Nemotron Web & API Server on PORT (port 8080) ----
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nemotron-3.5-Parse — Document OCR & Layout Intelligence</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  body { background: #0b0f19; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 24px 16px; }
  .card { background: #151c2c; border: 1px solid #2d3748; border-radius: 14px; width: 100%; max-width: 900px; padding: 28px; box-shadow: 0 12px 32px rgba(0,0,0,0.5); margin-bottom: 24px; }
  h1 { font-size: 1.7rem; color: #fff; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }
  .badge { font-size: 0.75rem; background: #10b981; color: #fff; padding: 3px 10px; border-radius: 6px; font-weight: 700; }
  p.subtitle { color: #94a3b8; font-size: 0.92rem; margin-bottom: 20px; line-height: 1.4; }
  .drop-zone { border: 2px dashed #3b82f6; border-radius: 10px; padding: 32px 16px; text-align: center; cursor: pointer; transition: background 0.2s; background: rgba(59, 130, 246, 0.05); margin-bottom: 18px; }
  .drop-zone:hover { background: rgba(59, 130, 246, 0.12); }
  .drop-zone input { display: none; }
  .drop-zone p { color: #93c5fd; font-size: 0.95rem; font-weight: 600; }
  .drop-zone span { font-size: 0.8rem; color: #64748b; margin-top: 4px; display: block; }
  #thumbPreview { max-height: 220px; max-width: 100%; border-radius: 8px; margin: 12px auto; display: none; box-shadow: 0 4px 14px rgba(0,0,0,0.4); }
  .options { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 18px; }
  label { display: block; font-size: 0.82rem; font-weight: 600; color: #cbd5e1; margin-bottom: 6px; }
  select, input { width: 100%; background: #0b0f19; border: 1px solid #334155; border-radius: 8px; color: #fff; padding: 10px; font-size: 0.9rem; }
  button { width: 100%; background: #2563eb; color: #fff; border: none; padding: 14px 20px; font-size: 1.05rem; font-weight: 700; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #475569; cursor: not-allowed; }
  #status { margin-top: 14px; font-size: 0.92rem; color: #60a5fa; text-align: center; }
  .output-box { margin-top: 20px; background: #0b0f19; border: 1px solid #334155; border-radius: 10px; padding: 18px; }
  .output-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #1e293b; padding-bottom: 8px; }
  .output-title { font-size: 0.95rem; font-weight: 700; color: #f1f5f9; }
  .copy-btn { width: auto; background: #334155; padding: 6px 14px; font-size: 0.82rem; border-radius: 6px; }
  .copy-btn:hover { background: #475569; }
  pre { white-space: pre-wrap; word-break: break-word; color: #cbd5e1; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.88rem; max-height: 480px; overflow-y: auto; line-height: 1.5; }
</style>
</head>
<body>
<div class="card">
  <h1>Nemotron-3.5-Parse <span class="badge">NVIDIA OCR</span></h1>
  <p class="subtitle">Sub-1B Vision-Language Document Intelligence — parses complex documents, multi-column PDFs, tables, receipts, and charts into structured Markdown.</p>

  <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
    <input type="file" id="fileInput" accept="image/*" onchange="handleFile(event)">
    <p>Drop document image here or click to browse</p>
    <span>Supports PNG, JPG, WebP (up to 1664x2048)</span>
    <img id="thumbPreview" alt="Preview">
  </div>

  <div class="options">
    <div>
      <label for="modeSelect">Extraction Target</label>
      <select id="modeSelect">
        <option value="markdown">Clean Markdown (Tables, Lists, Headings)</option>
        <option value="full">Full Layout (Markdown + Bounding Boxes + Classes)</option>
        <option value="chart">Charts & Tables Focused</option>
      </select>
    </div>
    <div>
      <label for="maxTokens">Max Output Tokens</label>
      <input type="number" id="maxTokens" value="4096" min="256" max="8192" step="256">
    </div>
  </div>

  <button id="parseBtn" onclick="parseDoc()">Parse Document with Nemotron</button>
  <div id="status"></div>

  <div class="output-box" id="outputBox" style="display:none;">
    <div class="output-header">
      <span class="output-title">Parsed Output</span>
      <button class="copy-btn" onclick="copyResult()">Copy Markdown</button>
    </div>
    <pre id="resultText"></pre>
  </div>
</div>

<script>
let currentImageB64 = '';

function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    currentImageB64 = evt.target.result;
    const img = document.getElementById('thumbPreview');
    img.src = currentImageB64;
    img.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

// Drag & drop handlers
const dropZone = document.querySelector('.drop-zone');
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.background = 'rgba(59, 130, 246, 0.2)'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.background = 'rgba(59, 130, 246, 0.05)'; });
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.style.background = 'rgba(59, 130, 246, 0.05)';
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    document.getElementById('fileInput').files = e.dataTransfer.files;
    handleFile({ target: { files: e.dataTransfer.files } });
  }
});

async function parseDoc() {
  if (!currentImageB64) return alert('Please upload or drag an image first');
  const btn = document.getElementById('parseBtn');
  const status = document.getElementById('status');
  const outBox = document.getElementById('outputBox');
  const resPre = document.getElementById('resultText');
  const mode = document.getElementById('modeSelect').value;
  const maxTokens = parseInt(document.getElementById('maxTokens').value) || 4096;

  btn.disabled = true;
  outBox.style.display = 'none';
  status.textContent = 'Parsing document on Kaggle GPU with Nemotron-3.5-Parse...';
  const t0 = Date.now();

  try {
    const resp = await fetch('/v1/parse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: currentImageB64, mode: mode, max_new_tokens: maxTokens })
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${err}`);
    }
    const data = await resp.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
    status.textContent = `Completed in ${elapsed}s (${data.elapsed_seconds}s model inference)`;
    resPre.textContent = data.text || data.raw || 'No text extracted';
    outBox.style.display = 'block';
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

function copyResult() {
  const text = document.getElementById('resultText').textContent;
  navigator.clipboard.writeText(text).then(() => alert('Copied to clipboard!'));
}
</script>
</body>
</html>
"""

AUTH_USER = "__AUTH_USER__" or os.environ.get("CLUSTER_AUTH_USER", "")
AUTH_PASS = "__AUTH_PASS__" or os.environ.get("CLUSTER_AUTH_PASS", "")
AUTH_KEY = "__AUTH_KEY__" or os.environ.get("CLUSTER_API_KEY", "")
AUTH_ENABLED = bool((AUTH_USER and AUTH_PASS) or AUTH_KEY)

class ServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _check_auth(self):
        if not AUTH_ENABLED:
            return True
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health", "/healthz"):
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                creds = base64.b64decode(auth[6:].strip()).decode("utf-8")
                u, p = creds.split(":", 1)
                if u == AUTH_USER and p == AUTH_PASS:
                    return True
            except Exception:
                pass
        elif auth.startswith("Bearer "):
            token = auth[7:].strip()
            if (AUTH_KEY and token == AUTH_KEY) or (AUTH_PASS and token == AUTH_PASS):
                return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="AI Cluster Protected"')
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b'{"error": "Unauthorized"}\n')
        return False

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
        if not self._check_auth(): return
        self.do_GET()

    def do_GET(self):
        if not self._check_auth(): return
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
                "model": MODEL_ALIAS,
                "architecture": "nemotron-parse-3.5",
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            })
        elif path in ("/models", "/v1/models"):
            self._send_json(200, {
                "data": [{
                    "id": MODEL_ALIAS,
                    "object": "model",
                    "type": "document-parsing-ocr"
                }]
            })
        else:
            self._send_json(404, {"error": f"Path {path} not found"})

    def do_POST(self):
        if not self._check_auth(): return
        path = self.path.split("?")[0].rstrip("/")
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            req = json.loads(post_data.decode("utf-8"))
        except Exception:
            return self._send_json(400, {"error": "Invalid JSON payload"})

        if path in ("/v1/parse", "/parse"):
            raw_img = req.get("image", "")
            if not raw_img:
                return self._send_json(400, {"error": "Missing image"})

            # Load image from base64 or URL
            try:
                if raw_img.startswith("data:image"):
                    raw_b64 = raw_img.split(",", 1)[-1]
                    img = PIL.Image.open(io.BytesIO(base64.b64decode(raw_b64)))
                elif raw_img.startswith(("http://", "https://")):
                    with urllib.request.urlopen(raw_img, timeout=30) as r:
                        img = PIL.Image.open(io.BytesIO(r.read()))
                else:
                    img = PIL.Image.open(io.BytesIO(base64.b64decode(raw_img)))
            except Exception as e:
                return self._send_json(400, {"error": f"Failed to load image: {e}"})

            mode = req.get("mode", "markdown")
            if mode == "markdown":
                prompt_str = "</s><s><output_markdown>"
            elif mode == "chart":
                prompt_str = "</s><s><predict_classes><output_markdown><predict_text_in_pic>"
            else:
                prompt_str = req.get("prompt") or "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"

            max_tokens = int(req.get("max_new_tokens", DEFAULT_MAX_TOKENS))

            t0 = time.time()
            with INFER_LOCK:
                clean_text, raw_text = parse_image(img, prompt_str=prompt_str, max_tokens=max_tokens)
            elapsed = round(time.time() - t0, 3)

            return self._send_json(200, {
                "status": "ok",
                "text": clean_text,
                "raw": raw_text,
                "elapsed_seconds": elapsed
            })

        elif path in ("/v1/chat/completions", "/chat/completions"):
            messages = req.get("messages", [])
            # Search for image in message content
            img = None
            prompt_str = ""
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            url_val = item.get("image_url", {}).get("url", "")
                            if url_val.startswith("data:image"):
                                b64_data = url_val.split(",", 1)[-1]
                                img = PIL.Image.open(io.BytesIO(base64.b64decode(b64_data)))
                            elif url_val.startswith(("http://", "https://")):
                                with urllib.request.urlopen(url_val, timeout=30) as r:
                                    img = PIL.Image.open(io.BytesIO(r.read()))
                        elif isinstance(item, dict) and item.get("type") == "text":
                            prompt_str += item.get("text", "") + " "
                elif isinstance(content, str):
                    prompt_str += content + " "

            if img is not None:
                t0 = time.time()
                with INFER_LOCK:
                    clean_text, _ = parse_image(img)
                reply = clean_text
            else:
                if "CLUSTER ONLINE" in prompt_str.upper():
                    reply = "CLUSTER ONLINE - Nemotron-3.5-Parse document OCR server is online."
                else:
                    reply = "Nemotron-3.5-Parse OCR server is online. Please attach an image to parse document layout, tables, and text."

            return self._send_json(200, {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ALIAS,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": len(reply.split()),
                    "total_tokens": 10 + len(reply.split())
                }
            })
        else:
            self._send_json(404, {"error": f"POST {path} not supported"})

server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), ServerHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print("Nemotron Parse server running on port", PORT, flush=True)

# ---- publish both tunnels ----
CF_TOKEN = "__CLOUDFLARE_TUNNEL_TOKEN__" or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
STATIC_URL = "__STATIC_URL__" or os.environ.get("CLUSTER_URL", "")

def start_public_tunnel(url, log_path):
    fh = open(log_path, "w")
    if CF_TOKEN:
        print("Starting named Cloudflare tunnel with token...", flush=True)
        proc = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "run", "--token", CF_TOKEN, "--url", url],
                                stdout=fh, stderr=subprocess.STDOUT)
        time.sleep(5)
        if proc.poll() is not None:
            fh.flush()
            print("cloudflared failed to start! Log tail:", flush=True)
            print(open(log_path).read()[-2000:], flush=True)
            raise RuntimeError("named cloudflared tunnel exited immediately")
        return STATIC_URL or "https://ai.883742.xyz"
    else:
        subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", url],
                         stdout=fh, stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(2)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path).read())
            if m:
                return m.group(0)
        return None

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

public_url = start_public_tunnel(f"http://127.0.0.1:{PORT}", CF_LOG)
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
print("MODEL=" + MODEL_REPO)
print("=" * 72)
print(f"Web UI         : {public_url}")
print(f"API endpoint   : {public_url}/v1/parse")
print(f"Chat endpoint  : {public_url}/v1/chat/completions")
print(f"Health         : {public_url}/health")
print("Stop from PC   : ctr.sh stop   (uses the control tunnel above)")
'''

PARSE_CELLS = [CELL_1_ENV, CELL_2_PARSE_DEPS, CELL_3_PARSE_SERVE, CELL_5_KEEPALIVE]
