"""Edge0 latent-MoE notebook cells (Strata engine, not llama.cpp)."""

CELL_EDGE0_BUILD = r'''
# --- Cell 2: build Edge0 (strata + edge0-infer) ----------------------------
import os, subprocess, shutil, sys, glob, time

SRC = "/kaggle/temp/Edge0"
STRATA = os.path.join(SRC, "edge0-strata")
INFER = os.path.join(SRC, "edge0-infer")
LLAMA_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"

def run(cmd, cwd=None, env=None):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=True,
                          capture_output=False)

print("--- toolchain ---", flush=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout)
nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
print("nvcc:", nvcc, subprocess.run([nvcc, "--version"], capture_output=True,
                                    text=True).stdout.strip().splitlines()[-1:])
run(["pip", "install", "-q", "cmake", "ninja", "uv"])

if not os.path.isdir(SRC):
    run(["git", "clone", "--depth", "1",
         "https://github.com/Edge0-AI/Edge0.git", SRC])

LLAMA_DIR = os.path.join(STRATA, "third_party", "llama.cpp")
if not os.path.exists(os.path.join(LLAMA_DIR, "CMakeLists.txt")):
    os.makedirs(LLAMA_DIR, exist_ok=True)
    run(["bash", "-lc",
         "curl -L https://github.com/ggml-org/llama.cpp/archive/%s.tar.gz | "
         "tar -xz --strip-components=1 -C %s" % (LLAMA_COMMIT, LLAMA_DIR)])

env = dict(os.environ)
env["PATH"] = os.path.dirname(nvcc) + ":" + env["PATH"]
run(["cmake", "-S", STRATA, "-B", os.path.join(STRATA, "build"), "-G", "Ninja",
     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_COMPILER=" + nvcc,
     "-DCMAKE_CUDA_ARCHITECTURES=75", "-DSTRATA_ENABLE_CUDA=ON",
     "-DSTRATA_GGML_DIR=" + LLAMA_DIR], env=env)
run(["cmake", "--build", os.path.join(STRATA, "build"), "--target", "strata"], env=env)

run(["cmake", "-S", INFER, "-B", os.path.join(INFER, "build", "cpu"),
     "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"], env=env)
run(["cmake", "--build", os.path.join(INFER, "build", "cpu"), "-j", "4"], env=env)

venv = os.path.join(STRATA, ".venv")
run(["uv", "venv", "--python", sys.executable, venv])
python = os.path.join(venv, "bin", "python")
run(["uv", "pip", "install", "--python", python,
     "-r", os.path.join(STRATA, "requirements.txt")])
print("build done", flush=True)
'''

CELL_EDGE0_SERVE = r'''
# --- Cell 3: download the pack, serve, publish tunnels ----------------------
import os, re, json, time, glob, subprocess, threading, http.server, socket, urllib.request

PORT = int("__PORT__")
CTL_PORT = int("__CTL_PORT__")
CTX = int("__CTX__")
MODEL_REPO = "__MODEL_REPO__"
MODEL_ALIAS = "__MODEL_ALIAS__"
STOP_REQUESTED = threading.Event()
WORK = "/kaggle/temp/run"
PACK = "/kaggle/temp/edge0-pack"
os.makedirs(WORK, exist_ok=True)

from huggingface_hub import snapshot_download
pack = snapshot_download(MODEL_REPO, local_dir=PACK, max_workers=8)
print("pack:", pack, flush=True)

CF = "/kaggle/temp/cloudflared"
if not os.path.exists(CF):
    subprocess.run(["curl", "-4", "-fSL", "--retry", "3", "-o", CF,
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"],
                   check=True)
    os.chmod(CF, 0o755)

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

STOP_TOKEN = os.environ.get("CLUSTER_STOP_TOKEN") or os.urandom(8).hex()
class Ctl(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, {"ok": True, "stopping": STOP_REQUESTED.is_set()})
        elif self.path.startswith("/stop/" + STOP_TOKEN):
            STOP_REQUESTED.set(); self._send(200, {"ok": True})
        else:
            self._send(404, {"ok": False})
    def do_POST(self): self.do_GET()
ctl = http.server.ThreadingHTTPServer(("127.0.0.1", CTL_PORT), Ctl)
threading.Thread(target=ctl.serve_forever, daemon=True).start()

SRC = "/kaggle/temp/Edge0"
BIN = glob.glob(os.path.join(SRC, "edge0-infer", "build", "cpu", "bin", "edge0-infer"))[0]
LOG = os.path.join(WORK, "edge0.log")
SPLIT = os.environ.get("EDGE0_SPLIT", "1") == "1"
base = [BIN, "serve", "--model", pack, "--context", str(CTX)]
variants = []
if SPLIT:
    variants.append(base + ["--device", "cuda",
                            "--strata-arg", "--layer-split", "--strata-arg", "auto",
                            "--strata-arg", "--vram-reserve-mib", "--strata-arg", "700"])
variants.append(base + ["--device", "cuda:0"])
ready = False
server = None
for attempt, cmd in enumerate(variants, 1):
    print(f"launching (attempt {attempt}):", " ".join(cmd), flush=True)
    fh = open(LOG, "w")
    server = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    for i in range(240):
        time.sleep(5)
        if server.poll() is not None:
            print(open(LOG).read()[-3000:], flush=True)
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=4) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        if i % 12 == 0:
            print(f"[{i*5}s] loading...", flush=True)
    if ready:
        break
    print(f"attempt {attempt} failed; trying the next launch variant", flush=True)
if not ready:
    print(open(LOG).read()[-4000:], flush=True)
    raise RuntimeError("edge0 did not become healthy")
print("server is UP on", PORT, flush=True)

public_url = start_tunnel(f"http://127.0.0.1:{PORT}", os.path.join(WORK, "cf.log"))
control_url = start_tunnel(f"http://127.0.0.1:{CTL_PORT}", os.path.join(WORK, "cf-ctl.log"))
if not public_url or not control_url:
    raise RuntimeError("could not obtain tunnel URLs")
open("/kaggle/working/CONTROL_URL.txt", "w").write(public_url + "\n")
open("/kaggle/working/STOP_URL.txt", "w").write(control_url + "/stop/" + STOP_TOKEN + "\n")
print("=" * 72)
print("CONTROL_URL=" + public_url)
print("STOP_URL=" + control_url + "/stop/" + STOP_TOKEN)
print("MODEL=" + MODEL_ALIAS)
print("=" * 72)
'''
