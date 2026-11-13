#!/usr/bin/env python3
"""bench — measure real prompt/decode throughput of a remote GPU server.

Answers "why is my tokens/sec slow?" with numbers instead of guesses:

  * asks the server for its own timings (`prompt_per_second`, `predicted_per_second`)
  * separates PREFILL (context processing) from DECODE (token generation)
  * re-runs the same prompt to prove prefix-cache behaviour
  * optionally measures a long-context prefill, which is what agent sessions feel
  * reads the server log tail for offloaded-layer / per-GPU buffer / CUDA lines

Usage
-----
    python3 lib/bench.py                       # uses <project>/url.txt via $CLUSTER_PROJECT
    python3 lib/bench.py --url https://x.trycloudflare.com
    python3 lib/bench.py --long 16000          # 16k-token prefill test
    python3 lib/bench.py --decode 256          # generate more tokens

Stdlib only, like the rest of the clients.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Load .env (auth keys) exactly like cluster.py does.
env_path = os.path.join(ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k not in os.environ and v:
                os.environ[k] = v

PROJECT = os.environ.get("CLUSTER_PROJECT", "")
TIMEOUT = int(os.environ.get("CLUSTER_TIMEOUT") or "1800")

C = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
}


def color(name: str, text: str) -> str:
    return f"{C.get(name, '')}{text}{C['reset']}"


def resolve_url(explicit: str | None) -> str:
    url = explicit or os.environ.get("CLUSTER_URL") or ""
    if not url:
        path = os.path.join(PROJECT, "url.txt") if PROJECT else os.path.join(ROOT, "url.txt")
        if os.path.exists(path):
            url = open(path).read().strip()
    if not url:
        sys.exit(
            "No server URL.\n"
            "  - run './ctr.sh url' first, or pass --url https://xxxx.trycloudflare.com"
        )
    return url.rstrip("/")


def auth_headers() -> dict[str, str]:
    headers = {"User-Agent": "cluster-bench/1.0"}
    user, passwd = os.environ.get("CLUSTER_AUTH_USER"), os.environ.get("CLUSTER_AUTH_PASS")
    api_key = os.environ.get("CLUSTER_API_KEY")
    if user and passwd:
        import base64
        headers["Authorization"] = "Basic " + base64.b64encode(f"{user}:{passwd}".encode()).decode()
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_json(url: str, payload: dict, timeout: int = TIMEOUT) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **auth_headers()}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers=auth_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def get_json(url: str, timeout: int = 60) -> dict:
    return json.loads(get_text(url, timeout))


def tokenize(url: str, text: str) -> int:
    try:
        obj = post_json(url + "/tokenize", {"content": text, "add_special": False}, timeout=120)
        return len(obj.get("tokens", []))
    except Exception:
        return max(1, len(text.split()))


FILLER = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. "
    "A wizard's job is to vex chumps quickly in fog. Bright vixens jump; dozy fowl quack. "
)


def make_prompt(url: str, target_tokens: int) -> str:
    """Grow the filler until it tokenizes to roughly `target_tokens`."""
    if target_tokens <= 0:
        return "Reply with exactly: CLUSTER BENCH OK"
    per_repeat = max(1, tokenize(url, FILLER))
    repeats = max(1, target_tokens // per_repeat)
    text = FILLER * repeats
    return (
        f"Here is reference material:\n{text}\n\n"
        "Using only the material above, reply with exactly: CLUSTER BENCH OK"
    )


def chat(url: str, prompt: str, max_tokens: int, cache_prompt: bool) -> tuple[dict, float]:
    payload = {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "cache_prompt": cache_prompt,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    obj = post_json(url + "/v1/chat/completions", payload)
    return obj, time.monotonic() - t0


def fmt_timings(label: str, obj: dict, wall: float) -> dict:
    t = obj.get("timings") or {}
    u = obj.get("usage") or {}
    prompt_n = t.get("prompt_n", u.get("prompt_tokens", 0))
    pred_n = t.get("predicted_n", u.get("completion_tokens", 0))
    pps = t.get("prompt_per_second", 0.0)
    dps = t.get("predicted_per_second", 0.0)
    print(f"\n{color('bold', label)}")
    print(f"  prompt tokens     : {prompt_n:>6}   {pps:>8.1f} tok/s  "
          f"({t.get('prompt_ms', 0) / 1000:.1f}s)")
    if t.get("cache_n"):
        print(f"  from prefix cache : {t['cache_n']:>6} tokens")
    print(f"  generated tokens  : {pred_n:>6}   {dps:>8.1f} tok/s  "
          f"({t.get('predicted_ms', 0) / 1000:.1f}s)")
    print(f"  wall clock        : {wall:>6.1f}s   (includes tunnel + proxy)")
    if "draft_n" in t or "draft_n_accepted" in t:
        dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        acc = (da / dn * 100) if dn else 0.0
        print(f"  speculative draft : {da}/{dn} accepted ({acc:.0f}%)")
    text = (obj.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    print(f"  reply             : {text.strip()[:120]!r}")
    return {"prompt_n": prompt_n, "prompt_per_second": pps,
            "predicted_n": pred_n, "predicted_per_second": dps, "wall": wall}


def log_diagnostics(url: str) -> None:
    """Best-effort read of the llama-server log via the auth proxy /log route."""
    try:
        log = get_text(url + "/log", timeout=30)
    except urllib.error.HTTPError as e:
        print(f"  {color('dim', f'server log not exposed (HTTP {e.code}); enable CLUSTER_AUTH_* to use /log')}")
        return
    except Exception as e:
        print(f"  {color('dim', f'server log unavailable: {e}')}")
        return

    def hits(pattern: str, limit: int = 6) -> list[str]:
        seen = [ln.strip() for ln in log.splitlines() if re.search(pattern, ln, re.I)]
        return seen[-limit:]

    checks = [
        ("CUDA device / layer offload", r"offloaded \d+/\d+ layers|offloading .* layers|load_tensors"),
        ("per-GPU model buffers", r"CUDA[0-9] model buffer size|model buffer size"),
        ("KV / compute buffers", r"(KV|compute) buffer size"),
        ("flash attention", r"flash_attn|flash attention|attn_type"),
        ("split mode", r"split_mode|tensor_split"),
        ("errors", r"failed to allocate|out of memory|CUDA error"),
    ]
    for title, pattern in checks:
        found = hits(pattern)
        if found:
            print(f"  {color('cyan', title)}:")
            for line in found:
                print(f"    {line[:180]}")
        else:
            print(f"  {color('cyan', title)}: {color('dim', 'no matching log line')}")


def verdict(results: dict, long_result: dict | None) -> None:
    print(f"\n{color('bold', 'Interpreting the numbers')}")
    dps = results["predicted_per_second"]
    pps = results["prompt_per_second"]

    if dps >= 60:
        print(f"  decode {dps:.0f} tok/s — {color('green', 'excellent')} for a T4 pair (small MoE territory).")
    elif dps >= 25:
        print(f"  decode {dps:.0f} tok/s — {color('green', 'expected')} for a ~3B-active MoE on dual T4.")
    elif dps >= 10:
        print(f"  decode {dps:.0f} tok/s — {color('yellow', 'dense-model territory')} "
              "(a 24–32B dense Q4 model tops out here: ~17 tok/s is the T4 memory-bandwidth ceiling).")
    elif dps >= 4:
        print(f"  decode {dps:.0f} tok/s — {color('red', 'low')} for anything. Check the log lines above for "
              "CPU-offloaded layers, and `nvidia-smi` for clock throttling.")
    else:
        print(f"  decode {dps:.0f} tok/s — {color('red', 'very low')}: most layers are almost certainly "
              "running on CPU (model did not fit in VRAM) or the tunnel is saturated.")

    if pps and pps < 120:
        print(f"  prefill {pps:.0f} tok/s — {color('red', 'slow')}; raise `-ub` (512 → 1024/2048) "
              "or check PCIe/CPU offload.")
    elif pps:
        print(f"  prefill {pps:.0f} tok/s — {color('green', 'healthy')} for Q4 on dual T4.")

    if long_result:
        prefill_s = long_result["prompt_n"] / max(1.0, long_result["prompt_per_second"])
        print(f"  a {long_result['prompt_n']}-token prompt costs ~{prefill_s:.0f}s of prefill. "
              "Agent sessions pay this whenever the prompt prefix changes; keep system "
              "prompts / tool lists stable so `cache_prompt` can hit.")
    print("  Rule of thumb on 2×T4 (320 GB/s each, PCIe only): decode time ≈ bytes of weights "
          "read per token ÷ one GPU's bandwidth, because layer-split runs the GPUs in series.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="override the tunnel URL")
    ap.add_argument("--decode", type=int, default=128, help="tokens to generate (default 128)")
    ap.add_argument("--context", type=int, default=1500, help="short-prompt size in tokens (default 1500)")
    ap.add_argument("--long", type=int, default=0,
                    help="also test a long prefill of N tokens (e.g. 16000)")
    ap.add_argument("--skip-log", action="store_true", help="do not fetch the server log")
    a = ap.parse_args()

    url = resolve_url(a.url)
    print(f"bench: {url}")

    try:
        props = get_json(url + "/props")
        print(f"  model   : {props.get('model_path', '?')}")
        print(f"  alias   : {props.get('model_alias', '?')}")
        print(f"  n_ctx   : {props.get('n_ctx', '?')}   slots: {props.get('total_slots', '?')}")
        print(f"  build   : {(props.get('build_info') or '?')}")
        dgs = props.get("default_generation_settings") or {}
        if dgs.get("n_ctx"):
            print(f"  slot ctx: {dgs.get('n_ctx')}")
    except Exception as e:
        print(f"  {color('yellow', f'/props unavailable: {e}')}")

    if not a.skip_log:
        print(f"\n{color('bold', 'Server-side log checks')}")
        log_diagnostics(url)

    prompt = make_prompt(url, a.context)
    print(f"\n{color('bold', 'Cold prefill + decode')}")
    obj, wall = chat(url, prompt, a.decode, cache_prompt=False)
    cold = fmt_timings("cold request (cache_prompt=false)", obj, wall)

    print(f"\n{color('bold', 'Warm request (same prompt)')}")
    obj, wall = chat(url, prompt, 32, cache_prompt=True)
    warm = fmt_timings("warm request (cache_prompt=true)", obj, wall)
    if warm["prompt_per_second"] > cold["prompt_per_second"] * 1.5:
        print(f"  {color('green', 'prefix cache is working')} — repeated turns are cheap.")
    else:
        print(f"  {color('yellow', 'prefix cache did not help')} — the prompt prefix changed "
              "or the cache was erased.")

    long_result = None
    if a.long > 0:
        print(f"\n{color('bold', f'Long prefill test ({a.long} tokens)')}")
        lp = make_prompt(url, a.long)
        obj, wall = chat(url, lp, 32, cache_prompt=False)
        long_result = fmt_timings("long context", obj, wall)

    verdict(cold, long_result)
    print()


if __name__ == "__main__":
    main()
