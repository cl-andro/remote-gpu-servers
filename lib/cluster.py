#!/usr/bin/env python3
"""cluster — control a Kaggle-hosted inference server from your PC.

The model runs on Kaggle. This script only sends HTTP requests to the public
tunnel URL, so nothing large is ever downloaded locally.

Typical flow
------------
    ./ctr.sh push          # push the kernel to Kaggle and start it
    ./ctr.sh url           # read the public tunnel URL out of the run logs
    python3 lib/cluster.py health
    python3 lib/cluster.py chat "Explain gated delta networks in two sentences."
    python3 lib/cluster.py chat --no-think "2+2?"
    python3 lib/cluster.py stream "Write a haiku about GPUs."
    python3 lib/cluster.py tools "What's the weather in Paris and Tokyo?"

URL resolution order: --url flag > $CLUSTER_URL > url.txt in the project dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

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

TIMEOUT = int(os.environ.get("CLUSTER_TIMEOUT") or os.environ.get("SALUKI_TIMEOUT", "600"))
DEFAULT_MODEL = os.environ.get("CLUSTER_MODEL", "model")
# `--project <dir>` points at the project folder that owns url.txt.
PROJECT = os.environ.get("CLUSTER_PROJECT", "")


def url_file() -> str:
    if PROJECT:
        return os.path.join(PROJECT, "url.txt")
    return os.path.join(ROOT, "url.txt")

RESET = "\033[0m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def resolve_url(explicit: str | None) -> str:
    url = explicit or os.environ.get("CLUSTER_URL") or os.environ.get("SALUKI_URL") or ""
    path = url_file()
    if not url and os.path.exists(path):
        url = open(path).read().strip()
    if not url:
        sys.exit(
            "No server URL.\n"
            "  - run './ctr.sh url' to fetch it from the Kaggle logs, or\n"
            "  - pass --url https://xxxx.trycloudflare.com, or\n"
            "  - export CLUSTER_URL=..."
        )
    return url.rstrip("/")


def auth_headers() -> dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) cluster/1.0"}
    user = os.environ.get("CLUSTER_AUTH_USER")
    passwd = os.environ.get("CLUSTER_AUTH_PASS")
    api_key = os.environ.get("CLUSTER_API_KEY")
    if user and passwd:
        import base64
        token = base64.b64encode(f"{user}:{passwd}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def authed_request(url: str, data: bytes | None = None, headers: dict | None = None, method: str | None = None) -> urllib.request.Request:
    h = dict(headers or {})
    h.update(auth_headers())
    return urllib.request.Request(url, data=data, headers=h, method=method)


def post_stream(url: str, payload: dict, show_reasoning: bool = True) -> None:
    """Streaming SSE request to an OpenAI-compatible endpoint."""
    data = json.dumps(payload).encode()
    req = authed_request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    thinking = False
    saw_reason = False
    out_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reason = delta.get("reasoning_content")
            content = delta.get("content")
            if reason and show_reasoning:
                if not thinking:
                    # dim once at the start; a reset per token would fragment words
                    sys.stdout.write(f"{DIM}[thinking] ")
                    thinking = True
                saw_reason = True
                sys.stdout.write(reason)
            if content:
                if thinking:
                    sys.stdout.write(f"{RESET}\n")
                    thinking = False
                out_parts.append(content)
                sys.stdout.write(content)
            sys.stdout.flush()
    if thinking:
        sys.stdout.write(RESET)
    print()
    return "".join(out_parts)


def cmd_health(a) -> None:
    url = resolve_url(a.url)
    with urllib.request.urlopen(authed_request(f"{url}/health"), timeout=30) as r:
        print(f"{GREEN}{r.status}{RESET} {r.read().decode()}")


def cmd_props(a) -> None:
    url = resolve_url(a.url)
    with urllib.request.urlopen(authed_request(f"{url}/props"), timeout=30) as r:
        print(json.dumps(json.load(r), indent=2))


def cmd_models(a) -> None:
    url = resolve_url(a.url)
    with urllib.request.urlopen(authed_request(f"{url}/v1/models"), timeout=30) as r:
        print(json.dumps(json.load(r), indent=2))


def cmd_slots(a) -> None:
    url = resolve_url(a.url)
    with urllib.request.urlopen(authed_request(f"{url}/slots"), timeout=30) as r:
        print(json.dumps(json.load(r), indent=2))


def build_messages(system: str | None, prompt: str, history: list) -> list:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(history)
    msgs.append({"role": "user", "content": prompt})
    return msgs


def run_chat(a, history: list) -> None:
    url = resolve_url(a.url)
    payload = {
        "model": a.model,
        "messages": build_messages(a.system, a.prompt, history),
        "temperature": a.temperature,
        "top_p": a.top_p,
        "max_tokens": a.max_tokens,
        "stream": True,
        "cache_prompt": True,
    }
    if a.no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    post_stream(url + "/v1/chat/completions", payload, show_reasoning=not a.no_think)


def cmd_chat(a) -> None:
    run_chat(a, [])


def cmd_tools(a) -> None:
    """Demonstrate the tool-calling path this model is tuned for."""
    url = resolve_url(a.url)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": a.prompt}],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    req = authed_request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        obj = json.load(r)
    msg = obj["choices"][0]["message"]
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc["function"]
            print(f"{GREEN}tool_call{RESET}: {fn['name']}({fn['arguments']})")
    else:
        print(f"{YELLOW}no tool call{RESET}")
    if msg.get("content"):
        print("content:", msg["content"])


def cmd_tokenize(a) -> None:
    url = resolve_url(a.url)
    payload = {"content": a.text, "add_special": False}
    req = authed_request(
        url + "/tokenize",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        obj = json.load(r)
    print("n_tokens:", len(obj.get("tokens", [])))
    print(obj.get("tokens"))


def cmd_gen(a) -> None:
    """Generate an image from a prompt via /v1/images/generations."""
    import base64
    url = resolve_url(a.url)
    payload = {
        "prompt": a.prompt,
        "size": a.size or "1024x1024",
        "n": 1,
        "response_format": "b64_json",
        "use_kv_cache": False,
    }
    if getattr(a, "steps", None) is not None:
        payload["steps"] = int(a.steps)
    if getattr(a, "seed", None) is not None:
        payload["seed"] = int(a.seed)
    if getattr(a, "guidance_scale", None) is not None:
        payload["guidance_scale"] = float(a.guidance_scale)
    if getattr(a, "img_guidance_scale", None) is not None:
        payload["img_guidance_scale"] = float(a.img_guidance_scale)
    if getattr(a, "max_input_image_size", None) is not None:
        payload["max_input_image_size"] = int(a.max_input_image_size)
    if getattr(a, "offload_model", None) is not None:
        payload["offload_model"] = bool(a.offload_model)
    if getattr(a, "image", None):
        import mimetypes
        encoded = []
        max_dim = int(getattr(a, "max_input_image_size", None) or 512)
        for p in a.image:
            if os.path.exists(p):
                mime = mimetypes.guess_type(p)[0] or "image/png"
                try:
                    from PIL import Image
                    import io
                    im = Image.open(p)
                    if max(im.size) > max_dim:
                        im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                        buf = io.BytesIO()
                        im.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        mime = "image/png"
                    else:
                        with open(p, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                encoded.append(f"data:{mime};base64,{b64}")
            else:
                encoded.append(p)
        payload["input_images"] = encoded

    def _encode_img(p, max_dim=512):
        if not p:
            return None
        if os.path.exists(p):
            import mimetypes, io
            from PIL import Image
            mime = mimetypes.guess_type(p)[0] or "image/png"
            try:
                im = Image.open(p)
                if max(im.size) > max_dim:
                    im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    return f"data:image/png;base64,{b64}"
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"data:{mime};base64,{b64}"
            except Exception:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"data:{mime};base64,{b64}"
        return p

    if getattr(a, "face_left", None):
        payload["face_left"] = _encode_img(a.face_left)
    if getattr(a, "face_right", None):
        payload["face_right"] = _encode_img(a.face_right)
    if getattr(a, "face", None):
        payload["face_images"] = [_encode_img(x) for x in a.face]

    print(f"{CYAN}generating image{RESET} ({payload['size']}) -> {a.out} ...")
    data = json.dumps(payload).encode()
    req = authed_request(
        f"{url}/v1/images/generations",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    initial_hash = None
    try:
        init_req = authed_request(f"{url}/last_image", headers={}, method="GET")
        with urllib.request.urlopen(init_req, timeout=10) as p_init:
            if p_init.status == 200:
                import hashlib
                initial_hash = hashlib.sha256(p_init.read()).hexdigest()
    except Exception:
        pass

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            res = json.load(resp)
            b64_data = res["data"][0]["b64_json"]
            img_bytes = base64.b64decode(b64_data)
            out_path = a.out or "output.png"
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            print(f"{GREEN}saved image{RESET} ({len(img_bytes)} bytes) to: {out_path}")
            return
    except urllib.error.HTTPError as e:
        if e.code == 524:
            print(f"{YELLOW}Cloudflare connection timed out (524), but generation is running on Kaggle GPU!{RESET}")
            print(f"{CYAN}Polling for completed image from Kaggle...{RESET}")
            import time, hashlib
            out_path = a.out or "output.png"
            for poll_idx in range(180):
                time.sleep(10)
                try:
                    poll_req = authed_request(f"{url}/last_image", headers={}, method="GET")
                    with urllib.request.urlopen(poll_req, timeout=30) as p_resp:
                        if p_resp.status == 200:
                            img_bytes = p_resp.read()
                            if len(img_bytes) > 2000:
                                cur_hash = hashlib.sha256(img_bytes).hexdigest()
                                if initial_hash is None or cur_hash != initial_hash:
                                    with open(out_path, "wb") as f:
                                        f.write(img_bytes)
                                    print(f"{GREEN}saved image{RESET} ({len(img_bytes)} bytes) to: {out_path}")
                                    return
                except Exception:
                    pass
                if poll_idx % 3 == 0:
                    print(f"{DIM}still generating on GPU... ({poll_idx * 10 + 10}s elapsed){RESET}")
            print(f"{RED}Timed out waiting for generated image{RESET}", file=sys.stderr)
            sys.exit(1)
        raise


def cmd_raw(a) -> None:
    """POST arbitrary JSON to an arbitrary path:  cluster.py raw /props '{}'"""
    url = resolve_url(a.url)
    req = authed_request(
        url + a.path,
        data=(a.body or "{}").encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        print(r.read().decode())


def cmd_repl(a) -> None:
    """Interactive multi-turn chat. Commands inside: /new /think /nothink /quit."""
    url = resolve_url(a.url)
    history: list = []
    think = not a.no_think
    print(f"{CYAN}saluki repl{RESET} @ {url}")
    print(f"{DIM}commands: /new  /think  /nothink  /quit{RESET}")
    while True:
        try:
            line = input(f"{CYAN}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/new":
            history = []
            print(f"{DIM}history cleared{RESET}")
            continue
        if line == "/think":
            think = True
            print(f"{DIM}thinking on{RESET}")
            continue
        if line == "/nothink":
            think = False
            print(f"{DIM}thinking off{RESET}")
            continue

        payload = {
            "model": a.model,
            "messages": build_messages(a.system, line, history),
            "temperature": a.temperature,
            "top_p": a.top_p,
            "max_tokens": a.max_tokens,
            "stream": True,
            "cache_prompt": True,
        }
        if not think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        sys.stdout.write(f"{GREEN}saluki>{RESET} ")
        answer = post_stream(url + "/v1/chat/completions", payload, show_reasoning=think)

        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": answer or "(no content)"})


def main() -> None:
    p = argparse.ArgumentParser(prog="cluster", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", help="override the tunnel URL")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="model id sent to the server (any value; llama-server ignores it)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, need_prompt=True):
        if need_prompt:
            sp.add_argument("prompt", help="the user message")
        else:
            sp.add_argument("prompt", nargs="?", default=None,
                            help="the user message (optional; a default is used)")
        sp.add_argument("--system", help="system prompt")
        sp.add_argument("--temperature", type=float, default=0.6)
        sp.add_argument("--top-p", type=float, default=0.95)
        sp.add_argument("--max-tokens", type=int, default=2048)
        sp.add_argument("--no-think", action="store_true",
                        help="disable the thinking phase (faster, better for tools)")

    sub.add_parser("health", help="server health").set_defaults(func=cmd_health)
    sub.add_parser("props", help="server properties / model info").set_defaults(func=cmd_props)
    sub.add_parser("models", help="list models").set_defaults(func=cmd_models)
    sub.add_parser("slots", help="show context slots").set_defaults(func=cmd_slots)

    sp = sub.add_parser("chat", help="single-shot chat")
    add_common(sp)
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("stream", help="alias of chat (streams by default)")
    add_common(sp)
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("tools", help="demonstrate tool calling")
    add_common(sp, need_prompt=False)
    sp.set_defaults(func=cmd_tools)

    sp = sub.add_parser("repl", help="interactive multi-turn chat")
    add_common(sp, need_prompt=False)
    sp.set_defaults(func=cmd_repl)

    sp = sub.add_parser("tokenize", help="count tokens for a string")
    sp.add_argument("text")
    sp.set_defaults(func=cmd_tokenize)

    sp = sub.add_parser("raw", help="raw POST to a path")
    sp.add_argument("path")
    sp.add_argument("body", nargs="?", default="{}")
    sp.set_defaults(func=cmd_raw)

    sp = sub.add_parser("gen", help="generate image via /v1/images/generations")
    sp.add_argument("prompt", help="prompt text describing the image")
    sp.add_argument("--image", "-i", action="append", default=[],
                    help="reference input image file (can be specified multiple times)")
    sp.add_argument("--size", default="1024x1024", help="image size (e.g. 1024x1024, 768x768, 512x512)")
    sp.add_argument("--steps", default=None, help="inference steps")
    sp.add_argument("--seed", default=None, help="random seed (integer)")
    sp.add_argument("--guidance-scale", type=float, default=None, help="classifier-free guidance scale")
    sp.add_argument("--img-guidance-scale", type=float, default=None, help="image guidance scale")
    sp.add_argument("--max-input-image-size", type=int, default=512, help="max reference image dimension (default 512)")
    sp.add_argument("--offload-model", action="store_true", default=None, help="enable layer offload to save VRAM")
    sp.add_argument("--face-left", default=None, help="reference face image to swap onto the left/foreground person")
    sp.add_argument("--face-right", default=None, help="reference face image to swap onto the right/background person")
    sp.add_argument("--face", action="append", default=[], help="reference face image to swap onto detected faces")
    sp.add_argument("--out", "-o", default="output.png", help="output file path (default output.png)")
    sp.set_defaults(func=cmd_gen)

    a = p.parse_args()
    if a.cmd == "tools" and not getattr(a, "prompt", None):
        a.prompt = "What is the weather in Paris and Tokyo?"
    try:
        a.func(a)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"{RED}HTTP {e.code}{RESET} {e.reason}\n{body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"{RED}cannot reach server{RESET}: {e.reason}\n"
              f"Is the Kaggle kernel still running? Try './ctr.sh url'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()