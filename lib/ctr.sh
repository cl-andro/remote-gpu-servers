#!/usr/bin/env bash
# lib/ctr.sh — generic lifecycle driver for remote-GPU inference projects.
#
# It is invoked by a thin per-project wrapper that passes --project <dir>:
#
#     ./saluki/ctr.sh push
#
#   push      regenerate notebook.ipynb and push it to Kaggle, starting the run
#   url       fetch the public tunnel URL from the live run logs -> url.txt
#   status    kernel run status
#   logs      print the log so far
#   follow    live-tail the kernel logs (Ctrl-C to detach)
#   stop      ask the kernel to exit, then wait for COMPLETE
#   stop-url  print/cache the stop URL for the current run
#   output    download the run's output files
#   test      health check + a sample chat through the tunnel
#
# The Kaggle token is read from <repo-root>/.env (KGAT_...).
set -euo pipefail

PROJECT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    *) break ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "usage: ctr.sh --project <dir> <command>" >&2
  exit 2
fi
PROJECT="$(cd "$PROJECT" && pwd)"
CMD="${1:-}"
ROOT="$(cd "$PROJECT/.." && pwd)"
ENV_FILE="$ROOT/.env"
export CLUSTER_PROJECT="$PROJECT"

KERNEL="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["id"])' \
          "$PROJECT/kernel-metadata.json")"
MODEL_ALIAS="$(python3 -c 'import sys;sys.path.insert(0,sys.argv[1]);import kernel_config as k;print(k.CFG.get("__MODEL_ALIAS__","model"))' \
          "$PROJECT" 2>/dev/null || echo model)"
MODEL_TYPE="$(python3 -c 'import sys;sys.path.insert(0,sys.argv[1]);import kernel_config as k;print(k.CFG.get("__MODEL_TYPE__","llm"))' \
          "$PROJECT" 2>/dev/null || echo llm)"
CLIENT="$ROOT/lib/cluster.py"
URL_FILE="$PROJECT/url.txt"
STOP_URL_FILE="$PROJECT/stop_url.txt"

if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    if [[ "$line" == *=* ]]; then
      key="${line%%=*}"
      val="${line#*=}"
      key="$(echo "$key" | tr -d '[:space:]')"
      val="$(echo "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
      if [[ -n "$key" ]]; then
        export "$key"="$val"
      fi
    fi
  done < "$ENV_FILE"
  TOKEN="$(grep -m1 -oE 'KGAT_[A-Za-z0-9]+' "$ENV_FILE" || true)"
  [[ -n "$TOKEN" ]] && export KAGGLE_API_TOKEN="$TOKEN"
fi

pretty() {
  # `kaggle kernels logs` emits JSON events; forward their `data` fields and
  # pass plain text through unchanged.
  python3 -c '
import sys, json
for raw in sys.stdin:
    line = raw.rstrip("\n").lstrip(",")
    if not line.lstrip().startswith("{"):
        sys.stdout.write(raw)
        continue
    try:
        rec = json.loads(line)
    except Exception:
        sys.stdout.write(raw)
        continue
    items = rec if isinstance(rec, list) else [rec]
    for r in items:
        if isinstance(r, dict) and "data" in r:
            sys.stdout.write(r.get("data", ""))
        else:
            sys.stdout.write(json.dumps(r) + "\n")
'
}

# The non-follow log call returns nothing while a run is live, so scrapers
# attach to the follow stream (which replays from the start) then detach.
kernel_log_text() {
  local target="$1"
  timeout "${SCRAPE_SECONDS:-25}" kaggle kernels logs -f "$target" 2>/dev/null | pretty || true
}

# Pull CONTROL_URL out of the log stream and cache it locally.
extract_url() {
  local target="$1" out marker
  out="$(kernel_log_text "$target")"
  marker="$(printf '%s\n' "$out" \
    | grep -oE 'CONTROL_URL=https://[a-zA-Z0-9.-]+' \
    | tail -1 | sed 's/^CONTROL_URL=//')"
  if [[ -n "$marker" ]]; then echo "$marker"; return 0; fi
  printf '%s\n' "$out" \
    | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' \
    | tail -1
}

# Pull the full STOP_URL (control tunnel + secret path) out of the logs.
extract_stop_url() {
  local target="$1"
  kernel_log_text "$target" \
    | grep -oE 'STOP_URL=https://[a-z0-9-]+\.trycloudflare\.com/stop/[0-9a-f]+' \
    | tail -1 \
    | sed 's/^STOP_URL=//'
}

case "$CMD" in
  push)
    python3 "$ROOT/lib/make_kernel.py" "$PROJECT"
    ( cd "$PROJECT" && kaggle kernels push -p . )
    rm -f "$URL_FILE" "$STOP_URL_FILE"
    ;;
  up)
    python3 "$ROOT/lib/make_kernel.py" "$PROJECT"
    ( cd "$PROJECT" && kaggle kernels push -p . )
    rm -f "$URL_FILE" "$STOP_URL_FILE"
    echo "waiting for the tunnel URL (first run downloads the model) ..."
    URL=""
    for _ in $(seq 1 "${UP_WAIT_TRIES:-90}"); do
      URL="$(extract_url "$KERNEL" || true)"
      [[ -n "$URL" ]] && break
      sleep "${UP_POLL_SECONDS:-20}"
    done
    if [[ -z "$URL" ]]; then
      echo "no tunnel URL yet; keep watching with ./bin/cluster <project> follow" >&2
      exit 1
    fi
    echo "$URL" | tee "$URL_FILE"
    ;;
  url)
    URL=""
    RETRIES="${URL_RETRIES:-3}"
    for i in $(seq 1 "$RETRIES"); do
      URL="$(extract_url "$KERNEL" || true)"
      [[ -n "$URL" ]] && break
      if [[ "$i" -lt "$RETRIES" ]]; then
        sleep "${URL_POLL_SECONDS:-5}"
      fi
    done
    if [[ -z "$URL" ]]; then
      echo "no tunnel URL yet. Check status with: ./bin/cluster $(basename "$PROJECT") status" >&2
      echo "or watch startup with:             ./bin/cluster $(basename "$PROJECT") follow" >&2
      exit 1
    fi
    echo "$URL" | tee "$URL_FILE"
    ;;
  status)
    kaggle kernels status "$KERNEL" 2>/dev/null | tail -1
    ;;
  logs)
    timeout "${LOGS_SECONDS:-20}" kaggle kernels logs -f "$KERNEL" 2>/dev/null | pretty || true
    ;;
  follow)
    kaggle kernels logs -f "$KERNEL" 2>/dev/null | pretty
    ;;
  stop)
    STOP_URL="$(cat "$STOP_URL_FILE" 2>/dev/null || true)"
    if [[ -z "$STOP_URL" ]]; then
      STOP_URL="$(extract_stop_url "$KERNEL" || true)"
      [[ -n "$STOP_URL" ]] && echo "$STOP_URL" > "$STOP_URL_FILE"
    fi
    if [[ -z "$STOP_URL" ]]; then
      echo "could not find STOP_URL. Is the run live? Try: ./bin/cluster $(basename "$PROJECT") stop-url" >&2
      echo "Fallback: https://www.kaggle.com/code/$KERNEL -> Stop Session" >&2
      exit 1
    fi
    echo "Requesting shutdown via $STOP_URL ..."
    python3 - "$STOP_URL" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=60) as r:
        print(r.read().decode().strip() or f"HTTP {r.status}")
except Exception as e:
    print("stop request failed:", e)
    raise SystemExit(1)
PY
    echo "Waiting for the kernel to stop (this can take a few minutes) ..."
    STATE=""
    for _ in $(seq 1 "${STOP_WAIT_TRIES:-40}"); do
      STATE="$(kaggle kernels status "$KERNEL" 2>/dev/null | tail -1 || true)"
      case "$STATE" in
        *COMPLETE*|*ERROR*|*CANCEL*)
          echo "Stopped: $STATE"
          echo "GPU released."
          exit 0
          ;;
      esac
      sleep "${STOP_POLL_SECONDS:-15}"
    done
    echo "Still showing: $STATE" >&2
    echo "It usually finishes shortly; re-check with: ./bin/cluster $(basename "$PROJECT") status" >&2
    ;;
  stop-url)
    STOP_URL="$(extract_stop_url "$KERNEL" || true)"
    if [[ -z "$STOP_URL" ]]; then
      echo "(no STOP_URL yet - run is not live)" >&2
      exit 1
    fi
    echo "$STOP_URL" | tee "$STOP_URL_FILE"
    ;;
  output)
    ( cd "$PROJECT" && kaggle kernels output "$KERNEL" -p ./output )
    ;;
  test)
    URL="$(cat "$URL_FILE" 2>/dev/null || extract_url "$KERNEL" || true)"
    [[ -z "$URL" ]] && { echo "no URL; run ./bin/cluster $(basename "$PROJECT") url first" >&2; exit 1; }
    echo "== health =="
    python3 "$CLIENT" --url "$URL" --model "$MODEL_ALIAS" health
    if [[ "$MODEL_TYPE" == "flux" || "$MODEL_TYPE" == "sdxl" || "$MODEL_TYPE" == "omnigen" || "$MODEL_TYPE" == "janus" ]]; then
      echo "== test image generation (512x512) =="
      python3 "$CLIENT" --url "$URL" --model "$MODEL_ALIAS" gen \
        "a golden metallic robot in a futuristic desert, sharp focus, 8k" \
        --size 512x512 \
        --out "$PROJECT/test_output.png"
    else
      echo "== chat (no reasoning) =="
      python3 "$CLIENT" --url "$URL" --model "$MODEL_ALIAS" chat --no-think --max-tokens 128 \
        "Reply with exactly: CLUSTER ONLINE"
    fi
    ;;
  chat|stream|repl|tools|health|props|models|slots|tokenize|raw|gen)
    # Direct passthrough to the stdlib client (one-shot chat, REPL, image gen, etc.).
    # `$@` already begins with the command name.
    URL="$(cat "$URL_FILE" 2>/dev/null || extract_url "$KERNEL" || true)"
    [[ -z "$URL" ]] && { echo "no URL; run ./bin/cluster $(basename "$PROJECT") url first" >&2; exit 1; }
    exec python3 "$CLIENT" --url "$URL" --model "$MODEL_ALIAS" "$@"
    ;;
  *)
    sed -n '2,20p' "$0"
    exit 2
    ;;
esac