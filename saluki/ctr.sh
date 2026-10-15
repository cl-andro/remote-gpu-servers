#!/usr/bin/env bash
# ctr.sh — manage the Kaggle-hosted Saluki server from your PC.
#
#   ./ctr.sh push     push notebook.ipynb to Kaggle and start the run
#   ./ctr.sh url      fetch the public tunnel URL from the live run logs
#   ./ctr.sh status   kernel run status
#   ./ctr.sh logs     print the log so far
#   ./ctr.sh follow   live-tail the kernel logs (Ctrl-C to detach)
#   ./ctr.sh stop     stop the running kernel session (frees the GPU)
#   ./ctr.sh stop-url print/cache the stop URL for the current run
#   ./ctr.sh output   download the run's output files
#   ./ctr.sh test     run health + a sample chat against the tunnel
#
# Token is read from ../.env  (KGAT_...). Nothing large is downloaded locally.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$HERE/../.env"
KERNEL="zkmohammad/saluki-server"
URL_FILE="$HERE/url.txt"
STOP_URL_FILE="$HERE/stop_url.txt"

if [[ -f "$ENV_FILE" ]]; then
  TOKEN="$(grep -m1 -oE 'KGAT_[A-Za-z0-9]+' "$ENV_FILE" || true)"
  [[ -n "$TOKEN" ]] && export KAGGLE_API_TOKEN="$TOKEN"
fi

pretty() {
  # The CLI prints plain log text for scripts and JSON events for `kaggle
  # kernels logs`; forward JSON `data` fields and pass everything else through.
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
  # Prefer the explicit marker so the control tunnel URL can never win.
  marker="$(printf '%s\n' "$out" \
    | grep -oE 'CONTROL_URL=https://[a-z0-9-]+\.trycloudflare\.com' \
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

case "${1:-}" in
  push)
    ( cd "$HERE" && python3 make_notebook.py )
    ( cd "$HERE" && kaggle kernels push -p . )
    ;;
  url)
    URL="$(extract_url "$KERNEL" || true)"
    if [[ -z "$URL" ]]; then
      echo "no tunnel URL yet. Check status with: ./ctr.sh status" >&2
      echo "or watch startup with:             ./ctr.sh follow" >&2
      exit 1
    fi
    echo "$URL" | tee "$URL_FILE"
    ;;
  status)
    kaggle kernels status "$KERNEL" 2>/dev/null | tail -1
    ;;
  logs)
    # Kaggle's non-follow log call can be empty while a run is live, so we
    # briefly attach to the follow stream and then detach.
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
      echo "could not find STOP_URL. Is the run live? Try: ./ctr.sh stop-url" >&2
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
    echo "It usually finishes shortly; re-check with: ./ctr.sh status" >&2
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
    ( cd "$HERE" && kaggle kernels output "$KERNEL" -p ./output )
    ;;
  test)
    URL="$(cat "$URL_FILE" 2>/dev/null || extract_url "$KERNEL" || true)"
    [[ -z "$URL" ]] && { echo "no URL; run ./ctr.sh url first" >&2; exit 1; }
    echo "== health =="
    python3 "$HERE/saluki.py" --url "$URL" health
    echo "== chat (thinking off) =="
    python3 "$HERE/saluki.py" --url "$URL" chat --no-think --max-tokens 128 \
      "Reply with exactly: SALUKI ONLINE"
    ;;
  *)
    sed -n '2,16p' "$0"
    exit 2
    ;;
esac