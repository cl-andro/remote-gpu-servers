#!/usr/bin/env bash
# Thin wrapper: all logic lives in ../lib/ctr.sh (shared by every project).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../lib/ctr.sh" --project "$HERE" "$@"