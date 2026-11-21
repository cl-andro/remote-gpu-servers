#!/usr/bin/env bash
# sd-checkpoint/ctr.sh — thin wrapper for the shared driver.
# All commands delegate to lib/ctr.sh with --project set to this directory.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../lib/ctr.sh" --project "$SCRIPT_DIR" "$@"