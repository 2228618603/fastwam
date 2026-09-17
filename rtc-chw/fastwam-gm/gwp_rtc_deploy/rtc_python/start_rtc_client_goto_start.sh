#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEED="${SPEED:-30}" MODE=goto-start bash "${ROOT}/deploy/start_client.sh"
