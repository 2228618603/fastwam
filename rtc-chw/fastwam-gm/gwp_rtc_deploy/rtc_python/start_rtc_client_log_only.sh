#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${SERVER:-ws://127.0.0.1:8000}" MODE=log-only bash "${ROOT}/deploy/start_client.sh"
