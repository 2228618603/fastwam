#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${SERVER:-ws://127.0.0.1:8000}" SPEED="${SPEED:-30}" HZ="${HZ:-10}" REPLAN="${REPLAN:-8}" DURATION="${DURATION:-20}" JOINT_DELTA="${JOINT_DELTA:-1.0}" VIOLATIONS="${VIOLATIONS:-1}" MODE=closed-loop bash "${ROOT}/deploy/start_client.sh"
