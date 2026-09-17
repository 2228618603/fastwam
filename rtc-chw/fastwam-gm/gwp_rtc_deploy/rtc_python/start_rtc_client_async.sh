#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${SERVER:-ws://127.0.0.1:8000}" \
SPEED="${SPEED:-50}" \
HZ="${HZ:-10}" \
REPLAN="${REPLAN:-16}" \
RTC_PREFIX="${RTC_PREFIX:-8}" \
DURATION="${DURATION:-30}" \
JOINT_DELTA="${JOINT_DELTA:-1.0}" \
VIOLATIONS="${VIOLATIONS:-1}" \
MODE=rtc-async bash "${ROOT}/deploy/start_client.sh"
