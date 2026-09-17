#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SERVER="${SERVER:-ws://10.11.0.93:8000}"
MODE="${MODE:-log-only}"
SPEED="${SPEED:-30}"
DURATION="${DURATION:-30}"
HZ="${HZ:-10}"
REPLAN="${REPLAN:-8}"
JOINT_DELTA="${JOINT_DELTA:-1.0}"
VIOLATIONS="${VIOLATIONS:-1}"
ENABLE_TIMEOUT="${ENABLE_TIMEOUT:-20}"

cd "$(dirname "$0")"

case "${MODE}" in
    log-only)
        exec "${PYTHON}" robot_client.py --server "${SERVER}" --mode log-only
        ;;
    goto-start)
        exec "${PYTHON}" goto_start.py \
            --confirm-safety --speed-percent "${SPEED}" --enable-timeout-s "${ENABLE_TIMEOUT}"
        ;;
    step)
        exec "${PYTHON}" robot_client.py \
            --server "${SERVER}" --mode step --max-steps 1 --goto-start \
            --confirm-safety --speed-percent "${SPEED}" --enable-timeout-s "${ENABLE_TIMEOUT}"
        ;;
    closed-loop)
        exec "${PYTHON}" robot_client.py \
            --server "${SERVER}" --mode closed-loop --goto-start \
            --confirm-safety --i-am-watching --max-duration "${DURATION}" \
            --enable-timeout-s "${ENABLE_TIMEOUT}" \
            --speed-percent "${SPEED}" --action-hz "${HZ}" \
            --replan-steps "${REPLAN}" \
            --max-joint-delta-rad "${JOINT_DELTA}" \
            --max-violations "${VIOLATIONS}"
        ;;
    rtc-async)
        exec "${PYTHON}" robot_client.py \
            --server "${SERVER}" --mode rtc-async --goto-start \
            --confirm-safety --i-am-watching --max-duration "${DURATION}" \
            --enable-timeout-s "${ENABLE_TIMEOUT}" \
            --speed-percent "${SPEED}" --action-hz "${HZ}" \
            --replan-steps "${REPLAN}" --rtc-prefix-steps "${RTC_PREFIX:-8}" \
            --max-joint-delta-rad "${JOINT_DELTA}" \
            --max-violations "${VIOLATIONS}"
        ;;
    *)
        echo "MODE must be log-only, goto-start, step, closed-loop, or rtc-async" >&2
        exit 2
        ;;
esac

