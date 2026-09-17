#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
"${PYTHON}" -m pip install 'numpy<2' websockets msgpack msgpack-numpy pyarrow piper_sdk

echo "Client dependencies installed. In each new terminal also run:"
echo "  source /opt/ros/humble/setup.bash"
echo "  conda activate gwp_client"
