#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_ROOT}/../.." && pwd)"
DIST_ROOT="${REPO_ROOT}/deploy-all-on-4090/dist"
STAGE_ROOT="${DIST_ROOT}/stage-$(date +%Y%m%d-%H%M%S)"
CODE_STAGE="${STAGE_ROOT}/code/fastwam-load-giga"
MODEL_STAGE="${STAGE_ROOT}/models/fastwam-load-giga"

mkdir -p "${CODE_STAGE}" "${MODEL_STAGE}" "${DIST_ROOT}"

rsync -a --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "${CODE_ROOT}/app" \
  "${CODE_ROOT}/assets" \
  "${CODE_ROOT}/scripts" \
  "${CODE_ROOT}/README.md" \
  "${CODE_ROOT}/requirements_4090.txt" \
  "${CODE_ROOT}/MANIFEST.sources.json" \
  "${CODE_STAGE}/"

mkdir -p \
  "${CODE_STAGE}/configs/data" \
  "${CODE_STAGE}/configs/model" \
  "${CODE_STAGE}/configs/task" \
  "${CODE_STAGE}/src"

cp "${CODE_ROOT}/configs/train.yaml" "${CODE_STAGE}/configs/train.yaml"
cp "${CODE_ROOT}/configs/data/agilex_empty_box.yaml" "${CODE_STAGE}/configs/data/agilex_empty_box.yaml"
cp "${CODE_ROOT}/configs/model/fastwam.yaml" "${CODE_STAGE}/configs/model/fastwam.yaml"
cp "${CODE_ROOT}/configs/task/agilex_empty_box_uncond_3cam384.yaml" \
  "${CODE_STAGE}/configs/task/agilex_empty_box_uncond_3cam384.yaml"

rsync -a --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.egg-info/' \
  "${REPO_ROOT}/src/fastwam" \
  "${CODE_STAGE}/src/"

rsync -a --delete \
  "${CODE_ROOT}/weights" \
  "${CODE_ROOT}/model_cache" \
  "${MODEL_STAGE}/"

tar -C "${STAGE_ROOT}/code" -czf "${DIST_ROOT}/fastwam-load-giga-code.tar.gz" fastwam-load-giga
tar -C "${STAGE_ROOT}/models" --use-compress-program "zstd -1 -T0" \
  -cf "${DIST_ROOT}/fastwam-load-giga-models.tar.zst" fastwam-load-giga

echo "Created:"
du -h "${DIST_ROOT}/fastwam-load-giga-code.tar.gz" "${DIST_ROOT}/fastwam-load-giga-models.tar.zst"
echo
echo "Code archive extracts to: /home/geekplus/chw/fastwam-load-giga"
echo "Model archive extracts to: /media/geekplus/PortableSSD/chw/fastwam-load-giga"
