#!/usr/bin/env bash
# 一次性补齐：把此前从未同步到 OSS 的部分传过去。
#
# 为什么不直接跑 sync_to_oss.sh 全量：ossfs2 不保存 mtime，rsync -rt 的
# size+mtime 快速检查会对每个已存在的文件都判定"需要重传"，全量一跑就是
# checkpoints(65G) + 470(28G) + runs(610G) 全部重上传一遍。这里只点名新增项。
#
# 跑完之后这些路径已写入 sync_to_oss.sh，日常由 autosync 守护进程接管。

set -uo pipefail

SRC="/home/gaomeng"
DST="/mnt/data/gaomeng"

RSYNC_OPTS=(-rt --no-perms --no-owner --no-group --omit-dir-times
            --human-readable --stats)

echo "=========================================================="
echo " 一次性补齐  $SRC -> $DST  (ossfs2)"
echo " 开始: $(date '+%F %T')"
echo "=========================================================="

# 1) FastWAM 的同级工作目录（~283M，最大的是 giga-world-policy 的 .git）
for d in giga-world-policy wam.cpp GigaWorldPolicy解读 geekrl openpi .claude; do
  echo ""
  echo "---- 同级目录: $d ----"
  rsync "${RSYNC_OPTS[@]}" \
    --exclude='**/__pycache__/' --exclude='*.pyc' \
    --exclude='.venv/' --exclude='node_modules/' \
    "$SRC/$d" "$DST/"
done

# 2) 漏掉的原始数据集（40G）——静态，传一次即可
echo ""
echo "---- 原始数据集 542_0711 (40G) ----"
rsync "${RSYNC_OPTS[@]}" \
  "$SRC/FastWAM/data/agilex_empty_the_box_all_542_0711" \
  "$DST/FastWAM/data/"

# 3) 转换后 542 数据集的 meta+parquet（videos 全是软链接，OSS 建不了，需时重新生成）
echo ""
echo "---- 转换后 542 数据集 meta+parquet (227M) ----"
rsync "${RSYNC_OPTS[@]}" \
  --exclude='videos/' \
  "$SRC/FastWAM/data/agilex_empty_the_box_542_fastwam" \
  "$DST/FastWAM/data/"

echo ""
echo "=========================================================="
echo " 补齐完成: $(date '+%F %T')"
echo "=========================================================="
