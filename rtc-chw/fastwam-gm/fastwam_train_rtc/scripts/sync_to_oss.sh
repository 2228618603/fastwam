#!/usr/bin/env bash
# 把 /home 的本地工作目录镜像到 /mnt/data (OSS) 做持久化。
#
# 布局决策（混合方案）：
#   执行目录  = /home/gaomeng/FastWAM      容器 overlay，快、支持软链接
#   持久镜像  = /mnt/data/gaomeng/FastWAM  ossfs2 对象存储，慢但容器重建不丢
#
# 除 FastWAM 外，/home/gaomeng 下的同级仓库/笔记（giga-world-policy、wam.cpp 等）
# 也一起镜像到 /mnt/data/gaomeng/ —— 见下面的 SIBLINGS。它们同样只存在于容器
# overlay，重建即丢，且都是代码/文档量级，纳入每轮同步的成本可以忽略。
#
# 为什么执行留在 /home 而不是 OSS：
#   1. 数据集随机读 —— dataloader 每样本 99 次 seek 解码，OSS 实测慢 1.3x（且该
#      测量偏乐观，真实要随机读 1413 个文件、缓存命中率更低）
#   2. checkpoint 写入 —— weights 约 12GB、ZeRO state 更大，8 个 rank 并发写 FUSE
#   3. 文本 embedding 缓存 —— robot_video_dataset.py:261 每个样本都 torch.load 一次
#   4. OSS 不支持软链接/硬链接（ossfs2 固有限制），而转换后的数据集依赖软链接
#
# 用法:
#   bash scripts/sync_to_oss.sh              # 同步（不含 ZeRO state）
#   bash scripts/sync_to_oss.sh --dry-run    # 只看会传什么
#   bash scripts/sync_to_oss.sh --with-state # 连 ZeRO state 一起（体积很大）
#   bash scripts/sync_to_oss.sh --code-only  # 只同步代码/配置/文档 + 同级目录（秒级）
#   bash scripts/sync_to_oss.sh --verify     # 逐字节校验镜像是否漂移（只报不写，慢）

set -euo pipefail

SRC="/home/gaomeng/FastWAM"
DST="/mnt/data/gaomeng/FastWAM"

# FastWAM 的同级工作目录：源 /home/gaomeng/<d> -> 镜像 /mnt/data/gaomeng/<d>
HOME_SRC="/home/gaomeng"
HOME_DST="/mnt/data/gaomeng"
SIBLINGS=(wam.cpp GigaWorldPolicy解读 geekrl openpi .claude)

DRY=""
WITH_STATE=0
CODE_ONLY=0
CHANGING_ONLY=0
VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY="--dry-run" ;;
    --with-state)    WITH_STATE=1 ;;
    --code-only)     CODE_ONLY=1 ;;
    --changing-only) CHANGING_ONLY=1 ;;
    --verify)        VERIFY=1 ;;
    *) echo "未知参数: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$DST"

# ossfs2 不支持 chown/chmod/utime，也不支持软链接：
#   -r 递归
#   -t 保留 mtime —— **在 ossfs2 上无效**（写入后 mtime 被重置为上传时刻），保留它
#      只为让本地侧语义一致；正因为无效才需要下面的 --size-only，见 RSYNC_OPTS_BIG
#   刻意不加 -l/-a —— 那会尝试创建软链接并报 "Operation not supported"
#   --no-perms/--no-owner/--no-group 避免 FUSE 上的属性设置失败
#   用 --stats 而非 --info=progress2：后者在几千个文件时输出量爆炸
RSYNC_OPTS=(-rt --no-perms --no-owner --no-group --omit-dir-times
            --human-readable --stats $DRY)

# 大块静态数据额外加 --size-only。
#
# 为什么必须加：ossfs2 不支持 utime，目标 mtime 永远等于上传时刻，源 mtime 一个
# 都落不下来（实测 README.md 源 1787735739 / 目标 1788487350）。rsync 的默认快速
# 检查要求 size 和 mtime **都**匹配才跳过，mtime 永不匹配 ⇒ 每一轮把所有文件重传
# 一遍。修之前 daemon 日志每轮稳定报 98,096 文件 / 6.81G（代码）+ 2,083 文件 /
# 397.66G（runs），那不是"变了这么多"，是"又白传了这么多"。
#
# 为什么这些路径可以只比 size：权重/检查点/原始数据集/git 对象/逐样本评测产物都是
# 写一次不再改，日志是追加（长度变 ⇒ size 变）。会被原地覆盖的小文件（refs、
# summary.json、dataset_stats.json 等）留在精确比对里，见第 1) 段。
#
# 已知代价：同尺寸原地覆盖会被漏掉 —— 典型场景是同一 config 重跑训练，
# runs/<exp>/<variant>/checkpoints/step_XXXX/ 下生成路径相同、字节数也相同但内容
# 不同的权重。这种情况先删掉 OSS 上对应目录再同步，或用 --verify 查漂移。
RSYNC_OPTS_BIG=("${RSYNC_OPTS[@]}" --size-only)

if [[ $VERIFY -eq 1 ]]; then
  # 逐字节校验（-c）+ 只报不写。会把 OSS 上的数据读回来算 checksum，很慢很贵，
  # 只在怀疑镜像漂移时手动跑，不要放进定时任务。
  RSYNC_OPTS=("${RSYNC_OPTS[@]}" -c --dry-run)
  RSYNC_OPTS_BIG=("${RSYNC_OPTS[@]}")
fi

EXCLUDES=(
  # 转换后数据集的视频目录，里面全是软链接（OSS 建不了，指向原始数据集）。
  # 需要时用 scripts/convert_agilex_to_fastwam.py 一条命令重新生成，无需备份。
  "--exclude=data/agilex_empty_the_box_fastwam/videos/"
  "--exclude=data/agilex_empty_the_box_542_fastwam/videos/"
  # 凭证不同步到 OSS（对象存储上多一份 secret 没必要）
  "--exclude=.wandb_key"
  # 字节码，纯派生物
  "--exclude=**/__pycache__/"
  "--exclude=*.pyc"
)
# 注：.cache/ 与 runs/**/wandb/ 曾被排除（可重建 / 指标已上传云端），现按"全量镜像"
# 的要求纳入。两者合计约 345MB，wandb 里的软链接会被 rsync 跳过（RSYNC_OPTS 无 -l）。

# 同级目录用独立的排除集 —— 上面那些是 FastWAM 相对路径，套到别的仓库上会误伤
# （例如 runs/**/ 规则会碰到 openpi/runs/）。
SIBLING_EXCLUDES=(
  "--exclude=**/__pycache__/"
  "--exclude=*.pyc"
  "--exclude=.venv/"
  "--exclude=node_modules/"
)

if [[ $WITH_STATE -eq 0 ]]; then
  # ZeRO 完整训练状态，体积远大于 weights；只用于断点续训，不用于持久化。
  EXCLUDES+=("--exclude=runs/**/checkpoints/state/")
fi

echo "=========================================================="
echo " 源:   $SRC  (+ 同级目录 ${SIBLINGS[*]})"
echo " 目标: $HOME_DST   (ossfs2 / OSS)"
[[ -n "$DRY" ]]      && echo " 模式: DRY-RUN（不实际写入）"
[[ $VERIFY -eq 1 ]]  && echo " 模式: VERIFY（逐字节校验，只报不写）"
[[ $CODE_ONLY -eq 1 ]] && echo " 模式: 仅代码/配置/文档 + 同级目录"
[[ $WITH_STATE -eq 1 ]] && echo " 含 ZeRO state: 是"
echo "=========================================================="

# "$3" = "exact" 时用 size+mtime 精确比对（小文件，每轮重传的代价可以接受）；
# 默认用 --size-only（大块静态数据，理由见 RSYNC_OPTS_BIG 的注释）。
sync_one() {
  local rel="$1" desc="$2" mode="${3:-size-only}"
  if [[ ! -e "$SRC/$rel" ]]; then
    echo "[skip] $desc ($rel 不存在)"
    return 0
  fi
  local -a opts
  if [[ "$mode" == "exact" ]]; then
    opts=("${RSYNC_OPTS[@]}")
  else
    opts=("${RSYNC_OPTS_BIG[@]}")
  fi
  echo ""
  echo "---- $desc  ($rel) ----"
  rsync "${opts[@]}" "${EXCLUDES[@]}" "$SRC/$rel" "$DST/$(dirname "$rel")/"
}

# 1) 代码 / 配置 / 文档 / 脚本（小，永远同步，精确比对）
#
# 这里刻意排掉四块"藏在代码目录里的大块静态数据"，它们占了本段 99% 的文件数和
# 99% 的字节数（98,141 文件 / 6.63G，排掉后只剩约 1,300 文件 / 41M），交给 1.2)
# 用 --size-only 处理：
#   .git/objects/            git 对象按内容寻址（SHA 命名），写一次永不原地改
#   eval_offline 的 npz/mp4  按 step_XXXXXX 分目录的逐样本产物，48,248 个 / 3.2G
#   .cache/huggingface/      HF datasets 缓存，路径里就是内容指纹，293M
# 注意 .git 的**其余部分**必须留在精确比对里 —— refs/heads/* 是固定 41 字节的 SHA，
# 分支移动后内容变、字节数不变，--size-only 会漏掉它，恢复出来的仓库就指向旧提交。
# 同理 eval_offline 的 summary.json / metrics.csv / sweep.json 和 .cache/deploy_*/ 下的
# dataset_stats.json 会被原地覆盖，也都留在这里。
echo ""
echo "---- 代码 / 配置 / 文档 ----"
rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" \
  --exclude=data/ --exclude=checkpoints/ --exclude=runs/ \
  --exclude='.git/objects/' --exclude='.cache/huggingface/' \
  --exclude='eval_offline/**/*.npz' --exclude='eval_offline/**/*.mp4' \
  "$SRC/" "$DST/"

# 1.2) 上面排掉的静态大块，用 --size-only（理由见 RSYNC_OPTS_BIG 注释）
echo ""
echo "---- git objects（内容寻址，只比 size） ----"
rsync "${RSYNC_OPTS_BIG[@]}" "$SRC/.git/objects/" "$DST/.git/objects/"

echo ""
echo "---- HF datasets 缓存（内容指纹，只比 size） ----"
rsync "${RSYNC_OPTS_BIG[@]}" "$SRC/.cache/huggingface/" "$DST/.cache/huggingface/"

echo ""
echo "---- 离线评测逐样本产物 npz/mp4（只比 size） ----"
rsync "${RSYNC_OPTS_BIG[@]}" \
  --include='*/' --include='*.npz' --include='*.mp4' --exclude='*' \
  "$SRC/eval_offline/" "$DST/eval_offline/"

# 1.5) 同级工作目录（都是代码/文档量级；giga-world-policy 的 269M 里 254M 是 .git）
#      .git/objects/ 同样单独走 --size-only，理由见 1) 段的注释。
echo ""
echo "---- 同级工作目录 ----"
for d in "${SIBLINGS[@]}"; do
  if [[ ! -e "$HOME_SRC/$d" ]]; then
    echo "[skip] $d（不存在）"
    continue
  fi
  # 用 "---- ... ----" 而不是 "· xxx" —— autosync_daemon.sh 的日志 grep 只放行
  # ^---- 开头的行，否则日志里会出现一串认不出归属的 stats 块。
  echo ""
  echo "---- 同级目录: $d ----"
  rsync "${RSYNC_OPTS[@]}" "${SIBLING_EXCLUDES[@]}" --exclude='.git/objects/' \
    "$HOME_SRC/$d" "$HOME_DST/"
  if [[ -d "$HOME_SRC/$d/.git/objects" ]]; then
    rsync "${RSYNC_OPTS_BIG[@]}" \
      "$HOME_SRC/$d/.git/objects/" "$HOME_DST/$d/.git/objects/"
  fi
done

if [[ $CODE_ONLY -eq 1 ]]; then
  echo ""
  echo "完成（仅代码模式）。"
  exit 0
fi

# 2) 归一化统计量 + 文本缓存（小，但重算要时间 —— 用精确比对，别为省流量漏掉更新）
sync_one "runs/_shared"              "归一化统计量 dataset_stats.json" exact
sync_one "data/text_embeds_cache"    "T5 文本 embedding 缓存"          exact

# 6) 训练输出的 weights（state 默认排除）
sync_one "runs"                      "训练输出（weights/日志；state 按开关）"

# --changing-only：只同步会变的部分（上面这些 + 代码 + 同级目录）。
# checkpoints/ 与原始数据集是**静态**的，同步过一次就不会再变，
# 每轮再去 stat 133GB（65G 权重 + 28G 470 + 40G 542_0711）会让单次同步在 OSS FUSE
# 上白耗好几分钟 —— 定时任务不该付这个成本。
if [[ $CHANGING_ONLY -eq 1 ]]; then
  echo ""
  echo "完成（仅变动部分；checkpoints/ 与原始数据集已静态，跳过）。"
  exit 0
fi

# 3) 底座权重（65G，重建容器后免重下）
sync_one "checkpoints"               "Wan 底座 + ActionDiT backbone + RoboTwin 权重"

# 4) 原始数据集（备份用；训练不从 OSS 读）
sync_one "data/agilex_empty_the_box_all_470"      "原始数据集 470（28G，备份用，训练仍读本地）"
sync_one "data/agilex_empty_the_box_all_542_0711" "原始数据集 542_0711（40G，备份用，训练仍读本地）"

# 5) 转换后数据集的 meta+parquet（videos 已排除，可重新生成）
sync_one "data/agilex_empty_the_box_fastwam"     "转换后数据集 meta+parquet（172M；videos 需重建）"
sync_one "data/agilex_empty_the_box_542_fastwam" "转换后 542 数据集 meta+parquet（227M；videos 需重建）"

echo ""
echo "=========================================================="
echo " 同步完成"
du -sh "$DST" 2>/dev/null || true
echo ""
echo " 容器重建后的恢复步骤见 AGILEX_TRAINING_PLAN.md 的「灾难恢复」小节。"
echo "=========================================================="
