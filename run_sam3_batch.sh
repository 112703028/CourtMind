#!/usr/bin/env bash
# run_sam3_batch.sh — 在 vipl-spark 主機批次跑多支影片的 SAM3 tracks + inference → events pkl。
#
# 放在 vipl-spark 主機 /home/vipl/timwu/basketball_analysis/ 執行（該路徑掛載到兩個 container 的 /workspace）。
#
# 兩個 container（都在 vipl-spark）：
#   sam3_env            (cu13.1)  跑 run_sam3_tracks.py（SAM3 追蹤，只有 SAM3 能跑）
#   basketball_analysis           跑 inference.py --sam3_tracks（court/分隊/擋拆下游模組）
#
# 產出：output_videos/screens_v<vid>_sam3_events.pkl（掛載，主機直接看得到）
#
# 用法（在 vipl-spark 主機上）：
#   cd /home/vipl/timwu/basketball_analysis
#   nohup bash run_sam3_batch.sh 116 117 118 119 121 122 123 124 126 127 > sam3_batch.log 2>&1 &
#   tail -f sam3_batch.log
# 不給參數則跑預設 10 支（已排除 SAM2 版 events=0 的 120/125）。

set -u
VIDS=("$@")
if [ ${#VIDS[@]} -eq 0 ]; then
  VIDS=(116 117 118 119 121 122 123 124 126 127)
fi

CKPT=nba_data_pipeline/checkpoints/finetune_manual_best.pt
HANDLER=models/handler_detector.pt

echo "=== 批次跑 SAM3：${VIDS[*]} ==="
date

for vid in "${VIDS[@]}"; do
  VIDEO=input_videos/screen/screen_${vid}.mp4
  TRACKS=stubs/screen_${vid}_sam3_tracks.pkl
  OUTVID=output_videos/screens_v${vid}_sam3.avi
  EVENTS=output_videos/screens_v${vid}_sam3_events.pkl

  echo ""
  echo "################## v${vid}  $(date +%H:%M:%S) ##################"

  # --- 步驟 1：SAM3 追蹤（sam3_env, cu13.1）---
  if docker exec sam3_env test -f /workspace/${TRACKS} 2>/dev/null; then
    echo "[v${vid}] tracks 已存在，跳過 SAM3 追蹤：${TRACKS}"
  else
    echo "[v${vid}] SAM3 追蹤 → ${TRACKS}"
    docker exec -w /workspace sam3_env bash -c "
      export PYTHONIOENCODING=utf-8
      python -X utf8 nba_data_pipeline/run_sam3_tracks.py \
        --video ${VIDEO} --out ${TRACKS} --single_gpu
    " || { echo "[v${vid}] SAM3 追蹤失敗，跳過"; continue; }
  fi

  # --- 步驟 2：inference 讀 tracks 跑下游（basketball_analysis）---
  # ⚠ 換追蹤來源後刪舊 stub，避免讀到 SAM2 版分隊結果
  echo "[v${vid}] inference（讀 SAM3 tracks）→ ${EVENTS}"
  docker exec -w /workspace basketball_analysis bash -c "
    export PYTHONIOENCODING=utf-8
    rm -f stubs/player_assignment_stub.pkl
    python -X utf8 nba_data_pipeline/inference.py \
      --video ${VIDEO} \
      --sam3_tracks ${TRACKS} \
      --ckpt ${CKPT} \
      --handler_model ${HANDLER} \
      --output_video ${OUTVID} \
      --threshold 0.70
  " || { echo "[v${vid}] inference 失敗，跳過"; continue; }

  echo "[v${vid}] 完成 → ${EVENTS}"
done

echo ""
echo "=== 全部跑完 $(date) ==="
echo "從本地 scp 回："
echo "  scp 'vipl-spark:/home/vipl/timwu/basketball_analysis/output_videos/screens_v*_sam3_events.pkl' output_videos/"
echo "本地驗： bash verify_hmm_batch.sh ${VIDS[*]}"
