#!/usr/bin/env bash
# rerun_inference_batch.sh — 只重跑 inference（讀現成 SAM3 tracks），不重跑 SAM3 追蹤。
#
# 用途：改了 inference.py（如加幀號標示、修 fps）後，重出標框影片 + events pkl。
# SAM3 tracks 已在 stubs/，這步只跑下游（court/分隊/擋拆/畫框），快。
#
# 放在 vipl-spark 主機 /home/vipl/timwu/basketball_analysis/ 執行。
# 用法：
#   cd /home/vipl/timwu/basketball_analysis
#   nohup bash rerun_inference_batch.sh 116 117 118 119 121 122 123 124 126 127 129 130 > rerun.log 2>&1 &
#   tail -f rerun.log

set -u
VIDS=("$@")
if [ ${#VIDS[@]} -eq 0 ]; then
  VIDS=(116 117 118 119 121 122 123 124 126 127 129 130)
fi

CKPT=nba_data_pipeline/checkpoints/finetune_manual_best.pt
HANDLER=models/handler_detector.pt

echo "=== 重跑 inference（讀現成 SAM3 tracks）：${VIDS[*]} ==="
date

for vid in "${VIDS[@]}"; do
  VIDEO=input_videos/screen/screen_${vid}.mp4
  TRACKS=stubs/screen_${vid}_sam3_tracks.pkl
  OUTVID=output_videos/screens_v${vid}_sam3.avi
  EVENTS=output_videos/screens_v${vid}_sam3_events.pkl

  echo ""
  echo "################## v${vid}  $(date +%H:%M:%S) ##################"

  if ! docker exec basketball_analysis test -f /workspace/${TRACKS} 2>/dev/null; then
    echo "[v${vid}] ⚠ tracks 不存在：${TRACKS}，跳過（需先跑 SAM3 追蹤）"
    continue
  fi

  # ⚠ 換追蹤來源/改分隊後刪舊 stub，避免讀到舊分隊結果
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

  echo "[v${vid}] 完成 → ${EVENTS} + ${OUTVID%.avi}.mp4（含幀號）"
done

echo ""
echo "=== 全部重跑完 $(date) ==="
echo "從本地 scp 回影片 + pkl："
echo "  scp 'vipl-spark:/home/vipl/timwu/basketball_analysis/output_videos/screens_v*_sam3.mp4' output_videos/"
echo "  scp 'vipl-spark:/home/vipl/timwu/basketball_analysis/output_videos/screens_v*_sam3_events.pkl' output_videos/"
