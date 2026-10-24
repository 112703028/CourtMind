#!/usr/bin/env bash
# run_peak_5fps.sh — 用「每秒 5 張的訓練影格」跑新 ScreenNet 尖峰偵測，完全不用跑 SAM3（不需要 GPU）。
#
# 為什麼可以：新模型本來就只吃 5fps；stubs/train_sam3/ 已有每支 clip 在 5fps 影格上的 SAM3 追蹤
# （做訓練資料時跑的）。這裡把 input_videos/screen_frames/<clip>/ 的影格組成 5fps 影片，
# 推論時直接讀那份追蹤，其餘（持球者偵測、分隊、球場座標、擋拆偵測）用 CPU 跑。
#
# 注意：輸出的幀號是 5fps 影格索引（= contact_labels.json 的 idx），不是原片幀號。
#       原片幀號 ≈ 索引 × (原片 fps / 5)。
#
# 在 vipl-spark 主機 /home/vipl/timwu/basketball_analysis/ 執行：
#   nohup bash run_peak_5fps.sh nba_data_pipeline/val_clips.txt > peak5.log 2>&1 &

set -u
LIST=${1:-nba_data_pipeline/val_clips.txt}
PEAK_THR=${PEAK_THR:-0.6}
CKPT=nba_data_pipeline/checkpoints/finetune_contact_clean_final_best.pt

echo "=== $(date +%H:%M) 開始：$(grep -c . "$LIST") 支（5fps 影格，不跑 SAM3）==="
for clip in $(tr -d '\r' < "$LIST"); do
  v=${clip#screen_}
  FRAMES=input_videos/screen_frames/${clip}
  V5=input_videos/screen_5fps/${clip}.mp4
  TRACKS=stubs/train_sam3/${clip}_sam3_tracks.pkl
  EVENTS=output_videos/screens_v${v}_peak5_events.pkl

  [ -f "$EVENTS" ] && { echo "[v$v] 已跑過，跳過"; continue; }
  [ -f "$TRACKS" ] || { echo "[v$v] 沒有 5fps 追蹤 $TRACKS，跳過"; continue; }
  [ -d "$FRAMES" ] || { echo "[v$v] 沒有影格 $FRAMES，跳過"; continue; }

  echo ""
  echo "########## v$v  $(date +%H:%M) ##########"

  # 1. 影格 → 5fps 影片（用 sam3_env 裡的 ffmpeg，不需要 GPU）
  if [ ! -f "$V5" ]; then
    docker exec -w /workspace sam3_env bash -c "mkdir -p input_videos/screen_5fps && \
      ffmpeg -y -loglevel error -framerate 5 -start_number 0 -i ${FRAMES}/${clip}_%04d.jpg \
             -c:v libx264 -crf 18 -pix_fmt yuv420p $V5" \
      || { echo "[v$v] 組影片失敗，跳過"; continue; }
  fi

  # 2. 推論（CPU）。持球者平滑的幀數參數改成 5fps 的尺度：
  #    --ba_min_run 1（不刪短持球；30fps 的 4 幀在 5fps 會變 0.8 秒）、--ba_fill_gap 2（補 0.4 秒內的空洞）
  docker exec -e CUDA_VISIBLE_DEVICES= -w /workspace basketball_analysis python -u -X utf8 \
    nba_data_pipeline/inference.py \
    --video "$V5" --sam3_tracks "$TRACKS" --ckpt "$CKPT" \
    --handler_model models/handler_detector.pt \
    --output_video output_videos/screens_v${v}_peak5.avi \
    --peak_threshold "$PEAK_THR" --ba_min_run 1 --ba_fill_gap 2 --no_stub \
    || { echo "[v$v] 推論失敗，跳過"; continue; }
  echo "[v$v] 完成"
done
echo ""
echo "=== $(date +%H:%M) 全部完成 ==="
