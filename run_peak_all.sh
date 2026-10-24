#!/usr/bin/env bash
# run_peak_all.sh — 批次對多支影片跑「SAM3 追蹤（沒有才補）+ 新 ScreenNet 尖峰偵測」。
#
# 在 vipl-spark 主機 /home/vipl/timwu/basketball_analysis/ 執行（掛載成兩個 container 的 /workspace）。
#   步驟 1：stubs/screen_N_sam3_tracks.pkl 不存在 → sam3_env 跑 SAM3（要 GPU）
#   步驟 2：output_videos/screens_vN_peak_events.pkl 不存在 → basketball_analysis 跑 inference（尖峰模式）
# 跑過的自動跳過，中斷了重新執行就會接著跑。
#
# 用法：
#   bash run_peak_all.sh nba_data_pipeline/val_clips.txt     # 先跑 21 支驗證用影片
#   bash run_peak_all.sh nba_data_pipeline/train_clips.txt   # 全部 146 支
# 背景跑：
#   nohup bash run_peak_all.sh nba_data_pipeline/val_clips.txt > peak_all.log 2>&1 &
#
# 環境變數：
#   INFER_CPU=1（預設）推論用 CPU（GPU 被佔時）；INFER_CPU=0 用 GPU
#   PEAK_THR=0.6（預設）尖峰門檻

set -u
LIST=${1:-nba_data_pipeline/val_clips.txt}
INFER_CPU=${INFER_CPU:-1}
PEAK_THR=${PEAK_THR:-0.6}
CKPT=nba_data_pipeline/checkpoints/finetune_contact_clean_final_best.pt
GPU_FLAG=""
[ "$INFER_CPU" = "1" ] && GPU_FLAG="-e CUDA_VISIBLE_DEVICES="

echo "=== $(date +%H:%M) 開始：$(grep -c . "$LIST") 支（清單 $LIST）==="
for clip in $(tr -d '\r' < "$LIST"); do
  v=${clip#screen_}
  VIDEO=input_videos/screen/screen_${v}.mp4
  TRACKS=stubs/screen_${v}_sam3_tracks.pkl
  EVENTS=output_videos/screens_v${v}_peak_events.pkl

  if [ ! -f "$VIDEO" ]; then echo "[v$v] 沒有影片，跳過"; continue; fi
  if [ -f "$EVENTS" ]; then echo "[v$v] 已跑過，跳過"; continue; fi

  echo ""
  echo "########## v$v  $(date +%H:%M) ##########"

  # AV1 影片：container 的影片讀取解不好（screen_18 448 幀只讀到 15 幀）→ 先轉 H.264。
  # passthrough 保留每一幀，幀號跟原片（使用者標記時看的）一致。
  CODEC=$(docker exec -w /workspace sam3_env ffprobe -v error -select_streams v:0 \
          -show_entries stream=codec_name -of csv=p=0 "$VIDEO" | tr -d '\r')
  if [ "$CODEC" = "av1" ]; then
    H264=input_videos/screen_h264/screen_${v}.mp4
    if [ ! -f "$H264" ]; then
      echo "[v$v] AV1 → 轉 H.264..."
      docker exec -w /workspace sam3_env bash -c "mkdir -p input_videos/screen_h264 && \
        ffmpeg -y -loglevel error -i $VIDEO -c:v libx264 -crf 18 -preset fast -fps_mode passthrough -an $H264" \
        || { echo "[v$v] 轉檔失敗，跳過"; continue; }
    fi
    VIDEO=$H264
  fi

  # 步驟 1：SAM3 追蹤（要 GPU；記憶體不夠會失敗 → 跳過這支，之後重跑會再試）
  if [ ! -f "$TRACKS" ]; then
    echo "[v$v] SAM3 追蹤..."
    docker exec -w /workspace sam3_env python -u -X utf8 nba_data_pipeline/run_sam3_tracks.py \
      --video "$VIDEO" --out "$TRACKS" --single_gpu \
      || { echo "[v$v] SAM3 失敗（GPU 記憶體不夠？），跳過"; continue; }
  fi

  # 步驟 2：尖峰模式偵測擋拆（--no_stub：不讀別支影片留下的快取）
  echo "[v$v] 偵測擋拆（尖峰模式）..."
  docker exec $GPU_FLAG -w /workspace basketball_analysis python -u -X utf8 nba_data_pipeline/inference.py \
    --video "$VIDEO" --sam3_tracks "$TRACKS" --ckpt "$CKPT" \
    --handler_model models/handler_detector.pt \
    --output_video output_videos/screens_v${v}_peak.avi \
    --peak_threshold "$PEAK_THR" --no_stub \
    || { echo "[v$v] 推論失敗，跳過"; continue; }
  echo "[v$v] 完成"
done
echo ""
echo "=== $(date +%H:%M) 全部完成 ==="
