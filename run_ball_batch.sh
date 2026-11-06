#!/bin/bash
# 原片段補追球（持球者改用 SAM3 追球 + Viterbi 用）：SAM3 只追 10fps 球員 + 球（--ball_only，不動擋拆追蹤）
# → Roboflow 選球 + 碰球的人 → stubs/ball_batch/<片段>_touch.pkl。做過的跳過，斷線重跑會接著做。
# 用法：nohup bash run_ball_batch.sh > logs/ball_batch.log 2>&1 &
cd "$(dirname "$0")"
source sam3_env.sh
export HF_HUB_OFFLINE=1
command -v ffmpeg > /dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) > /dev/null 2>&1
mkdir -p stubs/ball_batch logs
for V in $(ls input_videos/screen/*.mp4 | sort -V); do
  N=$(basename "$V" .mp4)
  [ -f stubs/ball_batch/${N}_touch.pkl ] && continue
  echo "=== $(date '+%m-%d %H:%M') $N"
  if [ ! -f stubs/ball_batch/${N}_ball.pkl ]; then
    python -u nba_data_pipeline/run_sam3_tracks.py --video "$V" --out stubs/ball_batch/${N}_ball.pkl --single_gpu --ball_only \
      2>&1 | grep -E "只追球|Error|Traceback" || true
  fi
  [ -f stubs/ball_batch/${N}_ball.pkl ] || { echo "  ❌ SAM3 失敗"; continue; }
  python -u nba_data_pipeline/possession_sam3.py prep "$V" stubs/ball_batch/${N}_ball.pkl stubs/ball_batch/${N}_touch.pkl \
    2>&1 | grep -E "Error|Traceback" || true
  [ -f stubs/ball_batch/${N}_touch.pkl ] && echo "  ✓" || echo "  ❌ 碰球計算失敗"
done
echo "BALL_DONE $(date '+%m-%d %H:%M')"
