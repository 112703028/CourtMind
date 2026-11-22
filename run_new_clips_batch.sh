#!/bin/bash
# 2026-11-22 新片段（使用者從 YouTube 挑的）跑整條流程：input_videos/offball（無球掩護 105 段）+ input_videos/dho（手遞手 29 段）
#   每段 run_pipeline.sh（SAM3 → 推論 → 傳球 / 手遞手手腕證據）→ output_videos/pipeline/<片段>_events.pkl；做過的跳過
# 用法：nohup bash run_new_clips_batch.sh > logs/new_clips.log 2>&1 &
cd "$(dirname "$0")"
command -v ffmpeg > /dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) > /dev/null 2>&1
mkdir -p logs
for V in input_videos/dho/*.mp4 input_videos/offball/*.mp4; do
  N=$(basename "$V" .mp4)
  grep -q "^$N$" logs/new_clips_done.txt 2>/dev/null && continue
  echo "=== $(date '+%m-%d %H:%M') $N"
  bash run_pipeline.sh "$V" > logs/new_clips_$N.log 2>&1 && echo $N >> logs/new_clips_done.txt || echo "  ❌ $N 失敗（logs/new_clips_$N.log）"
done
echo "完成 $(wc -l < logs/new_clips_done.txt) 段"
echo NEW_CLIPS_DONE
