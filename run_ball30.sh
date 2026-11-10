#!/bin/bash
# 30fps 追球實驗（2026-11-10）：抽查 30 支 + 端到端評估出手錯的 → stubs/ball30/<片段>_ball.pkl + Roboflow 30fps（stubs/rf_cache/batch30_*）
# 用法：nohup bash run_ball30.sh > logs/ball30.log 2>&1 &
cd "$(dirname "$0")"
source sam3_env.sh
export HF_HUB_OFFLINE=1
command -v ffmpeg > /dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) > /dev/null 2>&1
mkdir -p stubs/ball30 logs
CLIPS=$(python -c "import json; print(' '.join(json.load(open('output_videos/shot_check_batch/clips.json')) + ['screen_54','screen_117','screen_136','screen_441','screen_34']))")
for N in $CLIPS; do
  [ -f stubs/ball30/${N}_ball.pkl ] && continue
  echo "=== $(date '+%H:%M') $N"
  python -u nba_data_pipeline/run_sam3_tracks.py --video input_videos/screen/${N}.mp4 --out stubs/ball30/${N}_ball.pkl \
    --single_gpu --ball_only --ball_fps 30 2>&1 | grep -E "只追球|Error|Traceback" || true
done
python -u nba_data_pipeline/shot_eval.py cache batch30 e2e30 2>&1 | grep -vi warn | tail -2
echo BALL30_DONE
