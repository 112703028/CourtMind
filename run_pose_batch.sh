#!/bin/bash
# 原片段補手腕姿態（傳球偵測用）：有 stubs/ball_batch/<片段>_ball.pkl 的片段跑 pose_hands → stubs/pose_hands/<片段>.pkl。
# 做過的跳過，斷線重跑會接著做。傳球本身（pose_pass）在本地算：拉回 pose_hands/*.pkl 就好（很小）。
# 用法：nohup bash run_pose_batch.sh > logs/pose_batch.log 2>&1 &
cd "$(dirname "$0")"
source sam3_env.sh
mkdir -p stubs/pose_hands logs
CLIPS=$(ls stubs/ball_batch/screen_*_touch.pkl | sed 's/.*screen_//;s/_touch.pkl//' | sort -n | tr '\n' ',' | sed 's/,$//')
python -u nba_data_pipeline/pose_hands.py --clips "$CLIPS" 2>&1 | grep -vE "Warning|warn"
echo "POSE_DONE $(date '+%m-%d %H:%M')"
