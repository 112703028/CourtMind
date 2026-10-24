#!/bin/bash
# 重啟 GPU 保持忙碌（不限時間）
cd /root/basketball_analysis
source sam3_env.sh
pkill -f gpu_keepalive.py
sleep 1
nohup python -u gpu_keepalive.py > logs/keepalive.log 2>&1 < /dev/null &
disown
sleep 5
cat logs/keepalive.log
pgrep -fa gpu_keepalive.py
