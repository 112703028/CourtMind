#!/bin/bash
# 在 GPU pod 上跑 coverage_net.py（ScreenNet 主幹 + 防守應對分類頭）
# 用法（pod 上，專案資料夾）：
#   tar -xf /tmp/covnet_bundle.tar                       # 第一次：解開本地打包的程式 + 資料
#   nohup bash run_coverage_net.sh frozen,finetune 5 > logs/covnet.log 2>&1 &
#   tail -f logs/covnet.log                              # 看進度（Ctrl+C 只停止查看）
# 參數：$1 = 訓練方式（frozen / finetune，逗號分隔），$2 = 切法數（預設 5）
cd "$(dirname "$0")"
source sam3_env.sh
mkdir -p logs
rm -f logs/covnet_done logs/covnet_failed
python -u nba_data_pipeline/coverage_net.py --modes "${1:-frozen,finetune}" --seeds "${2:-5}" \
  && touch logs/covnet_done || touch logs/covnet_failed
