#!/bin/bash
# 查 pod 上實驗進度：ssh -p <port> root@140.119.163.112 bash /mnt/data/basketball_analysis/status.sh
cd "$(dirname "$0")"     # 腳本所在的資料夾（/mnt/data/basketball_analysis 或 /root/basketball_analysis）
ck() { ls nba_data_pipeline/checkpoints | grep -c "contact_$1_s[1-5]_cv"; }
res() {  # 最後一份評估的 ±0.4 秒 / ±1 秒 結果
  grep -A3 "±0.4 秒\|±1 秒" "$1" 2>/dev/null | grep -E "秒 =====|3 個平均|同一組" | tail -6
}
echo "===== $(date '+%m-%d %H:%M') ====="
echo "[逐窗持球者] 訓練 $(ck poss)/25；$(grep -q POSS_DONE logs/poss_eval.log 2>/dev/null && echo '✅ 評估完成' || echo '評估未完成')"
grep -q POSS_DONE logs/poss_eval.log 2>/dev/null && res logs/poss_eval.log
echo "[座標第 2 版] 訓練 $(ck court2)/25；$(grep -q COURT2_DONE logs/court2_eval.log 2>/dev/null && echo '✅ 評估完成' || echo '評估未完成')"
grep -q COURT2_DONE logs/court2_eval.log 2>/dev/null && res logs/court2_eval.log
for f in logs/poss_failed logs/court2_failed; do [ -f $f ] && echo "❌ $f"; done
echo "[GPU] $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
