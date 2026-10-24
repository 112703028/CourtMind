#!/bin/bash
# run_new_clips.sh — 在 GPU pod 上一鍵處理一支 YouTube 影片的擋拆片段：
#   切片段 + 5fps 影格 → SAM3 追蹤 → 前處理（持球者 / 分隊 / 球場座標）→ ScreenNet 預測 → 30fps 確認影片 + 表格
# 用法（在專案資料夾，sam3_env.sh 由 setup_pod.sh 產生）：
#   bash run_new_clips.sh <YouTube 網址> <時間段檔> <起始編號>
#   bash run_new_clips.sh "https://www.youtube.com/watch?v=xxxx" nba_data_pipeline/new_clips_xxxx.txt 283
# 下載失敗（YouTube 擋機房 IP）時：在本地下載 input_videos/yt_src/<id>.mp4 傳上來，再重跑（會跳過下載）
set -e
source "$(cd "$(dirname "$0")" && pwd)/sam3_env.sh"
URL="$1"; RANGES="$2"; START="$3"
if [ -z "$START" ]; then echo "用法：bash run_new_clips.sh <網址> <時間段檔> <起始編號>"; exit 1; fi
N=$(grep -cvE '^\s*(#|$)' "$RANGES")
END=$((START + N - 1))
echo "== screen_${START} ~ screen_${END}（${N} 段）"
# 給本地 run_on_pod.ps1 看的狀態記號：成功 → logs/done_起_迄，中途出錯 → logs/failed_起_迄
mkdir -p logs
rm -f "logs/done_${START}_${END}" "logs/failed_${START}_${END}"
trap 'rc=$?; [ $rc -ne 0 ] && touch "logs/failed_${START}_${END}"' EXIT

python -u nba_data_pipeline/make_new_clips.py --url "$URL" --ranges "$RANGES" --start_id "$START"

LIST=/tmp/clips_${START}_${END}.txt
seq "$START" "$END" | sed 's/^/screen_/' > "$LIST"
python -u nba_data_pipeline/run_sam3_tracks.py --frames_root input_videos/screen_frames \
    --clips_file "$LIST" --out_dir stubs/train_sam3 --single_gpu

python -u nba_data_pipeline/prep_new_clips.py --start "$START" --end "$END" --device cuda --render30
echo "== 完成：output_videos/new_clip_check/（screen_${START}~${END} 的影片、screens_${START}_${END}.csv、clips_${START}_${END}.csv）"
touch "logs/done_${START}_${END}"