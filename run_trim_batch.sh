#!/bin/bash
# 片頭剪掉重跑（2026-11-14）：clip_trim.json 的片段（使用者標「從第幾張開始保留」）重跑整條流程
#   SAM3（剪片頭、--ball）→ 編號換回舊的（remap_tracks，舊 = output_videos/batch 的 events.pkl）→ 推論（final5bgru + 救回 + Viterbi 持球者 + 投籃）
#   → 選球 / 碰球 → 手腕骨架 → 傳球。輸出 stubs/trim/<片段>_{sam3,touch,pose}.pkl、output_videos/trim/<片段>_events.pkl（+ 籃框放大圖）
# 用法：nohup bash run_trim_batch.sh [片段 ...] > logs/trim_batch.log 2>&1 &     # 不給片段 = clip_trim.json 全部；做過的跳過
cd "$(dirname "$0")"
source sam3_env.sh
export HF_HUB_OFFLINE=1
command -v ffmpeg > /dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) > /dev/null 2>&1
CK=nba_data_pipeline/checkpoints
GRU="$CK/finetune_seq_final5bgru_s1_best.pt $CK/finetune_seq_final5bgru_s2_best.pt $CK/finetune_seq_final5bgru_s3_best.pt"
mkdir -p stubs/trim output_videos/trim logs
CLIPS="$@"
[ -z "$CLIPS" ] && CLIPS=$(python -c "import json; print(' '.join(k for k in json.load(open('nba_data_pipeline/clip_trim.json', encoding='utf-8')) if not k.startswith('_')))")
for N in $CLIPS; do
  V=input_videos/screen/$N.mp4
  [ -f output_videos/trim/${N}_events.pkl ] && grep -q "^$N$" logs/trim_done.txt 2>/dev/null && continue
  echo "=== $(date '+%m-%d %H:%M') $N"
  if [ ! -f stubs/trim/${N}_sam3.pkl ]; then
    python -u nba_data_pipeline/run_sam3_tracks.py --video $V --out stubs/trim/${N}_sam3.pkl --single_gpu --ball \
      2>&1 | grep -E "片頭剪掉|SAM3 tracked|Error|Traceback" || true
    [ -f stubs/trim/${N}_sam3.pkl ] || { echo "  ❌ SAM3 失敗"; continue; }
    python -u nba_data_pipeline/remap_tracks.py stubs/trim/${N}_sam3.pkl output_videos/batch/${N}_events.pkl
  fi
  python -u nba_data_pipeline/inference.py --video $V --sam3_tracks stubs/trim/${N}_sam3.pkl --ckpt $GRU \
    --output_video output_videos/trim/${N}.mp4 --stub_dir stubs/trim_${N} --team_prior output_videos/batch/${N}_events.pkl \
    2>&1 | grep -E "沒持球者|分隊不確定|Detected|撞上|出手|事件資料 dump|Error|Traceback" || true
  [ -f output_videos/trim/${N}_events.pkl ] || { echo "  ❌ 推論失敗"; continue; }
  ( python -u nba_data_pipeline/possession_sam3.py prep $V stubs/trim/${N}_sam3.pkl stubs/trim/${N}_touch.pkl
    python -u nba_data_pipeline/pose_hands.py --ball_pkl stubs/trim/${N}_sam3.pkl --video $V --out stubs/trim/${N}_pose.pkl
    python -u nba_data_pipeline/pose_pass.py add output_videos/trim/${N}_events.pkl stubs/trim/${N}_sam3.pkl \
      stubs/trim/${N}_touch.pkl stubs/trim/${N}_pose.pkl $V ) 2>&1 | grep -E "骨架|傳球|Error|Traceback" || true
  rm -rf stubs/trim_${N}
  echo $N >> logs/trim_done.txt
done
echo TRIM_DONE
