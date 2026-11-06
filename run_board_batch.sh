#!/bin/bash
# 全部影片批次：SAM3 → 推論（擋拆 GRU + 分隊塌掉修正 + 存球員框）→ 動態戰術板（原片畫框 + 編號）
# 先跑延長片段（screen_ext，含追球 / 投籃），再跑原片段（screen，不追球）。做過的跳過 → pod 斷線後重跑會接著做。
# 用法：nohup bash run_board_batch.sh > logs/board_batch.log 2>&1 &
# 輸出：output_videos/batch/<名稱>_events.pkl、output_videos/board_batch/<名稱>_anim.mp4（延長片段名稱前面加 ext_）
cd "$(dirname "$0")"
source sam3_env.sh
export HF_HUB_OFFLINE=1
[ -f /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc ] || (apt-get update -qq && apt-get install -y -qq fonts-noto-cjk) > /dev/null 2>&1
command -v ffmpeg > /dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) > /dev/null 2>&1
CK=nba_data_pipeline/checkpoints
GRU="$CK/finetune_seq_final4gru_s1_best.pt $CK/finetune_seq_final4gru_s2_best.pt $CK/finetune_seq_final4gru_s3_best.pt"
mkdir -p stubs/batch output_videos/batch output_videos/board_batch logs
one() {  # $1 = 影片、$2 = 輸出名稱、$3 = --ball 或空
  local V=$1 N=$2 BALL=$3
  [ -f output_videos/board_batch/${N}_anim.mp4 ] && return 0
  echo "=== $(date '+%m-%d %H:%M') $N"
  if [ ! -f stubs/batch/${N}_sam3.pkl ]; then
    python -u nba_data_pipeline/run_sam3_tracks.py --video "$V" --out stubs/batch/${N}_sam3.pkl --single_gpu $BALL \
      2>&1 | grep -E "SAM3 tracked|Error|Traceback" || true
  fi
  [ -f stubs/batch/${N}_sam3.pkl ] || { echo "  ❌ SAM3 失敗"; return 1; }
  if [ ! -f output_videos/batch/${N}_events.pkl ]; then
    python -u nba_data_pipeline/inference.py --video "$V" --sam3_tracks stubs/batch/${N}_sam3.pkl --ckpt $GRU \
      --output_video output_videos/batch/${N}.mp4 --stub_dir stubs/batch_${N} 2>&1 \
      | grep -E "分隊檢查|Detected|出手 [0-9]|Error|Traceback" || true
    rm -f output_videos/batch/${N}.mp4                     # 推論自己的標註影片不需要（動態板會畫）
  fi
  [ -f output_videos/batch/${N}_events.pkl ] || { echo "  ❌ 推論失敗"; return 1; }
  python -u nba_data_pipeline/tactics_board_anim.py output_videos/batch/${N}_events.pkl --video "$V" \
    --out_dir output_videos/board_batch --no_vlm 2>&1 | grep -E "→|Error|Traceback" || true
}
for V in $(ls input_videos/screen_ext/*.mp4 | sort -V); do one "$V" ext_$(basename "$V" .mp4) --ball; done
for V in $(ls input_videos/screen/*.mp4 | sort -V); do one "$V" $(basename "$V" .mp4) ""; done
echo "BATCH_DONE $(date '+%m-%d %H:%M')"
