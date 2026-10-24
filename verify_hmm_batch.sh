#!/usr/bin/env bash
# verify_hmm_batch.sh — 本地一次驗多支影片的 HMM 防守配對。
# scp 回 events pkl 後跑： bash verify_hmm_batch.sh 116 117 118 119 121 122 123 124 126 127
export PYTHONIOENCODING=utf-8
VIDS=("$@")
if [ ${#VIDS[@]} -eq 0 ]; then
  VIDS=(116 117 118 119 121 122 123 124 126 127)
fi
for v in "${VIDS[@]}"; do
  PKL=output_videos/screens_v${v}_sam3_events.pkl
  if [ -f "$PKL" ]; then
    python -X utf8 nba_data_pipeline/verify_hmm.py --pkl "$PKL"
    echo ""
    echo "############################################################"
    echo ""
  else
    echo "=== v${v}: $PKL 不存在（可能還沒 scp 回或 SAM3 沒產出）==="
    echo ""
  fi
done
