#!/bin/bash
# 一支影片跑完 GPU 部分：SAM3（擋拆 5fps + 投籃 10fps 追球）→ inference（擋拆 + 投籃 + 球場座標第 2 版）
# 用法：bash run_pipeline.sh input_videos/xxx.mp4
# 輸出：output_videos/pipeline/<名稱>.mp4（標註影片）、<名稱>_events.pkl（給 tactics_tools / tactics_agent）
set -e
cd "$(dirname "$0")"     # 腳本所在的資料夾（/mnt/data/basketball_analysis 或 /root/basketball_analysis）
V="$1"; [ -f "$V" ] || { echo "找不到影片 $V"; exit 1; }
NAME=$(basename "$V" .mp4)
source sam3_env.sh
export HF_HUB_OFFLINE=1
CK=nba_data_pipeline/checkpoints
mkdir -p stubs/pipeline output_videos/pipeline
echo "== 1/2 SAM3 追蹤（球員 + 球）"
python -u nba_data_pipeline/run_sam3_tracks.py --video "$V" --out stubs/pipeline/${NAME}_sam3.pkl --single_gpu --ball \
  2>&1 | grep -E "抽出|只追蹤|投籃偵測|SAM3 tracked|裁判排除|Error|Traceback" || true
echo "== 2/2 推論（擋拆 + 投籃）"
python -u nba_data_pipeline/inference.py --video "$V" --sam3_tracks stubs/pipeline/${NAME}_sam3.pkl \
  --ckpt $CK/finetune_contact_final2_s1_best.pt $CK/finetune_contact_final2_s2_best.pt $CK/finetune_contact_final2_s3_best.pt \
  --output_video output_videos/pipeline/${NAME}.mp4 --stub_dir stubs/pipeline_${NAME} \
  2>&1 | grep -E "第 2 版|have homography|Detected|撞上|鏡頭切點|出手|^    f|事件資料 dump|Error|Traceback" || true
echo "✅ 完成：output_videos/pipeline/${NAME}.mp4、output_videos/pipeline/${NAME}_events.pkl"
