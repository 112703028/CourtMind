#!/bin/bash
# 一支影片跑完 GPU 部分：SAM3（擋拆 5fps + 投籃 10fps 追球）→ inference（擋拆 + 投籃 + 球場座標第 2 版）→ 傳球（手腕）
# 擋拆模型：final4（翻轉增強）+ 整段 GRU（finetune_seq.py），解碼 = 連續一段算一次（門檻 0.7、掉到 0.5 以下才結束）
# 要換回不用 GRU 的版本：--ckpt 改成 finetune_contact_final4_s{1,2,3}_best.pt（自動用找峰值 + 3 秒去重）
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
echo "== 1/3 SAM3 追蹤（球員 + 球）"
if [ -f stubs/pipeline/${NAME}_sam3.pkl ] && [ -z "$REDO_SAM3" ]; then echo "  已有 SAM3 結果，跳過（REDO_SAM3=1 重跑）"; else
python -u nba_data_pipeline/run_sam3_tracks.py --video "$V" --out stubs/pipeline/${NAME}_sam3.pkl --single_gpu --ball \
  2>&1 | grep -E "抽出|只追蹤|投籃偵測|SAM3 tracked|裁判排除|Error|Traceback" || true
fi
echo "== 2/3 推論（擋拆 + 投籃）"
python -u nba_data_pipeline/inference.py --video "$V" --sam3_tracks stubs/pipeline/${NAME}_sam3.pkl \
  --ckpt $CK/finetune_seq_final4gru_s1_best.pt $CK/finetune_seq_final4gru_s2_best.pt $CK/finetune_seq_final4gru_s3_best.pt \
  --output_video output_videos/pipeline/${NAME}.mp4 --stub_dir stubs/pipeline_${NAME} \
  2>&1 | grep -E "第 2 版|have homography|Detected|撞上|鏡頭切點|出手|^    f|事件資料 dump|Error|Traceback" || true
for f in stubs/pipeline/${NAME}_sam3.pkl output_videos/pipeline/${NAME}_events.pkl; do
  [ -f "$f" ] || { echo "❌ 失敗：沒有產生 $f（上面的 Traceback 是原因）"; exit 1; }
done
echo "== 3/3 傳球偵測（選球 + 碰球的人 → 每人手腕姿態 → 手腕找持球者 → 傳球，寫進 events.pkl 的 passes）"
# 失敗不影響前面的結果（events.pkl 沒有 passes，tactics_tools / 主播就不提傳球）
( set -e
  [ -f stubs/pipeline/${NAME}_touch.pkl ] || python -u nba_data_pipeline/possession_sam3.py prep "$V" stubs/pipeline/${NAME}_sam3.pkl stubs/pipeline/${NAME}_touch.pkl
  [ -f stubs/pipeline/${NAME}_pose.pkl ] || python -u nba_data_pipeline/pose_hands.py --ball_pkl stubs/pipeline/${NAME}_sam3.pkl --video "$V" --out stubs/pipeline/${NAME}_pose.pkl
  python -u nba_data_pipeline/pose_pass.py add output_videos/pipeline/${NAME}_events.pkl stubs/pipeline/${NAME}_sam3.pkl     stubs/pipeline/${NAME}_touch.pkl stubs/pipeline/${NAME}_pose.pkl "$V"
) 2>&1 | grep -E "骨架|傳球|^  (pass|turnover)|Error|Traceback" || true
python -c "import pickle,sys; sys.exit(0 if 'passes' in pickle.load(open('output_videos/pipeline/${NAME}_events.pkl','rb')) else 1)"   || echo "⚠ 傳球偵測沒成功（擋拆 / 投籃結果不受影響）"
echo "✅ 完成：output_videos/pipeline/${NAME}.mp4、output_videos/pipeline/${NAME}_events.pkl"
