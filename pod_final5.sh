 #!/bin/bash
# final5（2026-11-13）：補標「同一掩護者的第二次擋拆」4 個（44 / 146 / 156 / 157，missing_label_check）+ 修正標記第 3 輪（final4 沒包含）
#   1. 修正寫進訓練快取 2. 重建 ScreenNet 訓練資料（court2fix4）+ GRU 序列快取 3. ScreenNet（final4 設定：翻轉增強、17 epoch、全部資料）× 3
#   4. GRU（起點 final5、9 輪、正樣本權重 10、門檻 0.7）× 3 → checkpoints/finetune_seq_final5gru_s{1,2,3}_best.pt
# 用法：nohup bash pod_final5.sh > logs/final5.log 2>&1 &
cd "$(dirname "$0")"
source sam3_env.sh
export HF_HUB_OFFLINE=1
set -e
python -u nba_data_pipeline/apply_cache_fixes.py
python -u nba_data_pipeline/build_court2.py output_contact_court2fix4_503
python -u nba_data_pipeline/finetune_seq.py --build --data stubs/seq_data_fix4.pkl
set +e
for s in 1 2 3; do
  python -u nba_data_pipeline/finetune.py --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
    --data_dir nba_data_pipeline/output_contact_court2fix4_503 --out_dir nba_data_pipeline/checkpoints \
    --tag contact_final5_s$s --pairwise --epochs 17 --batch 64 --lr 3e-4 --pos_weight_cap 5 --val_frac 0 --train_seed $s --flip_aug \
    > logs/final5_s$s.log 2>&1 &
done
wait
for s in 1 2 3; do
  python -u nba_data_pipeline/finetune_seq.py --final --init final5 --seed $s --tag final5gru --epochs 9 \
    --pos_weight 10 --flip_aug --threshold 0.7 --data stubs/seq_data_fix4.pkl > logs/final5gru_s$s.log 2>&1 &
done
wait
ls -la nba_data_pipeline/checkpoints/finetune_contact_final5_s*_best.pt nba_data_pipeline/checkpoints/finetune_seq_final5gru_s*_best.pt
echo FINAL5_DONE
