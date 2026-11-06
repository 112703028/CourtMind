#!/bin/bash
# A2：推論翻轉平均（court2fix2 現有 25 個交叉驗證模型）；A1a：翻轉增強重訓（court2fix2flip，5 種子 × 5 份）
cd "$(dirname "$0")"
source sam3_env.sh
rm -f logs/a12_*
python -u nba_data_pipeline/five_seed_eval.py --tags court2fix2,court2fix2+tta --workers 6 --threads 2 > logs/a12_tta_eval.log 2>&1 &
run() {  # $1 = 種子
  for k in 0 1 2 3 4; do
    python -u nba_data_pipeline/finetune.py --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
      --data_dir nba_data_pipeline/output_contact_court2fix2_503 \
      --out_dir nba_data_pipeline/checkpoints --tag contact_court2fix2flip_s$1_cv$k --pairwise --epochs 40 --batch 64 \
      --lr 3e-4 --patience 8 --pos_weight_cap 5 --n_folds 5 --fold $k --train_seed $1 --flip_aug || return 1
  done
}
for s in 1 2 3 4 5; do run $s > logs/a12_flip_s$s.log 2>&1 & done
wait
echo TTA_DONE >> logs/a12_tta_eval.log
python -u nba_data_pipeline/five_seed_eval.py --tags court2fix2,court2fix2flip --workers 10 --threads 2 > logs/a12_flip_eval.log 2>&1
echo FLIP_DONE >> logs/a12_flip_eval.log
