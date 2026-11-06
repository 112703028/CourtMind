#!/bin/bash
# A1b：混 SportVU（雜訊對齊轉播實測，output_contact_nbabc，只進訓練）→ court2fix2nbabc，5 種子 × 5 份
cd "$(dirname "$0")"
source sam3_env.sh
rm -f logs/a1b_*
run() {  # $1 = 種子
  for k in 0 1 2 3 4; do
    python -u nba_data_pipeline/finetune.py --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
      --data_dir nba_data_pipeline/output_contact_court2fix2_503 --extra_data_dir nba_data_pipeline/output_contact_nbabc \
      --out_dir nba_data_pipeline/checkpoints --tag contact_court2fix2nbabc_s$1_cv$k --pairwise --epochs 40 --batch 64 \
      --lr 3e-4 --patience 8 --pos_weight_cap 5 --n_folds 5 --fold $k --train_seed $1 || return 1
  done
}
for s in 1 2 3 4 5; do run $s > logs/a1b_s$s.log 2>&1 & done
wait
# 等 A1a 的評估寫完曲線快取再評（同一個 five_seed_curves.pkl，避免兩個程式同時寫）
until grep -q FLIP_DONE logs/a12_flip_eval.log 2>/dev/null; do sleep 60; done
python -u nba_data_pipeline/five_seed_eval.py --tags court2fix2,court2fix2nbabc --workers 10 --threads 2 > logs/a1b_eval.log 2>&1
echo NBABC_DONE >> logs/a1b_eval.log
