#!/bin/bash
# 2026-11-16 本機 CPU（pod 連不上）：ScreenNet 只算持球者那 4 組 loss（--mask_offpairs）vs 原本，court2fix2 + 翻轉增強，3 種子 × 5 份
# 用法：nohup bash run_mask_cv_local.sh > logs/mask_cv.log 2>&1 &
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
job() {  # $1 = tag  $2 = 種子  $3 = 份
  ck=nba_data_pipeline/checkpoints/finetune_contact_$1_s$2_cv$3_best.pt
  [ -f "$ck" ] && return 0
  extra=""; [ "$1" = court2fix2maskL ] && extra="--mask_offpairs"
  python -u nba_data_pipeline/finetune.py --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
    --data_dir nba_data_pipeline/output_contact_court2fix2_503 --out_dir nba_data_pipeline/checkpoints \
    --tag contact_$1_s$2_cv$3 --pairwise --epochs 40 --batch 64 --lr 3e-4 --patience 8 --pos_weight_cap 5 \
    --n_folds 5 --fold $3 --train_seed $2 --flip_aug $extra > logs/maskcv_$1_s$2_cv$3.log 2>&1
  echo "$(date '+%m-%d %H:%M') $1 s$2 cv$3 $([ -f "$ck" ] && echo ok || echo FAIL)"
}
export -f job
for s in 1 2 3; do for k in 0 1 2 3 4; do for t in court2fix2flipL court2fix2maskL; do echo "$t $s $k"; done; done; done \
  | xargs -P 4 -L 1 bash -c 'job $0 $1 $2'
python -u nba_data_pipeline/five_seed_eval.py --tags court2fix2flipL,court2fix2maskL --seeds 1,2,3 --workers 4 --threads 4
echo MASK_CV_DONE
