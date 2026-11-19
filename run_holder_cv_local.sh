#!/bin/bash
# 2026-11-19 本機 CPU（pod 等待建立中）：持球通道 + 手遞手頭 vs 對照，同一批 458 支、3 種子 × 5 份
#   court2baseL      ：x, y（對照）
#   court2holdL      ：x, y, 持球通道 + 手遞手頭（我們的 9 個手遞手）
#   court2holdnodhoL ：同上但手遞手 loss 權重 0（把「持球通道」和「手遞手頭」的影響分開）
# 用法：nohup bash run_holder_cv_local.sh > logs/holder_cv.log 2>&1 &
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
job() {  # $1 = tag  $2 = 種子  $3 = 份
  ck=nba_data_pipeline/checkpoints/finetune_contact_$1_s$2_cv$3_best.pt
  [ -f "$ck" ] && return 0
  case $1 in
    court2baseL) extra="--data_dir nba_data_pipeline/output_contact_court2base_503" ;;
    court2holdL) extra="--data_dir nba_data_pipeline/output_contact_court2holder_503" ;;
    court2holdnodhoL) extra="--data_dir nba_data_pipeline/output_contact_court2holder_503 --dho_weight 0" ;;
  esac
  python -u nba_data_pipeline/finetune.py --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
    --out_dir nba_data_pipeline/checkpoints --tag contact_$1_s$2_cv$3 --pairwise --epochs 40 --batch 64 --lr 3e-4 \
    --patience 8 --pos_weight_cap 5 --n_folds 5 --fold $3 --train_seed $2 --flip_aug $extra > logs/holdcv_$1_s$2_cv$3.log 2>&1
  echo "$(date '+%m-%d %H:%M') $1 s$2 cv$3 $([ -f "$ck" ] && echo ok || echo FAIL)"
}
export -f job
# 先兩組主要的
for s in 1 2 3; do for k in 0 1 2 3 4; do for t in court2baseL court2holdL; do echo "$t $s $k"; done; done; done \
  | xargs -P 4 -L 1 bash -c 'job $0 $1 $2'
python -u nba_data_pipeline/five_seed_eval.py --tags court2baseL,court2holdL --seeds 1,2,3 --workers 4 --threads 4 2>&1 | grep -v "曲線完成"
echo MAIN_DONE
# 再分開兩個因素
for s in 1 2 3; do for k in 0 1 2 3 4; do echo "court2holdnodhoL $s $k"; done; done | xargs -P 4 -L 1 bash -c 'job $0 $1 $2'
python -u nba_data_pipeline/five_seed_eval.py --tags court2baseL,court2holdnodhoL --seeds 1,2,3 --workers 4 --threads 4 2>&1 | grep -v "曲線完成"
echo "== 手遞手（各份最後一個 epoch）"
for f in logs/holdcv_court2holdL_s*_cv*.log; do echo "$f $(grep -a '手遞手（真的掩護' "$f" | tail -1)"; done
echo HOLDER_CV_DONE
