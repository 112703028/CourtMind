#!/bin/bash
# 2026-11-19 手遞手 + 持球通道實驗（交叉驗證 3 種子 × 5 份，同一批片段 = 目前訓練快取 458 支）
#   base   ：court2base（x, y；不加手遞手）                                  ← 對照
#   hold   ：court2holder（x, y, 持球通道 + 手遞手頭；我們自己的 9 個手遞手）    ← 持球通道對擋拆有沒有幫助
#   holdnodho：hold 但手遞手 loss 權重 0                                         ← 把持球通道和手遞手頭的影響分開
#   holdnba：hold + NBA 手遞手弱標籤（output_contact_nbadho，只進訓練）          ← NETS 的弱標籤
#   seqnba ：先只用 NBA 訓練（nbastage1），再用自己的標記接著訓練（--init_ckpt）  ← NETS 最好的順序
# 評估：擋拆 F1 用 five_seed_eval.py（本機也能跑，要 output_videos/batch 的 events.pkl 算持球通道）；手遞手看各 log 的 [Val] 手遞手
# 用法：nohup bash pod_dho.sh > logs/dho.log 2>&1 &
cd "$(dirname "$0")"
source sam3_env.sh
CK=nba_data_pipeline/checkpoints
PRE=$CK/finetune_weak_best.pt
COMMON="--pairwise --epochs 40 --batch 64 --lr 3e-4 --patience 8 --pos_weight_cap 5 --n_folds 5 --flip_aug"
mkdir -p logs
run() {  # $1 = tag  $2 = 種子  $3... = 額外參數
  tag=$1; s=$2; shift 2
  for k in 0 1 2 3 4; do
    [ -f $CK/finetune_contact_${tag}_s${s}_cv${k}_best.pt ] && continue
    python -u nba_data_pipeline/finetune.py --pretrain_ckpt $PRE --out_dir $CK --tag contact_${tag}_s${s}_cv${k} \
      $COMMON --fold $k --train_seed $s "$@" > logs/dho_${tag}_s${s}_cv${k}.log 2>&1
  done
}
# 第 1 段：NBA 單獨訓練（固定 8 個 epoch、沒有 val）→ seqnba 的起點
if [ ! -f $CK/finetune_nbastage1_best.pt ]; then
  python -u nba_data_pipeline/finetune.py --pretrain_ckpt $PRE --out_dir $CK --tag nbastage1 \
    --data_dir nba_data_pipeline/output_contact_nbadho --pairwise --epochs 8 --batch 128 --lr 3e-4 \
    --pos_weight_cap 5 --val_frac 0 --flip_aug --train_seed 1 > logs/dho_nbastage1.log 2>&1
fi
for s in 1 2 3; do
  run base $s --data_dir nba_data_pipeline/output_contact_court2base_503 &
  run hold $s --data_dir nba_data_pipeline/output_contact_court2holder_503 &
  run holdnodho $s --data_dir nba_data_pipeline/output_contact_court2holder_503 --dho_weight 0 &
  wait
  run holdnba $s --data_dir nba_data_pipeline/output_contact_court2holder_503 --extra_data_dir nba_data_pipeline/output_contact_nbadho &
  run seqnba $s --data_dir nba_data_pipeline/output_contact_court2holder_503 --init_ckpt $CK/finetune_nbastage1_best.pt &
  wait
done
echo "訓練完 $(ls $CK | grep -c 'contact_\(base\|hold\|holdnodho\|holdnba\|seqnba\)_s[1-3]_cv') / 75"
for t in hold holdnba seqnba; do
  echo "== $t 最後一個 epoch 的手遞手（各份加總）"
  for f in logs/dho_${t}_s*_cv*.log; do grep -a "手遞手（真的掩護" "$f" | tail -1; done
done
echo DHO_DONE
