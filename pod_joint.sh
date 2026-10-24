#!/bin/bash
# 新 pod：安裝環境 → GPU 保持忙碌 → 新舊片段一起當考題的交叉驗證（A 只用舊片段訓練 / B 新舊都用）× 3 種子
cd /root/basketball_analysis
mkdir -p logs
rm -f logs/joint_done logs/joint_failed
bash setup_pod.sh > logs/setup.log 2>&1 || { touch logs/joint_failed; echo setup failed; exit 1; }
bash start_keepalive.sh > logs/ka_start.log 2>&1
source sam3_env.sh
for s in 1 2 3; do
  for cfg in A B; do
    lim=""
    [ "$cfg" = "A" ] && lim="--train_clip_max 185"
    for k in 0 1 2 3 4; do
      python -u nba_data_pipeline/finetune.py --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
        --data_dir nba_data_pipeline/output_contact_all $lim \
        --out_dir nba_data_pipeline/checkpoints --tag contact_joint${cfg}_s${s}_cv$k --pairwise --epochs 40 --batch 64 \
        --lr 3e-4 --patience 8 --pos_weight_cap 5 --n_folds 5 --fold $k --train_seed $s \
        || { touch logs/joint_failed; exit 1; }
      echo "== 完成 ${cfg} 種子 ${s} 第 $((k+1)) 份"
    done
  done
done
touch logs/joint_done
