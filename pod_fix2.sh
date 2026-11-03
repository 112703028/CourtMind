#!/bin/bash
# 2026-11-03 修正標記後重訓：court2fix2（新標記）5 種子 × 5 份，跟 court2fix（舊標記訓練）用同一份新答案比
cd "$(dirname "$0")"
source sam3_env.sh
rm -f logs/fix2_*
run() {  # $1 = 種子
  for k in 0 1 2 3 4; do
    python -u nba_data_pipeline/finetune.py --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
      --data_dir nba_data_pipeline/output_contact_court2fix2_503 \
      --out_dir nba_data_pipeline/checkpoints --tag contact_court2fix2_s$1_cv$k --pairwise --epochs 40 --batch 64 \
      --lr 3e-4 --patience 8 --pos_weight_cap 5 --n_folds 5 --fold $k --train_seed $1 || return 1
  done
}
for s in 1 2 3 4 5; do run $s > logs/fix2_s$s.log 2>&1 & done
wait
# 答案（快取）改了 → 舊模型的分數曲線也要用新快取重算
python - <<'PY'
import pickle, pathlib
p = pathlib.Path('stubs/five_seed_curves.pkl')
if p.exists():
    d = pickle.load(open(p, 'rb')); d = {k: v for k, v in d.items() if k[0] not in ('court2fix', 'court2fix2')}
    pickle.dump(d, open(p, 'wb'))
PY
echo "court2fix2 $(ls nba_data_pipeline/checkpoints | grep -c 'contact_court2fix2_s[1-5]_cv')/25" > logs/fix2_eval.log
python -u nba_data_pipeline/five_seed_eval.py --tags court2fix,court2fix2 --workers 10 --threads 2 >> logs/fix2_eval.log 2>&1
echo FIX2_DONE >> logs/fix2_eval.log
