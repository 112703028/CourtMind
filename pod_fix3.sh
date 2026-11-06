#!/bin/bash
# 2026-11-04 修正標記第 3 輪（無球掩護 5 個拿掉、持球者 3 支改、443 整支不用、278 非掩護）→ court2fix3flip，跟 court2fix2flip 同一份新答案比
cd "$(dirname "$0")"
source sam3_env.sh
rm -f logs/fix3_*
run() {
  for k in 0 1 2 3 4; do
    python -u nba_data_pipeline/finetune.py --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
      --data_dir nba_data_pipeline/output_contact_court2fix3_503 \
      --out_dir nba_data_pipeline/checkpoints --tag contact_court2fix3flip_s$1_cv$k --pairwise --epochs 40 --batch 64 \
      --lr 3e-4 --patience 8 --pos_weight_cap 5 --n_folds 5 --fold $k --train_seed $1 --flip_aug || return 1
  done
}
for s in 1 2 3 4 5; do run $s > logs/fix3_s$s.log 2>&1 & done
wait
python - <<'PY'
import pickle, pathlib
p = pathlib.Path('stubs/five_seed_curves.pkl')
d = pickle.load(open(p, 'rb')); d = {k: v for k, v in d.items() if k[0] not in ('court2fix2flip', 'court2fix3flip')}
pickle.dump(d, open(p, 'wb'))
PY
python -u nba_data_pipeline/five_seed_eval.py --tags court2fix2flip,court2fix3flip --workers 10 --threads 2 > logs/fix3_eval.log 2>&1
python -u nba_data_pipeline/decode_compare.py court2fix2flip:nms court2fix3flip:nms court2fix3flip:hyst >> logs/fix3_eval.log 2>&1
echo FIX3_DONE >> logs/fix3_eval.log
