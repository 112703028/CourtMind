#!/bin/bash
# final5b（2026-11-13）：第 2 輪標記 + 補標「同一掩護者的第二次擋拆」4 個（44 / 146 / 156 / 157），**不含第 3 輪**
# （第 3 輪交叉驗證反而略差：±0.4 秒 0.858 → 0.850，見實驗紀錄 22）。快取已經套了第 3 輪、沒辦法退回 → 用組合的：
#   ScreenNet 資料 = court2fix2（final4 用的）把 4 支換成 court2fix4 的版本（這 4 支第 3 輪沒動過）
#   GRU 序列 = seq_data_fix4 把第 3 輪動過的 10 支換回舊的 seq_data（11-03 建、第 3 輪之前；含 443）
# 用法：nohup bash pod_final5b.sh > logs/final5b.log 2>&1 &
cd "$(dirname "$0")"
source sam3_env.sh
export HF_HUB_OFFLINE=1
set -e
python - <<'PY'
import numpy as np, pickle
from pathlib import Path
ADD = {44, 146, 156, 157}
R3 = {109, 278, 136, 214, 251, 328, 204, 242, 440, 443}
base, new = Path('nba_data_pipeline/output_contact_court2fix2_503'), Path('nba_data_pipeline/output_contact_court2fix4_503')
out = Path('nba_data_pipeline/output_contact_court2fix2add_503'); out.mkdir(exist_ok=True)
for name in ('positive_sequences.npz', 'negative_sequences.npz'):
    a, b = np.load(base / name, allow_pickle=True), np.load(new / name, allow_pickle=True)
    ka = ~np.isin(a['clip_ids'], list(ADD)); kb = np.isin(b['clip_ids'], list(ADD))
    arr = {k: (np.concatenate([a[k][ka], b[k][kb]]) if a[k].ndim and len(a[k]) == len(a['clip_ids']) else a[k]) for k in a.files}
    np.savez_compressed(out / name, **arr)
    print(name, '第 2 輪', len(a['clip_ids']), '→ 拿掉 4 支', int((~ka).sum()), '+ 新版 4 支', int(kb.sum()), '=', len(arr['clip_ids']))
s4, old = pickle.load(open('stubs/seq_data_fix4.pkl', 'rb')), pickle.load(open('stubs/seq_data.pkl', 'rb'))
back = [c for c in R3 if c in old]
for c in back:
    s4[c] = old[c]
pickle.dump(s4, open('stubs/seq_data_fix2add.pkl', 'wb'))
print('GRU 序列：fix4', len(s4) - len([c for c in back if c not in pickle.load(open('stubs/seq_data_fix4.pkl', 'rb'))]), '支、換回第 3 輪前', sorted(back), '→', len(s4), '支')
PY
set +e
for s in 1 2 3; do
  python -u nba_data_pipeline/finetune.py --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
    --data_dir nba_data_pipeline/output_contact_court2fix2add_503 --out_dir nba_data_pipeline/checkpoints \
    --tag contact_final5b_s$s --pairwise --epochs 17 --batch 64 --lr 3e-4 --pos_weight_cap 5 --val_frac 0 --train_seed $s --flip_aug \
    > logs/final5b_s$s.log 2>&1 &
done
wait
for s in 1 2 3; do
  python -u nba_data_pipeline/finetune_seq.py --final --init final5b --seed $s --tag final5bgru --epochs 9 \
    --pos_weight 10 --flip_aug --threshold 0.7 --data stubs/seq_data_fix2add.pkl > logs/final5bgru_s$s.log 2>&1 &
done
wait
ls -la nba_data_pipeline/checkpoints/finetune_contact_final5b_s*_best.pt nba_data_pipeline/checkpoints/finetune_seq_final5bgru_s*_best.pt
echo FINAL5B_DONE
