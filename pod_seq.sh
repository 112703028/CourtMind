#!/bin/bash
# C1：整段時序模型（finetune_seq.py），5 種子 × 5 份，跟 court2fix2 比
cd "$(dirname "$0")"
source sam3_env.sh
rm -f logs/seq_*
[ -f stubs/seq_data.pkl ] || python -u nba_data_pipeline/finetune_seq.py --build > logs/seq_build.log 2>&1 || { echo FAIL > logs/seq_eval.log; exit 1; }
for s in 1 2 3 4 5; do python -u nba_data_pipeline/finetune_seq.py --seed $s --tag seq > logs/seq_s$s.log 2>&1 & done
wait
python -u nba_data_pipeline/five_seed_eval.py --tags court2fix2,seq --workers 2 --threads 2 > logs/seq_eval.log 2>&1
echo SEQ_DONE >> logs/seq_eval.log
