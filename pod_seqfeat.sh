#!/bin/bash
# GRU 改法 1：每窗多 10 個幾何特徵（finetune_seq.py --feat），起點 court2fix2flip、翻轉、正樣本權重 10
cd "$(dirname "$0")"
source sam3_env.sh
rm -f logs/seqfeat_*
python -u nba_data_pipeline/finetune_seq.py --build --data stubs/seq_data_feat.pkl > logs/seqfeat_build.log 2>&1 || { echo FAIL > logs/seqfeat_eval.log; exit 1; }
for s in 1 2 3 4 5; do
  python -u nba_data_pipeline/finetune_seq.py --seed $s --tag seqflipfeat --init court2fix2flip --flip_aug --pos_weight 10 \
    --feat --data stubs/seq_data_feat.pkl > logs/seqfeat_s$s.log 2>&1 &
done
wait
python -u nba_data_pipeline/decode_compare.py court2fix2flip:nms seqflippw:hyst seqflipfeat:hyst court2fix2flip:hyst > logs/seqfeat_eval.log 2>&1
echo "=== 漏抓分類（seqflipfeat）" >> logs/seqfeat_eval.log
python /tmp/miss_analysis.py seqflipfeat hyst 0.7 >> logs/seqfeat_eval.log 2>&1
echo "=== 漏抓分類（seqflippw）" >> logs/seqfeat_eval.log
python /tmp/miss_analysis.py seqflippw hyst 0.7 >> logs/seqfeat_eval.log 2>&1
echo SEQFEAT_DONE >> logs/seqfeat_eval.log
