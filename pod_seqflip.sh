#!/bin/bash
# GRU 第 2 輪：起點換成翻轉增強模型（court2fix2flip）+ GRU 訓練也翻轉；seqflip = 正樣本權重 5、seqflippw = 10
cd "$(dirname "$0")"
source sam3_env.sh
rm -f logs/seqflip_*
for s in 1 2 3 4 5; do
  python -u nba_data_pipeline/finetune_seq.py --seed $s --tag seqflip --init court2fix2flip --flip_aug > logs/seqflip_s$s.log 2>&1 &
  python -u nba_data_pipeline/finetune_seq.py --seed $s --tag seqflippw --init court2fix2flip --flip_aug --pos_weight 10 > logs/seqflip_pw_s$s.log 2>&1 &
done
wait
python -u nba_data_pipeline/decode_compare.py court2fix2flip:nms seqflip:nms seqflip:run seqflippw:nms seqflippw:run court2fix2:nms > logs/seqflip_eval.log 2>&1
echo SEQFLIP_DONE >> logs/seqflip_eval.log
