"""
cv_eval.py — 彙整交叉驗證：每份模型只考自己沒訓練過的 clip，5 份加總成整體成績。

每支 clip 剛好被考一次，所以加總後等於「全部 clip 都是沒看過的考題」，比單一份 21 支可靠得多。
考法同 eval_contact.py：每個進攻候選人輪流放 slot 1 → 分數曲線 → 找尖峰
→ 尖峰在接觸幀 ±tol 張內算抓對，並記錄掩護者是否也認對。

用法：
  python nba_data_pipeline/cv_eval.py --pattern "nba_data_pipeline/checkpoints/finetune_contact_cv{}_best.pt" --n_folds 5
"""
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from eval_contact import load_model, score_curves, find_peaks, CACHE_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', required=True, help='checkpoint 路徑樣板，{} 會換成份數 0~n-1')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--tol', type=int, default=2, help='±幾張內算抓對（5fps，2 = 0.4 秒）')
    ap.add_argument('--nms', type=int, default=15, help='同一人兩個尖峰至少隔幾張（15 = 3 秒，同 inference 預設）')
    args = ap.parse_args()

    thrs = (0.5, 0.6, 0.7, 0.8, 0.9)
    tot = {t: [0, 0, 0, 0, 0] for t in thrs}     # 掩護數、抓對、人也對、抓錯、漏抓
    epochs = []
    for k in range(args.n_folds):
        path = args.pattern.format(k)
        if not Path(path).exists():
            print(f"第 {k + 1} 份：找不到 {path}，跳過")
            continue
        model, ck = load_model(path)
        anchor = int(ck.get('data_meta', {}).get('anchor', 3))
        epochs.append(ck.get('epoch'))
        clips = [f'screen_{c}' for c in ck.get('val_clips', [])]
        data = []
        for clip in clips:
            c = pickle.load(open(CACHE_DIR / f'{clip}.pkl', 'rb'))
            gts = [(e['idx'], e['tid']) for e in c['events']
                   if e['tid'] is not None and e['idx'] >= c['valid_start_idx']]
            data.append((score_curves(model, c, anchor), gts, c['valid_start_idx']))
        line = []
        for t in thrs:
            n = tp = idok = fp = 0
            for curves, gts, vs in data:
                pk = [(f, cand) for cand, cv in curves.items()
                      for f, _ in find_peaks(cv, t, args.nms) if f >= vs]
                used = set()
                for g, tid in gts:
                    m = [(abs(f - g), i) for i, (f, _) in enumerate(pk)
                         if i not in used and abs(f - g) <= args.tol]
                    if m:
                        i = min(m)[1]
                        used.add(i)
                        tp += 1
                        idok += (pk[i][1] == tid)
                fp += len(pk) - len(used)
                n += len(gts)
            for j, x in enumerate((n, tp, idok, fp, n - tp)):
                tot[t][j] += x
            if t == 0.6:
                line = f"掩護 {n}、抓對 {tp}、抓錯 {fp}、漏抓 {n - tp}"
        print(f"第 {k + 1} 份（{len(clips)} 支，最佳第 {ck.get('epoch')} 輪）門檻 0.6：{line}")

    print(f"\n── {len(epochs)} 份加總（每支 clip 都只在沒訓練過時被考）──")
    print("  門檻 | 掩護 抓對(人也對) 抓錯 漏抓 | 準確率 抓到率  F1")
    for t in thrs:
        n, tp, idok, fp, fn = tot[t]
        pr = tp / max(1, tp + fp)
        rc = tp / max(1, n)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        print(f"  {t:.1f}  | {n:4d} {tp:4d}({idok:3d}) {fp:4d} {fn:4d} |  {pr:.2f}   {rc:.2f}  {f1:.2f}")
    if epochs:
        print(f"\n各份最佳輪數 {epochs} → 最終模型（全部資料）建議訓練 {int(round(np.mean(epochs)))} 輪")


if __name__ == '__main__':
    main()
