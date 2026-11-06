"""
decode_compare.py — 兩種「模型 + 解碼規則」的 5 種子配對比較（10 種 3 個模型平均組合）。

解碼規則（分數曲線 → 擋拆時間點）：
  nms  ：找峰值（≥ 門檻的局部最高）+ 同一人兩個尖峰至少隔 3 秒（現行 inference / five_seed_eval）
  peak ：只找峰值、不去重
  run  ：連續超過門檻的一段算一次（取最高那張）——沒有間隔參數
  hyst / hyst1 / hyst3：雙門檻——到門檻才開始一段，掉到「門檻 − 0.2 / 0.1 / 0.3」以下才結束（中間小凹陷不會切成兩次）
  後面加 -mid：一段裡取中間那張（不取最高）；-w：取分數加權中心（權重 = 分數 − 結束門檻）
  例 hyst-mid、run-w
門檻掃 0.3~0.9 各自取最佳（同 five_seed_eval）。

曲線來源：tag 開頭是 seq → stubs/seq_curves/<tag>_s<種子>.pkl（finetune_seq.py）；其他 → stubs/five_seed_curves.pkl。

用法：python nba_data_pipeline/decode_compare.py court2fix2flip:nms seqflip:run seqflip:nms
      （第一個 = 基準，其餘各自跟它比）
"""
import sys
import pickle
import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from eval_contact import find_peaks, CACHE_DIR

SEEDS, THRS, NMS = (1, 2, 3, 4, 5), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 15


def decode(curve, thr, mode):
    if mode == 'nms':
        return [f for f, _ in find_peaks(curve, thr, NMS)]
    if mode == 'peak':
        return [f for f, _ in find_peaks(curve, thr, 1)]
    mode, _, pick = mode.partition('-')
    low = thr - {'hyst': 0.2, 'hyst1': 0.1, 'hyst3': 0.3}.get(mode, 0.0)

    def choose(run):
        if pick == 'mid':
            return run[len(run) // 2][0]
        if pick == 'w':
            w = [max(1e-6, s - low) for _, s in run]
            c = sum(f * x for (f, _), x in zip(run, w)) / sum(w)
            return min(run, key=lambda r: abs(r[0] - c))[0]
        return max(run, key=lambda x: x[1])[0]

    out, run = [], []                                   # run / hyst
    for f, s in sorted(curve.items()) + [(None, -1.0)]:
        if f is not None and run and s >= low and f == run[-1][0] + 1:
            run.append((f, s))
            continue
        if f is not None and s >= thr and not run:
            run.append((f, s))
            continue
        if run:
            out.append(choose(run))
        run = [(f, s)] if (f is not None and s >= thr) else []
    return out


def load(tag, cache):
    if tag.startswith('seq'):
        return {s: pickle.load(open(ROOT / 'stubs' / 'seq_curves' / f'{tag}_s{s}.pkl', 'rb')) for s in SEEDS}
    return {s: cache[(tag, s)] for s in SEEDS}


def avg(cvs):
    return {k: {f: float(np.mean([cv[k][f] for cv in cvs if k in cv and f in cv[k]])) for f in cvs[0][k]}
            for k in cvs[0]}


def main():
    specs = [a.split(':') for a in sys.argv[1:]]
    cache = pickle.load(open(ROOT / 'stubs' / 'five_seed_curves.pkl', 'rb'))
    M = {tag: load(tag, cache) for tag, _ in specs}
    gts = {}
    common = set.intersection(*(set(M[tag][s]) for tag, _ in specs for s in SEEDS))
    for cid in sorted(common):
        c = pickle.load(open(CACHE_DIR / f'screen_{cid}.pkl', 'rb'))
        if c.get('pending'):
            continue
        vs = c['valid_start_idx']
        gts[cid] = (vs, [e['idx'] for e in c['events'] if e['tid'] is not None and e['idx'] >= vs])

    def f1(curves, tol, mode):
        best = (0.0, None)
        for thr in THRS:
            tp = fp = fn = 0
            for cid, (vs, g) in gts.items():
                pk = [f for c in curves[cid].values() for f in decode(c, thr, mode) if f >= vs]
                used = set()
                for e in g:
                    mm = [(abs(f - e), i) for i, f in enumerate(pk) if i not in used and abs(f - e) <= tol]
                    if mm:
                        used.add(min(mm)[1])
                tp += len(used); fp += len(pk) - len(used); fn += len(g) - len(used)
            r = 2 * tp / max(1, 2 * tp + fp + fn)
            if r > best[0]:
                best = (r, (thr, tp, fp, fn))
        return best

    combos = list(itertools.combinations(SEEDS, 3))
    for tol, tn in ((2, '±0.4 秒'), (5, '±1 秒')):
        print(f'\n===== {tn} =====')
        res = {}
        for tag, mode in specs:
            r3 = [f1({cid: avg([M[tag][s][cid] for s in cb]) for cid in gts}, tol, mode)[0] for cb in combos]
            r5, info = f1({cid: avg([M[tag][s][cid] for s in SEEDS]) for cid in gts}, tol, mode)
            res[(tag, mode)] = r3
            print(f'  {tag}:{mode:5s} 3 個平均 {np.mean(r3):.3f}（{min(r3):.3f} ~ {max(r3):.3f}）  5 個平均 {r5:.3f}'
                  f'（門檻 {info[0]}：抓對 {info[1]} 誤報 {info[2]} 漏抓 {info[3]}）', flush=True)
        b = res[tuple(specs[0])]
        for spec in specs[1:]:
            d = [x - y for x, y in zip(res[tuple(spec)], b)]
            print(f'  {spec[0]}:{spec[1]} vs 基準：贏 {sum(x > 0 for x in d)}/10，差距平均 {np.mean(d):+.3f}'
                  f'（{min(d):+.3f} ~ {max(d):+.3f}）')


if __name__ == '__main__':
    main()
