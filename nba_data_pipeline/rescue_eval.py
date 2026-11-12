"""
rescue_eval.py — 中等分數的擋拆用「接觸幾何」救回，有沒有提升（2026-11-12，起因 screen_44 第二次掩護分數 0.64 < 門檻 0.7）。

基準：正式解碼（GRU 3 個平均，雙門檻：到 thr 開始、掉到 thr − 0.2 以下結束）。
救回：最高只到 [lo, thr) 的那一段，最高點前後 1 窗內掩護者離某個防守者 ≤ contact 呎、離持球者 ≤ near 呎 → 也算一次擋拆。
曲線 = pod 的交叉驗證（stubs/seq_curves/seqflip_s*.pkl），答案 = stubs/train_contact_cache（5fps 座標 + 標記）。
比較同 decode_compare：10 種 3 個模型平均組合，±0.4 秒（2 窗）/ ±1 秒（5 窗）事件 F1。
⚠ 答案有漏標（44 第二次掩護沒標；106 / 113 / 146 / 156 / 157 疑似）→ 救回對的也可能被算成誤報。

用法：python nba_data_pipeline/rescue_eval.py [曲線根目錄，預設 stubs/pod_eval/stubs]
"""
import sys
import math
import pickle
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'stubs' / 'pod_eval' / 'stubs'
SEEDS = (1, 2, 3, 4, 5)


def runs(curve, start, low):
    """雙門檻的每一段：[(最高窗, 最高分)]；start = 開始門檻。"""
    out, cur = [], []
    for f, s in sorted(curve.items()) + [(None, -1.0)]:
        if f is not None and cur and s >= low and f == cur[-1][0] + 1:
            cur.append((f, s)); continue
        if cur:
            out.append(max(cur, key=lambda x: x[1]))
        cur = [(f, s)] if (f is not None and s >= start) else []
    return out


def contact(c, cand, f, dist_c, near):
    ct, team = c['court_tracks'], c['track_team']
    H = c['handler_tid']
    for i in (f - 1, f, f + 1):
        if not 0 <= i < len(ct):
            continue
        fr = ct[i]
        if cand not in fr or H not in fr:
            continue
        s = fr[cand]
        if math.dist(s, fr[H]) > near:
            continue
        if any(team.get(q) == 'defense' and math.dist(s, p) <= dist_c for q, p in fr.items()):
            return True
    return False


def decode(curves_c, c, thr, lo, dist_c, near):
    out = []
    for cand, cv in curves_c.items():
        for f, s in runs(cv, lo if lo is not None else thr, thr - 0.2):
            if s >= thr or (lo is not None and contact(c, cand, f, dist_c, near)):
                out.append((f, cand))
    return out


def main():
    S = {s: pickle.load(open(BASE / 'seq_curves' / f'seqflip_s{s}.pkl', 'rb')) for s in SEEDS}
    common = set.intersection(*(set(v) for v in S.values()))
    caches = {}
    for cid in sorted(common):
        c = pickle.load(open(BASE / 'train_contact_cache' / f'screen_{cid}.pkl', 'rb'))
        if not c.get('pending'):
            caches[cid] = c

    def avg(cb):
        return {cid: {k: {f: float(np.mean([S[s][cid][k][f] for s in cb if k in S[s][cid] and f in S[s][cid][k]]))
                          for f in S[cb[0]][cid][k]} for k in S[cb[0]][cid]} for cid in caches}

    def f1(curves, tol, **kw):
        tp = fp = fn = 0
        for cid, c in caches.items():
            vs = c['valid_start_idx']
            g = [(e['idx'], e['tid']) for e in c['events'] if e['tid'] is not None and e['idx'] >= vs]
            pk = [(f, k) for f, k in decode(curves[cid], c, **kw) if f >= vs]
            used = set()
            for e, _ in g:
                mm = [(abs(f - e), i) for i, (f, _) in enumerate(pk) if i not in used and abs(f - e) <= tol]
                if mm:
                    used.add(min(mm)[1])
            tp += len(used); fp += len(pk) - len(used); fn += len(g) - len(used)
        return 2 * tp / max(1, 2 * tp + fp + fn), tp, fp, fn

    combos = list(itertools.combinations(SEEDS, 3))
    avgs = [avg(cb) for cb in combos]
    configs = [('基準（門檻 0.7）', dict(thr=0.7, lo=None, dist_c=3.5, near=10))]
    for lo in (0.5, 0.55, 0.6):
        for dist_c in (3.0, 3.5):
            configs.append((f'救回 [{lo}, 0.7) 接觸 ≤ {dist_c} 呎', dict(thr=0.7, lo=lo, dist_c=dist_c, near=10)))
    configs.append(('對照：直接降門檻到 0.6', dict(thr=0.6, lo=None, dist_c=3.5, near=10)))
    for tol, tn in ((2, '±0.4 秒'), (5, '±1 秒')):
        print(f'\n===== {tn} =====')
        base = None
        for name, kw in configs:
            r = [f1(a, tol, **kw) for a in avgs]
            m = np.mean([x[0] for x in r]); tp, fp, fn = (np.mean([x[i] for x in r]) for i in (1, 2, 3))
            line = f'  {name:28s} F1 {m:.3f}（抓對 {tp:.0f} 誤報 {fp:.0f} 漏抓 {fn:.0f}）'
            if base is None:
                base = [x[0] for x in r]
            else:
                d = [x[0] - y for x, y in zip(r, base)]
                line += f'  vs 基準 贏 {sum(v > 0 for v in d)}/10，平均 {np.mean(d):+.3f}'
            print(line, flush=True)


if __name__ == '__main__':
    main()
