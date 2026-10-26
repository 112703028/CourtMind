"""
five_seed_eval.py — 5 個種子驗證：cleanall（只用位置）vs pose（+ 姿態 / 遮罩通道）

每個 (版本, 種子) 用交叉驗證算每支片段的分數曲線（快取 stubs/five_seed_curves.pkl，沒算的才算、可平行），再比：
  - 單一模型 F1 平均（5 個種子）
  - 5 個模型平均
  - 5 選 3 的 10 種組合：3 個模型平均的 F1 分布、同一組種子下姿態版贏幾種
事件 F1：±0.4 秒（2 張）與 ±1 秒（5 張），門檻掃 0.3~0.8 取最佳

用法（pod 有 24 核心）：python nba_data_pipeline/five_seed_eval.py --workers 5
"""
import sys
import pickle
import argparse
import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
NMS, THRS, SEEDS = 15, (0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (1, 2, 3, 4, 5)
TAGS = ('cleanall', 'pose')                # --tags 可換；第一個 = 基準
NAME = {'cleanall': '只用位置', 'pose': '+ 姿態 / 遮罩', 'fill': '+ 座標內插'}
CACHE = ROOT / 'stubs' / 'five_seed_curves.pkl'


def curves_for(job):
    tag, s, threads = job
    import torch
    torch.set_num_threads(threads)
    sys.path[:0] = [str(HERE), str(ROOT)]
    from eval_contact import load_model, score_curves, CACHE_DIR
    d = {}
    for k in range(5):
        m, ck = load_model(ROOT / f'nba_data_pipeline/checkpoints/finetune_contact_{tag}_s{s}_cv{k}_best.pt')
        for cid in ck['val_clips']:
            c = pickle.load(open(CACHE_DIR / f'screen_{cid}.pkl', 'rb'))
            d[cid] = score_curves(m, c, 3)
    return (tag, s), d


def avg(cvs):
    return {cand: {f: float(np.mean([cv[cand][f] for cv in cvs if cand in cv and f in cv[cand]])) for f in cvs[0][cand]}
            for cand in cvs[0]}


def main():
    from multiprocessing import Pool
    from eval_contact import find_peaks, CACHE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--threads', type=int, default=4, help='每個 worker 的 torch 執行緒數')
    ap.add_argument('--tags', default='cleanall,pose', help='要比的兩個版本（第一個 = 基準）')
    args = ap.parse_args()
    global TAGS
    TAGS = tuple(args.tags.split(','))
    curves = pickle.load(open(CACHE, 'rb')) if CACHE.exists() else {}
    jobs = [(t, s, args.threads) for t in TAGS for s in SEEDS if (t, s) not in curves]
    print(f'已有 {len(curves)} 組曲線，要算 {len(jobs)} 組：{[(t, s) for t, s, _ in jobs]}', flush=True)
    if jobs:
        with Pool(min(args.workers, len(jobs))) as pool:
            for key, d in pool.imap_unordered(curves_for, jobs):
                curves[key] = d
                pickle.dump(curves, open(CACHE, 'wb'))
                print(f'{key[0]} 種子 {key[1]} 曲線完成', flush=True)

    gts = {}
    for cid in curves[('cleanall', 1)]:
        c = pickle.load(open(CACHE_DIR / f'screen_{cid}.pkl', 'rb'))
        vs = c['valid_start_idx']
        gts[cid] = (vs, [e['idx'] for e in c['events'] if e['tid'] is not None and e['idx'] >= vs])

    def f1(per_clip, tol):
        best = 0
        for t in THRS:
            tp = fp = fn = 0
            for cid, cv in per_clip.items():
                vs, g = gts[cid]
                pk = [f for c in cv.values() for f, s in find_peaks(c, t, NMS) if f >= vs]
                used = set()
                for e in g:
                    mm = [(abs(f - e), i) for i, f in enumerate(pk) if i not in used and abs(f - e) <= tol]
                    if mm:
                        used.add(min(mm)[1])
                tp += len(used); fp += len(pk) - len(used); fn += len(g) - len(used)
            best = max(best, 2 * tp / max(1, 2 * tp + fp + fn))
        return best

    ens = lambda tag, seeds: {cid: avg([curves[(tag, s)][cid] for s in seeds]) for cid in gts}
    res = {}
    for tag in TAGS:
        r = {}
        for tol in (2, 5):
            r[('single', tol)] = [f1(curves[(tag, s)], tol) for s in SEEDS]
            r[('five', tol)] = f1(ens(tag, SEEDS), tol)
            r[('three', tol)] = {cb: f1(ens(tag, cb), tol) for cb in itertools.combinations(SEEDS, 3)}
        res[tag] = r
    base, new = TAGS
    for tol, tn in ((2, '±0.4 秒'), (5, '±1 秒')):
        print(f'\n===== {tn} =====')
        for tag in TAGS:
            r = res[tag]
            th = np.array(list(r[('three', tol)].values()))
            print(f'  {NAME.get(tag, tag):10s} 單一模型 {np.mean(r[("single", tol)]):.3f}（{" / ".join(f"{x:.3f}" for x in r[("single", tol)])}）'
                  f'   5 個平均 {r[("five", tol)]:.3f}   3 個平均（10 種組合）{th.mean():.3f}（{th.min():.3f} ~ {th.max():.3f}）')
        diffs = [res[new][('three', tol)][cb] - res[base][('three', tol)][cb] for cb in res[new][('three', tol)]]
        win = sum(d > 0 for d in diffs)
        print(f'  同一組種子比較：{NAME.get(new, new)} 在 {win}/10 種 3 個平均組合裡贏；'
              f'差距 平均 {np.mean(diffs):+.3f}（{min(diffs):+.3f} ~ {max(diffs):+.3f}）')


if __name__ == '__main__':
    main()
