"""
coverage_eval.py — 用人工標的防守方式（coverage_labels.csv）驗證判斷規則。

資料：stubs/train_contact_cache/<clip>.pkl（5fps 球場座標、整支多數決分隊、持球者、
      每次掩護的接觸影格 idx 與掩護者 tid）。

規則（接觸影格 cf，持球者 H，掩護者 S）：
  撞上前 pre 秒：守 H 的 Dh、守 S 的 Ds = 這段平均距離最近的防守者
  撞上後 post 秒：
    switch（換防）：最後 0.5 秒 Ds 比 Dh 靠近 H，且 Dh 比 Ds 靠近 S
    兩人同時在 H 的 near 英尺內（兩人堵持球者）：
       Ds 最後退回 S 附近（return 英尺內）→ hedge（延誤）
       Ds 沒退回                         → blitz（包夾）
    Ds 一直沒靠近 H                      → drop（退防）
    其他                                  → unclear

用法：
  python nba_data_pipeline/coverage_eval.py
  python nba_data_pipeline/coverage_eval.py --near 7 --ret 6
"""
import csv
import pickle
import argparse
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE.parent / 'stubs' / 'train_contact_cache'
FPS = 5.0
NAME = {'switch': '換防', 'hedge': '延誤', 'blitz': '包夾', 'drop': '退防', 'unclear': '判不出'}
LABEL = {'換防': 'switch', '延誤': 'hedge', '包夾': 'blitz', '退防': 'drop'}


def load_labels():
    rows = list(csv.DictReader(open(HERE / 'coverage_labels.csv', encoding='utf-8-sig')))
    col = [k for k in rows[0] if k.startswith('防守方式')][0]
    out = collections.defaultdict(list)          # clip → [(原片接觸幀, 答案)]
    for r in rows:
        a = r[col].strip()
        out[r['clip']].append((int(r['接觸幀(原片)']), LABEL.get(a), a))
    return out


def classify(c, ev, pre, post, near, ret):
    ct, team, H, S, cf = c['court_tracks'], c['track_team'], c['handler_tid'], ev['tid'], ev['idx']
    n = len(ct)
    pre_f = [f for f in range(cf - int(pre * FPS), cf) if 0 <= f < n]
    post_f = [f for f in range(cf, cf + int(post * FPS) + 1) if 0 <= f < n]
    end_f = post_f[-3:]

    def d(a, b, f):
        return float(np.hypot(*np.subtract(ct[f][a], ct[f][b]))) if (a in ct[f] and b in ct[f]) else np.nan

    def mean_d(a, b, fs):
        v = [d(a, b, f) for f in fs]
        v = [x for x in v if not np.isnan(x)]
        return np.mean(v) if v else np.inf

    defs = [p for p, t in team.items() if t == 'defense']
    if H is None or S is None or not defs or not pre_f:
        return 'unclear', '缺資料', {}
    dh = min(defs, key=lambda p: mean_d(p, H, pre_f))
    rest = [p for p in defs if p != dh]
    if not rest:
        return 'unclear', '防守者不足', {}
    ds = min(rest, key=lambda p: mean_d(p, S, pre_f))

    dsH = [d(ds, H, f) for f in post_f]
    dhH = [d(dh, H, f) for f in post_f]
    feat = {'dh': dh, 'ds': ds,
            'pre_dhH': mean_d(dh, H, pre_f), 'pre_dsS': mean_d(ds, S, pre_f),
            'min_dsH': np.nanmin(dsH) if not np.all(np.isnan(dsH)) else np.nan,
            'end_dsS': mean_d(ds, S, end_f), 'end_dsH': mean_d(ds, H, end_f),
            'end_dhH': mean_d(dh, H, end_f), 'end_dhS': mean_d(dh, S, end_f),
            'coverage': np.mean([not np.isnan(x) for x in dsH])}
    if feat['coverage'] < 0.5:
        return 'unclear', '撞上後持球者/防守者座標缺太多', feat

    # 三個問題：Ds 有沒有靠近過持球者？最後 Ds 還在持球者旁嗎？最後 Dh 還在持球者旁嗎？
    #   （第一版用「Ds 比 Dh 近」判換防 → 包夾時兩人都貼著，Ds 稍近就被誤判成換防；
    #     真正的差別是 Dh 有沒有離開持球者。延誤則是 Ds 撞上後離開、Dh 回到持球者身上。）
    approached = feat['min_dsH'] <= near
    ds_on = feat['end_dsH'] <= ret
    dh_on = feat['end_dhH'] <= ret
    if not approached:
        return 'drop', 'Ds 沒靠近持球者', feat
    if ds_on and not dh_on:
        return 'switch', 'Ds 接手持球者、Dh 離開', feat
    if ds_on and dh_on:
        return 'blitz', 'Ds、Dh 最後都還在持球者旁', feat
    if dh_on:
        return 'hedge', 'Ds 上前後離開、Dh 回到持球者', feat
    return 'unclear', 'Ds 靠近過，但最後兩人都不在持球者旁', feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', type=float, default=1.0)
    ap.add_argument('--post', type=float, default=1.5)
    ap.add_argument('--near', type=float, default=6.5, help='Ds 進到持球者幾英尺內算「上前」')
    ap.add_argument('--ret', type=float, default=8.5, help='撞上後最後 0.5 秒離持球者幾英尺內算「還在持球者旁」')
    ap.add_argument('--show', action='store_true', help='印每一筆')
    args = ap.parse_args()

    labels = load_labels()
    conf = collections.defaultdict(collections.Counter)
    feats = collections.defaultdict(list)
    skipped = 0
    for clip, items in labels.items():
        p = CACHE / f'{clip}.pkl'
        if not p.exists():
            skipped += len(items)
            continue
        c = pickle.load(open(p, 'rb'))
        step = c['fps'] / FPS
        for frame, ans, raw in items:
            if ans is None:
                continue
            idx = round(frame / step)
            evs = [e for e in c['events'] if e['tid'] is not None and abs(e['idx'] - idx) <= 1]
            if not evs:
                skipped += 1
                continue
            pred, why, f = classify(c, evs[0], args.pre, args.post, args.near, args.ret)
            conf[ans][pred] += 1
            if f:
                feats[ans].append(f)
            if args.show:
                print(f"{clip:11s} {frame:4d} 答案 {NAME[ans]} → {NAME[pred]}（{why}）")

    preds = ['switch', 'hedge', 'blitz', 'drop', 'unclear']
    print(f"\n參數：撞上前 {args.pre}s、撞上後 {args.post}s、堵持球者 ≤{args.near}ft、退回 ≤{args.ret}ft（跳過 {skipped} 筆）")
    print("答案＼判斷  " + "  ".join(f"{NAME[p]:>4s}" for p in preds) + "   正確率")
    tot = ok = 0
    for a in ['switch', 'hedge', 'blitz', 'drop']:
        row = conf[a]
        n = sum(row.values())
        tot += n
        ok += row[a]
        print(f"  {NAME[a]}({n:3d}) " + "  ".join(f"{row[p]:5d}" for p in preds) + f"   {row[a] / max(1, n):.0%}")
    print(f"整體 {ok}/{tot} = {ok / max(1, tot):.0%}")

    print("\n各類別的特徵中位數（英尺）：")
    print("          Ds最靠近持球者  最後Ds離掩護者  最後Ds離持球者  最後Dh離持球者  撞上前Dh離持球者")
    for a in ['switch', 'hedge', 'blitz', 'drop']:
        fs = feats[a]
        if not fs:
            continue
        med = lambda k: np.nanmedian([x[k] for x in fs if np.isfinite(x[k])]) if fs else np.nan
        print(f"  {NAME[a]}   {med('min_dsH'):10.1f}    {med('end_dsS'):10.1f}    {med('end_dsH'):10.1f}"
              f"    {med('end_dhH'):10.1f}    {med('pre_dhH'):10.1f}")


if __name__ == '__main__':
    main()
