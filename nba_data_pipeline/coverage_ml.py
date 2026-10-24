"""
coverage_ml.py — 讓分類器從人工標記學「防守方式」，看追蹤資料夠不夠分辨四種防守。

特徵：撞上前 1 秒到撞上後 2.4 秒（5fps，共 18 個時間點），每個時間點 6 個距離
     Ds→持球者、Dh→持球者、Ds→掩護者、Dh→掩護者、Ds→籃框、Dh→籃框（缺值用前後補）。
     Dh / Ds 的找法同 coverage_eval.py（撞上前 1 秒最近的防守者）。
模型：隨機森林（class_weight=balanced），5 份按 clip 切的交叉驗證（考的都是沒看過的 clip）。
基準：全部猜最多的那類（退防）。

用法：
  python nba_data_pipeline/coverage_ml.py
"""
import pickle
import collections
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

import coverage_eval as ce

ROOT = Path(__file__).parent.parent
OFFS = list(range(-5, 13))     # 撞上前 1 秒 ~ 撞上後 2.4 秒（1.4 秒太短：換防/包夾的結果還沒出現；2.4~4 秒持平）


def basket_of(c):
    """進攻籃框：球場兩端籃框裡，持球者整段平均比較接近的那個。"""
    ct, H = c['court_tracks'], c['handler_tid']
    xs = [ct[f][H][0] for f in range(len(ct)) if H in ct[f]]
    w = 91.8
    return (5.25, 24.6) if (not xs or np.mean(xs) < w / 2) else (w - 5.25, 24.6)


_HMM_CACHE = {}


def hmm_matchup(c):
    """整支 clip 跑一次 HMM 防守配對（5fps：最短段落 5 張 = 1 秒）。"""
    if c['clip'] not in _HMM_CACHE:
        from tactics_tools import defensive_matchup_hmm
        pa = [{p: (1 if c['track_team'].get(p) == 'offense' else 2) for p in fd}
              for fd in c['court_tracks']]
        _HMM_CACHE[c['clip']] = defensive_matchup_hmm(
            c['court_tracks'], pa, c['handler_tid'], basket_of(c), min_seg=5)
    return _HMM_CACHE[c['clip']]


def features(c, ev, use_hmm=False, ds=None, dh=None):
    """ds / dh 有給就用指定的人（人工修正：程式用距離挑錯了），沒給照舊用撞上前 1 秒最近的防守者。"""
    ds_fix, dh_fix = ds, dh
    ct, team, H, S, cf = c['court_tracks'], c['track_team'], c['handler_tid'], ev['tid'], ev['idx']
    n = len(ct)
    pre_f = [f for f in range(cf - 5, cf) if 0 <= f < n]

    def d(a, b, f):
        if not (0 <= f < n) or a not in ct[f]:
            return np.nan
        pb = b if isinstance(b, tuple) else (ct[f][b] if b in ct[f] else None)
        return np.nan if pb is None else float(np.hypot(*np.subtract(ct[f][a], pb)))

    def mean_d(a, b):
        v = [x for x in (d(a, b, f) for f in pre_f) if not np.isnan(x)]
        return np.mean(v) if v else np.inf

    defs = [p for p, t in team.items() if t == 'defense']
    if H is None or S is None or len(defs) < 2 or not pre_f:
        return None
    dh = min(defs, key=lambda p: mean_d(p, H))
    ds = min((p for p in defs if p != dh), key=lambda p: mean_d(p, S))
    if use_hmm:
        # 擋拆前誰守誰改用 HMM 配對（擋拆前的配對 HMM 很準）；HMM 沒給出就沿用最近距離
        from tactics_tools import _matchup_mode
        m = hmm_matchup(c)
        a, b = max(0, cf - 5), max(0, cf - 1)
        on_h = [D for D in m if _matchup_mode(m, D, a, b) == H]
        on_s = [D for D in m if _matchup_mode(m, D, a, b) == S and D not in on_h]
        if on_h:
            dh = min(on_h, key=lambda p: mean_d(p, H))
        if on_s:
            ds = min((p for p in on_s if p != dh), key=lambda p: mean_d(p, S), default=ds)
        if ds == dh:
            ds = min((p for p in defs if p != dh), key=lambda p: mean_d(p, S))
    dh = dh_fix if dh_fix is not None else dh
    ds = ds_fix if ds_fix is not None else ds
    B = basket_of(c)
    M = np.array([[d(ds, H, cf + o), d(dh, H, cf + o), d(ds, S, cf + o),
                   d(dh, S, cf + o), d(ds, B, cf + o), d(dh, B, cf + o)] for o in OFFS])
    if np.isnan(M).mean() > 0.5:
        return None
    for j in range(M.shape[1]):                      # 缺值：前後最近的值補，全缺用該欄中位數
        col = M[:, j]
        ok = ~np.isnan(col)
        if ok.any():
            M[:, j] = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok])
    return M.flatten()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--hmm', action='store_true', help='擋拆前誰守誰改用 HMM 配對')
    ap.add_argument('--no_curve', action='store_true', help='不跑學習曲線')
    args = ap.parse_args()
    labels = ce.load_labels()
    X, y, g = [], [], []
    for clip, items in labels.items():
        p = ce.CACHE / f'{clip}.pkl'
        if not p.exists():
            continue
        c = pickle.load(open(p, 'rb'))
        step = c['fps'] / ce.FPS
        for frame, ans, _ in items:
            if ans is None:
                continue
            idx = round(frame / step)
            evs = [e for e in c['events'] if e['tid'] is not None and abs(e['idx'] - idx) <= 1]
            if not evs:
                continue
            f = features(c, evs[0], use_hmm=args.hmm)
            if f is None:
                continue
            X.append(f); y.append(ans); g.append(clip)
    X, y, g = np.array(X), np.array(y), np.array(g)
    classes = ['switch', 'hedge', 'blitz', 'drop']
    print(f"可用樣本 {len(y)}：", dict(collections.Counter(ce.NAME[v] for v in y)))

    pred = np.empty(len(y), dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in cv.split(X, y, g):
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                   class_weight='balanced', random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])

    cm = confusion_matrix(y, pred, labels=classes)
    print("\n答案＼判斷  " + "  ".join(f"{ce.NAME[p]:>4s}" for p in classes) + "   正確率")
    for i, a in enumerate(classes):
        n = cm[i].sum()
        print(f"  {ce.NAME[a]}({n:3d}) " + "  ".join(f"{v:5d}" for v in cm[i]) + f"   {cm[i, i] / max(1, n):.0%}")
    acc = (pred == y).mean()
    base = collections.Counter(y).most_common(1)[0][1] / len(y)
    bal = np.mean([cm[i, i] / max(1, cm[i].sum()) for i in range(len(classes))])
    print(f"\n整體 {acc:.0%}（全部猜退防 {base:.0%}）；四類平均正確率 {bal:.0%}（亂猜約 25%）")

    if args.no_curve:
        return
    # 學習曲線：訓練資料只用 50% / 75% / 100% 的 clip，看正確率是否還在上升（上升 = 多標資料有用）
    print("\n學習曲線（訓練資料比例 → 四類平均正確率，重複 10 次取平均）：")
    rng = np.random.RandomState(0)
    for frac in (0.5, 0.75, 1.0):
        scores = []
        for rep in range(10 if frac < 1 else 1):
            p2 = np.empty(len(y), dtype=object)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
            for tr, te in cv.split(X, y, g):
                clips_tr = np.unique(g[tr])
                keep = set(rng.choice(clips_tr, max(1, int(len(clips_tr) * frac)), replace=False))
                tr2 = [i for i in tr if g[i] in keep]
                m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                           class_weight='balanced', random_state=rep)
                m.fit(X[tr2], y[tr2])
                p2[te] = m.predict(X[te])
            c2 = confusion_matrix(y, p2, labels=classes)
            scores.append(np.mean([c2[i, i] / max(1, c2[i].sum()) for i in range(len(classes))]))
        print(f"  {frac:.0%}：{np.mean(scores):.0%}")


if __name__ == '__main__':
    main()
