"""
coverage_features.py — 評估在 coverage_model 的 108 個距離特徵上，再加其他特徵有沒有用。

設定同 coverage_model：三類（換防 / 包夾 / 退防+延誤）、撞上前 1 秒 ~ 後 2.4 秒、隨機森林。
每組特徵用同樣的 N 種按 clip 切法做 5 份交叉驗證，跟基準逐次配對比較（只有 ~129 筆，2~3% 內多半是雜訊）。

特徵組：
  A  基準：18 時間點 × 6 距離（Ds/Dh → 持球者、掩護者、籃框）
  G1 速度：Ds、Dh、持球者、掩護者每個時間點的速度（ft/s）
  G2 掩護座標：以接觸時掩護者為原點、籃框→掩護者為上下軸的位置（coverage_v2）+ Ds/Dh 朝持球者的速度
  G3 防守結構：持球者 6ft 內有幾個防守者、離持球者最近的是不是 Ds / Dh、離掩護者最近的是不是 Ds / Dh、
     第三個防守者（Ds、Dh 以外）離持球者多近
  G4 摘要（15 個純量）：Ds 最靠近持球者多近 / 何時、最後 Ds 離持球者、Ds 最高上提到哪、
     Dh 撞後最近 / 最後離持球者、撞後「最近持球者的是 Ds」「是 Dh」的比例、兩人以上貼持球者的比例、
     持球者切入（離籃框變化）、掩護者順下（離籃框變化）、最後 Ds / Dh 離掩護者、撞後 Ds / Dh 平均速度

用法：
  python nba_data_pipeline/coverage_features.py --seeds 10
"""
import pickle
import argparse
import collections

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import coverage_eval as ce
import coverage_ml as cm
import coverage_v2 as cv2_
import coverage_model as cmod

OFFS = cm.OFFS
POST = [i for i, o in enumerate(OFFS) if o >= 0]
END = [i for i, o in enumerate(OFFS) if o >= 10]
NEAR = 6.0


def series(ct, p, frames):
    """某人在各時間點的位置 (T, 2)，缺值前後內插。"""
    n = len(ct)
    P = np.array([ct[f][p] if (0 <= f < n and p in ct[f]) else (np.nan, np.nan) for f in frames], float)
    for j in range(2):
        ok = ~np.isnan(P[:, j])
        P[:, j] = np.interp(np.arange(len(P)), np.flatnonzero(ok), P[ok, j]) if ok.any() else 0.0
    return P


def extra_features(c, ev, ds, dh):
    ct, team, H, S, cf = c['court_tracks'], c['track_team'], c['handler_tid'], ev['tid'], ev['idx']
    n = len(ct)
    frames = [cf + o for o in OFFS]
    P = {k: series(ct, p, frames) for k, p in (('ds', ds), ('dh', dh), ('H', H), ('S', S))}
    B = np.array(cm.basket_of(c), float)
    d = lambda a, b: np.linalg.norm(P[a] - (P[b] if isinstance(b, str) else b), axis=1)

    # G1 速度
    spd = {k: np.r_[0, np.linalg.norm(np.diff(P[k], axis=0), axis=1) * ce.FPS] for k in P}
    g1 = np.column_stack([spd['ds'], spd['dh'], spd['H'], spd['S']])

    # G3 防守結構（用原始座標，其他防守者缺值就略過）
    defs = [p for p, t in team.items() if t == 'defense']
    g3 = []
    for i, f in enumerate(frames):
        if not (0 <= f < n):
            g3.append([np.nan] * 6)
            continue
        hp, sp = P['H'][i], P['S'][i]
        here = {p: np.array(ct[f][p]) for p in defs if p in ct[f]}
        here[ds], here[dh] = P['ds'][i], P['dh'][i]
        dH = {p: np.linalg.norm(q - hp) for p, q in here.items()}
        dS = {p: np.linalg.norm(q - sp) for p, q in here.items()}
        third = [v for p, v in dH.items() if p not in (ds, dh)]
        g3.append([sum(v <= NEAR for v in dH.values()),
                   float(min(dH, key=dH.get) == ds), float(min(dH, key=dH.get) == dh),
                   float(min(dS, key=dS.get) == ds), float(min(dS, key=dS.get) == dh),
                   min(third) if third else np.nan])
    g3 = np.array(g3, float)
    for j in range(g3.shape[1]):
        col = g3[:, j]
        ok = ~np.isnan(col)
        g3[:, j] = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok]) if ok.any() else 30.0

    # G4 摘要
    dsH, dhH = d('ds', 'H'), d('dh', 'H')
    up = P['S'][OFFS.index(0)] - B
    up = up / (np.linalg.norm(up) + 1e-6)
    ds_level = (P['ds'] - P['S'][OFFS.index(0)]) @ up
    g4 = np.array([
        dsH[POST].min(), OFFS[POST[int(dsH[POST].argmin())]] / ce.FPS, dsH[END].mean(),
        ds_level[POST].max(),
        dhH[POST].min(), dhH[END].mean(),
        g3[POST, 1].mean(), g3[POST, 2].mean(), (g3[POST, 0] >= 2).mean(),
        d('H', B)[END].mean() - d('H', B)[OFFS.index(0)],
        d('S', B)[END].mean() - d('S', B)[OFFS.index(0)],
        d('ds', 'S')[END].mean(), d('dh', 'S')[END].mean(),
        spd['ds'][POST].mean(), spd['dh'][POST].mean(),
    ])
    return g1, g3, g4


def build(S):
    rows = []
    for s in S:
        c = pickle.load(open(ce.CACHE / f"{s['clip']}.pkl", 'rb'))
        g1, g3, g4 = extra_features(c, s['ev'], s['ds'], s['dh'])
        rows.append({'A': s['old'].flatten(), 'G1': g1.flatten(), 'G2': s['new'].flatten(),
                     'G3': g3.flatten(), 'G4': g4})
    return rows


def run(X, y, g, seeds):
    accs, bals, per = [], [], collections.defaultdict(list)
    for sd in range(seeds):
        p = np.empty(len(y), dtype=object)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(X, y, g):
            p[te] = cmod._rf(sd).fit(X[tr], y[tr]).predict(X[te])
        accs.append((p == y).mean())
        r = [((p == c) & (y == c)).sum() / (y == c).sum() for c in cmod.CLASSES]
        bals.append(np.mean(r))
        for c, v in zip(cmod.CLASSES, r):
            per[c].append(v)
    return np.array(accs), np.array(bals), {c: np.mean(v) for c, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args()

    S = cv2_.collect()
    y = np.array([cmod.to3(s['y']) for s in S], dtype=object)
    g = np.array([s['clip'] for s in S])
    R = build(S)
    print(f"樣本 {len(y)}：", {cmod.NAME[k]: v for k, v in collections.Counter(y).items()})

    combos = {
        'A 基準（108 距離）': ['A'],
        'A + G1 速度': ['A', 'G1'],
        'A + G2 掩護座標': ['A', 'G2'],
        'A + G3 防守結構': ['A', 'G3'],
        'A + G4 摘要': ['A', 'G4'],
        'A + 全部': ['A', 'G1', 'G2', 'G3', 'G4'],
        '只用 G4 摘要（15 個）': ['G4'],
        'G3 + G4': ['G3', 'G4'],
    }
    base = None
    print(f"\n{'特徵':22s} 數量  整體   三類平均（±標準差）  換防  包夾  退防   vs 基準（三類平均，{args.seeds} 種切法配對）")
    for name, parts in combos.items():
        X = np.array([np.concatenate([r[k] for k in parts]) for r in R])
        X = cmod._fill_nan(X, np.nan_to_num(np.nanmedian(X, axis=0)))
        acc, bal, per = run(X, y, g, args.seeds)
        if base is None:
            base = bal
            cmp_txt = '—'
        else:
            dlt = bal - base
            cmp_txt = f"{dlt.mean():+.1%}（{(dlt > 0).sum()}/{len(dlt)} 次較好）"
        print(f"{name:22s} {X.shape[1]:4d}  {acc.mean():4.0%}   {bal.mean():4.0%} ±{bal.std():.0%}           "
              + "  ".join(f"{per[c]:4.0%}" for c in cmod.CLASSES) + f"   {cmp_txt}")


if __name__ == '__main__':
    main()
