"""
coverage_v2.py — 防守方式判斷的改進實驗（不需 GPU，讀 stubs/train_contact_cache）。

比較：
  A. 舊特徵（coverage_ml：6 個距離 × 13 時間點）+ 隨機森林
  B. 舊特徵 + 以掩護為中心的座標特徵 + 隨機森林
  C. B 的時間序列 + ROCKET（隨機卷積核 + Ridge，小資料友善）
  D. 分層：先分「退防 vs 積極」，積極再分換防/延誤/包夾；並畫「只回答有把握的」準確率曲線

以掩護為中心的座標（接觸那刻定好，整段固定）：
  原點 = 接觸時掩護者 S 的位置
  上下軸 = 籃框 → S 的方向；正值 = 比掩護者「高」（離籃框更遠、往持球者那邊上提），負值 = 退到掩護者後面
  左右軸 = 垂直上下軸，正向取「撞上前持球者所在的那一側」
  每個時間點記 Ds / Dh / H / S 的（上下, 左右）、Ds 與 Dh 朝持球者的速度

另外 --draw 會把判錯的樣本畫成確認圖（output_videos/coverage_check/），
看程式認定的 Ds（守掩護者）/ Dh（守持球者）有沒有認對。

用法：
  python nba_data_pipeline/coverage_v2.py
  python nba_data_pipeline/coverage_v2.py --draw 20
"""
import pickle
import argparse
import collections
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

import coverage_eval as ce
import coverage_ml as cm

ROOT = Path(__file__).parent.parent
FRAMES_DIR = ROOT / 'input_videos' / 'screen_frames'
OUT_DIR = ROOT / 'output_videos' / 'coverage_check_2p4s'
CLASSES = ['switch', 'hedge', 'blitz', 'drop']
OFFS = cm.OFFS


# ── 樣本 ──────────────────────────────────────────────────────────────────────

def pick_ds_dh(c, ev):
    """同 coverage_ml：撞上前 1 秒平均距離最近的防守者。"""
    ct, team, H, S, cf = c['court_tracks'], c['track_team'], c['handler_tid'], ev['tid'], ev['idx']
    n = len(ct)
    pre_f = [f for f in range(cf - 5, cf) if 0 <= f < n]

    def mean_d(a, b):
        v = [np.hypot(*np.subtract(ct[f][a], ct[f][b])) for f in pre_f if a in ct[f] and b in ct[f]]
        return np.mean(v) if v else np.inf

    defs = [p for p, t in team.items() if t == 'defense']
    if H is None or S is None or len(defs) < 2 or not pre_f:
        return None, None
    dh = min(defs, key=lambda p: mean_d(p, H))
    ds = min((p for p in defs if p != dh), key=lambda p: mean_d(p, S))
    return ds, dh


def interp_cols(M):
    for j in range(M.shape[1]):
        col = M[:, j]
        ok = ~np.isnan(col)
        if ok.any():
            M[:, j] = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok])
        else:
            M[:, j] = 0.0
    return M


def screen_frame_series(c, ev, ds, dh):
    """以掩護為中心的座標時間序列：(len(OFFS), 10)。"""
    ct, H, S, cf = c['court_tracks'], c['handler_tid'], ev['tid'], ev['idx']
    n = len(ct)

    def pos(p, f):
        return np.array(ct[f][p], float) if (0 <= f < n and p in ct[f]) else None

    # 接觸時 S 的位置（附近幾張找第一個有的）
    s0 = next((pos(S, cf + o) for o in (0, -1, 1, -2, 2) if pos(S, cf + o) is not None), None)
    if s0 is None:
        return None
    B = np.array(cm.basket_of(c), float)
    up = s0 - B
    if np.linalg.norm(up) < 1e-6:
        return None
    up /= np.linalg.norm(up)
    lat = np.array([-up[1], up[0]])
    pre_h = [pos(H, f) for f in range(cf - 5, cf)]
    pre_h = [p for p in pre_h if p is not None]
    if pre_h and np.dot(np.mean(pre_h, 0) - s0, lat) < 0:
        lat = -lat

    def proj(p, f):
        q = pos(p, f)
        if q is None:
            return (np.nan, np.nan)
        r = q - s0
        return (float(np.dot(r, up)), float(np.dot(r, lat)))

    rows = []
    for o in OFFS:
        f = cf + o
        row = []
        for p in (ds, dh, H, S):
            row += proj(p, f)
        rows.append(row)
    M = np.array(rows)
    # 朝持球者的速度（每張影格距離變化，負 = 靠近）
    def dist_series(a):
        return np.array([np.hypot(*(np.array(M[i, a:a + 2]) - M[i, 4:6])) for i in range(len(OFFS))])
    M = interp_cols(M)
    v_ds = np.gradient(dist_series(0))
    v_dh = np.gradient(dist_series(2))
    return np.column_stack([M, v_ds, v_dh])


def collect():
    labels = ce.load_labels()
    S = []
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
            ev = evs[0]
            old = cm.features(c, ev)
            ds, dh = pick_ds_dh(c, ev)
            if old is None or ds is None:
                continue
            new = screen_frame_series(c, ev, ds, dh)
            if new is None:
                continue
            S.append(dict(clip=clip, frame=frame, y=ans, ev=ev, ds=ds, dh=dh,
                          old=old.reshape(len(OFFS), 6), new=new))
    return S


# ── ROCKET（多變量，numpy 版）─────────────────────────────────────────────────

class Rocket:
    def __init__(self, n_kernels=2000, seed=0):
        self.n_kernels, self.seed = n_kernels, seed

    def fit(self, X):                       # X: (n, C, T)
        rng = np.random.RandomState(self.seed)
        _, C, T = X.shape
        self.mu = X.mean(axis=(0, 2), keepdims=True)
        self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
        self.k = []
        for _ in range(self.n_kernels):
            L = rng.choice([3, 5, 7])
            nch = rng.randint(1, min(C, 3) + 1)
            ch = rng.choice(C, nch, replace=False)
            w = rng.normal(size=(nch, L))
            w -= w.mean(axis=1, keepdims=True)
            b = rng.uniform(-1, 1)
            dil = int(2 ** rng.uniform(0, np.log2(max(1, (T - 1) / (L - 1)))))
            pad = ((L - 1) * dil) // 2 if rng.randint(2) else 0
            self.k.append((ch, w, b, dil, pad, L))
        return self

    def transform(self, X):
        X = (X - self.mu) / self.sd
        n, _, T = X.shape
        out = np.zeros((n, 2 * len(self.k)))
        for i, (ch, w, b, dil, pad, L) in enumerate(self.k):
            Xp = np.pad(X[:, ch, :], ((0, 0), (0, 0), (pad, pad)))
            Tout = Xp.shape[2] - (L - 1) * dil
            if Tout <= 0:
                continue
            conv = np.full((n, Tout), b)
            for j in range(L):
                conv += np.einsum('nct,c->nt', Xp[:, :, j * dil:j * dil + Tout], w[:, j])
            out[:, 2 * i] = (conv > 0).mean(axis=1)
            out[:, 2 * i + 1] = conv.max(axis=1)
        return out


# ── 評估 ──────────────────────────────────────────────────────────────────────

def make_X(S, kind):
    X = _make_X(S, kind)
    if np.isnan(X).any():                   # 整欄缺值（舊特徵沒補到的）→ 用全體中位數
        med = np.nanmedian(X, axis=0, keepdims=True)
        X = np.where(np.isnan(X), np.broadcast_to(np.nan_to_num(med), X.shape), X)
    return X


def _make_X(S, kind):
    if kind == 'old':
        return np.array([s['old'].flatten() for s in S])
    if kind == 'both':
        return np.array([np.concatenate([s['old'].flatten(), s['new'].flatten()]) for s in S])
    if kind == 'series':                    # (n, C, T)
        return np.array([np.concatenate([s['old'], s['new']], axis=1).T for s in S])
    raise ValueError(kind)


def cv_predict(X, y, g, model_fn, seed, proba=False):
    pred = np.empty(len(y), dtype=object)
    P = np.zeros((len(y), len(CLASSES)))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X if X.ndim == 2 else X[:, 0, :], y, g):
        m = model_fn(seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        if proba:
            pr = m.predict_proba(X[te])
            for j, cl in enumerate(m.classes_):
                P[te, CLASSES.index(cl)] = pr[:, j]
    return (pred, P) if proba else pred


class RocketRidge:
    def __init__(self, seed):
        self.r = Rocket(seed=seed)
        self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced')

    def fit(self, X, y):
        self.r.fit(X)
        self.clf.fit(self.r.transform(X), y)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, X):
        return self.clf.predict(self.r.transform(X))


def rf(seed):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight='balanced', random_state=seed)


def scores(y, pred):
    acc = (pred == y).mean()
    per = {c: ((pred == c) & (y == c)).sum() / max(1, (y == c).sum()) for c in CLASSES}
    bal = np.mean(list(per.values()))
    agg = lambda v: np.where(v == 'drop', 'drop', 'aggr')
    acc2 = (agg(pred) == agg(y)).mean()
    return acc, bal, acc2, per


def hierarchical(X, y, g, seed):
    """先分退防 vs 積極，積極再分三類；回傳 4 類預測與 4 類機率。"""
    pred = np.empty(len(y), dtype=object)
    P = np.zeros((len(y), len(CLASSES)))
    y2 = np.where(y == 'drop', 'drop', 'aggr')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, g):
        m1 = rf(seed).fit(X[tr], y2[tr])
        p_drop = m1.predict_proba(X[te])[:, list(m1.classes_).index('drop')]
        tr_a = tr[y[tr] != 'drop']
        m2 = rf(seed).fit(X[tr_a], y[tr_a])
        pa = m2.predict_proba(X[te])
        for j, cl in enumerate(m2.classes_):
            P[te, CLASSES.index(cl)] = (1 - p_drop) * pa[:, j]
        P[te, CLASSES.index('drop')] = p_drop
    pred[:] = [CLASSES[i] for i in P.argmax(1)]
    return pred, P


# ── 確認圖 ────────────────────────────────────────────────────────────────────

def draw(samples, preds, n_draw):
    import cv2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wrong = [(s, p) for s, p in zip(samples, preds) if p != s['y']][:n_draw]
    colors = {'H': (0, 200, 0), 'S': (0, 0, 255), 'Ds': (0, 140, 255), 'Dh': (255, 120, 0)}
    rows = []
    for k, (s, p) in enumerate(wrong):
        c = pickle.load(open(ce.CACHE / f"{s['clip']}.pkl", 'rb'))
        jpgs = sorted((FRAMES_DIR / s['clip']).glob('*.jpg'), key=lambda q: int(q.stem.rsplit('_', 1)[1]))
        cf = s['ev']['idx']
        who = {'H': c['handler_tid'], 'S': s['ev']['tid'], 'Ds': s['ds'], 'Dh': s['dh']}
        tiles = []
        for o in (-3, 0, 4):
            f = min(max(cf + o, 0), len(jpgs) - 1)
            img = cv2.imread(str(jpgs[f]))
            for name, pid in who.items():
                bb = c['raw_tracks'][f].get(pid)
                if bb is None:
                    continue
                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[name], 4)
                cv2.putText(img, f'{name} #{pid}', (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, colors[name], 3)
            cv2.putText(img, f'{s["clip"]} idx {f} ({o:+d})', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
            tiles.append(cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1]))))
        h = min(t.shape[0] for t in tiles)
        strip = np.hstack([t[:h] for t in tiles])
        name = f"{k + 1:02d}_{s['clip']}_f{s['frame']}_ans-{s['y']}_pred-{p}.jpg"
        cv2.imwrite(str(OUT_DIR / name), strip)
        rows.append(f"{k + 1:02d},{s['clip']},{s['frame']},{ce.NAME[s['y']]},{ce.NAME[p]},"
                    f"#{who['H']},#{who['S']},#{who['Ds']},#{who['Dh']},,")
    with open(OUT_DIR / 'check.csv', 'w', encoding='utf-8-sig') as fo:
        fo.write('編號,clip,接觸幀(原片),答案,判斷,H持球者,S掩護者,Ds守掩護者,Dh守持球者,Ds認對(t/f),Dh認對(t/f)\n')
        fo.write('\n'.join(rows) + '\n')
    print(f"\n已畫 {len(wrong)} 張判錯的確認圖 → {OUT_DIR}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5, help='交叉驗證重複幾次（不同切法）取平均')
    ap.add_argument('--draw', type=int, default=0, help='畫幾張判錯的確認圖')
    args = ap.parse_args()

    S = collect()
    y = np.array([s['y'] for s in S], dtype=object)
    g = np.array([s['clip'] for s in S])
    print(f"樣本 {len(y)}：", dict(collections.Counter(ce.NAME[v] for v in y)),
          f"（全猜退防 {np.mean(y == 'drop'):.0%}）")

    exps = {
        'A 舊特徵 + 隨機森林': ('old', rf),
        'B 加掩護座標 + 隨機森林': ('both', rf),
        'C 時間序列 + ROCKET': ('series', RocketRidge),
    }
    print(f"\n{'方法':24s} 整體  四類平均  退防/積極   換防  延誤  包夾  退防   （{args.seeds} 種切法平均）")
    keep = {}
    for name, (kind, fn) in exps.items():
        X = make_X(S, kind)
        res = [scores(y, cv_predict(X, y, g, fn, sd)) for sd in range(args.seeds)]
        acc, bal, acc2 = (np.mean([r[i] for r in res]) for i in range(3))
        per = {c: np.mean([r[3][c] for r in res]) for c in CLASSES}
        print(f"{name:24s} {acc:4.0%}   {bal:4.0%}      {acc2:4.0%}     "
              + "  ".join(f"{per[c]:4.0%}" for c in CLASSES))
        keep[name] = acc

    # D. 分層 + 只回答有把握的
    X = make_X(S, 'both')
    res, conf_curve = [], collections.defaultdict(list)
    for sd in range(args.seeds):
        pred, P = hierarchical(X, y, g, sd)
        res.append(scores(y, pred))
        top = P.max(1)
        for th in (0.0, 0.5, 0.6, 0.7, 0.8):
            m = top >= th
            conf_curve[th].append(((pred[m] == y[m]).mean() if m.any() else np.nan, m.mean()))
        p_drop = P[:, CLASSES.index('drop')]
        for th in (0.0, 0.6, 0.7, 0.8):            # 兩類：max(p_drop, 1-p_drop) >= th
            m = np.maximum(p_drop, 1 - p_drop) >= th
            pr2 = np.where(p_drop >= 0.5, 'drop', 'aggr')
            y2 = np.where(y == 'drop', 'drop', 'aggr')
            conf_curve[('2', th)].append(((pr2[m] == y2[m]).mean() if m.any() else np.nan, m.mean()))
    acc, bal, acc2 = (np.mean([r[i] for r in res]) for i in range(3))
    per = {c: np.mean([r[3][c] for r in res]) for c in CLASSES}
    print(f"{'D 分層（B 的特徵）':24s} {acc:4.0%}   {bal:4.0%}      {acc2:4.0%}     "
          + "  ".join(f"{per[c]:4.0%}" for c in CLASSES))

    print("\n只回答有把握的（分層模型）：")
    print("  四類  信心門檻  回答比例  回答的準確率")
    for th in (0.0, 0.5, 0.6, 0.7, 0.8):
        a = np.nanmean([v[0] for v in conf_curve[th]]); r = np.mean([v[1] for v in conf_curve[th]])
        print(f"        {th:.1f}      {r:4.0%}      {a:4.0%}")
    print("  兩類（退防/積極）")
    for th in (0.0, 0.6, 0.7, 0.8):
        a = np.nanmean([v[0] for v in conf_curve[('2', th)]]); r = np.mean([v[1] for v in conf_curve[('2', th)]])
        print(f"        {th:.1f}      {r:4.0%}      {a:4.0%}")

    if args.draw:
        pred = cv_predict(make_X(S, 'both'), y, g, rf, 0)
        draw(S, pred, args.draw)


if __name__ == '__main__':
    main()
