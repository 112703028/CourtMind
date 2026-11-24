"""
dho_skeleton.py — 骨架版手遞手判斷（方法 A：骨架特徵 + 小分類器），2026-11-24。

設計：docs/superpowers/specs/2026-11-24-skeleton-dho-design.md；計畫：docs/superpowers/plans/2026-11-24-skeleton-dho.md
手遞手最直接的證據是手的動作（手把球交出去、兩人的手同時在球上、交完轉身擋人），腳底座標看不到，遮罩版骨架看得到。
NBA 追蹤資料沒有骨架 → 只能用自己標的約 45 個手遞手 → 小分類器，先回答「骨架有沒有額外資訊」。

用法：python nba_data_pipeline/dho_skeleton.py eval
"""
import math

import numpy as np

CONF = 0.3                                  # 關節信心 < 0.3 當沒有
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
ARMS = ((L_EL, L_WR), (R_EL, R_WR))


def _pt(kp, j):
    if kp is None or kp[j, 2] < CONF:
        return None
    return np.asarray(kp[j, :2], dtype=float)


def _mid_shoulder(kp):
    a, b = _pt(kp, L_SH), _pt(kp, R_SH)
    if a is None and b is None:
        return None
    return (a + b) / 2 if a is not None and b is not None else (a if a is not None else b)


def _angle(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v) / (nu * nv))))))


def skeleton_features(kp_a, kp_b, ball, H):
    """前後 1 秒每張 10fps 的骨架（(17, 3) 或 None）+ 球中心 → 手遞手的骨架特徵（身高正規化，缺 = NaN）。
    both_hands：兩人手腕離球比較遠的那個的最小值；hand_to_hand：A、B 手腕最近距離；
    arm_angle：A 手臂（肘→腕）跟 A→B（肩中點）夾角最小值（0 = 指向 B），arm_ext：那張的手臂長度；
    ball_travel：有球的張裡第一顆到最後一顆的距離；turn：A 第一張與最後一張肩線方向的夾角（0~180）。"""
    nan = float('nan')
    f = dict(both_hands=nan, hand_to_hand=nan, arm_angle=nan, arm_ext=nan, ball_travel=nan, turn=nan,
             has_pose=int(any(k is not None and (k[:, 2] >= CONF).any() for k in kp_a)))
    H = float(H) if H else 1.0
    both, h2h, arm, shoulders, balls = [], [], [], [], []
    for a, b, c in zip(kp_a, kp_b, ball):
        wa = [p for p in (_pt(a, L_WR), _pt(a, R_WR)) if p is not None]
        wb = [p for p in (_pt(b, L_WR), _pt(b, R_WR)) if p is not None]
        if wa and wb:
            h2h.append(min(np.linalg.norm(x - y) for x in wa for y in wb))
        if c is not None:
            balls.append(np.asarray(c, dtype=float))
            if wa and wb:
                both.append(max(min(np.linalg.norm(x - c) for x in wa), min(np.linalg.norm(y - c) for y in wb)))
        ma, mb = _mid_shoulder(a), _mid_shoulder(b)
        if ma is not None and mb is not None:
            for el, wr in ARMS:
                e, w = _pt(a, el), _pt(a, wr)
                if e is not None and w is not None:
                    ang = _angle(w - e, mb - ma)
                    if ang is not None:
                        arm.append((ang, np.linalg.norm(w - e)))
        sl, sr = _pt(a, L_SH), _pt(a, R_SH)
        if sl is not None and sr is not None:
            shoulders.append(sr - sl)
    if both:
        f['both_hands'] = min(both) / H
    if h2h:
        f['hand_to_hand'] = min(h2h) / H
    if arm:
        ang, ln = min(arm, key=lambda x: x[0])
        f['arm_angle'], f['arm_ext'] = ang, ln / H
    if len(balls) >= 2:
        f['ball_travel'] = float(np.linalg.norm(balls[-1] - balls[0])) / H
    if len(shoulders) >= 2:
        t = _angle(shoulders[0], shoulders[-1])
        f['turn'] = t if t is not None else nan
    return f


# ── 候選 + 特徵列 + 標記 ─────────────────────────────────────────────────────

import sys
import pickle
import collections
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
EXIST = ['approach_ft', 'v_giver', 'hold_b_s', 'probe', 'from_screens', 'from_event']
SKEL = ['both_hands', 'hand_to_hand', 'arm_angle', 'arm_ext', 'ball_travel', 'turn', 'has_pose']


def merge_candidates(events, handoffs, fps, gap_s=0.5):
    """擋拆事件 / 持球者換手候選（都是 (給球者, 接球者, 影格)）→ 同一對、時間差 ≤ gap_s 秒的併成一個。
    回 [{giver, receiver, frame, from_event, from_handoff, event_frame}]（frame：有換手用換手張，否則用撞上張）。"""
    out = []
    for A, B, f in events:
        out.append(dict(giver=A, receiver=B, frame=f, from_event=True, from_handoff=False, event_frame=f))
    for A, B, f in handoffs:
        m = [c for c in out if c['giver'] == A and c['receiver'] == B and abs(c['frame'] - f) <= gap_s * fps]
        if m:
            m[0]['from_handoff'] = True
            m[0]['frame'] = f
        else:
            out.append(dict(giver=A, receiver=B, frame=f, from_event=False, from_handoff=True, event_frame=None))
    return out


def assign_labels(cands, labels, fps):
    """每個標記只給時間最近的那個候選（同一對、候選時間 − 交接張在 −0.6 ~ +2.1 秒）；同一個標記範圍內的其他候選 = None（重複，不評估）；
    沒落在任何標記範圍 = False（同一支片段其他候選當「不是」）。回跟 cands 同長度的 list。"""
    out = [False] * len(cands)
    for hf, A, B, t in labels:
        m = [i for i, c in enumerate(cands) if c['giver'] == A and c['receiver'] == B
             and -0.6 * fps <= c['frame'] - hf <= 2.1 * fps]
        if not m:
            continue
        best = min(m, key=lambda i: abs(cands[i]['frame'] - hf))
        for i in m:
            if i == best:
                out[i] = t
            elif out[i] is False:
                out[i] = None
    return out


def clip_rows(pk, need, labels):
    """一支片段 → 每個候選一列：骨架特徵 + 現有特徵 + 現在上線的做法有沒有標（prod）+ 標記。"""
    import dho
    import dho_model
    from shot_detect import map_ids
    d = pickle.load(open(pk, 'rb'))
    s = pickle.load(open(need[0], 'rb'))['shot_sam3']
    tch = pickle.load(open(need[1], 'rb'))[2]
    pose = pickle.load(open(need[2], 'rb'))
    fps = d['fps']
    cands_h = {(p['giver'], p['receiver'], p['frame']): p for p in dho.candidates(d)}
    cands = merge_candidates([(e['screener_id'], e['handler_id'], e['peak_frame']) for e in d.get('events') or []],
                             list(cands_h), fps)
    probe = dho_model.probe_scores(d)
    scr = {(p['giver'], p['receiver'], p['event_frame']) for p in d.get('dho_screens') or []
           if p.get('probe') is None or p['probe'] >= dho.PROBE_MIN}
    prod = [(e['dho_frame'], e['screener_id'], e['handler_id']) for e in dho.with_dho(d) if e.get('dho')]
    inv = {}
    for q, v in map_ids(s, d['player_tracks']).items():
        inv.setdefault(v, q)
    boxes = [b[3] - b[1] for fd in s['players'] for b in fd.values()]
    H = float(np.median(boxes)) if boxes else 100.0
    fi = s['frame_idx']
    labs = assign_labels(cands, labels, fps)
    rows = []
    for c, lab in zip(cands, labs):
        idx = [i for i in range(min(len(fi), len(pose), len(tch))) if abs(fi[i] - c['frame']) <= fps]
        sa, sb = inv.get(c['giver']), inv.get(c['receiver'])
        f = skeleton_features([pose[i].get(sa) if sa is not None else None for i in idx],
                              [pose[i].get(sb) if sb is not None else None for i in idx],
                              [tch[i][1] for i in idx], H)
        h = next((p for k, p in cands_h.items() if k[0] == c['giver'] and k[1] == c['receiver'] and k[2] == c['frame']), {})
        ek = (c['giver'], c['receiver'], c['event_frame'])
        f.update(approach_ft=h.get('approach_ft', float('nan')), v_giver=h.get('v_giver', float('nan')),
                 hold_b_s=h.get('hold_b_s', float('nan')), probe=probe.get(ek, float('nan')) if c['from_event'] else float('nan'),
                 from_screens=float(ek in scr), from_event=float(c['from_event']))
        f = {k: (float('nan') if v is None else v) for k, v in f.items()}
        f['prod'] = any(p[1] == c['giver'] and p[2] == c['receiver'] and abs(p[0] - c['frame']) <= 0.5 * fps
                        or (c['event_frame'] is not None and p[1] == c['giver'] and p[2] == c['receiver']
                            and -0.3 * fps <= c['event_frame'] - p[0] < 1.5 * fps) for p in prod)
        f['label'] = lab
        f['clip'] = Path(pk).stem.replace('_events', '')
        rows.append(f)
    return rows


def all_rows(split):
    """'old' = 舊 476 支（stubs/ball_batch + stubs/pose_hands）、'new' = 新來源 29 段（stubs/pipeline）。"""
    import glob
    import io
    import contextlib
    import dho_model
    if split == 'old':
        lab = dho_model.labels_old()
        items = [(pk, [ROOT / 'stubs' / 'ball_batch' / f'{c}_ball.pkl', ROOT / 'stubs' / 'ball_batch' / f'{c}_touch.pkl',
                       ROOT / 'stubs' / 'pose_hands' / f'{c}.pkl'], lab.get(c, []))
                 for pk in sorted(glob.glob(str(ROOT / 'output_videos' / 'batch' / 'screen_*_events.pkl')))
                 for c in [Path(pk).stem.replace('_events', '')]]
    else:
        lab = dho_model.labels_new()
        items = [(ROOT / 'output_videos' / 'pipeline' / f'{c}_events.pkl', [ROOT / 'stubs' / 'pipeline' / f'{c}_{k}.pkl'
                  for k in ('sam3', 'touch', 'pose')], lab.get(c, [])) for c in sorted(lab)]
    rows = []
    for pk, need, lb in items:
        if all(Path(x).exists() for x in need) and Path(pk).exists():
            with contextlib.redirect_stdout(io.StringIO()):
                rows += clip_rows(pk, need, lb)
    return rows


# ── 交叉驗證比較 ─────────────────────────────────────────────────────────────

def _models():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    return {'梯度提升': lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                                            class_weight='balanced', random_state=0),
            '羅吉斯迴歸': lambda: make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                                           LogisticRegression(C=0.3, class_weight='balanced', max_iter=2000))}


def _pick_threshold(prob, y, min_prec=0.95):
    """準確率 ≥ min_prec 下抓到最多的門檻（沒有的話取 F1 最高）。→ (門檻, 準確率, 抓到數, F1)"""
    best, best_f1 = None, None
    for t in np.arange(0.05, 0.96, 0.05):
        p = prob >= t
        tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
        prec = tp / max(1, tp + fp); f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if prec >= min_prec and (best is None or tp > best[2]):
            best = (float(t), prec, tp, f1)
        if best_f1 is None or f1 > best_f1[3]:
            best_f1 = (float(t), prec, tp, f1)
    return best or best_f1


def evaluate():
    import warnings
    from sklearn.model_selection import GroupKFold
    from sklearn.inspection import permutation_importance
    warnings.filterwarnings('ignore')
    W = ROOT / '.superpowers' / 'sdd' / '2026-11-24-skeleton-dho'
    cache = {sp: W / f'rows_{sp}.pkl' for sp in ('old', 'new')}
    rows = {sp: (pickle.load(open(cache[sp], 'rb')) if cache[sp].exists() else all_rows(sp)) for sp in cache}
    old = [r for r in rows['old'] if r['label'] is not None]
    new = [r for r in rows['new'] if r['label'] is not None]
    y_old = np.array([r['label'] is True for r in old]); g_old = np.array([r['clip'] for r in old])
    y_new = np.array([r['label'] is True for r in new])
    X = lambda rs, cols: np.array([[r[c] for c in cols] for r in rs], dtype=float)
    n_old, n_new = int(y_old.sum()), 27
    print(f'舊 476 支：候選 {len(old)}（手遞手 {n_old}）；新來源：候選 {len(new)}（手遞手 {int(y_new.sum())}，全部 {n_new} 個）')
    pa = np.array([r['prod'] for r in old]); tp = int((pa & y_old).sum()); fp = int((pa & ~y_old).sum())
    pn = np.array([r['prod'] for r in new]); tpn = int((pn & y_new).sum()); fpn = int((pn & ~y_new).sum())
    print(f'(a) 現在上線的做法：舊 準 {tp}/{tp + fp}、抓到 {tp}/{n_old}、F1 {2 * tp / max(1, tp + fp + n_old):.2f}｜新 對 {tpn}、錯 {fpn}、抓到 {tpn}/{n_new}')
    sets = {'(b) 現有特徵': EXIST, '(c) 骨架特徵': SKEL, '(d) 全部': EXIST + SKEL}
    for mname, mk in _models().items():
        for sname, cols in sets.items():
            prob = np.zeros(len(old))
            for tr, va in GroupKFold(n_splits=5).split(old, y_old, g_old):
                m = mk(); m.fit(X(old, cols)[tr], y_old[tr]); prob[va] = m.predict_proba(X(old, cols)[va])[:, 1]
            t, prec, tpc, f1 = _pick_threshold(prob, y_old)
            m = mk(); m.fit(X(old, cols), y_old)
            pnw = m.predict_proba(X(new, cols))[:, 1] >= t
            print(f'{sname} {mname}：舊（交叉驗證，門檻 {t:.2f}）準 {prec:.0%}、抓到 {tpc}/{n_old}、F1 {f1:.2f}'
                  f'｜新 對 {int((pnw & y_new).sum())}、錯 {int((pnw & ~y_new).sum())}、抓到 {int((pnw & y_new).sum())}/{n_new}')
            if sname == '(d) 全部' and mname == '梯度提升':
                imp = permutation_importance(m, X(old, cols), y_old, scoring='average_precision', n_repeats=10, random_state=0)
                order = np.argsort(-imp.importances_mean)
                print('   特徵重要性（舊資料、平均精確度下降）：' + '、'.join(f'{cols[i]} {imp.importances_mean[i]:+.3f}' for i in order))


if __name__ == '__main__':
    if sys.argv[1:2] == ['eval']:
        evaluate()
    else:
        print(__doc__)
