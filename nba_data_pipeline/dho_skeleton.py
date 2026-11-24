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
