"""
court_v2.py — 球場座標第 2 版：homography 只用可信的球場關鍵點。

原本（convert_manual.build_homography）把模型輸出的 18 點全部拿去算（只檢查 x, y > 0），
但模型對看不到的點也會給位置（信心 0.01~0.1），且常把左 / 右底線同時給中等信心 → 互相矛盾的點一起算，
homography 歪掉 / 左右翻（鏡像閃爍的來源）。

第 2 版：每點信心 ≥ THR、在畫面內、只用信心總和較高的那一邊（左 / 右底線 + 罰球線角）+ 中場線兩點；
不到 4 點 → 借前後 1~2 張的 homography（鏡頭連續移動）。腳的位置、邊界過濾同原本。

460 支訓練片段（清理後）：持球者沒座標 12.5% → 5.9%、相鄰 > 8 ft 跳動 5.8% → 1.4%、> 30 ft/s 9.3% → 3.3%
（門檻 0.3 / 0.4 / 0.5 × 借前後張 / 退回舊方法 比較見實驗紀錄）。

關鍵點：stubs/court_kp/<clip>.pkl（court_kp_dump.py，每張 18 點 xy + conf）。
"""
import pickle
from pathlib import Path

import cv2
import numpy as np

import convert_manual as cm

ROOT = Path(__file__).parent.parent
KP_DIR = ROOT / 'stubs' / 'court_kp'
THR = 0.4
LEFT, RIGHT, MID = [0, 1, 2, 3, 4, 5, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17], [6, 7]
TARGET = np.float32(cm.TARGET_KPS)


def homography(xy, conf, w, h, thr=THR):
    side = LEFT if conf[LEFT].sum() >= conf[RIGHT].sum() else RIGHT
    ks = [k for k in side + MID if conf[k] >= thr and 0 <= xy[k, 0] < w and 0 <= xy[k, 1] < h]
    if len(ks) < 4:
        return None
    H, _ = cv2.findHomography(np.float32(xy[ks]), TARGET[ks], cv2.RANSAC, 5.0)
    return H


def homographies(kp, n, thr=THR):
    w, h = kp['size']
    Hs = [homography(kp['xy'][i], kp['conf'][i], w, h, thr) if i < len(kp['xy']) else None for i in range(n)]
    return [H if H is not None else
            next((Hs[j] for d in (1, -1, 2, -2) for j in [i + d] if 0 <= j < n and Hs[j] is not None), None)
            for i, H in enumerate(Hs)]


def project(H, xyxy):
    if H is None:
        return None
    x, y = cm.tactical_to_feet(*cm.transform(H, *cm.foot_pos_xyxy(xyxy)))
    return (x, y) if (0 <= x <= cm.COURT_W_FT + 5 and 0 <= y <= cm.COURT_H_FT + 5) else None


def stable_feet(raw_tracks, win=5, h_tol=0.3, top_tol=0.3, w_max=1.5):
    """每人每張的腳點（像素）。框底部 = 腳，但兩人重疊時框會一下包兩個人、一下只剩上半身 → 底部亂跳
    （2026-11-01：清理後單人跳動 1319 次，71% 是框高度突變，其中 820 次只有底部變、頂部不動）。
      高度跟前後 win 張的中位數差 ≤ h_tol → 照舊用框底部中間；
      否則頭頂跟前後正常張內插的頭頂差 ≤ top_tol 倍身高、框沒有變寬 → 腳 = 頭頂 + 正常身高；
      都不行（連頭頂一起跳 / 框包兩個人）→ None（這張不給座標，交給清理內插）。
    回 list[dict[pid: (x, y) 或 None]]。"""
    n = len(raw_tracks)
    out = [dict() for _ in range(n)]
    pids = {p for fd in raw_tracks for p in fd}
    for pid in pids:
        fr = [i for i in range(n) if pid in raw_tracks[i]]
        H = {i: raw_tracks[i][pid][3] - raw_tracks[i][pid][1] for i in fr}
        W = {i: raw_tracks[i][pid][2] - raw_tracks[i][pid][0] for i in fr}
        med_h = {i: float(np.median([H[j] for j in fr if abs(j - i) <= win])) for i in fr}
        med_w = {i: float(np.median([W[j] for j in fr if abs(j - i) <= win])) for i in fr}
        normal = {i: abs(H[i] / max(med_h[i], 1) - 1) <= h_tol for i in fr}
        for i in fr:
            x1, y1, x2, y2 = raw_tracks[i][pid]
            if normal[i]:
                out[i][pid] = ((x1 + x2) / 2, y2); continue
            prv = next((j for j in reversed(fr) if j < i and i - j <= win and normal[j]), None)
            nxt = next((j for j in fr if j > i and j - i <= win and normal[j]), None)
            tops = [(j, raw_tracks[j][pid][1]) for j in (prv, nxt) if j is not None]
            if not tops:
                out[i][pid] = None; continue
            if len(tops) == 2:
                (ja, ta), (jb, tb) = tops
                exp_top = ta + (tb - ta) * (i - ja) / (jb - ja)
            else:
                exp_top = tops[0][1]
            if abs(y1 - exp_top) <= top_tol * med_h[i] and W[i] <= w_max * med_w[i]:
                out[i][pid] = ((x1 + x2) / 2, y1 + med_h[i])
            else:
                out[i][pid] = None
    return out


def stable_court_tracks(raw_tracks, Hs, gap=2):
    """stable_feet 投影到球場；被判不可信（None）的那張，前後 gap 張內都有座標 → 球場座標線性內插補回
    （只補這種張，不補本來就沒框 / 沒 homography 的）。"""
    F = stable_feet(raw_tracks)
    n = len(raw_tracks)
    out = [{pid: pt for pid, f in F[i].items() if (pt := project_pt(Hs[i], f)) is not None} for i in range(n)]
    for i in range(n):
        for pid, f in F[i].items():
            if f is not None:
                continue
            a = next((j for j in range(i - 1, max(-1, i - gap - 1), -1) if pid in out[j]), None)
            b = next((j for j in range(i + 1, min(n, i + gap + 1)) if pid in out[j]), None)
            if a is not None and b is not None:
                t = (i - a) / (b - a)
                out[i][pid] = tuple(pa + (pb - pa) * t for pa, pb in zip(out[a][pid], out[b][pid]))
    return out


def project_pt(H, pt):
    if H is None or pt is None:
        return None
    x, y = cm.tactical_to_feet(*cm.transform(H, *pt))
    return (x, y) if (0 <= x <= cm.COURT_W_FT + 5 and 0 <= y <= cm.COURT_H_FT + 5) else None


def court_tracks(c, thr=THR, feet='bbox'):
    """feet='stable'：腳點用 stable_feet（重疊時用頭頂 + 正常身高）；'bbox' = 框底部。
    訓練快取 c → 第 2 版 court_tracks（list[dict[pid: (x, y) ft]]）；沒有關鍵點檔回 None。"""
    p = KP_DIR / f"{c['clip']}.pkl"
    if not p.exists():
        return None
    Hs = homographies(pickle.load(open(p, 'rb')), len(c['raw_tracks']), thr)
    if feet == 'stable':
        return stable_court_tracks(c['raw_tracks'], Hs)
    return [{pid: pt for pid, b in fd.items() if (pt := project(Hs[i], b)) is not None}
            for i, fd in enumerate(c['raw_tracks'])]
