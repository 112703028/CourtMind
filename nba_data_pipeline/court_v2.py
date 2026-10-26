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


def court_tracks(c, thr=THR):
    """訓練快取 c → 第 2 版 court_tracks（list[dict[pid: (x, y) ft]]）；沒有關鍵點檔回 None。"""
    p = KP_DIR / f"{c['clip']}.pkl"
    if not p.exists():
        return None
    Hs = homographies(pickle.load(open(p, 'rb')), len(c['raw_tracks']), thr)
    return [{pid: pt for pid, b in fd.items() if (pt := project(Hs[i], b)) is not None}
            for i, fd in enumerate(c['raw_tracks'])]
