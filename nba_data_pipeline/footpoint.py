"""
footpoint.py — 用 SAM3 遮罩找腳的位置，重算訓練快取的球場座標（court_tracks）。

背景（2026-10-26 量）：清理後持球者「有框沒座標」12.5%（93% 是同張只有他沒有 → 投影超出球場邊界 5 ft 被丟），
相鄰兩張 > 8 ft 的單人跳動 2176 張。原本腳的位置 = 框底部中間（convert_manual.foot_pos_xyxy），
被擋住 / 兩人框疊在一起 / 框被畫面切掉時，框底部就不是腳。

做法：
  1. 每張的 homography 不用重跑球場關鍵點 —— 快取裡同一張每個人「框底部像素 → 球場座標」都是同一個 H 算的，
     ≥4 人就能反推回來（重投影誤差 > 0.5 ft 的當不可靠，那張保留原座標）。
  2. 腳 = 遮罩最下面 5% 高度那幾排像素：x 取平均、y 取最低。遮罩用 bbox IoU ≥ 0.5 對回快取 pid。
  3. 投影 + 同樣的邊界過濾（球場外 5 ft 丟掉）。找不到遮罩的人沿用框底部。

用法：from footpoint import mask_court_tracks；ct = mask_court_tracks(c)（遮罩在 stubs/train_sam3_masks/）
"""
import pickle
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
MASK_DIR = ROOT / 'stubs' / 'train_sam3_masks'
COURT_W_FT, COURT_H_FT = 94.0, 50.0
BAND = 0.05


def unpack_mask(packed):
    bbox, h, w, data = packed
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:h * w]
    return bbox, bits.reshape(h, w).astype(bool)


def bbox_foot(b):
    return ((b[0] + b[2]) / 2, b[3])


def mask_foot(packed):
    (x1, y1, _, _), m = unpack_mask(packed)
    rows = np.where(m.any(1))[0]
    if not len(rows):
        return None
    ymax = rows[-1]
    band = max(2, int(BAND * (rows[-1] - rows[0] + 1)))
    ys, xs = np.nonzero(m[max(0, ymax - band):ymax + 1])
    return (float(xs.mean()) + x1, float(ymax) + y1)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def recover_h(raw_f, ct_f, max_err=0.5):
    """同一張的 (框底部像素, 球場座標) ≥4 對 → homography（像素 → 英尺）；重投影誤差太大回 None。"""
    pids = [p for p in ct_f if p in raw_f]
    if len(pids) < 4:
        return None
    src = np.float32([bbox_foot(raw_f[p]) for p in pids]); dst = np.float32([ct_f[p] for p in pids])
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return None
    err = np.abs(cv2.perspectiveTransform(src[None], H)[0] - dst).max()
    return H if err <= max_err else None


def project(H, pt):
    x, y = cv2.perspectiveTransform(np.float32([[pt]]), H)[0, 0]
    return (float(x), float(y)) if (0 <= x <= COURT_W_FT + 5 and 0 <= y <= COURT_H_FT + 5) else None


def mask_court_tracks(c, return_stats=False):
    """→ 新的 court_tracks（list[dict[pid: (x, y) ft]]）。沒有遮罩檔 → None。"""
    mp, tp = MASK_DIR / f"{c['clip']}_sam3_masks.pkl", MASK_DIR / f"{c['clip']}_sam3_tracks.pkl"
    if not (mp.exists() and tp.exists()):
        return None
    masks = pickle.load(open(mp, 'rb'))['masks']
    tracks = pickle.load(open(tp, 'rb'))['player_tracks']
    out, st = [], dict(frames=0, no_h=0, mask=0, bbox=0)
    for i, (raw_f, ct_f) in enumerate(zip(c['raw_tracks'], c['court_tracks'])):
        H = recover_h(raw_f, ct_f)
        st['frames'] += 1
        if H is None or i >= len(masks):
            st['no_h'] += H is None
            out.append(dict(ct_f)); continue
        nf = {}
        for pid, b in raw_f.items():
            best = max(((_iou(b, v['bbox']), sp) for sp, v in tracks[i].items() if sp in masks[i]), default=(0, None))
            foot = mask_foot(masks[i][best[1]]) if best[0] >= 0.5 else None
            st['mask' if foot else 'bbox'] += 1
            pt = project(H, foot or bbox_foot(b))
            if pt is not None:
                nf[pid] = pt
        out.append(nf)
    return (out, st) if return_stats else out
