"""
possession.py — 訓練 / 評估用的逐張持球者（SAM3 球遮罩 × 球員遮罩），編號是訓練快取的 pid。

背景：ScreenNet 原本整支片段只有一個 handler_tid；傳球之後掩護是幫「新的持球者」擋的，
模型卻還以舊持球者為中心排列 → 「持球者離掩護者 > 15 ft」的漏抓（實驗紀錄一-8，28 個）。

做法（跟 shot_owner 同一套，但訓練片段沒有 Roboflow → 球物體改用「最常碰到球員的那顆」選）：
  每張：球遮罩跟哪個球員遮罩（往外擴 DILATE 像素）重疊最多 = 持球者；只碰 1 張、前後不是他的當沒碰到。
  SAM3 編號 → 快取 pid：同一張 bbox IoU ≥ 0.5。

輸入：stubs/train_sam3_masks/<clip>_sam3_masks.pkl、<clip>_sam3_tracks.pkl（球員）、
      stubs/train_sam3_ball/<clip>_ball.pkl（sam3_ball_train.py）
輸出：holders(c) → list（每張 5fps 影格的持球者快取 pid 或 None），存 stubs/train_possession/<clip>.pkl
"""
import pickle
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
MASK_DIR = ROOT / 'stubs' / 'train_sam3_masks'
BALL_DIR = ROOT / 'stubs' / 'train_sam3_ball'
OUT_DIR = ROOT / 'stubs' / 'train_possession'
DILATE = 6
MIN_WIN = 3          # 窗內（10 張）至少拿球幾張才算這個窗的持球者


def unpack_mask(packed):
    bbox, h, w, data = packed
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:h * w]
    return bbox, bits.reshape(h, w).astype(bool)


def _paste(packed, shape):
    (x1, y1, _, _), m = unpack_mask(packed)
    out = np.zeros(shape, bool)
    hh, ww = min(m.shape[0], shape[0] - y1), min(m.shape[1], shape[1] - x1)
    out[y1:y1 + hh, x1:x1 + ww] = m[:hh, :ww]
    return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def _touch(ball_packed, players, k):
    """→ (球遮罩跟哪個球員遮罩（往外擴）重疊最多的 SAM3 pid 或 None, 球中心)。"""
    boxes = [ball_packed[0]] + [p[0] for p in players.values()]
    shape = (max(b[3] for b in boxes) + 2, max(b[2] for b in boxes) + 2)
    ball = _paste(ball_packed, shape)
    ys, xs = np.nonzero(ball)
    center = (float(xs.mean()), float(ys.mean())) if len(xs) else None
    bb = ball_packed[0]
    best, bo = None, 0
    for pid, p in players.items():
        b = p[0]
        if b[2] < bb[0] - 3 * DILATE or b[0] > bb[2] + 3 * DILATE or b[3] < bb[1] - 3 * DILATE or b[1] > bb[3] + 3 * DILATE:
            continue                                        # 框離球太遠，不可能碰到
        ov = int((cv2.dilate(_paste(p, shape).astype(np.uint8), k).astype(bool) & ball).sum())
        if ov > bo: best, bo = pid, ov
    return best, center


def holders(c, recompute=False):
    """訓練快取 c → 每張 5fps 影格的持球者（快取 pid 或 None）。沒有遮罩檔 → None（呼叫端退回 handler_tid）。"""
    clip = c['clip']
    out_p = OUT_DIR / f'{clip}.pkl'
    if out_p.exists() and not recompute:
        return pickle.load(open(out_p, 'rb'))
    mp, tp, bp = MASK_DIR / f'{clip}_sam3_masks.pkl', MASK_DIR / f'{clip}_sam3_tracks.pkl', BALL_DIR / f'{clip}_ball.pkl'
    if not (mp.exists() and tp.exists() and bp.exists()):
        return None
    masks = pickle.load(open(mp, 'rb'))['masks']
    tracks = pickle.load(open(tp, 'rb'))['player_tracks']
    bd = pickle.load(open(bp, 'rb'))
    n = min(len(masks), len(bd['ball_masks']), c['n'])
    k = np.ones((2 * DILATE + 1, 2 * DILATE + 1), np.uint8)

    # 球物體：比分條上的球圖示 / 觀眾席的東西不會碰到球員 → 選「碰到球員次數最多」的那顆
    per_obj = {}
    for i in range(n):
        for o, bm in bd['ball_masks'][i].items():
            pid, cen = _touch(bm, masks[i], k)
            per_obj.setdefault(o, {})[i] = (pid, cen)
    order = sorted(per_obj, key=lambda o: -sum(1 for v in per_obj[o].values() if v[0] is not None))
    seq, prev = [], None
    for i in range(n):
        here = [o for o in order if i in per_obj[o]]
        o = here[0] if here else None
        if here and prev is not None and o != order[0]:
            o = min(here, key=lambda q: np.hypot(per_obj[q][i][1][0] - prev[0], per_obj[q][i][1][1] - prev[1])
                    if per_obj[q][i][1] else 1e9)
        pid, cen = per_obj[o][i] if o is not None else (None, None)
        if cen is not None: prev = cen
        # SAM3 pid → 快取 pid（同一張 bbox IoU）
        cp = None
        if pid is not None and pid in tracks[i] and i < len(c['raw_tracks']):
            sb = tracks[i][pid]['bbox']
            best = max(c['raw_tracks'][i].items(), key=lambda kv: _iou(sb, kv[1]), default=None)
            if best and _iou(sb, best[1]) >= 0.5:
                cp = best[0]
        seq.append(cp)
    seq += [None] * (c['n'] - len(seq))
    sm = list(seq)                                          # 只碰 1 張、前後都不是他 → 當沒碰到
    for i in range(1, len(seq) - 1):
        if seq[i] is not None and seq[i - 1] != seq[i] and seq[i + 1] != seq[i]:
            sm[i] = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(sm, open(out_p, 'wb'))
    return sm


def window_handler(poss, start, seq_len, team, H):
    """窗內拿球最多的進攻球員（≥ MIN_WIN 張）；不是 H 才回傳，否則 None。"""
    if not poss:
        return None
    cnt = Counter(p for p in poss[start:start + seq_len] if p is not None and team.get(p) == 'offense')
    if not cnt:
        return None
    p, k = cnt.most_common(1)[0]
    return p if (k >= MIN_WIN and p != H) else None
