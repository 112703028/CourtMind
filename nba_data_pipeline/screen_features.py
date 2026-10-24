"""
screen_features.py — 第二階段檢查器用的掩護特徵（一個候選掩護者 j 在影格 f）

座標特徵（court_tracks 要先 clean_court_tracks）：
  dist_def    離最近防守者（ft）
  dist_h      離持球者
  speed       前後 2 張的平均速度（站定程度）
  hit_hdef    撞到的防守者 = 持球者的防守者（前 1 秒平均離持球者最近的防守者）
  dist_hdef   離持球者的防守者
  seg_dist    離「持球者–其防守者」線段（NETS 的三角形條件）
  h_approach  前 3 張持球者往 j 靠近的速度
  h_pass      之後 1 秒持球者離 j 的最近距離
遮罩特徵（SAM3 遮罩，沒有就是 NaN）：
  contact     j 遮罩擴張後跟防守者遮罩重疊像素 ÷ j 身高（取接觸最大的防守者 = 撞到的防守者）
  upper       同上，只算身高上面 65%
  stance      j 遮罩最下面 20% 的寬度 ÷ 身高（站定張腳）
  aspect      j 遮罩寬 ÷ 高
  contact_dur 前後 ±3 張跟同一防守者接觸 > 1 的張數
  foot_dy     兩人遮罩最低點的畫面高度差 ÷ 身高

用法：
  feat = ClipFeatures(c, masks)        # c = 訓練快取 dict；masks = load_masks(c, mask_dir) 或 None
  x = feat(j, f)                        # → dict 或 None（j 那張沒座標）
"""
import math
import pickle
from pathlib import Path

import numpy as np
import cv2

from tactics_tools import clean_court_tracks

UPPER = 0.65
NAMES = ['dist_def', 'dist_h', 'speed', 'hit_hdef', 'dist_hdef', 'seg_dist', 'h_approach', 'h_pass',
         'contact', 'upper', 'stance', 'aspect', 'contact_dur', 'foot_dy']
MASK_NAMES = ['contact', 'upper', 'stance', 'aspect', 'contact_dur', 'foot_dy']


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def load_masks(c, mask_dir):
    """run_sam3_tracks --save_masks 的輸出 → list[{原本 pid: (bbox, bool 遮罩)}]。
    重跑的 SAM3 編號跟快取不完全一樣 → 每幀用 bbox IoU ≥ 0.5 對回快取的 pid。檔案不存在 → None。"""
    mask_dir = Path(mask_dir)
    tp, mp = mask_dir / f"{c['clip']}_sam3_tracks.pkl", mask_dir / f"{c['clip']}_sam3_masks.pkl"
    if not (tp.exists() and mp.exists()):
        return None
    tr = pickle.load(open(tp, 'rb'))['player_tracks']
    mk = pickle.load(open(mp, 'rb'))['masks']
    out = []
    for f in range(c['n']):
        fm = {}
        new = {o: d['bbox'] for o, d in tr[f].items()} if f < len(tr) else {}
        pairs = sorted(((_iou(b, nb), p, o) for p, b in c['raw_tracks'][f].items() for o, nb in new.items()),
                       reverse=True)
        used_p, used_o = set(), set()
        for v, p, o in pairs:
            if v < 0.5 or p in used_p or o in used_o:
                continue
            used_p.add(p); used_o.add(o)
            bb, h, w, data = mk[f][o]
            fm[p] = (bb, np.unpackbits(np.frombuffer(data, np.uint8))[:h * w].reshape(h, w).astype(bool))
        out.append(fm)
    return out


def contact_info(ma, mb):
    """→ (接觸長度/身高, 上半身接觸長度/身高)"""
    (ba, A), (bb, B) = ma, mb
    h = ba[3] - ba[1] + 1
    k = max(3, int(0.03 * h))
    if (ba[0] > bb[2] + k) or (bb[0] > ba[2] + k) or (ba[1] > bb[3] + k) or (bb[1] > ba[3] + k):
        return 0.0, 0.0
    x1, y1 = min(ba[0], bb[0]) - k, min(ba[1], bb[1]) - k
    x2, y2 = max(ba[2], bb[2]) + k, max(ba[3], bb[3]) + k
    ca = np.zeros((y2 - y1 + 1, x2 - x1 + 1), np.uint8)
    cb = np.zeros_like(ca)
    ca[ba[1] - y1:ba[3] - y1 + 1, ba[0] - x1:ba[2] - x1 + 1] = A
    cb[bb[1] - y1:bb[3] - y1 + 1, bb[0] - x1:bb[2] - x1 + 1] = B
    ys = np.nonzero(cv2.dilate(ca, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) & cb)[0]
    if not len(ys):
        return 0.0, 0.0
    rel = (ys + y1 - ba[1]) / h
    return len(ys) / h, float((rel <= UPPER).sum()) / h


def _shape(m):
    bb, M = m
    ys, xs = np.nonzero(M)
    if len(ys) < 20:
        return np.nan, np.nan, np.nan, np.nan
    h = ys.max() - ys.min() + 1
    low = xs[ys >= ys.min() + 0.8 * h]
    return (low.max() - low.min() + 1) / h, (xs.max() - xs.min() + 1) / h, bb[1] + ys.max(), h


def _seg_dist(p, a, b):
    p, a, b = np.array(p), np.array(a), np.array(b)
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


class ClipFeatures:
    def __init__(self, c, masks=None):
        self.c, self.ms = c, masks
        self.n, self.team, self.H = c['n'], c['track_team'], c['handler_tid']
        self.ct = clean_court_tracks(c['court_tracks'])[0]
        self.defs = [p for p, t in self.team.items() if t == 'defense']
        self._cont = {}

    def pos(self, p, f):
        return self.ct[f].get(p) if p is not None and 0 <= f < self.n else None

    def dist(self, p, q, f):
        a, b = self.pos(p, f), self.pos(q, f)
        return math.dist(a, b) if a and b else np.nan

    def speed(self, p, f0, f1):
        v = [math.dist(self.pos(p, g), self.pos(p, g + 1)) for g in range(f0, f1)
             if self.pos(p, g) and self.pos(p, g + 1)]
        return float(np.mean(v)) if v else np.nan

    def contacts(self, j, f):
        """{防守者: (接觸長度, 上半身)}，有快取"""
        if (j, f) not in self._cont:
            fm = self.ms[f] if self.ms is not None and 0 <= f < self.n else {}
            self._cont[(j, f)] = {d: contact_info(fm[j], fm[d]) for d in self.defs if d in fm} if j in fm else {}
        return self._cont[(j, f)]

    def hdef(self, f):
        cand = []
        for d in self.defs:
            ds = [self.dist(d, self.H, g) for g in range(f - 5, f)]
            ds = [x for x in ds if not np.isnan(x)]
            if ds:
                cand.append((np.mean(ds), d))
        return min(cand)[1] if cand else None

    def __call__(self, j, f):
        if not self.pos(j, f):
            return None
        dd = [(self.dist(j, d, f), d) for d in self.defs if self.pos(d, f)]
        if not dd:
            return None
        H = self.H
        cd = self.contacts(j, f)
        if cd and max(v[0] for v in cd.values()) > 0:
            Dc = max(cd, key=lambda d: cd[d][0])
        else:
            Dc = min(dd)[1]
        Dh = self.hdef(f)
        ph0, ph1 = self.dist(j, H, f - 3), self.dist(j, H, f)
        hp = [x for x in (self.dist(j, H, g) for g in range(f, f + 6)) if not np.isnan(x)]
        x = {
            'dist_def': min(dd)[0],
            'dist_h': self.dist(j, H, f),
            'speed': self.speed(j, f - 2, f + 2),
            'hit_hdef': float(Dh is not None and Dc == Dh),
            'dist_hdef': self.dist(j, Dh, f),
            'seg_dist': _seg_dist(self.pos(j, f), self.pos(H, f), self.pos(Dh, f))
            if self.pos(H, f) and self.pos(Dh, f) else np.nan,
            'h_approach': (ph0 - ph1) / 3 if not (np.isnan(ph0) or np.isnan(ph1)) else np.nan,
            'h_pass': min(hp) if hp else np.nan,
        }
        fm = self.ms[f] if self.ms is not None else {}
        if j in fm:
            st, asp, fy_j, hj = _shape(fm[j])
            fy_d = _shape(fm[Dc])[2] if Dc in fm else np.nan
            x.update({
                'contact': cd.get(Dc, (0.0, 0.0))[0], 'upper': cd.get(Dc, (0.0, 0.0))[1],
                'stance': st, 'aspect': asp,
                'contact_dur': float(sum(1 for g in range(f - 3, f + 4)
                                         if self.contacts(j, g).get(Dc, (0.0, 0.0))[0] > 1.0)),
                'foot_dy': abs(fy_j - fy_d) / hj if not np.isnan(fy_d) else np.nan,
            })
        else:
            x.update({k: np.nan for k in MASK_NAMES})
        return x
