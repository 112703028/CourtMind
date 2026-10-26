"""
pose_features.py — ScreenNet 的姿態 / 遮罩輸入（每人每張 6 個數字，接在 x, y 後面 → 8 維）

  0 面向持球者     遮罩版姿態：鼻子相對兩肩中點（左右）+ 左右肩順序（面向 / 背對鏡頭）→ 畫面上的面向，
                   跟「自己腳點 → 持球者腳點」的方向取 cos：1 正面朝向、0 側對、−1 背對
  1 面向候選掩護者 同上，對象改成 slot 1（每個窗每個候選人輪流放 slot 1 → 每次重算）
  2 站姿寬度       SAM3 遮罩最下面 20% 的寬度 ÷ 遮罩高度（掩護者站定張腳）
  3 身體正不正     左右肩 x 距離 ÷ 身高（寬 = 正面 / 背面、窄 = 側身）
  4 跟對手接觸量   遮罩擴張後跟對方任一人遮罩重疊的像素 ÷ 身高（取最大）
  5 有沒有資料     這張有姿態或遮罩 = 1；被擋住 / 沒追到 = 0（其他欄位此時都是 0）

資料：pose_kpts2.json（遮罩版姿態，pid 已對回快取）、stubs/train_sam3_masks/（screen_features.load_masks 對回快取 pid）。
每支片段算一次存 stubs/pose_feat_cache/<clip>.pkl：{影格: {pid: (fx, fy, 站姿, 正不正, 接觸, 腳點x, 腳點y)}}
"""
import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE_DIR = ROOT / 'stubs' / 'pose_feat_cache'
KP_PATH = ROOT / 'pose_kpts2.json'
MASK_DIR = ROOT / 'stubs' / 'train_sam3_masks'
N_CH = 6
TH = 0.4
_KP = None


def _kp():
    global _KP
    if _KP is None:
        _KP = json.load(open(KP_PATH))
    return _KP


def _face(k, h):
    """→ (fx, fy) 單位向量或 None；正不正（|左肩x − 右肩x| / h）或 nan"""
    if k[5][2] < TH or k[6][2] < TH:
        return None, np.nan
    mx = (k[5][0] + k[6][0]) / 2
    v = np.array([np.clip((k[0][0] - mx) / h * 4, -1, 1) if k[0][2] >= TH else 0.0,
                  np.clip((k[5][0] - k[6][0]) / h * 4, -1, 1)])
    n = np.linalg.norm(v)
    return (v / n if n > 0.15 else None), abs(k[5][0] - k[6][0]) / h


def build_clip(c):
    """一支片段 → {f: {pid: (fx, fy, stance, square, contact, foot_x, foot_y)}}（沒有的值用 nan）"""
    from screen_features import load_masks, contact_info, _shape
    kp = _kp().get(c['clip'])
    ms = load_masks(c, MASK_DIR)
    team = c['track_team']
    out = {}
    for f in range(c['n']):
        fd = {}
        for p, bb in c['raw_tracks'][f].items():
            h = max(1.0, bb[3] - bb[1])
            face, square = (None, np.nan)
            v = kp[f].get(str(p)) if kp is not None and f < len(kp) else None
            if v:
                face, square = _face(v['k'], h)
            stance = contact = np.nan
            if ms is not None and p in ms[f]:
                stance = _shape(ms[f][p])[0]
                opp = [q for q in ms[f] if q != p and team.get(q) and team.get(q) != team.get(p)]
                contact = max([contact_info(ms[f][p], ms[f][q])[0] for q in opp] or [0.0])
            fx, fy = (face if face is not None else (np.nan, np.nan))
            fd[p] = (fx, fy, stance, square, contact, (bb[0] + bb[2]) / 2, bb[3])
        out[f] = fd
    return out


def load_clip(c):
    p = CACHE_DIR / f"{c['clip']}.pkl"
    if p.exists():
        return pickle.load(open(p, 'rb'))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pf = build_clip(c)
    pickle.dump(pf, open(p, 'wb'))
    return pf


def channels(pf, frames, order, H, cand):
    """一個窗的姿態通道 → (10, len(frames), 6)；order = 10 個 slot 的 pid（None = 缺人）。"""
    out = np.zeros((len(order), len(frames), N_CH), np.float32)
    for t, f in enumerate(frames):
        fd = pf.get(f, {})
        for i, p in enumerate(order):
            if p is None or p not in fd:
                continue
            fx, fy, stance, square, contact, ux, uy = fd[p]
            have = False
            if not np.isnan(fx):
                for j, q in ((0, H), (1, cand)):
                    if q is not None and q != p and q in fd:
                        d = np.array([fd[q][5] - ux, fd[q][6] - uy])
                        n = np.linalg.norm(d)
                        if n > 1:
                            out[i, t, j] = float(np.dot([fx, fy], d) / n)
                have = True
            contact = min(contact, 10.0) if not np.isnan(contact) else contact     # 撞上那刻約 5~10，封頂避免單一值過大
            for j, v in ((2, stance), (3, square), (4, contact)):
                if not np.isnan(v):
                    out[i, t, j] = v
                    have = True
            out[i, t, 5] = float(have)
    return out
