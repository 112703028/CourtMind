"""
pass_gt_probe.py — 使用者標的真傳球（片段、原片影格起迄、#A→#B，編號 = 批次推論 player_tracks）：
看真的傳球在 10fps SAM3 追球資料裡長什麼樣（碰球的人、球看不看得到、球離 A / B 多遠、球速），找偵測方法。

用法：python nba_data_pipeline/pass_gt_probe.py
"""
import sys
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import shot_owner as so
from shot_detect import map_ids
from possession_sam3 import _near

# 2026-11-05 使用者標（framenum_boxes 影片）：(片段, 起, 迄, A, B)；起 > 迄的當寫反、第二個 31 當 32
GT = [(26, 186, 226, 3, 4), (27, 112, 188, 4, 1), (28, 197, 223, 2, 0), (29, 142, 193, 4, 8), (30, 218, 268, 7, 2),
      (31, 107, 135, 4, 2), (32, 143, 189, 3, 1), (33, 230, 286, 6, 1), (34, 255, 286, 8, 2), (35, 420, 471, 7, 2),
      (36, 228, 250, 1, 7), (37, 427, 465, 1, 7), (38, 292, 341, 3, 9), (39, 128, 185, 1, 8), (40, 169, 222, 4, 2),
      (41, 263, 282, 2, 6), (42, 233, 268, 0, 5), (43, 386, 452, 2, 11), (44, 355, 391, 2, 11), (45, 163, 211, 2, 0),
      (46, 256, 291, 3, 5)]


def load(c):
    d = pickle.load(open(ROOT / f'output_videos/batch/screen_{c}_events.pkl', 'rb'))
    s = pickle.load(open(ROOT / f'stubs/ball_batch/screen_{c}_ball.pkl', 'rb'))['shot_sam3']
    _, _, tch = pickle.load(open(ROOT / f'stubs/ball_batch/screen_{c}_touch.pkl', 'rb'))
    return d, s, tch


def main():
    for c, f0, f1, A, B in GT:
        d, s, tch = load(c)
        idmap = map_ids(s, d['player_tracks'])                   # SAM3 10fps 編號 → 批次編號
        fi = list(s['frame_idx'])
        fps = s['orig_fps']
        hs = [b[3] - b[1] for fd in d['player_tracks'] for b in fd.values()]
        Hh = float(np.median(hs)) * s['players'][0].__len__() ** 0 if hs else 150.0
        print(f'\n=== screen_{c}  影格 {f0}-{f1}（{f0 / fps:.1f}~{f1 / fps:.1f}s）#{A} → #{B}   A 有對到 SAM3：{A in idmap.values()}、B：{B in idmap.values()}')
        lo, hi = f0 - int(0.4 * fps), f1 + int(0.4 * fps)
        prev = None
        for i, f in enumerate(fi):
            if not (lo <= f <= hi):
                continue
            p, cen = tch[i]
            who = idmap.get(p, f'?{p}') if p is not None else '—'
            tr = d['player_tracks'][min(f, len(d['player_tracks']) - 1)]
            da = _near(tr.get(A), cen, Hh) if cen is not None else None
            db = _near(tr.get(B), cen, Hh) if cen is not None else None
            sp = (np.hypot(cen[0] - prev[0], cen[1] - prev[1]) / Hh) if (cen is not None and prev is not None) else None
            prev = cen if cen is not None else prev
            mark = '◀起' if abs(f - f0) <= 2 else ('◀迄' if abs(f - f1) <= 2 else '')
            print(f'  f{f:4d}  碰球 {str(who):>4s}  球{"✓" if cen is not None else "✗"}'
                  f'  離A {"—" if da is None else f"{da:4.1f}"}  離B {"—" if db is None else f"{db:4.1f}"}'
                  f'  速度 {"—" if sp is None else f"{sp:4.2f}"} {mark}')


if __name__ == '__main__':
    main()
