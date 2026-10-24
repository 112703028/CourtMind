"""
render_matchup_board.py — 把 HMM 防守配對畫成俯視戰術板影片，用眼睛驗。

讀 events pkl（court_coords/player_assignment/events/fps），跑 defensive_matchup_hmm，
每幀畫：球場 + 進攻紅點 / 防守藍點 / 持球者金圈 + 防守者連線到他守的進攻者 + 幀號。
不需原影片、不需 bbox、純本地（court_coords 本來就是鳥瞰座標）。

用法：
  python render_matchup_board.py --pkl output_videos/screens_v118_sam3_events.pkl \
      --out output_videos/board_v118.mp4
"""
import os
import sys
import pickle
import argparse
import collections as _c
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from tactics_tools import (defensive_matchup_hmm, matchup_at_frame,
                           infer_target_basket)
from utils import save_video

# 顏色 BGR
C_OFF = (60, 60, 220)     # 進攻 紅
C_DEF = (200, 130, 40)    # 防守 藍
C_HANDLER = (40, 200, 240)  # 持球者 金
C_LINE = (90, 90, 90)     # 連線 灰
C_COURT = (70, 120, 70)   # 球場線 綠
C_BG = (40, 55, 40)       # 底 深綠


def court_to_px(pos, cw, ch, W, H, pad):
    """球場座標(ft) → 畫布像素。"""
    x, y = pos
    px = pad + (x / cw) * (W - 2 * pad)
    py = pad + (y / ch) * (H - 2 * pad)
    return int(px), int(py)


def draw_court(W, H, pad, cw, ch, basket):
    img = np.full((H, W, 3), C_BG, np.uint8)
    cv2.rectangle(img, (pad, pad), (W - pad, H - pad), C_COURT, 2)
    # 中線
    midx = pad + (W - 2 * pad) // 2
    cv2.line(img, (midx, pad), (midx, H - pad), C_COURT, 1)
    # 中圈
    cv2.circle(img, (midx, H // 2), int((W - 2 * pad) * 0.06), C_COURT, 1)
    # 籃框
    bx, by = court_to_px(basket, cw, ch, W, H, pad)
    cv2.circle(img, (bx, by), 7, (0, 140, 230), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--w', type=int, default=960)
    ap.add_argument('--sigma', type=float, default=5.0)
    ap.add_argument('--switch_pen', type=float, default=8.0)
    ap.add_argument('--min_seg', type=int, default=30)
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, 'rb'))
    cc = d['court_coords']
    pa = d['player_assignment']
    ev = d.get('events', [])
    fps = d.get('fps', 30.0)
    cw, ch = d.get('court_dims', (94.0, 50.0))
    n = len(cc)
    if not ev:
        print("無 events，無法定 handler")
        return
    handler = ev[0]['handler_id']
    basket = infer_target_basket(cc, 0, n - 1)

    matchup = defensive_matchup_hmm(cc, pa, handler, basket,
                                    sigma=args.sigma, switch_pen=args.switch_pen,
                                    min_seg=args.min_seg)

    # 分隊（決定紅藍）
    votes = {}
    for fr in pa:
        for pid, t in fr.items():
            votes.setdefault(pid, _c.Counter())[t] += 1
    maj = {pid: c.most_common(1)[0][0] for pid, c in votes.items()}
    offt = maj.get(handler)
    defenders = set(matchup)
    offense = set(p for p in maj if maj[p] == offt)

    W = args.w
    H = int(W * ch / cw)
    pad = 40
    frames = []
    for f in range(n):
        img = draw_court(W, H, pad, cw, ch, basket)
        snap = matchup_at_frame(matchup, f)   # {defender: offender}

        # 先畫連線（在點下層）
        for D, O in snap.items():
            if D in cc[f] and O in cc[f]:
                p1 = court_to_px(cc[f][D], cw, ch, W, H, pad)
                p2 = court_to_px(cc[f][O], cw, ch, W, H, pad)
                cv2.line(img, p1, p2, C_LINE, 1, cv2.LINE_AA)

        # 畫球員點
        for pid, pos in cc[f].items():
            px, py = court_to_px(pos, cw, ch, W, H, pad)
            if pid == handler:
                col = C_HANDLER
            elif pid in offense:
                col = C_OFF
            elif pid in defenders:
                col = C_DEF
            else:
                col = (120, 120, 120)   # 非球員/被濾掉
            cv2.circle(img, (px, py), 11, col, -1)
            cv2.circle(img, (px, py), 11, (20, 20, 20), 1)
            # 持球者外圈金環
            if pid == handler:
                cv2.circle(img, (px, py), 15, C_HANDLER, 2)
            cv2.putText(img, str(pid), (px - 6, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 幀號 + 圖例
        cv2.rectangle(img, (6, 6), (190, 30), (0, 0, 0), -1)
        cv2.putText(img, f"frame {f}/{n-1}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "red=OFF  blue=DEF  gold=handler  line=guards",
                    (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        frames.append(img)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    save_video(frames, args.out, fps=fps)
    print(f"寫出 {args.out}  ({n} 幀 @ {fps:.0f}fps)  handler=#{handler}")
    print(f"  進攻{sorted(offense & set().union(*[set(cc[f]) for f in range(n)]))}")
    print(f"  防守{sorted(defenders)}")


if __name__ == '__main__':
    main()
