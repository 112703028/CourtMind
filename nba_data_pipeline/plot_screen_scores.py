"""
plot_screen_scores.py — 畫 ScreenNet 逐幀分數曲線，診斷「連續發射」。

讀 inference.py dump 的 events pkl 裡的 window_scores（每個視窗未經門檻的全部候選分數），
每個進攻候選人一條曲線，x=視窗中心幀，y=分數；標上門檻線與合併後的 events。

同時印出量化診斷：每段「連續超過門檻」的區間
  - 寬度（幀）
  - 貼頂幀數：分數距該段最高分 < near_top 的幀數
平台 = 一大片幀貼著最高分（峰值位置由雜訊決定）；尖峰 = 只有少數幾幀貼頂。

用法（本地，不用 GPU）：
  python nba_data_pipeline/plot_screen_scores.py \
      --pkl output_videos/screens_v130_sam3_events.pkl \
      --out output_videos/screen_scores_v130.png
"""
import os
import pickle
import argparse
import collections as _c

import numpy as np
import cv2

# BGR 色盤（候選人輪流用）
PALETTE = [(60, 60, 220), (40, 170, 40), (200, 120, 30), (180, 60, 180),
           (30, 170, 200), (120, 120, 120)]


def runs_above(series, thr):
    """series: list[(frame, score)] 按幀排序 → 連續（幀相鄰）且 >= thr 的區間。"""
    runs, cur = [], []
    for f, s in series:
        if s >= thr and (not cur or f == cur[-1][0] + 1):
            cur.append((f, s))
        elif s >= thr:
            runs.append(cur)
            cur = [(f, s)]
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--threshold', type=float, default=0.70)
    ap.add_argument('--near_top', type=float, default=0.05,
                    help='分數距該段最高分小於此值算「貼頂」')
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, 'rb'))
    ws = d.get('window_scores')
    if not ws:
        print("pkl 沒有 window_scores —— 請用新版 inference.py 重跑這支影片")
        return
    seq_len = d.get('seq_len', 10)
    n = len(d['court_coords'])
    events = d.get('events', [])

    # 每個候選人：[(視窗中心幀, 分數)]
    series = _c.defaultdict(list)
    for w in ws:
        center = w['start'] + seq_len // 2
        for pid, s in w['scores'].items():
            series[pid].append((center, float(s)))
    for pid in series:
        series[pid].sort()

    # ── 量化診斷 ──
    print(f"門檻 {args.threshold}，貼頂 = 距段內最高分 < {args.near_top}\n")
    for pid in sorted(series):
        runs = runs_above(series[pid], args.threshold)
        if not runs:
            continue
        print(f"候選 #{pid}: {len(runs)} 段超過門檻")
        for r in runs:
            fs = [f for f, _ in r]
            ss = np.array([s for _, s in r])
            top = ss.max()
            n_top = int((ss >= top - args.near_top).sum())
            if len(r) < 5:
                kind = "門檻抖動"      # 只有幾幀越過門檻，不是峰也不是平台
            elif n_top >= 0.5 * len(r):
                kind = "平台"
            else:
                kind = "尖峰"
            print(f"   f{fs[0]}-{fs[-1]}  寬 {len(r):3d} 幀  最高 {top:.3f}  "
                  f"貼頂 {n_top:3d} 幀  → {kind}")
    print()

    # ── 畫圖 ──
    pids = sorted(p for p in series if max(s for _, s in series[p]) >= 0.3)
    row_h, pad_l, pad_r, top_h = 110, 70, 20, 40
    W = 1400
    H = top_h + row_h * max(1, len(pids)) + 30
    img = np.full((H, W, 3), 255, np.uint8)
    plot_w = W - pad_l - pad_r

    def fx(f):
        return int(pad_l + f / max(1, n - 1) * plot_w)

    cv2.putText(img, os.path.basename(args.pkl), (pad_l, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    for row, pid in enumerate(pids):
        y0 = top_h + row * row_h          # 列頂
        y1 = y0 + row_h - 20              # 列底（分數 0）

        def fy(s):
            return int(y1 - s * (y1 - y0))

        col = PALETTE[row % len(PALETTE)]
        cv2.rectangle(img, (pad_l, y0), (pad_l + plot_w, y1), (225, 225, 225), 1)
        # 這個候選人當 screener 的合併 events（淡色底）
        for ev in events:
            if ev['screener_id'] == pid:
                cv2.rectangle(img, (fx(ev['start']), y0 + 1), (fx(ev['end']), y1 - 1),
                              (235, 228, 210), -1)
        # 門檻線
        cv2.line(img, (pad_l, fy(args.threshold)), (pad_l + plot_w, fy(args.threshold)),
                 (150, 150, 255), 1, cv2.LINE_AA)
        # 分數曲線：中間有沒資料的窗（不在候選名單）就斷開，不要直線連過去
        seg = []
        for f, s in series[pid]:
            if seg and f != seg[-1][0] + 1:
                if len(seg) > 1:
                    cv2.polylines(img, [np.array([(fx(a), fy(b)) for a, b in seg], np.int32)],
                                  False, col, 2, cv2.LINE_AA)
                seg = []
            seg.append((f, s))
        if len(seg) > 1:
            cv2.polylines(img, [np.array([(fx(a), fy(b)) for a, b in seg], np.int32)],
                          False, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"#{pid}", (12, (y0 + y1) // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        cv2.putText(img, "1.0", (pad_l - 34, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, "0", (pad_l - 20, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (120, 120, 120), 1, cv2.LINE_AA)

    # x 軸刻度（每 30 幀）
    yb = H - 12
    for f in range(0, n, 30):
        cv2.putText(img, f"f{f}", (fx(f) - 12, yb), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (90, 90, 90), 1, cv2.LINE_AA)

    out = args.out or os.path.splitext(args.pkl)[0].replace('_events', '') + '_scores.png'
    cv2.imwrite(out, img)
    print(f"圖存到 {out}（淡色底 = 合併後的 events，紅線 = 門檻）")


if __name__ == '__main__':
    main()
