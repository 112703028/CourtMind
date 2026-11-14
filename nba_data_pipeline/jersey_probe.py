"""
jersey_probe.py — 背號辨識試驗第 1 步（2026-11-14）：Roboflow 偵測模型的「number」類（之前 KEEP 濾掉了）找背號框 → 對應到球員 → 裁圖。

背號框中心落在某個球員框的上 70%（胸口 / 背）→ 屬於那個人（多個候選取最小的框）。每秒抽 sample_fps 張。
輸出：output_videos/jersey_probe/<片段>.jpg（每個球員一列、最多 8 張背號裁圖）+ 統計（每人找到幾次、背號框高度）。

用法：python nba_data_pipeline/jersey_probe.py screen_44 screen_117 ...
"""
import sys
import pickle
import collections
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
OUT = ROOT / 'output_videos' / 'jersey_probe'
NUMBER = 2


def probe(c, det, sample_fps=3.0, conf=0.3):
    d = pickle.load(open(ROOT / 'output_videos' / 'batch' / f'{c}_events.pkl', 'rb'))
    cap = cv2.VideoCapture(str(ROOT / 'input_videos' / 'screen' / f'{c}.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS) or d['fps']
    step = max(1, int(round(fps / sample_fps)))
    crops, heights, n_num = collections.defaultdict(list), [], 0
    f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if f % step == 0 and f < len(d['player_tracks']):
            nums = [x for x in det.infer(im, conf=0.25, iou=0.5) if x[0] == NUMBER and x[1] >= conf]
            n_num += len(nums)
            for _, sc, b in nums:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                owner = []
                for p, v in d['player_tracks'][f].items():
                    pb = v.get('bbox') if isinstance(v, dict) else v
                    if pb and pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[1] + 0.7 * (pb[3] - pb[1]):
                        owner.append(((pb[2] - pb[0]) * (pb[3] - pb[1]), p))
                if not owner:
                    continue
                p = min(owner)[1]
                h = b[3] - b[1]; heights.append(h)
                m = 0.2 * max(b[2] - b[0], h)
                x1, y1, x2, y2 = int(max(0, b[0] - m)), int(max(0, b[1] - m)), int(min(im.shape[1], b[2] + m)), int(min(im.shape[0], b[3] + m))
                crops[p].append((sc, f, im[y1:y2, x1:x2].copy()))
        f += 1
    return d, crops, heights, n_num, f


def sheet(c, crops):
    rows = []
    for p in sorted(crops, key=lambda q: -len(crops[q])):
        best = sorted(crops[p], key=lambda x: -x[0])[:8]
        tiles = []
        for sc, f, im in best:
            t = cv2.resize(im, (96, 96), interpolation=cv2.INTER_CUBIC)
            cv2.putText(t, f'{f}', (2, 92), 0, 0.4, (0, 255, 255), 1)
            tiles.append(t)
        tiles += [np.zeros((96, 96, 3), np.uint8)] * (8 - len(tiles))
        label = np.zeros((96, 80, 3), np.uint8)
        cv2.putText(label, f'#{p}', (6, 40), 0, 0.9, (255, 255, 255), 2)
        cv2.putText(label, f'n={len(crops[p])}', (6, 75), 0, 0.5, (200, 200, 200), 1)
        rows.append(np.hstack([label] + tiles))
    if rows:
        OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / f'{c}.jpg'), np.vstack(rows))


def main():
    from shot_detect_rf import OnnxDetector
    det = OnnxDetector(ROOT / 'models' / 'rf_basketball_det_v4.onnx')
    allh = []
    for c in sys.argv[1:]:
        d, crops, heights, n_num, nf = probe(c, det)
        sheet(c, crops)
        allh += heights
        teams = collections.Counter()
        main_players = collections.Counter(p for fd in d['player_tracks'] for p in fd)
        print(f'{c}：{nf} 張、背號框 {n_num} 個（對到球員 {sum(len(v) for v in crops.values())}）、'
              f'有背號的球員 {len(crops)}/{len([p for p, k in main_players.items() if k > nf / 3])}、'
              f'每人 {sorted((len(v) for v in crops.values()), reverse=True)}、背號高度中位數 {np.median(heights) if heights else 0:.0f} px')
    if allh:
        print(f'全部背號框高度：中位數 {np.median(allh):.0f} px、10% {np.percentile(allh, 10):.0f}、90% {np.percentile(allh, 90):.0f}')


if __name__ == '__main__':
    main()
