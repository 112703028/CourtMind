"""
pose_hands.py — 10fps 每個球員的姿態（找手腕，給持球者 / 傳球判斷用）。

跟之前擋拆實驗的「遮罩版姿態」同一招：每個人用框（外擴 15%）裁圖，框裡不屬於他 SAM3 遮罩（往外擴一點）的像素塗灰，
再跑 YOLO11x-pose → 17 個 COCO 關節點（9 = 左手腕、10 = 右手腕）轉回原片座標。直接裁框時兩人重疊骨架會拼錯（撞上時 8%），
塗灰後 0%（被擋住大半的人變成沒骨架，而不是畫錯）。
輸入：run_sam3_tracks --ball / --ball_only 的 pkl（shot_sam3：10fps 球員框 + 遮罩）+ 原片。編號是 shot_sam3 的 SAM3 編號。
輸出：stubs/pose_hands/<片段>.pkl：list（每張 10fps）{pid: (17, 3) 陣列 x, y, 信心}

用法（pod，有 GPU）：python nba_data_pipeline/pose_hands.py --clips 26-46
     單支：python nba_data_pipeline/pose_hands.py --ball_pkl <sam3.pkl> --video <影片> --out <pose.pkl>
"""
import sys
import pickle
import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from shot_owner import full_mask

OUT = ROOT / 'stubs' / 'pose_hands'
PAD, DILATE = 0.15, 9


def clip_pose(model, video, s):
    cap = cv2.VideoCapture(str(video))
    frames, f = {}, 0
    want = set(s['frame_idx'])
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if f in want:
            frames[f] = im
        f += 1
    k = np.ones((2 * DILATE + 1, 2 * DILATE + 1), np.uint8)
    out = []
    for i, fi in enumerate(s['frame_idx']):
        im = frames.get(fi)
        res = {}
        if im is None:
            out.append(res); continue
        Hh, Ww = im.shape[:2]
        crops, metas = [], []
        for pid, bb in s['players'][i].items():
            x1, y1, x2, y2 = bb
            w, h = x2 - x1, y2 - y1
            X1, Y1 = int(max(0, x1 - PAD * w)), int(max(0, y1 - PAD * h))
            X2, Y2 = int(min(Ww, x2 + PAD * w)), int(min(Hh, y2 + PAD * h))
            if X2 - X1 < 8 or Y2 - Y1 < 16:
                continue
            crop = im[Y1:Y2, X1:X2].copy()
            pm = s['player_masks'][i].get(pid)
            if pm is not None:
                m = cv2.dilate(full_mask(pm, (Hh, Ww)).astype(np.uint8), k)[Y1:Y2, X1:X2].astype(bool)
                crop[~m] = 128                                   # 不是他的像素塗灰（避免骨架拼到旁人）
            crops.append(crop); metas.append((pid, X1, Y1))
        if crops:
            for (pid, X1, Y1), r in zip(metas, model(crops, verbose=False)):
                if r.keypoints is None or len(r.keypoints) == 0:
                    continue
                j = int(np.argmax(r.boxes.conf.cpu().numpy())) if r.boxes is not None and len(r.boxes) else 0
                kp = r.keypoints.data[j].cpu().numpy().astype(np.float32)      # (17, 3)
                kp[:, 0] += X1; kp[:, 1] += Y1
                res[pid] = kp
        out.append(res)
    return out


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', default='26-46', help='例 26-46 或 26,30,41')
    ap.add_argument('--ball_dir', default=str(ROOT / 'stubs' / 'ball_batch'))
    ap.add_argument('--model', default='yolo11x-pose.pt')
    ap.add_argument('--ball_pkl', help='單支影片模式：run_sam3_tracks --ball / --ball_only 的 pkl（配 --video、--out）')
    ap.add_argument('--video')
    ap.add_argument('--out')
    args = ap.parse_args()
    if args.ball_pkl:
        s = pickle.load(open(args.ball_pkl, 'rb'))['shot_sam3']
        res = clip_pose(YOLO(args.model), args.video, s)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(res, open(args.out, 'wb'))
        print(f'{len(res)} 張、平均每張 {np.mean([len(r) for r in res]):.1f} 人有骨架 → {args.out}')
        return
    if '-' in args.clips:
        a, b = map(int, args.clips.split('-')); clips = list(range(a, b + 1))
    else:
        clips = [int(x) for x in args.clips.split(',')]
    model = YOLO(args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    for c in clips:
        o = OUT / f'screen_{c}.pkl'
        if o.exists():
            continue
        bp = Path(args.ball_dir) / f'screen_{c}_ball.pkl'
        if not bp.exists():
            print(f'screen_{c}：沒有追球資料，略過'); continue
        s = pickle.load(open(bp, 'rb'))['shot_sam3']
        res = clip_pose(model, ROOT / 'input_videos' / 'screen' / f'screen_{c}.mp4', s)
        pickle.dump(res, open(o, 'wb'))
        print(f'screen_{c}：{len(res)} 張、平均每張 {np.mean([len(r) for r in res]):.1f} 人有骨架', flush=True)


if __name__ == '__main__':
    main()
