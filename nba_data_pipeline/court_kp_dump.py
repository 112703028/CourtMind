"""
court_kp_dump.py — 訓練片段每張 5fps 影格的球場關鍵點（18 點的 xy + 每點信心）存起來，
之後試不同的 homography 規則（信心門檻 / 只用一邊 / 借前後張）不用重跑模型。
輸出：stubs/court_kp/<clip>.pkl = {'xy': (n, 18, 2), 'conf': (n, 18), 'box_conf': (n,), 'size': (w, h)}
用法：python nba_data_pipeline/court_kp_dump.py [--clips a,b]
"""
import argparse, glob, pickle
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
FRAMES = ROOT / 'input_videos' / 'screen_frames'
OUT = ROOT / 'stubs' / 'court_kp'

ap = argparse.ArgumentParser(); ap.add_argument('--clips', default=''); args = ap.parse_args()
m = YOLO(str(ROOT / 'models' / 'court_keypoint_detector.pt'))
OUT.mkdir(parents=True, exist_ok=True)
clips = [c for c in args.clips.split(',') if c] or sorted((p.name for p in FRAMES.iterdir() if p.is_dir()), key=lambda s: int(s.split('_')[1]))
for k, clip in enumerate(clips):
    o = OUT / f'{clip}.pkl'
    if o.exists(): continue
    jp = sorted(glob.glob(str(FRAMES / clip / '*.jpg')), key=lambda q: int(Path(q).stem.rsplit('_', 1)[1]))
    xy, cf, bc = [], [], []
    for i in range(0, len(jp), 16):
        ims = [cv2.imread(q) for q in jp[i:i + 16]]
        for r in m.predict(ims, conf=0.5, verbose=False):
            if r.keypoints is not None and len(r.keypoints.xy):
                xy.append(r.keypoints.xy[0].cpu().numpy()); cf.append(r.keypoints.conf[0].cpu().numpy()); bc.append(float(r.boxes.conf[0]))
            else:
                xy.append(np.zeros((18, 2), np.float32)); cf.append(np.zeros(18, np.float32)); bc.append(0.0)
    h, w = cv2.imread(jp[0]).shape[:2]
    pickle.dump({'xy': np.array(xy), 'conf': np.array(cf), 'box_conf': np.array(bc), 'size': (w, h)}, open(o, 'wb'))
    if k % 50 == 0: print(f'[{k + 1}/{len(clips)}] {clip}', flush=True)
print('KP_DONE', flush=True)
