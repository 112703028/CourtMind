"""
shot_detect_rf.py — 用 Roboflow 的 NBA 轉播偵測模型（basketball-player-detection-3-ycjdo/4，roboflow.ipynb 同一個）
逐幀偵測 跳投 / 上籃灌籃 / 球進框 / 球 / 籃框，存原始偵測給後續投籃判斷用（判斷規則在 shot_events.py，改規則不用重跑模型）。

類別：0 ball, 1 ball-in-basket, 2 number, 3 player, 4 player-in-possession,
      5 player-jump-shot, 6 player-layup-dunk, 7 player-shot-block, 8 referee, 9 rim

環境：獨立 venv（Roboflow `inference` 依賴多，不裝進主環境）：
  py -3.12 -m venv C:/Users/timwu/venvs/rf_inference
  C:/Users/timwu/venvs/rf_inference/Scripts/pip install inference supervision python-dotenv
金鑰：專案根目錄 .env 的 ROBOFLOW_API_KEY（Private key；只用來下載模型權重，推論在本機跑，畫面不上傳）

用法：
  C:/Users/timwu/venvs/rf_inference/Scripts/python nba_data_pipeline/shot_detect_rf.py \
      --video_dir input_videos/screen_ext --out output_videos/shot_check/rf_dets.pkl --stride 2
"""
import argparse
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / '.env')

MODEL_ID = 'basketball-player-detection-3-ycjdo/4'
CLASSES = ['ball', 'ball-in-basket', 'number', 'player', 'player-in-possession',
           'player-jump-shot', 'player-layup-dunk', 'player-shot-block', 'referee', 'rim']
KEEP = {0, 1, 3, 4, 5, 6, 7, 9}           # 號碼 / 裁判不需要


class OnnxDetector:
    """不靠 Roboflow 套件 / 金鑰：直接用快取的 weights.onnx（Ultralytics YOLO11s，輸出 [1, 4+10, N]）。
    前處理照 environment.json：直接拉伸到 1280x1280、RGB、/255；後處理：每類 NMS（同 inference 套件的 iou_threshold）。"""
    def __init__(self, weights):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(weights), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        print('[onnx] providers', self.sess.get_providers())

    def infer(self, img, conf=0.25, iou=0.9):
        H, W = img.shape[:2]
        x = cv2.resize(img, (1280, 1280))[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0][0].T     # (N, 14)
        scores = out[:, 4:]; cls = scores.argmax(1); sc = scores.max(1)
        keep = sc >= conf
        b, cls, sc = out[keep, :4], cls[keep], sc[keep]
        sx, sy = W / 1280, H / 1280
        boxes = np.stack([(b[:, 0] - b[:, 2] / 2) * sx, (b[:, 1] - b[:, 3] / 2) * sy,
                          (b[:, 0] + b[:, 2] / 2) * sx, (b[:, 1] + b[:, 3] / 2) * sy], 1)
        dets = []
        for c in np.unique(cls):
            m = np.where(cls == c)[0]
            idx = cv2.dnn.NMSBoxes([[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes[m]],
                                   sc[m].astype(float).tolist(), conf, iou)
            for k in np.array(idx).reshape(-1):
                j = m[k]; dets.append((int(c), round(float(sc[j]), 3), tuple(float(v) for v in boxes[j])))
        return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_dir', default='input_videos/screen_ext')
    ap.add_argument('--clips', default='', help='逗號分隔；不給就跑整個資料夾')
    ap.add_argument('--out', default='output_videos/shot_check/rf_dets.pkl')
    ap.add_argument('--stride', type=int, default=2, help='每幾張原片影格跑一次')
    ap.add_argument('--conf', type=float, default=0.25, help='先用低門檻全存，判斷時再調')
    ap.add_argument('--backend', default='inference', choices=['inference', 'onnx'],
                    help='onnx = 直接跑快取的 weights.onnx（pod 上新版 inference 不吃舊快取、又不放金鑰時用）')
    args = ap.parse_args()
    cached = Path(os.environ.get('MODEL_CACHE_DIR', '/tmp/cache')) / MODEL_ID / 'weights.onnx'
    if args.backend == 'onnx':
        det = OnnxDetector(cached)
        model = None
    elif not os.environ.get('ROBOFLOW_API_KEY') and not cached.exists():
        raise SystemExit(f'.env 沒有 ROBOFLOW_API_KEY，且 {cached} 沒有快取權重（pod 上可直接複製本機 C:/tmp/cache 的權重，不需金鑰）')

    import sys
    here = str(Path(__file__).parent.resolve())
    sys.path = [p for p in sys.path if str(Path(p or '.').resolve()) != here]   # 同資料夾的 inference.py 會蓋掉 Roboflow 套件
    if args.backend == 'inference':
        from inference import get_model
        model = get_model(model_id=MODEL_ID)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pickle.load(open(out, 'rb')) if out.exists() else {}
    vdir = ROOT / args.video_dir
    clips = [c for c in args.clips.split(',') if c] or sorted(p.stem for p in vdir.glob('*.mp4'))
    for clip in clips:
        if clip in res:
            print(f'{clip} 已有結果，跳過'); continue
        cap = cv2.VideoCapture(str(vdir / f'{clip}.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames, t0, f = [], time.time(), 0
        while True:
            ok, img = cap.read()
            if not ok: break
            if f % args.stride == 0:
                if model is None:
                    dets = [d for d in det.infer(img, conf=args.conf, iou=0.9) if d[0] in KEEP]
                else:
                    r = model.infer(img, confidence=args.conf, iou_threshold=0.9)[0]
                    dets = [(p.class_id, round(p.confidence, 3),
                             (p.x - p.width / 2, p.y - p.height / 2, p.x + p.width / 2, p.y + p.height / 2))
                            for p in r.predictions if p.class_id in KEEP]
                frames.append((f, dets))
            f += 1
        res[clip] = dict(fps=fps, n=n, stride=args.stride, size=(int(cap.get(3)), int(cap.get(4))), frames=frames)
        pickle.dump(res, open(out, 'wb'))                      # 每支存一次，中斷可續跑
        cnt = {CLASSES[c]: sum(1 for _, d in frames for x in d if x[0] == c and x[1] >= 0.4) for c in (1, 5, 6, 9, 0)}
        print(f'{clip}：{len(frames)} 張 {time.time() - t0:.0f} 秒 | conf≥0.4 影格數 {cnt}', flush=True)


if __name__ == '__main__':
    main()
