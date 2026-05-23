"""
test_ball_acquisition.py — 視覺化 handler 偵測，支援兩種模式：

(1) 預設（BallAquisitionDetector）：
    PlayerTracker + BallTracker → BallAquisitionDetector 猜誰持球。

(2) 新 YOLO 模式（--yolo_model PATH）：
    用 multi-class YOLO 直接偵測 ball_handler（class 0），不靠 ball detection。
    每幀獨立判斷，輸出每幀所有 class 的 bbox。

Usage 範例：
  # 舊模式
  python nba_data_pipeline/test_ball_acquisition.py \\
    --video         input_videos/screen/screen_13.mp4 \\
    --output_video  output_videos/ba_test_v13.avi --smooth

  # YOLO 模式
  python nba_data_pipeline/test_ball_acquisition.py \\
    --video         input_videos/screen/screen_13.mp4 \\
    --yolo_model    models/player_in_possession.pt \\
    --output_video  output_videos/ba_yolo_v13.avi
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from ball_aquisition import BallAquisitionDetector

# YOLO 模式才會用到（lazy import 避免沒 ultralytics 也能跑舊模式）
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# YOLO 模式的 class 設定
YOLO_CLASS_COLORS = {
    0: ((0, 255, 0),   'HANDLER',  3),    # 綠粗
    1: ((0, 0, 255),   'SCREENER', 3),    # 紅粗
    2: ((255, 0, 0),   'DEFENDER', 3),    # 藍粗
    3: ((180, 180, 180), 'OTHER',  1),    # 灰細
}


def smooth_ball_acquisition(ball_acquisition, window=15, min_total_frames=20):
    n = len(ball_acquisition)
    if n == 0:
        return ball_acquisition
    handler_freq = Counter(p for p in ball_acquisition if p is not None and p != -1)
    valid = {p for p, cnt in handler_freq.items() if cnt >= min_total_frames}
    filtered = [p if p in valid else -1 for p in ball_acquisition]
    smoothed = list(filtered)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        votes = Counter(p for p in filtered[lo:hi] if p != -1)
        if votes:
            smoothed[i] = votes.most_common(1)[0][0]
    n_changed = sum(1 for a, b in zip(ball_acquisition, smoothed) if a != b)
    print(f"  smoothed: {n_changed}/{n} frames changed, valid handlers={sorted(valid)}")
    return smoothed


def ball_center(ball_track_frame):
    """從 ball_tracks 的一幀取球中心。回 None 如果沒偵測到。"""
    if not ball_track_frame:
        return None
    # ball_tracks[frame] 是 {ball_id: {'bbox': [x1,y1,x2,y2]}}
    for _id, info in ball_track_frame.items():
        bbox = info.get('bbox') if isinstance(info, dict) else info
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    return None


def bbox_center_player(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def draw(frame, player_tracks_frame, ball_tracks_frame, handler_id):
    img = frame.copy()

    # 球員 bbox + id
    for pid, info in player_tracks_frame.items():
        if 'bbox' not in info:
            continue
        x1, y1, x2, y2 = [int(v) for v in info['bbox']]
        if pid == handler_id:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"HANDLER #{pid}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(img, f"#{pid}", (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # 球
    bc = ball_center(ball_tracks_frame)
    if bc is not None:
        cv2.circle(img, bc, 8, (0, 255, 255), -1)
        cv2.circle(img, bc, 12, (0, 200, 200), 2)

        # 連線：handler 中心 → 球
        if handler_id is not None and handler_id in player_tracks_frame:
            bbox = player_tracks_frame[handler_id].get('bbox')
            if bbox is not None:
                pc = bbox_center_player(bbox)
                cv2.line(img, pc, bc, (0, 255, 255), 2)

    return img


def draw_yolo(frame, detections):
    """detections: list[(xyxy, class_id, conf)]，畫 YOLO 多 class 結果。"""
    img = frame.copy()
    for (x1, y1, x2, y2), cls, conf in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color, label, thick = YOLO_CLASS_COLORS.get(
            int(cls), ((255, 255, 255), '?', 1)
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)),
                      (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, max(10, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def run_yolo_mode(args, frames):
    """跑新 YOLO 模型，每幀偵測所有 class，輸出視覺化影片。"""
    if YOLO is None:
        print("ERROR: ultralytics not installed (pip install ultralytics)")
        return

    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    print(f"Running YOLO inference on {len(frames)} frames (conf={args.yolo_conf})...")
    out_frames = []
    stats = Counter()
    n_with_handler = 0
    n_multi_handler = 0

    for frame in frames:
        result = model.predict(frame, conf=args.yolo_conf, verbose=False)[0]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls  = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            for box, c, p in zip(xyxy, cls, conf):
                detections.append((box, int(c), float(p)))
                stats[int(c)] += 1

            n_handler = sum(1 for _, c, _ in detections if c == 0)
            if n_handler >= 1:
                n_with_handler += 1
            if n_handler > 1:
                n_multi_handler += 1

        out_frames.append(draw_yolo(frame, detections))

    label_map = {0: 'ball_handler', 1: 'screener', 2: 'defender', 3: 'others'}
    print(f"\n── YOLO mode stats ──")
    print(f"  Frames with >=1 handler:   {n_with_handler}/{len(frames)} "
          f"({100*n_with_handler/max(len(frames),1):.1f}%)")
    print(f"  Frames with multi-handler: {n_multi_handler}")
    print(f"  Per-class total detections:")
    for c in (0, 1, 2, 3):
        print(f"    {label_map[c]:15s}: {stats[c]}")

    os.makedirs(os.path.dirname(args.output_video) or '.', exist_ok=True)
    save_video(out_frames, args.output_video)
    print(f"\nOutput: {args.output_video}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output_video', default='output_videos/ba_test.avi')
    parser.add_argument('--stub_dir', default='stubs')
    parser.add_argument('--no_stub', action='store_true')
    parser.add_argument('--smooth', action='store_true',
                        help='也輸出 smoothed 版本（檔名加 _smoothed）')
    parser.add_argument('--ba_smooth_window', type=int, default=15)
    parser.add_argument('--ba_min_frames', type=int, default=20)
    parser.add_argument('--yolo_model', default=None,
                        help='指定後切換到 YOLO 模式（multi-class 直接偵測 ball_handler）')
    parser.add_argument('--yolo_conf', type=float, default=0.4,
                        help='YOLO confidence 門檻')
    args = parser.parse_args()

    print(f"Reading video: {args.video}")
    frames = read_video(args.video)
    if not frames:
        print("ERROR: cannot read video")
        return
    print(f"  {len(frames)} frames")

    # ── YOLO 模式（用新訓練的 multi-class 模型直接偵測 handler）──────────────────
    if args.yolo_model is not None:
        run_yolo_mode(args, frames)
        return

    use_stub = not args.no_stub
    os.makedirs(args.stub_dir, exist_ok=True)

    print("Player tracking...")
    player_tracker = PlayerTracker(str(PROJECT_ROOT / 'models' / 'player_detector.pt'))
    player_tracks = player_tracker.get_object_tracks(
        frames, read_from_stub=use_stub,
        stub_path=os.path.join(args.stub_dir, 'player_tracks_stub.pkl'),
    )

    print("Ball tracking...")
    ball_tracker = BallTracker(str(PROJECT_ROOT / 'models' / 'ball_detector_model.pt'))
    ball_tracks = ball_tracker.get_object_tracks(
        frames, read_from_stub=use_stub,
        stub_path=os.path.join(args.stub_dir, 'ball_tracks_stub.pkl'),
    )
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    print("Ball possession...")
    ba = BallAquisitionDetector()
    ball_acquisition = ba.detect_ball_possession(player_tracks, ball_tracks)

    # 統計各 handler 出現次數
    counts = Counter(p for p in ball_acquisition if p is not None and p != -1)
    total_with_handler = sum(counts.values())
    print(f"\n── Raw ball_acquisition stats ──")
    print(f"  Frames with handler:    {total_with_handler}/{len(frames)}")
    print(f"  Frames with no handler: {len(frames) - total_with_handler}")
    print(f"  Top handlers:")
    for pid, cnt in counts.most_common():
        print(f"    #{pid:3d}: {cnt:4d} frames ({100*cnt/len(frames):.1f}%)")

    # 輸出 raw 版本
    print(f"\nDrawing raw...")
    out_raw = [draw(f, pt, bt, h) for f, pt, bt, h in
               zip(frames, player_tracks, ball_tracks, ball_acquisition)]
    raw_path = args.output_video
    os.makedirs(os.path.dirname(raw_path) or '.', exist_ok=True)
    save_video(out_raw, raw_path)
    print(f"  → {raw_path}")

    # 如果有 --smooth，也輸出平滑版本
    if args.smooth:
        ball_acquisition_s = smooth_ball_acquisition(
            ball_acquisition,
            window=args.ba_smooth_window,
            min_total_frames=args.ba_min_frames,
        )

        # 統計平滑後
        counts_s = Counter(p for p in ball_acquisition_s if p is not None and p != -1)
        print(f"\n── Smoothed ball_acquisition stats ──")
        for pid, cnt in counts_s.most_common():
            print(f"    #{pid:3d}: {cnt:4d} frames ({100*cnt/len(frames):.1f}%)")

        print(f"\nDrawing smoothed...")
        out_smoothed = [draw(f, pt, bt, h) for f, pt, bt, h in
                        zip(frames, player_tracks, ball_tracks, ball_acquisition_s)]
        base, ext = os.path.splitext(raw_path)
        smoothed_path = f"{base}_smoothed{ext}"
        save_video(out_smoothed, smoothed_path)
        print(f"  → {smoothed_path}")


if __name__ == '__main__':
    main()
