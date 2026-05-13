"""
visualize_tracks.py

對每個 screen_frames clip 跑 YOLO + ByteTrack，輸出帶 track_id 標籤的影片。
同時用現有 COCO 標記預先猜測 screener/defender/handler 的 track_id。

輸出：
  output_track_viz/screen_{id}.mp4  — 每個 clip 的追蹤影片（screener=紅框/defender=藍框/handler=綠框）
  track_annotations.json            — 預填的 track_id 標記，確認或修改後給 convert_manual.py 使用

使用方式：
  python visualize_tracks.py

看完影片後，修改 track_annotations.json 中錯誤的 track_id，
null 表示猜不到，需要你手動填入。

JSON 格式：
  {
    "11": {"screener": 3, "defender": 7, "handler": 1, "confident": true},
    "12": {"screener": null, "defender": 5, "handler": null, "confident": false},
    ...
  }
"""

import sys, re, json, collections
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
import supervision as sv

FRAMES_DIR        = PROJECT_ROOT / 'input_videos' / 'screen_frames'
COCO_PATH         = PROJECT_ROOT / 'screen_data' / 'screen_v2.coco' / 'train' / '_annotations.coco.json'
PLAYER_MODEL_PATH = PROJECT_ROOT / 'models' / 'player_detector.pt'
OUT_DIR           = Path(__file__).parent / 'output_track_viz'
ANN_PATH          = Path(__file__).parent / 'track_annotations.json'

# 顏色：screener=紅, defender=藍, handler=綠, 其他=白
ROLE_COLORS = {
    'screener': (0, 0, 255),
    'defender': (255, 0, 0),
    'handler':  (0, 255, 0),
    'other':    (200, 200, 200),
}


def bbox_center(bbox_xywh):
    x, y, w, h = [float(v) for v in bbox_xywh]
    return x + w / 2, y + h / 2


def xyxy_center(xyxy):
    return (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2


def nearest_track(coco_cx, coco_cy, frame_dict, max_dist=150):
    best_tid, best_dist = None, float('inf')
    for tid, xyxy in frame_dict.items():
        tx, ty = xyxy_center(xyxy)
        d = ((coco_cx - tx) ** 2 + (coco_cy - ty) ** 2) ** 0.5
        if d < best_dist:
            best_dist, best_tid = d, tid
    if best_dist > max_dist:
        return None
    return best_tid


def draw_tracks(frame, frame_dict, role_map):
    """
    frame_dict: {track_id: [x1,y1,x2,y2]}
    role_map:   {track_id: 'screener'|'defender'|'handler'|'other'}
    """
    img = frame.copy()
    for tid, xyxy in frame_dict.items():
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        role  = role_map.get(tid, 'other')
        color = ROLE_COLORS[role]
        thick = 3 if role != 'other' else 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

        label = f"{tid}"
        if role != 'other':
            label += f" [{role[:3].upper()}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main():
    import json as _json
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 載入 COCO 標記
    with open(COCO_PATH) as f:
        coco = _json.load(f)

    cat_map = {c['id']: c['name'] for c in coco['categories']}
    ann_by_img = collections.defaultdict(list)
    for ann in coco['annotations']:
        ann_by_img[ann['image_id']].append(
            (cat_map[ann['category_id']], ann['bbox'])
        )

    coco_clips = collections.defaultdict(lambda: collections.defaultdict(list))
    for img in coco['images']:
        m = re.match(r'screen_(\d+)_(\d{4})', img['file_name'])
        if m:
            coco_clips[int(m.group(1))][int(m.group(2))] = ann_by_img[img['id']]

    print("Loading player model...")
    player_model = YOLO(str(PLAYER_MODEL_PATH))

    clip_dirs = sorted(
        [d for d in FRAMES_DIR.iterdir() if d.is_dir() and re.match(r'screen_\d+', d.name)],
        key=lambda p: int(re.search(r'screen_(\d+)', p.name).group(1))
    )

    # 讀取已存在的標記（避免覆蓋已填好的）
    existing = {}
    if ANN_PATH.exists():
        with open(ANN_PATH) as f:
            existing = _json.load(f)

    annotations = dict(existing)
    tracks_out   = {}

    for clip_dir in clip_dirs:
        clip_id = int(re.search(r'screen_(\d+)', clip_dir.name).group(1))
        frame_paths = sorted(
            clip_dir.glob('*.jpg'),
            key=lambda p: int(re.search(r'_(\d{4})\.jpg$', p.name).group(1))
        )
        if not frame_paths:
            continue

        # 已標記過且 confident=true 就跳過
        if str(clip_id) in annotations and annotations[str(clip_id)].get('confident'):
            print(f"  Clip {clip_id}: already annotated (confident), skipping video gen")
            continue

        print(f"Processing clip {clip_id} ({len(frame_paths)} frames)...")

        frames = []
        frame_nums = []
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
                frame_nums.append(int(re.search(r'_(\d{4})\.jpg$', p.name).group(1)))

        if not frames:
            continue

        # YOLO + ByteTrack
        BATCH = 20
        all_det_results = []
        for i in range(0, len(frames), BATCH):
            all_det_results += player_model.predict(
                frames[i:i+BATCH], conf=0.4, verbose=False
            )

        tracker = sv.ByteTrack()
        raw_tracks = []
        for result in all_det_results:
            det = sv.Detections.from_ultralytics(result)
            det_tracked = tracker.update_with_detections(det)
            fd = {}
            for frame_det in det_tracked:
                tid = frame_det[4]
                if tid is not None:
                    fd[int(tid)] = frame_det[0].tolist()
            raw_tracks.append(fd)

        # 用 COCO 標記猜測 track_id（中心距離）
        coco_by_frame = dict(coco_clips.get(clip_id, {}))
        screener_votes = collections.Counter()
        defender_votes = collections.Counter()
        handler_votes  = collections.Counter()

        for f_num, anns in coco_by_frame.items():
            if f_num not in frame_nums:
                continue
            f_idx = frame_nums.index(f_num)
            fd = raw_tracks[f_idx]
            for cat, bbox in anns:
                cx, cy = bbox_center(bbox)
                tid = nearest_track(cx, cy, fd)
                if tid is None:
                    continue
                if cat == 'screener':
                    screener_votes[tid] += 1
                elif cat == 'defender':
                    defender_votes[tid] += 1
                elif cat == 'ball_handler':
                    handler_votes[tid] += 1

        best_screener = screener_votes.most_common(1)[0][0] if screener_votes else None
        best_defender = defender_votes.most_common(1)[0][0] if defender_votes else None
        best_handler  = handler_votes.most_common(1)[0][0]  if handler_votes  else None

        confident = (best_screener is not None and best_handler is not None)
        annotations[str(clip_id)] = {
            'screener':  best_screener,
            'defender':  best_defender,
            'handler':   best_handler,
            'confident': confident,
        }

        print(f"  → screener={best_screener}  defender={best_defender}  "
              f"handler={best_handler}  confident={confident}")

        # 輸出影片
        h, w = frames[0].shape[:2]
        out_path = OUT_DIR / f'screen_{clip_id}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, 8, (w, h))

        for frame, fd in zip(frames, raw_tracks):
            role_map = {}
            for tid in fd:
                if tid == best_screener:
                    role_map[tid] = 'screener'
                elif tid == best_defender:
                    role_map[tid] = 'defender'
                elif tid == best_handler:
                    role_map[tid] = 'handler'
                else:
                    role_map[tid] = 'other'
            writer.write(draw_tracks(frame, fd, role_map))
        writer.release()

        # 存 tracks 資料供 annotate_tracks.py 使用（不需重跑 YOLO）
        tracks_out[str(clip_id)] = {
            'frame_nums': frame_nums,
            'tracks': [{str(tid): xyxy for tid, xyxy in fd.items()} for fd in raw_tracks],
        }

    # 存標記檔
    with open(ANN_PATH, 'w') as f:
        _json.dump(annotations, f, indent=2, ensure_ascii=False)

    # 存 tracks 資料
    tracks_path = OUT_DIR / 'tracks_data.json'
    with open(tracks_path, 'w') as f:
        _json.dump(tracks_out, f)

    total     = len(annotations)
    confident = sum(1 for v in annotations.values() if v.get('confident'))
    print(f"\nDone. {confident}/{total} clips auto-annotated with confidence.")
    print(f"Videos     → {OUT_DIR}/")
    print(f"Tracks     → {tracks_path}")
    print(f"Annotation → {ANN_PATH}")
    print(f"\n執行互動標記工具：python annotate_tracks.py")


if __name__ == '__main__':
    main()
