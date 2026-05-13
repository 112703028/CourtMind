"""
visualize_tracks_sam2.py

使用 YOLO（第一幀偵測提示）+ SAM2（跨幀追蹤）替代 YOLO+ByteTrack。
SAM2 用記憶體庫維持 track_id，遮擋後不會跳號。

輸出格式與 visualize_tracks.py 相同，可接 annotate_tracks.py。

執行：
  python visualize_tracks_sam2.py

SAM2 模型下載（models/ 目錄下）：
  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P models/

安裝套件：
  pip install git+https://github.com/facebookresearch/sam2.git
"""

from __future__ import annotations

import sys, re, json, collections, os, tempfile, shutil
import numpy as np
import cv2
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("ERROR: sam2 not installed.")
    print("  pip install git+https://github.com/facebookresearch/sam2.git")
    sys.exit(1)

# ── 路徑設定 ──────────────────────────────────────────────────────────────────

FRAMES_DIR        = PROJECT_ROOT / 'input_videos' / 'screen_frames'
COCO_PATH         = PROJECT_ROOT / 'screen_data' / 'screen_v2.coco' / 'train' / '_annotations.coco.json'
PLAYER_MODEL_PATH = PROJECT_ROOT / 'models' / 'player_detector.pt'
OUT_DIR           = Path(__file__).parent / 'output_track_viz'
ANN_PATH          = Path(__file__).parent / 'track_annotations.json'

SAM2_CHECKPOINT = PROJECT_ROOT / 'models' / 'sam2.1_hiera_large.pt'
SAM2_CONFIG     = 'configs/sam2.1/sam2.1_hiera_l.yaml'
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

ROLE_COLORS = {
    'screener': (0, 0, 255),
    'defender': (255, 0, 0),
    'handler':  (0, 255, 0),
    'other':    (200, 200, 200),
}


# ── SAM2 追蹤（標準 API）─────────────────────────────────────────────────────

def run_sam2_tracks(clip_dir: Path, frames, player_model, predictor) -> list:
    result = player_model.predict(frames[0], conf=0.4, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
    if len(boxes) == 0:
        return []
    obj_ids = list(range(1, len(boxes) + 1))

    # SAM2 init_state requires frames named as plain integers (00000.jpg, 00001.jpg, ...)
    frame_paths = sorted(
        clip_dir.glob('*.jpg'),
        key=lambda p: int(re.search(r'_(\d{4})\.jpg$', p.name).group(1))
    )
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        for i, fp in enumerate(frame_paths):
            dst = tmp_dir / f'{i:05d}.jpg'
            try:
                os.link(fp, dst)
            except OSError:
                shutil.copy2(fp, dst)

        inference_state = predictor.init_state(video_path=str(tmp_dir))
        predictor.reset_state(inference_state)
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            for obj_id, xyxy in zip(obj_ids, boxes):
                predictor.add_new_points_or_box(
                    inference_state, frame_idx=0, obj_id=obj_id,
                    box=xyxy.astype(np.float32))
        raw_tracks = [{} for _ in frames]
        for obj_id, xyxy in zip(obj_ids, boxes):
            raw_tracks[0][obj_id] = xyxy.tolist()
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
                for obj_id, mask in zip(object_ids, masks):
                    mask_np = (mask[0].cpu().numpy() > 0)
                    if mask_np.any() and frame_idx < len(raw_tracks):
                        ys, xs = np.where(mask_np)
                        raw_tracks[frame_idx][int(obj_id)] = [
                            float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max())]
        predictor.reset_state(inference_state)
        return raw_tracks
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── 工具函數 ──────────────────────────────────────────────────────────────────

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
    return best_tid if best_dist <= max_dist else None


def draw_tracks(frame, frame_dict, role_map):
    img = frame.copy()
    for tid, xyxy in frame_dict.items():
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        role  = role_map.get(tid, 'other')
        color = ROLE_COLORS[role]
        thick = 3 if role != 'other' else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = f"{tid}" + (f" [{role[:3].upper()}]" if role != 'other' else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 載入 COCO
    with open(COCO_PATH) as f:
        coco = json.load(f)
    
    '''

    data['categories'] : 
    
    === categories ===
   [{'id': 0, 'name': 'screen', 'supercategory': 'none'},
    {'id': 1, 'name': 'ball_handler', 'supercategory': 'screen'},
    {'id': 2, 'name': 'defender', 'supercategory': 'screen'},
    {'id': 3, 'name': 'others', 'supercategory': 'screen'},
    {'id': 4, 'name': 'player', 'supercategory': 'screen'},
    {'id': 5, 'name': 'screen', 'supercategory': 'screen'},
    {'id': 6, 'name': 'screener', 'supercategory': 'screen'}]
    '''
    cat_map = {c['id']: c['name'] for c in coco['categories']}  
    ann_by_img = collections.defaultdict(list)

    '''

    data['images'][0] :

    === images[0] ===
    {'date_captured': '2026-04-30T13:43:44+00:00',
    'extra': {'name': 'screen_89_0024.jpg'},
    'file_name': 'screen_89_0024_jpg.rf.i47NJVSm3gD4JsNCRs5P.jpg',
    'height': 720,
    'id': 0,
    'license': 1,
    'width': 1280}
    '''

    '''

    data['annotations'][:3] : 

    === annotations[0:3] ===
    [{'area': 8766.124,
    'bbox': [844, 236, '74.22', '118.11'],
    'category_id': 6,
    'id': 1,
    'image_id': 0,
    'iscrowd': 0,
    'segmentation': []},
    {'area': 8485.005,
    'bbox': [780, 299, '61.33', '138.35'],
    'category_id': 2,
    'id': 2,
    'image_id': 0,
    'iscrowd': 0,
    'segmentation': []},
    {'area': 12944.518,
    'bbox': [699, 306, '79.59', '162.64'],
    'category_id': 1,
    'id': 3,
    'image_id': 0,
    'iscrowd': 0,
    'segmentation': []}]
    '''
    for ann in coco['annotations']:
        ann_by_img[ann['image_id']].append((cat_map[ann['category_id']], ann['bbox'])) # key : 哪張圖片， value : 這張圖片的標註（類別、bbox）

    coco_clips = collections.defaultdict(lambda: collections.defaultdict(list))
    for img in coco['images']:
        m = re.match(r'screen_(\d+)_(\d{4})', img['file_name'])
        if m:
            coco_clips[int(m.group(1))][int(m.group(2))] = ann_by_img[img['id']]  # [第幾個影片][影片中的第幾個frame] -> 這張圖片的標註（類別、bbox）

    # 載入模型
    print("Loading player model...")
    player_model = YOLO(str(PLAYER_MODEL_PATH))

    if not SAM2_CHECKPOINT.exists():
        print(f"ERROR: SAM2 model not found at {SAM2_CHECKPOINT}")
        print("Download with:")
        print(f"  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt "
              f"-P {PROJECT_ROOT}/models/")
        sys.exit(1)

    print(f"Loading SAM2 on {DEVICE}...")
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=DEVICE)
    print("Models loaded.")

    existing = {}
    if ANN_PATH.exists():
        with open(ANN_PATH) as f:
            existing = json.load(f)
    annotations = dict(existing)

    # 載入已有的 tracks，避免重跑已處理的 clip
    tracks_path = OUT_DIR / 'tracks_data_sam2.json'
    tracks_out  = {}
    if tracks_path.exists():
        with open(tracks_path) as f:
            tracks_out = json.load(f)

    clip_dirs = sorted(
        [d for d in FRAMES_DIR.iterdir() if d.is_dir() and re.match(r'screen_\d+', d.name)],
        key=lambda p: int(re.search(r'screen_(\d+)', p.name).group(1))
    )

    # 遍歷每個影片資料夾
    for clip_dir in clip_dirs:
        clip_id = int(re.search(r'screen_(\d+)', clip_dir.name).group(1))
        frame_paths = sorted(
            clip_dir.glob('*.jpg'),
            key=lambda p: int(re.search(r'_(\d{4})\.jpg$', p.name).group(1))
        )
        if not frame_paths:
            continue

        already_confident = str(clip_id) in annotations and annotations[str(clip_id)].get('confident')
        if already_confident and str(clip_id) in tracks_out:
            print(f"  Clip {clip_id}: already confident + has tracks, skipping")
            continue

        print(f"Processing clip {clip_id} ({len(frame_paths)} frames)...")

        # 遍歷一個影片中的frame
        frames, frame_nums = [], []
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
                frame_nums.append(int(re.search(r'_(\d{4})\.jpg$', p.name).group(1)))

        if not frames:
            continue

        # SAM2 追蹤（標準 API）
        try:
            raw_tracks = run_sam2_tracks(clip_dir, frames, player_model, predictor)
        except Exception as e:
            print(f"  Clip {clip_id}: SAM2 failed: {e}, skipping")
            continue

        if not raw_tracks:
            print(f"  Clip {clip_id}: no detections in first frame, skipping")
            continue

        print(f"  SAM2 tracked {len(raw_tracks)} frames")

        # COCO 投票猜角色（用中心距離）
        coco_by_frame = dict(coco_clips.get(clip_id, {})) # key = frame number , value = 該 frame 的標註
        screener_votes = collections.Counter()
        defender_votes = collections.Counter()
        handler_votes  = collections.Counter()
        target_votes   = collections.Counter()

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
                elif cat == 'target':
                    target_votes[tid] += 1

        # 取每幀最多幾個 screener，決定要取幾票
        max_scr_per_frame = max(
            (sum(1 for cat, _ in anns if cat == 'screener')
             for anns in coco_by_frame.values()), default=1)
        n_screeners = max(1, max_scr_per_frame)
        best_screeners = [t for t, _ in screener_votes.most_common(n_screeners)] if screener_votes else []

        best_defender = defender_votes.most_common(1)[0][0] if defender_votes else None
        best_handler  = handler_votes.most_common(1)[0][0]  if handler_votes  else None
        best_target   = target_votes.most_common(1)[0][0]   if target_votes   else None
        confident = (bool(best_screeners) and
                     (best_handler is not None or best_target is not None))

        if not already_confident:
            annotations[str(clip_id)] = {
                'screener':  best_screeners,   # list
                'defender':  best_defender,
                'handler':   best_handler,
                'target':    best_target,
                'confident': confident,
            }
        screen_type = 'off-ball' if best_target is not None else 'ball'
        print(f"  → [{screen_type}] screener={best_screeners}  handler={best_handler}  "
              f"target={best_target}  defender={best_defender}  confident={confident}")

        # 輸出影片（檔名加 _sam2 以區分）
        h, w = frames[0].shape[:2]
        out_path = OUT_DIR / f'screen_{clip_id}_sam2.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, 8, (w, h))
        for frame, fd in zip(frames, raw_tracks):
            role_map = {}
            for tid in fd:
                if tid in best_screeners:  role_map[tid] = 'screener'
                elif tid == best_defender: role_map[tid] = 'defender'
                elif tid == best_handler:  role_map[tid] = 'handler'
                else:                      role_map[tid] = 'other'
            writer.write(draw_tracks(frame, fd, role_map))
        writer.release()

        tracks_out[str(clip_id)] = {
            'frame_nums': frame_nums,
            'tracks': [{str(tid): xyxy for tid, xyxy in fd.items()} for fd in raw_tracks],
        }

    # 存檔
    with open(ANN_PATH, 'w') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    with open(tracks_path, 'w') as f:
        json.dump(tracks_out, f)

    total     = len(annotations)
    confident = sum(1 for v in annotations.values() if v.get('confident'))
    print(f"\nDone. {confident}/{total} clips auto-annotated.")
    print(f"Videos     → {OUT_DIR}/*_sam2.mp4")
    print(f"Tracks     → {tracks_path}")
    print(f"Annotation → {ANN_PATH}")
    print(f"\n標記工具（使用 SAM2 tracks）：")
    print(f"  python annotate_tracks.py --tracks {tracks_path}")


if __name__ == '__main__':
    main()
