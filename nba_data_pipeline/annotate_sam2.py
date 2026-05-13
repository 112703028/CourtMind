"""
annotate_sam2.py

整合 SAM2 追蹤 + 互動標記的單一工具。
取代兩步驟流程（visualize_tracks_sam2.py → annotate_tracks.py）。

需要 GPU（SAM2）+ 顯示器（cv2 GUI）。
建議在 AI Stack 上透過 X11 forwarding 執行：
  ssh -X -p 31682 root@your-server
  python nba_data_pipeline/annotate_sam2.py

操作方式：
  滑鼠左鍵   選取球員（框變黃色）
  S          screener（設掩護的人）
  H          handler（持球手）
  T          target（off-ball screen 的被掩護者，ball screen 時不需要按）
  D          defender（target 或 handler 的防守者）
  C          清除選取球員的所有角色
  ← / →      上一幀 / 下一幀
  Space      播放 / 暫停
  N          儲存並跳到下一個 clip
  B          跳回上一個 clip（不儲存）
  Q          儲存全部並退出

Ball screen：只標 S / H / D
Off-ball screen：標 S / H / T / D（T = 被掩護的非持球球員）

執行：
  python annotate_sam2.py
  python annotate_sam2.py --skip_confident   # 跳過已標記的 clip
"""

from __future__ import annotations

import argparse, json, os, re, shutil, sys, tempfile
import cv2
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
import supervision as sv

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("ERROR: sam2 not installed.")
    print("  pip install git+https://github.com/facebookresearch/sam2.git")
    sys.exit(1)

# ── 路徑設定 ──────────────────────────────────────────────────────────────────

SCRIPT_DIR        = Path(__file__).parent
FRAMES_DIR        = PROJECT_ROOT / 'input_videos' / 'screen_frames'
PLAYER_MODEL_PATH = PROJECT_ROOT / 'models' / 'player_detector.pt'
ANN_PATH          = SCRIPT_DIR / 'track_annotations.json'
TRACKS_CACHE_PATH = SCRIPT_DIR / 'output_track_viz' / 'tracks_data_sam2.json'

SAM2_CHECKPOINT = PROJECT_ROOT / 'models' / 'sam2.1_hiera_large.pt'
SAM2_CONFIG     = 'configs/sam2.1/sam2.1_hiera_l.yaml'
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

ROLE_COLORS = {
    'screener': (0,   0,   255),   # 紅
    'handler':  (0,   255, 0),     # 綠
    'target':   (0,   128, 255),   # 橘
    'defender': (255, 0,   0),     # 藍
    'selected': (0,   255, 255),   # 黃
    'other':    (180, 180, 180),
}

WIN = ('annotate_sam2  '
       '[S=screener  H=handler  T=target(off-ball)  D=defender  '
       'C=clear  ←→=frame  N=next  B=back  Q=quit]')


# ── SAM2 追蹤（標準 init_state API）─────────────────────────────────────────

def run_sam2_tracks(clip_dir: Path, frames, player_model, predictor) -> list:
    """
    用 SAM2 標準 API 追蹤 clip_dir 裡的所有幀。
    回傳 raw_tracks: list[{int_tid: [x1,y1,x2,y2]}]，長度 = len(frames)
    """
    result = player_model.predict(frames[0], conf=0.4, verbose=False)[0]
    det0 = sv.Detections.from_ultralytics(result)
    if len(det0) == 0:
        return [{} for _ in frames]

    obj_ids = list(range(1, len(det0) + 1))

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
            for obj_id, xyxy in zip(obj_ids, det0.xyxy):
                predictor.add_new_points_or_box(
                    inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=xyxy.astype(np.float32),
                )

        # frame 0 直接用 YOLO 結果
        raw_tracks = [{} for _ in frames]
        for obj_id, xyxy in zip(obj_ids, det0.xyxy):
            raw_tracks[0][obj_id] = xyxy.tolist()

        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
                for obj_id, mask in zip(object_ids, masks):
                    mask_np = (mask[0].cpu().numpy() > 0)
                    if mask_np.any() and frame_idx < len(raw_tracks):
                        ys, xs = np.where(mask_np)
                        raw_tracks[frame_idx][int(obj_id)] = [
                            float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max()),
                        ]

        predictor.reset_state(inference_state)
        return raw_tracks
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── SAM2 追蹤（含快取）───────────────────────────────────────────────────────

def get_or_compute_tracks(clip_id: str, clip_dir: Path, frames, frame_nums,
                           player_model, predictor, tracks_cache: dict) -> list:
    """
    若 tracks_cache 已有此 clip 直接回傳；否則跑 SAM2 並更新快取。
    回傳 raw_tracks: list[{int_tid: [x1,y1,x2,y2]}]，長度 = len(frames)
    """
    if clip_id in tracks_cache:
        pt = tracks_cache[clip_id]
        pt_by_fnum = {fn: {int(k): v for k, v in fd.items()}
                      for fn, fd in zip(pt['frame_nums'], pt['tracks'])}
        return [pt_by_fnum.get(fn, {}) for fn in frame_nums]

    print(f"  Running SAM2 on clip {clip_id}...")
    try:
        raw_tracks = run_sam2_tracks(clip_dir, frames, player_model, predictor)
    except Exception as e:
        print(f"  SAM2 failed: {e}")
        return [{} for _ in frames]

    # 寫入快取
    tracks_cache[clip_id] = {
        'frame_nums': frame_nums,
        'tracks': [{str(tid): xyxy for tid, xyxy in fd.items()} for fd in raw_tracks],
    }
    TRACKS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKS_CACHE_PATH, 'w') as f:
        json.dump(tracks_cache, f)
    print(f"  SAM2 done, cached")

    return raw_tracks


# ── 繪製 ──────────────────────────────────────────────────────────────────────

def draw_frame(base_frame, frame_dict, assignments, selected_tid):
    img = base_frame.copy()
    role_of = {v: k for k, v in assignments.items() if v is not None}

    for tid, xyxy in frame_dict.items():
        x1, y1, x2, y2 = [int(v) for v in xyxy]

        if tid == selected_tid:
            color, thick, label = ROLE_COLORS['selected'], 3, f"{tid} ?"
        elif tid in role_of:
            role = role_of[tid]
            color, thick = ROLE_COLORS[role], 3
            label = f"{tid} [{role[:3].upper()}]"
        else:
            color, thick, label = ROLE_COLORS['other'], 1, str(tid)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    s = assignments.get('screener')
    h = assignments.get('handler')
    t = assignments.get('target')
    d = assignments.get('defender')
    screen_type = 'off-ball' if t is not None else 'ball'
    status = f"SCR:{s}  HDL:{h}  TGT:{t}  DEF:{d}  [{screen_type}]"
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    return img


# ── 滑鼠 callback ─────────────────────────────────────────────────────────────

def make_mouse_cb(frame_dict_ref, click_state):
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        hit = None
        for tid, xyxy in frame_dict_ref[0].items():
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            if x1 <= x <= x2 and y1 <= y <= y2:
                hit = tid
                break
        click_state['tid'] = hit
    return on_mouse


# ── 標記單個 clip ─────────────────────────────────────────────────────────────

def annotate_clip(clip_id: str, frames, frame_nums, raw_tracks, annotations) -> str:
    n_frames = len(frames)
    ann = annotations.get(clip_id, {})
    assignments = {
        'screener': ann.get('screener'),
        'handler':  ann.get('handler'),
        'target':   ann.get('target'),    # None = ball screen
        'defender': ann.get('defender'),
    }

    frame_idx      = 0
    selected_tid   = None
    playing        = False
    frame_dict_ref = [{}]
    click_state    = {'tid': None}

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, make_mouse_cb(frame_dict_ref, click_state))

    while True:
        fd = raw_tracks[frame_idx]
        frame_dict_ref[0] = fd

        if click_state['tid'] is not None:
            selected_tid = click_state['tid']
            click_state['tid'] = None

        display = draw_frame(frames[frame_idx], fd, assignments, selected_tid)

        h_img, w_img = display.shape[:2]
        bar_y = h_img - 20
        cv2.rectangle(display, (0, bar_y), (w_img, h_img), (40, 40, 40), -1)
        bar_x = int(w_img * frame_idx / max(n_frames - 1, 1))
        cv2.rectangle(display, (0, bar_y), (bar_x, h_img), (100, 200, 100), -1)
        cv2.putText(display, f"clip {clip_id}  frame {frame_idx+1}/{n_frames}",
                    (10, h_img - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(WIN, display)
        wait = 80 if playing else 30
        key = cv2.waitKey(wait) & 0xFF

        if playing:
            frame_idx = (frame_idx + 1) % n_frames
            if frame_idx == 0:
                playing = False

        if key in (ord('q'), ord('Q')):
            _save_clip(clip_id, assignments, annotations)
            return 'quit'
        elif key in (ord('n'), ord('N')):
            _save_clip(clip_id, assignments, annotations)
            return 'next'
        elif key in (ord('b'), ord('B')):
            return 'back'
        elif key == ord(' '):
            playing = not playing
        elif key in (81, 2):    # ←
            frame_idx = max(0, frame_idx - 1)
            playing = False
        elif key in (83, 3):    # →
            frame_idx = min(n_frames - 1, frame_idx + 1)
            playing = False
        elif key in (ord('s'), ord('S')):
            if selected_tid is not None:
                assignments['screener'] = selected_tid
        elif key in (ord('h'), ord('H')):
            if selected_tid is not None:
                assignments['handler'] = selected_tid
        elif key in (ord('t'), ord('T')):
            if selected_tid is not None:
                assignments['target'] = selected_tid
        elif key in (ord('d'), ord('D')):
            if selected_tid is not None:
                assignments['defender'] = selected_tid
        elif key in (ord('c'), ord('C')):
            if selected_tid is not None:
                for role in ('screener', 'handler', 'target', 'defender'):
                    if assignments.get(role) == selected_tid:
                        assignments[role] = None
            selected_tid = None


def _save_clip(clip_id: str, assignments, annotations):
    confident = (assignments.get('screener') is not None and
                 assignments.get('handler') is not None)
    annotations[clip_id] = {
        'screener':  assignments.get('screener'),
        'handler':   assignments.get('handler'),
        'target':    assignments.get('target'),   # None = ball screen
        'defender':  assignments.get('defender'),
        'confident': confident,
    }
    with open(ANN_PATH, 'w') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    screen_type = 'off-ball' if assignments.get('target') is not None else 'ball'
    status = 'confident' if confident else 'incomplete'
    print(f"  Clip {clip_id} saved ({status}, {screen_type} screen): "
          f"scr={assignments['screener']} hdl={assignments['handler']} "
          f"tgt={assignments['target']} def={assignments['defender']}")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_confident', action='store_true',
                        help='自動跳過已 confident=True 的 clips')
    args = parser.parse_args()

    if not SAM2_CHECKPOINT.exists():
        print(f"ERROR: SAM2 model not found at {SAM2_CHECKPOINT}")
        sys.exit(1)

    print("Loading models...")
    player_model = YOLO(str(PLAYER_MODEL_PATH))
    predictor    = build_sam2_video_predictor(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=DEVICE)
    print(f"Models loaded (SAM2 on {DEVICE})\n")

    tracks_cache = {}
    if TRACKS_CACHE_PATH.exists():
        with open(TRACKS_CACHE_PATH) as f:
            tracks_cache = json.load(f)
        print(f"Tracks cache loaded: {len(tracks_cache)} clips")

    annotations = {}
    if ANN_PATH.exists():
        with open(ANN_PATH) as f:
            annotations = json.load(f)

    clip_dirs = sorted(
        [d for d in FRAMES_DIR.iterdir()
         if d.is_dir() and re.match(r'screen_\d+', d.name)],
        key=lambda p: int(re.search(r'screen_(\d+)', p.name).group(1))
    )
    clip_ids = [str(int(re.search(r'screen_(\d+)', d.name).group(1)))
                for d in clip_dirs]

    confident_n = sum(1 for v in annotations.values() if v.get('confident'))
    print(f"{len(clip_ids)} clips total, {confident_n} already confident\n")

    idx = 0
    while 0 <= idx < len(clip_ids):
        cid = clip_ids[idx]

        if args.skip_confident and annotations.get(cid, {}).get('confident'):
            idx += 1
            continue

        clip_dir = FRAMES_DIR / f'screen_{cid}'
        frame_paths = sorted(
            clip_dir.glob('*.jpg'),
            key=lambda p: int(re.search(r'_(\d{4})\.jpg$', p.name).group(1))
        )
        if not frame_paths:
            idx += 1
            continue

        frames, frame_nums = [], []
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
                frame_nums.append(int(re.search(r'_(\d{4})\.jpg$', p.name).group(1)))
        if not frames:
            idx += 1
            continue

        raw_tracks = get_or_compute_tracks(
            cid, clip_dir, frames, frame_nums, player_model, predictor, tracks_cache)

        result = annotate_clip(cid, frames, frame_nums, raw_tracks, annotations)

        if result == 'quit':
            break
        elif result == 'next':
            idx += 1
        elif result == 'back':
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()
    confident_n = sum(1 for v in annotations.values() if v.get('confident'))
    print(f"\nDone. {confident_n}/{len(clip_ids)} clips annotated.")
    print(f"Saved → {ANN_PATH}")


if __name__ == '__main__':
    main()
