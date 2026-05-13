"""
annotate_tracks.py

互動式 track_id 標記工具。
需要先執行 visualize_tracks.py 或 visualize_tracks_sam2.py 產生 output_track_viz/。

操作方式：
  滑鼠左鍵   選取球員（框變黃色）
  S          把選取的球員設為 screener（紅框）
  D          把選取的球員設為 defender（藍框）
  H          把選取的球員設為 handler（綠框）
  C          清除選取球員的所有角色
  ← / →      上一幀 / 下一幀
  Space      播放 / 暫停
  N          儲存並跳到下一個 clip
  B          跳回上一個 clip（不儲存）
  Q          儲存全部並退出

執行方式（本機）：
  python annotate_tracks.py
  python annotate_tracks.py --tracks output_track_viz/tracks_data_sam2.json
"""

import argparse, json, re, sys
import cv2
import numpy as np
from pathlib import Path

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tracks', type=Path, default=None,
                   help='tracks_data.json 路徑（預設：output_track_viz/tracks_data.json）')
    return p.parse_args()

SCRIPT_DIR  = Path(__file__).parent
OUT_DIR     = SCRIPT_DIR / 'output_track_viz'
ANN_PATH    = SCRIPT_DIR / 'track_annotations.json'

_args       = _parse_args()
TRACKS_PATH = _args.tracks if _args.tracks else OUT_DIR / 'tracks_data.json'
FRAMES_DIR  = SCRIPT_DIR.parent / 'input_videos' / 'screen_frames'

ROLE_COLORS = {
    'screener': (0,   0,   255),
    'defender': (255, 0,   0),
    'handler':  (0,   255, 0),
    'selected': (0,   255, 255),
    'other':    (180, 180, 180),
}

WIN = 'annotate_tracks  [S=screener  D=defender  H=handler  C=clear  ←→=frame  N=next  B=back  Q=quit]'


# ── 載入資料 ──────────────────────────────────────────────────────────────────

def load_data():
    if not TRACKS_PATH.exists():
        print(f"ERROR: {TRACKS_PATH} not found. Run visualize_tracks.py first.")
        sys.exit(1)

    with open(TRACKS_PATH) as f:
        tracks_data = json.load(f)

    annotations = {}
    if ANN_PATH.exists():
        with open(ANN_PATH) as f:
            annotations = json.load(f)

    return tracks_data, annotations


def save_annotations(annotations):
    with open(ANN_PATH, 'w') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


# ── 繪製 ──────────────────────────────────────────────────────────────────────

def draw_frame(base_frame, frame_dict, assignments, selected_tid):
    """
    base_frame  : numpy BGR image
    frame_dict  : {str(tid): [x1,y1,x2,y2]}
    assignments : {'screener': tid/None, 'defender': tid/None, 'handler': tid/None}
    selected_tid: int or None
    """
    img = base_frame.copy()
    role_of = {}
    for role in ('screener', 'defender', 'handler'):
        v = assignments.get(role)
        if v is not None:
            role_of[v] = role

    for tid_str, xyxy in frame_dict.items():
        tid = int(tid_str)
        x1, y1, x2, y2 = [int(v) for v in xyxy]

        if tid == selected_tid:
            color, thick = ROLE_COLORS['selected'], 3
            label = f"{tid} ?"
        elif tid in role_of:
            color, thick = ROLE_COLORS[role_of[tid]], 3
            label = f"{tid} [{role_of[tid][:3].upper()}]"
        else:
            color, thick = ROLE_COLORS['other'], 1
            label = str(tid)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 右上角顯示目前指派狀態
    s = assignments.get('screener')
    d = assignments.get('defender')
    h = assignments.get('handler')
    status = f"SCR:{s}  DEF:{d}  HDL:{h}"
    cv2.putText(img, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return img


# ── 滑鼠 callback ─────────────────────────────────────────────────────────────

_click_state = {'tid': None}

def make_mouse_cb(frame_dict_ref, click_state):
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        fd = frame_dict_ref[0]
        hit = None
        for tid_str, xyxy in fd.items():
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            if x1 <= x <= x2 and y1 <= y <= y2:
                hit = int(tid_str)
                break
        click_state['tid'] = hit
    return on_mouse


# ── 主迴圈 ────────────────────────────────────────────────────────────────────

def annotate_clip(clip_id_str, clip_tracks, annotations):
    """
    Returns True to go to next clip, False to go back.
    """
    frame_nums = clip_tracks['frame_nums']
    tracks_list = clip_tracks['tracks']  # list of {str(tid): xyxy}
    n_frames = len(frame_nums)

    # 載入影格圖片
    clip_dir = FRAMES_DIR / f'screen_{clip_id_str}'
    frames = []
    for fn in frame_nums:
        p = clip_dir / f'screen_{clip_id_str}_{fn:04d}.jpg'
        if not p.exists():
            # 嘗試找任何符合的檔案
            candidates = list(clip_dir.glob(f'*_{fn:04d}.jpg'))
            p = candidates[0] if candidates else None
        img = cv2.imread(str(p)) if p and p.exists() else None
        if img is None:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
        frames.append(img)

    # 取得目前的標記
    ann = annotations.get(clip_id_str, {})
    assignments = {
        'screener': ann.get('screener'),
        'defender': ann.get('defender'),
        'handler':  ann.get('handler'),
    }

    frame_idx   = 0
    selected_tid = None
    playing     = False
    frame_dict_ref = [tracks_list[frame_idx]]
    click_state    = {'tid': None}

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, make_mouse_cb(frame_dict_ref, click_state))

    while True:
        fd = tracks_list[frame_idx]
        frame_dict_ref[0] = fd

        # 更新被點擊的 tid
        if click_state['tid'] is not None:
            selected_tid = click_state['tid']
            click_state['tid'] = None

        display = draw_frame(frames[frame_idx], fd, assignments, selected_tid)

        # 底部進度條
        h, w = display.shape[:2]
        bar_y = h - 20
        cv2.rectangle(display, (0, bar_y), (w, h), (40, 40, 40), -1)
        bar_x = int(w * frame_idx / max(n_frames - 1, 1))
        cv2.rectangle(display, (0, bar_y), (bar_x, h), (100, 200, 100), -1)
        cv2.putText(display, f"clip {clip_id_str}  frame {frame_idx+1}/{n_frames}",
                    (10, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(WIN, display)
        wait = 80 if playing else 30
        key = cv2.waitKey(wait) & 0xFF

        if playing:
            frame_idx = (frame_idx + 1) % n_frames
            if frame_idx == 0:
                playing = False

        if key == ord('q') or key == ord('Q'):
            _save_clip(clip_id_str, assignments, annotations)
            return 'quit'

        elif key == ord('n') or key == ord('N'):
            _save_clip(clip_id_str, assignments, annotations)
            return 'next'

        elif key == ord('b') or key == ord('B'):
            return 'back'

        elif key == ord(' '):
            playing = not playing

        elif key == 81 or key == 2:   # ← (Linux/Windows)
            frame_idx = max(0, frame_idx - 1)
            playing = False

        elif key == 83 or key == 3:   # →
            frame_idx = min(n_frames - 1, frame_idx + 1)
            playing = False

        elif key == ord('s') or key == ord('S'):
            if selected_tid is not None:
                assignments['screener'] = selected_tid

        elif key == ord('d') or key == ord('D'):
            if selected_tid is not None:
                assignments['defender'] = selected_tid

        elif key == ord('h') or key == ord('H'):
            if selected_tid is not None:
                assignments['handler'] = selected_tid

        elif key == ord('c') or key == ord('C'):
            if selected_tid is not None:
                for role in ('screener', 'defender', 'handler'):
                    if assignments.get(role) == selected_tid:
                        assignments[role] = None
            selected_tid = None


def _save_clip(clip_id_str, assignments, annotations):
    confident = (assignments.get('screener') is not None and
                 assignments.get('handler')  is not None)
    annotations[clip_id_str] = {
        'screener':  assignments.get('screener'),
        'defender':  assignments.get('defender'),
        'handler':   assignments.get('handler'),
        'confident': confident,
    }
    save_annotations(annotations)
    status = 'confident' if confident else 'incomplete'
    print(f"  Clip {clip_id_str} saved ({status}): {assignments}")


def main():
    tracks_data, annotations = load_data()

    clip_ids = sorted(tracks_data.keys(), key=lambda x: int(x))
    idx = 0

    print(f"Loaded {len(clip_ids)} clips.")
    print(f"Already annotated: "
          f"{sum(1 for v in annotations.values() if v.get('confident'))}/{len(clip_ids)}")
    print(f"\nPress N to skip already-confident clips, or start from the first.\n")

    while 0 <= idx < len(clip_ids):
        cid = clip_ids[idx]
        result = annotate_clip(cid, tracks_data[cid], annotations)

        if result == 'quit':
            break
        elif result == 'next':
            idx += 1
        elif result == 'back':
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()
    confident = sum(1 for v in annotations.values() if v.get('confident'))
    print(f"\nDone. {confident}/{len(clip_ids)} clips annotated.")
    print(f"Saved → {ANN_PATH}")
    print(f"\n接下來執行：python convert_manual.py")


if __name__ == '__main__':
    main()
