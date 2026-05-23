"""
convert_manual.py

YOLO（第一幀偵測）+ SAM2（穩定跨幀追蹤）+ TeamClassifier（SigLIP 球隊分類）
→ output_manual/*.npz

Output:
  output_manual/positive_sequences.npz
  output_manual/negative_sequences.npz

Each npz:
  trajectories   : (N, 10, 10, 2)
  roles          : (N, 10)   0=handler, 1=other_offense, 2=defense
  labels         : (N,)      binary
  pairwise_labels: (N, 20)   pair (i,j): j screens for i

Usage:
  python convert_manual.py
"""

import sys, re, json, collections, os, tempfile, shutil
import numpy as np
import cv2
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

FRAMES_DIR        = PROJECT_ROOT / 'input_videos' / 'screen_frames'
COCO_PATH         = PROJECT_ROOT / 'screen_data' / 'screen_v2.coco' / 'train' / '_annotations.coco.json'
OUT_DIR           = Path(__file__).parent / 'output_manual'
PLAYER_MODEL_PATH = PROJECT_ROOT / 'models' / 'player_detector.pt'
COURT_MODEL_PATH  = PROJECT_ROOT / 'models' / 'court_keypoint_detector.pt'
SAM2_CHECKPOINT   = PROJECT_ROOT / 'models' / 'sam2.1_hiera_large.pt'
SAM2_CONFIG       = 'configs/sam2.1/sam2.1_hiera_l.yaml'
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'

SEQ_LEN   = 10
N_OFF     = 5
N_DEF     = 5
N_PLAYERS = N_OFF + N_DEF
STRIDE    = 1

COURT_W_PX, COURT_H_PX = 300, 161
COURT_W_FT, COURT_H_FT = 91.8, 49.2

_W, _H = COURT_W_PX, COURT_H_PX
_AW, _AH = 28.0, 15.0
TARGET_KPS = [
    (0, 0),
    (0, int((0.91/_AH)*_H)),
    (0, int((5.18/_AH)*_H)),
    (0, int((10/_AH)*_H)),
    (0, int((14.1/_AH)*_H)),
    (0, int(_H)),
    (int(_W/2), _H),
    (int(_W/2), 0),
    (int((5.79/_AW)*_W),  int((5.18/_AH)*_H)),
    (int((5.79/_AW)*_W),  int((10/_AH)*_H)),
    (_W, int(_H)),
    (_W, int((14.1/_AH)*_H)),
    (_W, int((10/_AH)*_H)),
    (_W, int((5.18/_AH)*_H)),
    (_W, int((0.91/_AH)*_H)),
    (_W, 0),
    (int(((_AW-5.79)/_AW)*_W), int((5.18/_AH)*_H)),
    (int(((_AW-5.79)/_AW)*_W), int((10/_AH)*_H)),
]


# ── SAM2 追蹤 ─────────────────────────────────────────────────────────────────

def run_sam2_tracks(clip_dir: Path, frames: list, player_model, predictor) -> list:
    """YOLO 偵測第一幀 → SAM2 跨幀傳播。返回 list[{obj_id: [x1,y1,x2,y2]}]。"""
    result = player_model.predict(frames[0], conf=0.4, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return []
    obj_ids = list(range(1, len(boxes) + 1))

    frame_paths_sorted = sorted(
        clip_dir.glob('*.jpg'),
        key=lambda p: int(re.search(r'_(\d{4})\.jpg$', p.name).group(1))
    )
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        for i, fp in enumerate(frame_paths_sorted):
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

        raw_tracks = [{} for _ in frame_paths_sorted]
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_homography(kp_result):
    if kp_result.keypoints is None:
        return None
    kp_list = kp_result.keypoints.xy.tolist()
    if not kp_list:
        return None
    kp_xy = kp_list[0]
    valid = [(i, kp) for i, kp in enumerate(kp_xy)
             if kp[0] > 0 and kp[1] > 0 and i < len(TARGET_KPS)]
    if len(valid) < 4:
        return None
    src = np.array([kp for _, kp in valid], dtype=np.float32)
    dst = np.array([TARGET_KPS[i] for i, _ in valid], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def foot_pos_xyxy(xyxy):
    return ((xyxy[0]+xyxy[2])/2, xyxy[3])


def transform(H, x, y):
    if H is None:
        return None
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0,0,0]), float(out[0,0,1])


def tactical_to_feet(x_px, y_px):
    return x_px / COURT_W_PX * COURT_W_FT, y_px / COURT_H_PX * COURT_H_FT


def pair_index(i, j):
    """Linear index of pair (i,j) where j screens for i."""
    idx = 0
    for ii in range(N_OFF):
        for jj in range(N_OFF):
            if ii == jj:
                continue
            if ii == i and jj == j:
                return idx
            idx += 1
    return -1


def classify_teams_siglip(frames, raw_tracks, screener_tid, handler_tid, defender_tid):
    """sv.TeamClassifier (SigLIP + UMAP + k-means) 分球隊。失敗回傳 None。"""
    crops, crop_tids = [], []
    for frame, fd in zip(frames, raw_tracks):
        for tid, xyxy in fd.items():
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            h = y2 - y1
            crop = frame[y1:y1+max(1, h//2), x1:x2]  # 上半身（球衣區域）
            if crop.shape[0] > 5 and crop.shape[1] > 5:
                crops.append(crop)
                crop_tids.append(tid)

    if len(crops) < 4:
        return None

    try:
        classifier = sv.TeamClassifier(device=DEVICE)
        classifier.fit(crops)
        predictions = classifier.predict(crops)
    except Exception as e:
        print(f"    TeamClassifier failed: {e}")
        return None

    # 每個 tid 多幀投票
    tid_votes = collections.defaultdict(lambda: [0, 0])
    for tid, team in zip(crop_tids, predictions):
        tid_votes[tid][int(team)] += 1

    # 決定哪個 cluster 是 offense（以 screener/handler/defender 為錨點）
    track_team_raw = {tid: (0 if v[0] >= v[1] else 1) for tid, v in tid_votes.items()}

    offense_cluster = None
    if screener_tid is not None and screener_tid in track_team_raw:
        offense_cluster = track_team_raw[screener_tid]
    elif handler_tid is not None and handler_tid in track_team_raw:
        offense_cluster = track_team_raw[handler_tid]
    elif defender_tid is not None and defender_tid in track_team_raw:
        offense_cluster = 1 - track_team_raw[defender_tid]
    if offense_cluster is None:
        offense_cluster = 0

    return {tid: ('offense' if c == offense_cluster else 'defense')
            for tid, c in track_team_raw.items()}


def classify_teams_hsv_fallback(frames, raw_tracks, screener_tid, handler_tid, defender_tid):
    """HSV k-means fallback。

    若有 screener + defender anchor → 用顏色距離直接比對每個 player。
    否則 → 所有 player 一起做 k-means，用 screener_tid 定義 offense cluster。
    """
    def get_color(frame, xyxy):
        x1, y1, x2, y2 = [int(float(v)) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2_half = min(frame.shape[0], y1 + (y2-y1)//2)
        crop = frame[y1:y2_half, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return hsv.reshape(-1, 3).mean(axis=0)

    all_tids = list(set(tid for fd in raw_tracks for tid in fd))

    # 每個 tid 的代表顏色（取前 3 幀平均）
    tid_colors = {}
    for tid in all_tids:
        colors = []
        for f_idx, fd in enumerate(raw_tracks):
            if tid in fd:
                c = get_color(frames[f_idx], fd[tid])
                if c is not None:
                    colors.append(c)
                if len(colors) >= 3:
                    break
        if colors:
            tid_colors[tid] = np.mean(colors, axis=0)

    if not tid_colors:
        return {}

    # ── Case 1：有 screener + defender anchor → 直接顏色距離比對 ────────────
    screener_color = tid_colors.get(screener_tid)
    defender_color = tid_colors.get(defender_tid)

    if screener_color is not None and defender_color is not None:
        track_team = {}
        for tid, color in tid_colors.items():
            track_team[tid] = ('offense' if np.linalg.norm(color - screener_color)
                               < np.linalg.norm(color - defender_color) else 'defense')
        return track_team

    # ── Case 2：只有 screener anchor（無 defender）→ k-means，screener 定 offense ──
    tids_with_color = [(tid, c) for tid, c in tid_colors.items()]
    if len(tids_with_color) < 2:
        # 太少 player，screener=offense，其餘不確定
        return {tid: ('offense' if tid in (screener_tid, handler_tid) else 'unknown')
                for tid in all_tids}

    arr = np.array([c for _, c in tids_with_color], dtype=np.float32)
    _, labels_km, _ = cv2.kmeans(arr, 2, None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0),
                                 10, cv2.KMEANS_RANDOM_CENTERS)
    labels_flat = labels_km.flatten()

    # 找 screener（或 handler）所在的 cluster → offense
    offense_cluster = 0
    for i, (tid, _) in enumerate(tids_with_color):
        if tid == screener_tid or tid == handler_tid:
            offense_cluster = labels_flat[i]
            break

    return {tid: ('offense' if lbl == offense_cluster else 'defense')
            for (tid, _), lbl in zip(tids_with_color, labels_flat)}


# ── Main pipeline per clip ─────────────────────────────────────────────────────

def process_clip(clip_id, clip_dir, frame_paths, coco_by_frame,
                 player_model, court_model, predictor):
    frames, frame_nums = [], []
    for p in sorted(frame_paths, key=lambda x: int(re.search(r'_(\d{4})\.jpg$', x).group(1))):
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
            frame_nums.append(int(re.search(r'_(\d{4})\.jpg$', p).group(1)))

    if len(frames) < 3:
        return []

    print(f"  Clip {clip_id}: {len(frames)} frames, COCO frames={sorted(coco_by_frame.keys())[:6]}...")

    # SAM2 追蹤（穩定 ID）
    try:
        raw_tracks = run_sam2_tracks(clip_dir, frames, player_model, predictor)
    except Exception as e:
        print(f"  Clip {clip_id}: SAM2 failed ({e}), skipping")
        return []

    if not raw_tracks:
        print(f"  Clip {clip_id}: no detections in first frame, skipping")
        return []

    # 補足到 SEQ_LEN 幀
    while len(frames) < SEQ_LEN:
        frames.append(frames[-1].copy())
        frame_nums.append(frame_nums[-1])
        raw_tracks.append(dict(raw_tracks[-1]) if raw_tracks else {})

    # Court keypoints → homography → feet
    kp_results   = court_model.predict(frames, conf=0.5, verbose=False)
    H_list       = [build_homography(kp) for kp in kp_results]
    court_tracks = []
    for frame_dict, H in zip(raw_tracks, H_list):
        ct = {}
        if H is not None:
            for tid, xyxy in frame_dict.items():
                fx, fy = foot_pos_xyxy(xyxy)
                pt = transform(H, fx, fy)
                if pt is not None:
                    tx, ty = tactical_to_feet(*pt)
                    if 0 <= tx <= COURT_W_FT + 5 and 0 <= ty <= COURT_H_FT + 5:
                        ct[tid] = (tx, ty)
        court_tracks.append(ct)

    # ── COCO 跨幀投票：決定 screener/defender/handler 的代表 tid ────────────────
    # 支援多 screener：同一幀的不同 COCO bbox 強制配給不同 track_id（避免雙重指派）。
    # 最後依「任一幀內出現的最大 screener 數」決定要保留幾個 screener tid。
    screener_votes = collections.Counter()
    defender_votes = collections.Counter()
    handler_votes  = collections.Counter()
    screener_frames = set()
    max_screeners_per_frame = 0

    for f_num, anns in coco_by_frame.items():
        f_idx = frame_nums.index(f_num) if f_num in frame_nums else -1
        if f_idx < 0:
            continue
        frame_dict = raw_tracks[f_idx]
        if not frame_dict:
            continue

        n_scr_here = sum(1 for c, _ in anns if c == 'screener')
        if n_scr_here > max_screeners_per_frame:
            max_screeners_per_frame = n_scr_here

        used_tids_this_frame = set() # 防止重複分配
        for cat, bbox in anns:
            if cat not in ('screener', 'defender', 'ball_handler'):
                continue
            coco_cx = float(bbox[0]) + float(bbox[2]) / 2
            coco_cy = float(bbox[1]) + float(bbox[3]) / 2
            best_tid, best_dist = None, float('inf')
            for tid, xyxy in frame_dict.items():
                if tid in used_tids_this_frame:
                    continue  # ← 新增：跳過已被本幀其他標注用掉的 tid
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                d = ((coco_cx - cx) ** 2 + (coco_cy - cy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best_tid = d, tid

            if best_tid is None:
                continue

            used_tids_this_frame.add(best_tid)

            if cat == 'screener':
                screener_votes[best_tid] += 1
                screener_frames.add(f_num)

            elif cat == 'defender':
                defender_votes[best_tid] += 1

            elif cat == 'ball_handler':
                handler_votes[best_tid] += 1

    n_keep = max(1, max_screeners_per_frame)
    screener_tids = [tid for tid, _ in screener_votes.most_common(n_keep)] # 改成保留多個 screener tid
    defender_tid  = defender_votes.most_common(1)[0][0] if defender_votes else None
    handler_tid   = handler_votes.most_common(1)[0][0]  if handler_votes  else None

    primary_screener_tid = screener_tids[0] if screener_tids else None
    if max_screeners_per_frame > 1:
        print(f"    multi-screener clip: keeping {len(screener_tids)} screener tids {screener_tids}")

    # ── 球隊分類：SigLIP TeamClassifier，失敗退回 HSV ─────────────────────────
    track_team = classify_teams_siglip(frames, raw_tracks, primary_screener_tid, handler_tid, defender_tid)
    if track_team is None:
        track_team = classify_teams_hsv_fallback(frames, raw_tracks, primary_screener_tid, handler_tid, defender_tid)

    # COCO 已知角色強制覆蓋球隊分類結果
    for s_tid in screener_tids:
        track_team[s_tid] = 'offense'
    if handler_tid is not None:
        track_team[handler_tid] = 'offense'
    if defender_tid is not None:
        track_team[defender_tid] = 'defense'

    # ── Sliding window → sequences ─────────────────────────────────────────────
    clip_has_screener = len(screener_tids) > 0
    results = []

    for start in range(0, len(frames) - SEQ_LEN + 1, STRIDE):
        window_fnums = [frame_nums[start + k] for k in range(SEQ_LEN)]
        win_screener = screener_frames & set(window_fnums)

        if win_screener:
            label = 1
        elif clip_has_screener:
            label = 0
        else:
            continue

        window_tracks = [court_tracks[start + k] for k in range(SEQ_LEN)]
        window_raw    = [raw_tracks[start + k]   for k in range(SEQ_LEN)]

        tid_counts   = collections.Counter(tid for fd in window_raw for tid in fd)
        present_tids = set(tid for tid, cnt in tid_counts.items() if cnt >= SEQ_LEN // 2)

        # SAM2 ID 穩定，但仍強制把 COCO 已知角色加入（邊緣情況保護）
        for forced in screener_tids + [handler_tid, defender_tid]:
            if forced is not None and tid_counts.get(forced, 0) >= 1:
                present_tids.add(forced)
        present_tids = list(present_tids)

        offense_tids_w = [t for t in present_tids if track_team.get(t) == 'offense']
        defense_tids_w = [t for t in present_tids if track_team.get(t) == 'defense']

        need_defense = defender_tid is not None
        if len(offense_tids_w) < 2 or (need_defense and len(defense_tids_w) < 1):
            continue

        handler_in_window = handler_tid if (handler_tid is not None
                                            and handler_tid in offense_tids_w) else None
        screeners_in_window = [s for s in screener_tids if s in offense_tids_w]

        special    = {t for t in [handler_in_window] + screeners_in_window if t is not None}
        others_off = [t for t in offense_tids_w if t not in special]

        # 順序：handler → screener(s) → others。truncation 才不會把 screener 砍掉。
        offense_ordered = [handler_in_window] + screeners_in_window + others_off
        if not screeners_in_window:
            offense_ordered.insert(1, None)  # 預留 screener slot
        while len(offense_ordered) < N_OFF:
            offense_ordered.append(None)
        offense_ordered = offense_ordered[:N_OFF]

        defense_ordered = defense_tids_w[:]
        while len(defense_ordered) < N_DEF:
            defense_ordered.append(None)
        defense_ordered = defense_ordered[:N_DEF]

        player_order = offense_ordered + defense_ordered

        traj       = np.zeros((N_PLAYERS, SEQ_LEN, 2), dtype=np.float32)
        last_known = [None] * N_PLAYERS
        for k in range(SEQ_LEN):
            ct = window_tracks[k]
            for p_idx, tid in enumerate(player_order):
                if tid is not None and tid in ct:
                    traj[p_idx, k] = ct[tid]
                    last_known[p_idx] = ct[tid]
                elif last_known[p_idx] is not None:
                    traj[p_idx, k] = last_known[p_idx]

        roles = np.array([0] + [1]*4 + [2]*5, dtype=np.int64)

        pw = np.zeros(20, dtype=np.int64)
        if label == 1:
            for screener in screeners_in_window:
                s_slot = offense_ordered.index(screener) if screener in offense_ordered else -1
                if 0 <= s_slot < N_OFF:
                    idx = pair_index(0, s_slot)
                    if idx >= 0:
                        pw[idx] = 1

        if label == 1 and not pw.any():
            continue

        results.append((traj, roles, pw, label))

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(COCO_PATH) as f:
        coco = json.load(f)

    cat_map    = {c['id']: c['name'] for c in coco['categories']}
    ann_by_img = collections.defaultdict(list)
    for ann in coco['annotations']:
        ann_by_img[ann['image_id']].append((cat_map[ann['category_id']], ann['bbox']))

    coco_clips = collections.defaultdict(lambda: collections.defaultdict(list))
    for img in coco['images']:
        m = re.match(r'screen_(\d+)_(\d{4})', img['file_name'])
        if m:
            coco_clips[int(m.group(1))][int(m.group(2))] = ann_by_img[img['id']]

    print("Loading models...")
    player_model = YOLO(str(PLAYER_MODEL_PATH))
    court_model  = YOLO(str(COURT_MODEL_PATH))

    if not SAM2_CHECKPOINT.exists():
        print(f"ERROR: SAM2 checkpoint not found: {SAM2_CHECKPOINT}")
        print(f"  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
              f" -P {PROJECT_ROOT}/models/")
        sys.exit(1)

    print(f"Loading SAM2 on {DEVICE}...")
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=DEVICE)
    print("Models loaded.")

    all_pos, all_neg = [], []

    for clip_id in sorted(coco_clips.keys()):
        frames_dir  = FRAMES_DIR / f'screen_{clip_id}'
        if not frames_dir.exists():
            continue
        frame_paths   = [str(p) for p in sorted(frames_dir.glob('*.jpg'))]
        coco_by_frame = dict(coco_clips[clip_id])

        seqs = process_clip(clip_id, frames_dir, frame_paths, coco_by_frame,
                            player_model, court_model, predictor)

        for traj, roles, pw, label in seqs:
            if label == 1:
                all_pos.append((traj, roles, pw))
            else:
                all_neg.append((traj, roles, pw))

        print(f"  → {sum(1 for _,_,_,l in seqs if l==1)} pos, "
              f"{sum(1 for _,_,_,l in seqs if l==0)} neg windows")

    def save_npz(path, samples, label_val):
        if not samples:
            print(f"No samples for label={label_val}, skipping {path}")
            return
        trajs  = np.stack([s[0] for s in samples])
        roles  = np.stack([s[1] for s in samples])
        pws    = np.stack([s[2] for s in samples])
        labels = np.full(len(samples), label_val, dtype=np.float32)
        np.savez_compressed(str(path),
                            trajectories=trajs, roles=roles,
                            labels=labels, pairwise_labels=pws)
        print(f"Saved {len(samples):,} sequences → {path}")

    save_npz(OUT_DIR / 'positive_sequences.npz', all_pos, 1)
    save_npz(OUT_DIR / 'negative_sequences.npz', all_neg, 0)
    print(f"\nDone. Total pos={len(all_pos)}, neg={len(all_neg)}")


if __name__ == '__main__':
    main()
