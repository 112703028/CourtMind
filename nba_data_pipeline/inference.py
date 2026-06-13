"""
inference.py — 用訓練好的 ScreenNetPairwise 在新影片上預測掩護。

Pipeline（新版）:
  1. YOLO 第一幀偵測 + SAM2 跨幀傳播 → 穩定 player_id
  2. handler_detector.pt（multi-class YOLO）每幀偵測 ball_handler bbox
     → IOU 配對 SAM2 tracked id → 得到每幀的 handler player_id
  3. CourtKeypointDetector → homography → 場上座標 (feet)
  4. TeamAssigner → 分隊
  5. 滑動 10 幀視窗 → 預測 screener
  6. 影片輸出（HANDLER 綠框、SCREENER 紅框）

Usage:
  python nba_data_pipeline/inference.py \
    --video         input_videos/screen/screen_100.mp4 \
    --ckpt          nba_data_pipeline/checkpoints/finetune_manual_best.pt \
    --handler_model models/handler_detector.pt \
    --output_video  output_videos/screens_v100.avi \
    --threshold     0.70 \
    --debug

"""

import os
import sys
import re
import argparse
import tempfile
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import read_video, save_video
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector

from ultralytics import YOLO

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None

# 必須在 sys.path 設好後才能匯入
from nba_data_pipeline.model import ScreenNetPairwise, N_OFFENSE, N_DEFENSE, N_PLAYERS

# ── 常數（與 convert_manual.py 對齊）─────────────────────────────────────────────
SEQ_LEN = 10

SAM2_CHECKPOINT_DEFAULT = str(PROJECT_ROOT / 'models' / 'sam2.1_hiera_large.pt')
SAM2_CONFIG_DEFAULT     = 'configs/sam2.1/sam2.1_hiera_l.yaml'
HANDLER_MODEL_DEFAULT   = str(PROJECT_ROOT / 'models' / 'handler_detector.pt')
PLAYER_MODEL_DEFAULT    = str(PROJECT_ROOT / 'models' / 'player_detector.pt')

COURT_W_PX, COURT_H_PX = 300, 161
COURT_W_FT, COURT_H_FT = 91.8, 49.2

_W, _H = COURT_W_PX, COURT_H_PX
_AW, _AH = 28.0, 15.0
TARGET_KPS = [
    (0, 0),
    (0, int((0.91 / _AH) * _H)),
    (0, int((5.18 / _AH) * _H)),
    (0, int((10 / _AH) * _H)),
    (0, int((14.1 / _AH) * _H)),
    (0, int(_H)),
    (int(_W / 2), _H),
    (int(_W / 2), 0),
    (int((5.79 / _AW) * _W), int((5.18 / _AH) * _H)),
    (int((5.79 / _AW) * _W), int((10 / _AH) * _H)),
    (_W, int(_H)),
    (_W, int((14.1 / _AH) * _H)),
    (_W, int((10 / _AH) * _H)),
    (_W, int((5.18 / _AH) * _H)),
    (_W, int((0.91 / _AH) * _H)),
    (_W, 0),
    (int(((_AW - 5.79) / _AW) * _W), int((5.18 / _AH) * _H)),
    (int(((_AW - 5.79) / _AW) * _W), int((10 / _AH) * _H)),
]


# ── SAM2 追蹤 ─────────────────────────────────────────────────────────────────

def _pick_init_frame(frames, player_model, min_players=5, max_search=30, conf=0.4):
    """
    在前 max_search 幀裡找一個偵測球員最多的當 SAM2 init frame。
    避免拿第 0 幀（可能是黑屏 / scene 切換 / 少數人入鏡）導致 SAM2 追錯。
    """
    best_idx, best_count, best_boxes = 0, 0, None
    n_search = min(max_search, len(frames))
    for i in range(n_search):
        r = player_model.predict(frames[i], conf=conf, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        n = len(boxes)
        if n > best_count:
            best_idx, best_count, best_boxes = i, n, boxes
            if n >= 10:  # 已經 10 人 = 完整 5v5，就用這幀
                break
    if best_count < min_players:
        print(f"  ⚠ 前 {n_search} 幀都偵測不到 >={min_players} 個球員 "
              f"(最佳 frame {best_idx} 有 {best_count} 個)")
    return best_idx, best_boxes


def run_sam2_tracks(frames, player_model, predictor, device='cuda'):
    """
    YOLO 在「球員最多的幀」偵測 → SAM2 跨幀傳播。
    回傳 list[dict[player_id: {'bbox': [x1,y1,x2,y2]}]]，與舊 player_tracks 格式相容。
    """
    if not frames:
        return []

    init_idx, boxes = _pick_init_frame(frames, player_model)
    if boxes is None or len(boxes) == 0:
        print("  ⚠ SAM2: 找不到能初始化的幀")
        return [{} for _ in frames]

    print(f"  SAM2 init at frame {init_idx} with {len(boxes)} player bboxes")
    obj_ids = list(range(1, len(boxes) + 1))

    # SAM2 video predictor 需要影格目錄，把 numpy frames 寫成 jpg
    tmp_dir = Path(tempfile.mkdtemp(prefix='sam2_frames_'))
    try:
        for i, frame in enumerate(frames):
            cv2.imwrite(str(tmp_dir / f'{i:05d}.jpg'), frame)

        inference_state = predictor.init_state(video_path=str(tmp_dir))
        predictor.reset_state(inference_state)

        autocast_dtype = torch.bfloat16 if device == 'cuda' else torch.float32

        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            for obj_id, xyxy in zip(obj_ids, boxes):
                predictor.add_new_points_or_box(
                    inference_state, frame_idx=init_idx, obj_id=obj_id,
                    box=xyxy.astype(np.float32))

        raw_tracks = [{} for _ in frames]
        # init frame 直接用 YOLO 結果填入
        for obj_id, xyxy in zip(obj_ids, boxes):
            raw_tracks[init_idx][int(obj_id)] = {'bbox': xyxy.tolist()}

        def _absorb(frame_idx, object_ids, masks):
            for obj_id, mask in zip(object_ids, masks):
                mask_np = (mask[0].cpu().numpy() > 0)
                if mask_np.any() and 0 <= frame_idx < len(raw_tracks):
                    ys, xs = np.where(mask_np)
                    raw_tracks[frame_idx][int(obj_id)] = {
                        'bbox': [float(xs.min()), float(ys.min()),
                                 float(xs.max()), float(ys.max())]
                    }

        # 從 init_idx 往「後」propagate
        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
                _absorb(frame_idx, object_ids, masks)

        # 從 init_idx 往「前」propagate（如果 init_idx > 0）
        if init_idx > 0:
            with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
                for frame_idx, object_ids, masks in predictor.propagate_in_video(
                        inference_state, reverse=True):
                    _absorb(frame_idx, object_ids, masks)

        predictor.reset_state(inference_state)
        return raw_tracks
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Handler 偵測 + IOU 配對 ───────────────────────────────────────────────────

def iou(a, b):
    """兩個 xyxy bbox 的 IOU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def verify_sam2_with_yolo(frames, player_tracks, player_model,
                          iou_threshold=0.5, batch=16, conf=0.4):
    """
    每幀用 YOLO 重新偵測，跟 SAM2 給的 bbox 配對。
    SAM2 mask 飄掉 / 跟其他 ID 撞 → IOU 太低或被搶 → 標記為 drifted。

    每個 SAM2 player_id 在每幀：
      - 找最大 IOU 的 YOLO bbox
      - IOU < iou_threshold → drifted
      - 多個 SAM2 ID 搶同一個 YOLO bbox → 只留 IOU 最高的，其餘 drifted

    drifted 的會把 bbox 設為 None，並加 'drifted': True flag。
    下游（team_assigner / court_coords / draw）原本就有 `if bbox is None` 跳過邏輯。
    """
    print(f"  YOLO drift verification ({len(frames)} frames)...")
    yolo_results = []
    for i in range(0, len(frames), batch):
        yolo_results.extend(
            player_model.predict(frames[i:i+batch], conf=conf, verbose=False)
        )

    n_drift = 0
    n_total = 0

    for f_idx, result in enumerate(yolo_results):
        frame_tracks = player_tracks[f_idx]
        if not frame_tracks:
            continue

        yolo_bboxes = (result.boxes.xyxy.cpu().numpy()
                       if result.boxes is not None and len(result.boxes) > 0
                       else [])

        # 每個 SAM2 pid 找最佳 YOLO 配對
        match = {}   # pid -> (yolo_idx, iou)
        for pid, info in frame_tracks.items():
            n_total += 1
            sam_bbox = info.get('bbox') if isinstance(info, dict) else None
            if sam_bbox is None:
                continue
            best_iou, best_idx = 0.0, -1
            for y_idx, y_bbox in enumerate(yolo_bboxes):
                v = iou(sam_bbox, y_bbox)
                if v > best_iou:
                    best_iou, best_idx = v, y_idx
            match[pid] = (best_idx, best_iou)

        # 衝突解決：每個 YOLO bbox 只能配對一個 SAM2 ID（IOU 最高的勝出）
        yolo_winner = {}    # yolo_idx -> (winner_pid, winner_iou)
        drifted_pids = set()
        for pid, (y_idx, iou_v) in match.items():
            if y_idx == -1 or iou_v < iou_threshold:
                drifted_pids.add(pid)
                continue
            if y_idx in yolo_winner:
                prev_pid, prev_iou = yolo_winner[y_idx]
                if iou_v > prev_iou:
                    drifted_pids.add(prev_pid)
                    yolo_winner[y_idx] = (pid, iou_v)
                else:
                    drifted_pids.add(pid)
            else:
                yolo_winner[y_idx] = (pid, iou_v)

        # 把 drifted 的 bbox 清掉
        for pid in drifted_pids:
            if pid in frame_tracks:
                frame_tracks[pid]['bbox'] = None
                frame_tracks[pid]['drifted'] = True
                n_drift += 1

    print(f"  → {n_drift}/{n_total} player-frames marked as drifted "
          f"({100*n_drift/max(n_total,1):.1f}%)")
    return player_tracks


def detect_player_class_votes(frames, player_tracks, handler_model,
                              conf=0.4, iou_threshold=0.3):
    """
    對每幀跑 multi-class YOLO，IOU 配對到 SAM2 player_id，
    記錄每個 player_id 整段被預測為各 class 的票數。

    回傳 dict[player_id: Counter({class_id: count})]。
    class 0=ball_handler, 1=screener, 2=defender, 3=others
    """
    from collections import defaultdict
    votes = defaultdict(Counter)
    n = len(frames)

    BATCH = 16
    results = []
    for i in range(0, n, BATCH):
        batch_res = handler_model.predict(frames[i:i+BATCH], conf=conf, verbose=False)
        results.extend(batch_res)

    for f_idx, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        cls  = result.boxes.cls.cpu().numpy().astype(int)
        xyxy = result.boxes.xyxy.cpu().numpy()

        # 對每個 YOLO detection 配對最匹配的 SAM2 player_id（避免重複指派）
        used_pids = set()
        for det_idx in range(len(cls)):
            det_class = int(cls[det_idx])
            det_bbox  = xyxy[det_idx]

            best_pid, best_iou = -1, 0.0
            for pid, info in player_tracks[f_idx].items():
                if pid in used_pids:
                    continue
                bbox = info.get('bbox')
                if bbox is None:
                    continue
                v = iou(det_bbox, bbox)
                if v > best_iou:
                    best_iou, best_pid = v, pid

            if best_iou >= iou_threshold and best_pid != -1:
                used_pids.add(best_pid)
                votes[best_pid][det_class] += 1

    return dict(votes)


def detect_handler_per_frame(frames, player_tracks, handler_model,
                             conf=0.4, iou_threshold=0.3):
    """
    對每幀跑 handler_detector.pt（multi-class YOLO），找 class==0（ball_handler）。
    與該幀 player_tracks 的 player_id 做 IOU 配對，回 list[player_id or -1]。
    """
    n = len(frames)
    out = [-1] * n
    print(f"  YOLO handler detection ({n} frames)...")

    BATCH = 16
    results = []
    for i in range(0, n, BATCH):
        batch_res = handler_model.predict(frames[i:i+BATCH], conf=conf, verbose=False)
        results.extend(batch_res)

    n_detected = 0
    for f_idx, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        cls  = result.boxes.cls.cpu().numpy().astype(int)
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        # 找 class==0 (ball_handler)，conf 最高的
        handler_idx = [i for i, c in enumerate(cls) if c == 0]
        if not handler_idx:
            continue

        best = max(handler_idx, key=lambda i: confs[i])
        handler_bbox = xyxy[best]

        # 與 SAM2 tracked player 做 IOU 配對
        best_pid, best_iou = -1, 0.0
        for pid, info in player_tracks[f_idx].items():
            bbox = info.get('bbox') # SAM2 偵測的bbox
            if bbox is None:
                continue
            v = iou(handler_bbox, bbox)
            if v > best_iou:
                best_iou, best_pid = v, pid

        if best_iou >= iou_threshold:
            out[f_idx] = best_pid
            n_detected += 1

    print(f"  handler matched in {n_detected}/{n} frames "
          f"({100*n_detected/max(n,1):.1f}%)")
    
    return out


# ── Homography helpers ─────────────────────────────────────────────────────────

def build_homography_from_keypoints(keypoints_obj):
    """從 ultralytics Keypoints 物件建 homography。失敗回 None。"""
    if keypoints_obj is None:
        return None
    try:
        xy = keypoints_obj.xy.tolist()
    except AttributeError:
        return None
    
    if not xy:
        return None
    
    kp_xy = xy[0]
    valid = [(i, kp) for i, kp in enumerate(kp_xy)
             if kp[0] > 0 and kp[1] > 0 and i < len(TARGET_KPS)]
    
    if len(valid) < 4:
        return None
    
    src = np.array([kp for _, kp in valid], dtype=np.float32)
    dst = np.array([TARGET_KPS[i] for i, _ in valid], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    
    return H


def foot_pos_xyxy(xyxy):
    return ((xyxy[0] + xyxy[2]) / 2, xyxy[3])


def transform_pt(H, x, y):
    if H is None:
        return None
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)

    return float(out[0, 0, 0]), float(out[0, 0, 1])


def tactical_to_feet(x_px, y_px):
    return x_px / COURT_W_PX * COURT_W_FT, y_px / COURT_H_PX * COURT_H_FT


def player_court_coords(player_tracks, court_keypoints):
    """list[dict[player_id: (x_ft, y_ft)]]，過濾掉場外與無 homography 的 frame。"""
    n = len(player_tracks)
    out = []
    for f_idx in range(n):
        H = build_homography_from_keypoints(court_keypoints[f_idx])
        frame_out = {}
        if H is not None:
            for pid, info in player_tracks[f_idx].items():
                bbox = info.get('bbox') if isinstance(info, dict) else info
                if bbox is None:
                    continue
                fx, fy = foot_pos_xyxy(bbox)
                pt = transform_pt(H, fx, fy)
                if pt is None:
                    continue
                tx, ty = tactical_to_feet(*pt)
                if 0 <= tx <= COURT_W_FT + 5 and 0 <= ty <= COURT_H_FT + 5:
                    frame_out[pid] = (tx, ty)
        out.append(frame_out)
        
    return out


# ── Ball acquisition 平滑 ──────────────────────────────────────────────────────

def smooth_ball_acquisition(ball_acquisition, window=15, min_total_frames=20):
    """
    對 ball_acquisition 做時間平滑 + 噪音過濾。

    1. 過濾：在整段影片出現少於 min_total_frames 幀的 player 視為誤偵測，全部設為 -1。
    2. 平滑：每幀取 [i-window, i+window] 範圍內的多數決 player 當作 handler。
    """
    n = len(ball_acquisition)
    if n == 0:
        return ball_acquisition

    handler_freq = Counter(p for p in ball_acquisition if p is not None and p != -1)
    valid = {p for p, cnt in handler_freq.items() if cnt >= min_total_frames}
    filtered = [p if p in valid else -1 for p in ball_acquisition]

    smoothed = list(filtered)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1) # 取以 i 為中心、半寬 window 的視窗範圍
        votes = Counter(p for p in filtered[lo:hi] if p != -1)
        if votes:
            smoothed[i] = votes.most_common(1)[0][0]

    n_changed = sum(1 for a, b in zip(ball_acquisition, smoothed) if a != b)
    print(f"  smoothed ball_acquisition: {n_changed}/{n} frames changed, "
          f"valid handlers={sorted(valid)}")
    return smoothed


# ── Sliding window inference ───────────────────────────────────────────────────

def positions_for(pid, court_coords, window):
    """取 pid 在 window 內 SEQ_LEN 幀的位置序列。漏幀用前一個已知值填。"""
    seq = np.zeros((SEQ_LEN, 2), dtype=np.float32)
    last_known = None
    for k, f_idx in enumerate(window):
        if pid in court_coords[f_idx]:
            seq[k] = court_coords[f_idx][pid]
            last_known = seq[k]
        elif last_known is not None:
            seq[k] = last_known
    return seq


def detect_screens(model, device, court_coords, player_assignment, ball_acquisition,
                   threshold=0.70, debug=False, yolo_defender_pids=None):
    """
    對每張 frame 預測掩護。回傳 list[None | dict]。
    debug=True 時，對每個 score >= threshold 的視窗印出 offense/defense 分配與所有候選人分數。
    yolo_defender_pids: set of player_id，YOLO 主要把他們認成 defender 的 id，會從 offense 候選剔除。
    """
    if yolo_defender_pids is None:
        yolo_defender_pids = set()
    num_frames = len(court_coords)
    per_frame = [None] * num_frames

    # pair (0, 1) 的索引：枚舉順序 (i=0,j=1) 是第一對 → index 0
    pair_idx_01 = 0

    roles_arr = np.array([0] + [1] * (N_OFFENSE - 1) + [2] * N_DEFENSE, dtype=np.int64)
    roles_t = torch.tensor(roles_arr).unsqueeze(0).to(device)  # (1, 10)

    for start in range(0, num_frames - SEQ_LEN + 1):
        window = list(range(start, start + SEQ_LEN))

        # Handler：視窗內 ball_aquisition 多數決
        handler_votes = Counter()
        for f_idx in window:
            pid = ball_acquisition[f_idx]
            if pid is not None and pid != -1:
                handler_votes[pid] += 1

        if not handler_votes:
            continue
        handler_id, _ = handler_votes.most_common(1)[0]

        # Handler team（視窗內 majority vote，避免 SAM2 mask drift 早期幀污染）
        handler_team_votes = Counter()
        for f_idx in window:
            assign = player_assignment[f_idx]
            if handler_id in assign:
                handler_team_votes[assign[handler_id]] += 1
        if not handler_team_votes:
            continue
        handler_team = handler_team_votes.most_common(1)[0][0]

        # 蒐集視窗內出現過、有 court coords 的所有球員
        all_pids = set()
        for f_idx in window:
            all_pids.update(court_coords[f_idx].keys())
        all_pids.discard(handler_id)

        # 分隊（同樣用 majority vote，避免一兩幀漂移就把球員分錯邊）
        offense_pids, defense_pids = [], []
        for pid in all_pids:
            team_votes = Counter()
            for f_idx in window:
                if pid in player_assignment[f_idx]:
                    team_votes[player_assignment[f_idx][pid]] += 1
                    
            if not team_votes:
                continue

            team = team_votes.most_common(1)[0][0]

            if team == handler_team:
                # YOLO 主要認為這個 player 是 defender → 強制改判 defense
                if pid in yolo_defender_pids:
                    defense_pids.append(pid)
                else:
                    offense_pids.append(pid)
            else:
                defense_pids.append(pid)

        if not offense_pids or handler_id not in court_coords[window[-1]]:
            continue

        handler_pos = positions_for(handler_id, court_coords, window)

        # 取最靠近 handler 的 5 個防守球員（不夠就補 0）
        defense_sorted = sorted(
            defense_pids,
            key=lambda p: np.linalg.norm(positions_for(p, court_coords, window)[-1] - handler_pos[-1])
        )[:N_DEFENSE]

        defense_positions = np.zeros((N_DEFENSE, SEQ_LEN, 2), dtype=np.float32)
        for i, p in enumerate(defense_sorted):
            defense_positions[i] = positions_for(p, court_coords, window)

        # 對每個進攻候選人 c：把 c 放到 slot 1，其他進攻補 slot 2~4
        best_score = -1.0
        best_screener = None
        candidate_scores = {}  # for debug: {pid: score}

        offense_position_cache = {p: positions_for(p, court_coords, window) for p in offense_pids}

        # 遍歷所有可能的進攻球員（offense_pids），並將他們模擬為「潛在掩護者」
        for candidate in offense_pids:
            others = [p for p in offense_pids if p != candidate]
            # slot 2~4：按與 handler 的距離排序，最近的排前面
            others_sorted = sorted(
                others,
                key=lambda p: np.linalg.norm(offense_position_cache[p][-1] - handler_pos[-1])
            )[:N_OFFENSE - 2]

            traj = np.zeros((N_PLAYERS, SEQ_LEN, 2), dtype=np.float32)
            traj[0] = handler_pos
            traj[1] = offense_position_cache[candidate]
            for i, p in enumerate(others_sorted):
                traj[2 + i] = offense_position_cache[p]
            for i in range(N_DEFENSE):
                traj[N_OFFENSE + i] = defense_positions[i]

            traj_t = torch.tensor(traj).unsqueeze(0).to(device)  # (1, 10, T, 2)

            with torch.no_grad():
                logits = model(traj_t, roles_t)  # (1, 20)

            prob = torch.sigmoid(logits)[0, pair_idx_01].item()
            candidate_scores[candidate] = prob

            if prob > best_score:
                best_score = prob
                best_screener = candidate

        if debug and best_score >= threshold and best_screener is not None:
            scores_str = ', '.join(f'{p}={s:.2f}' for p, s in
                                   sorted(candidate_scores.items(), key=lambda x: -x[1]))
            print(f"  [DEBUG] window start={start:4d}  "
                  f"handler={handler_id}(team={handler_team})  "
                  f"offense={offense_pids}  defense={defense_sorted}  "
                  f"→ scores: {scores_str}  "
                  f"→ PICK screener={best_screener} ({best_score:.3f})")

        if best_score >= threshold and best_screener is not None:
            # 掩護結果套用到視窗的中間幾幀
            mid_start = start + SEQ_LEN // 3
            mid_end = start + 2 * SEQ_LEN // 3
            for f_idx in range(mid_start, mid_end + 1):
                # 已有的話保留分數較高的
                if per_frame[f_idx] is None or per_frame[f_idx]['score'] < best_score:
                    per_frame[f_idx] = {
                        'handler_id': handler_id,
                        'screener_id': best_screener,
                        'score': float(best_score),
                    }

    return per_frame


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_predictions(frames, player_tracks, per_frame_screens):
    out = []
    for f_idx, frame in enumerate(frames):
        img = frame.copy()
        tracks = player_tracks[f_idx]
        ev = per_frame_screens[f_idx]

        handler_id = ev['handler_id'] if ev is not None else None
        screener_id = ev['screener_id'] if ev is not None else None

        # 1. 所有球員：細灰框 + ID（drifted 的不畫）
        for pid, info in tracks.items():
            bbox = info.get('bbox')
            if bbox is None:
                continue
            if pid == handler_id or pid == screener_id:
                continue  # handler / screener 之後用粗框畫
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(img, f"#{pid}", (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # 2. Handler：粗綠框（drifted 那幀不畫，避免框錯人）
        if handler_id is not None and handler_id in tracks:
            bbox = tracks[handler_id].get('bbox')
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"HANDLER #{handler_id}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 3. Screener：粗紅框 + 信心分數（drifted 那幀不畫，避免框到 defender）
        if screener_id is not None and screener_id in tracks:
            bbox = tracks[screener_id].get('bbox')
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, f"SCREENER #{screener_id} {ev['score']:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.append(img)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--ckpt', required=True,
                        help='Path to finetune_manual_best.pt')
    parser.add_argument('--output_video', default='output_videos/screens_pred.avi')
    parser.add_argument('--threshold', type=float, default=0.70)
    parser.add_argument('--stub_dir', default='stubs')
    parser.add_argument('--no_stub', action='store_true',
                        help='Disable stub caching (recompute everything)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug info per predicted window')
    parser.add_argument('--ba_smooth_window', type=int, default=15,
                        help='Handler smoothing window (frames each side)')
    parser.add_argument('--ba_min_frames', type=int, default=20,
                        help='Min total frames a player must be handler to be considered valid')
    parser.add_argument('--handler_model', default=HANDLER_MODEL_DEFAULT,
                        help='multi-class YOLO 模型路徑（class 0 = ball_handler）')
    parser.add_argument('--player_model', default=PLAYER_MODEL_DEFAULT,
                        help='YOLO 第一幀球員偵測模型（單 class）')
    parser.add_argument('--sam2_ckpt', default=SAM2_CHECKPOINT_DEFAULT)
    parser.add_argument('--sam2_config', default=SAM2_CONFIG_DEFAULT)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Video ──────────────────────────────────────────────────────────────
    print(f"Reading video: {args.video}")
    frames = read_video(args.video)
    if not frames:
        print("ERROR: cannot read video")
        return
    print(f"  {len(frames)} frames")

    use_stub = not args.no_stub
    os.makedirs(args.stub_dir, exist_ok=True)

    # ── Player tracking: YOLO 第一幀 + SAM2 跨幀傳播 ─────────────────────────
    if build_sam2_video_predictor is None:
        print("ERROR: sam2 not installed. pip install git+https://github.com/facebookresearch/sam2.git")
        return

    print(f"Loading YOLO player model: {args.player_model}")
    player_model = YOLO(args.player_model)

    print(f"Loading SAM2 predictor on {device}...")
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_ckpt, device=str(device))

    print("Player tracking via SAM2...")
    player_tracks = run_sam2_tracks(frames, player_model, predictor, device=str(device))
    n_tracked = sum(1 for ft in player_tracks if ft)
    print(f"  SAM2 tracked {n_tracked}/{len(frames)} frames")

    # 用 YOLO 每幀重新偵測，清掉 SAM2 mask drift 的 bbox
    player_tracks = verify_sam2_with_yolo(frames, player_tracks, player_model)

    # ── Handler detection: multi-class YOLO 直接抓 ball_handler ──────────────
    print(f"Loading handler detector: {args.handler_model}")
    handler_model = YOLO(args.handler_model)
    ball_acquisition = detect_handler_per_frame(frames, player_tracks, handler_model)

    # ── YOLO 整段 class 投票（class 0=handler, 1=screener, 2=defender, 3=other）──
    print("YOLO per-player class voting...")
    class_votes = detect_player_class_votes(frames, player_tracks, handler_model)
    yolo_defender_pids = set()
    for pid, counter in class_votes.items():
        if not counter:
            continue
        top_class, _ = counter.most_common(1)[0]
        if top_class == 2:  # defender
            yolo_defender_pids.add(pid)
    print(f"  YOLO 主要認為以下 player_id 是 defender（會從 offense 候選剔除）: "
          f"{sorted(yolo_defender_pids)}")

    # ── Court keypoints ────────────────────────────────────────────────────
    print("Court keypoints...")
    court_detector = CourtKeypointDetector(
        str(PROJECT_ROOT / 'models' / 'court_keypoint_detector.pt')
    )
    court_kps = court_detector.get_court_keypoints(
        frames, read_from_stub=use_stub,
        stub_path=os.path.join(args.stub_dir, 'court_key_points_stub.pkl'),
    )

    # ── Team assignment（吃 SAM2 的 player_tracks）────────────────────────────
    print("Team assignment...")
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        frames, player_tracks, read_from_stub=use_stub,
        stub_path=os.path.join(args.stub_dir, 'player_assignment_stub.pkl'),
    )

    # ── 時間平滑 handler（YOLO 偵測可能還有少數漏抓，用多數決補）─────────────
    ball_acquisition = smooth_ball_acquisition(
        ball_acquisition, window=args.ba_smooth_window, min_total_frames=args.ba_min_frames
    )

    # ── Court coords ───────────────────────────────────────────────────────
    print("Projecting players to court coordinates (feet)...")
    court_coords = player_court_coords(player_tracks, court_kps)
    n_with_coords = sum(1 for fc in court_coords if fc)
    print(f"  {n_with_coords}/{len(frames)} frames have homography")

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading model: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    d_model = ckpt.get('d_model', 64)
    n_heads = ckpt.get('n_heads', 4)
    n_layers = ckpt.get('n_layers', 2)
    model = ScreenNetPairwise(d_model, n_heads, n_layers)
    model.load_state_dict(ckpt['model'])  # 載入完整權重（含 head）
    model = model.to(device).eval()
    print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    # ── Inference ──────────────────────────────────────────────────────────
    print(f"Sliding-window inference (threshold={args.threshold})...")
    per_frame = detect_screens(
        model, device, court_coords, player_assignment, ball_acquisition,
        threshold=args.threshold, debug=args.debug,
        yolo_defender_pids=yolo_defender_pids,
    )

    n_screen_frames = sum(1 for e in per_frame if e is not None)
    print(f"  → {n_screen_frames}/{len(frames)} frames flagged as screen")

    # 印出事件段
    events = []
    cur = None
    for i, e in enumerate(per_frame):
        if e is not None:
            if cur is None or cur['end'] + 1 != i or cur['screener_id'] != e['screener_id']:
                if cur is not None:
                    events.append(cur)
                cur = {'start': i, 'end': i,
                       'handler_id': e['handler_id'],
                       'screener_id': e['screener_id'],
                       'score': e['score']}
            else:
                cur['end'] = i
                cur['score'] = max(cur['score'], e['score'])
    if cur is not None:
        events.append(cur)

    print(f"\nDetected {len(events)} screen events:")
    for ev in events:
        print(f"  Frames {ev['start']:4d}-{ev['end']:4d}: "
              f"handler={ev['handler_id']}, screener={ev['screener_id']}, "
              f"score={ev['score']:.3f}")

    # ── Draw + save ────────────────────────────────────────────────────────
    print(f"\nDrawing predictions...")
    output_frames = draw_predictions(frames, player_tracks, per_frame)

    out_path = args.output_video
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    print(f"Saving → {out_path}")
    save_video(output_frames, out_path)
    print("Done.")


if __name__ == '__main__':
    main()
