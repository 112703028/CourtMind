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

def _filter_player_like_bboxes(boxes, min_height=25, min_aspect=1.0):
    """
    過濾掉「明顯不像球員」的 bbox（如籃球、極小雜訊）。
    放寬條件：min_aspect=1.0 容許跳投/側身姿勢；min_height=25 容許遠景球員。
    球員特徵：高 > min_height pixel + 高/寬 >= min_aspect（不是橫躺/球形）
    """
    if len(boxes) == 0:
        return boxes
    filtered = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        if h < min_height:
            continue              # 太小 = 雜訊 / 遠景觀眾
        if h / w < min_aspect:
            continue              # 橫躺 / 完全球形（h/w < 1）
        filtered.append(box)
    return np.array(filtered) if filtered else np.empty((0, 4), dtype=float)


def _pick_init_frame(frames, player_model, min_players=5, max_search=60, conf=0.4):
    """
    在前 max_search 幀裡找一個偵測球員最多的當 SAM2 init frame。
    避免拿第 0 幀（可能是黑屏 / scene 切換 / 少數人入鏡）導致 SAM2 追錯。
    過濾「球員形狀」的 bbox，避免籃球 / 雜訊被當 player。
    """
    best_idx, best_count, best_boxes = 0, 0, None
    n_search = min(max_search, len(frames))
    for i in range(n_search):
        r = player_model.predict(frames[i], conf=conf, verbose=False)[0]
        raw_boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        # 過濾掉籃球/雜訊（短/方的 bbox 不是球員）
        boxes = _filter_player_like_bboxes(raw_boxes)
        n = len(boxes)
        if n > best_count:
            best_idx, best_count, best_boxes = i, n, boxes
            if n >= 10:  # 已經 10 人 = 完整 5v5，就用這幀
                break
    if best_count < min_players:
        print(f"  ⚠ 前 {n_search} 幀都偵測不到 >={min_players} 個球員 "
              f"(最佳 frame {best_idx} 有 {best_count} 個)")
    return best_idx, best_boxes


def run_sam2_tracks(frames, player_model, predictor, device='cuda',
                    scan_interval=60, iou_match_threshold=0.5,
                    max_new_per_scan=2, max_total_obj=15):
    """
    多幀 init SAM2 追蹤：
      1. 找球員最多的幀做基礎 init
      2. 第一輪 propagate
      3. 整段影片均勻掃描（每 scan_interval 幀），找出 SAM2 沒追到的新球員
      4. 加新 obj_id 到對應幀，重新 propagate

    回傳 list[dict[player_id: {'bbox': [x1,y1,x2,y2]}]]，與舊格式相容。
    """
    if not frames:
        return []

    init_idx, init_boxes = _pick_init_frame(frames, player_model)
    if init_boxes is None or len(init_boxes) == 0:
        print("  ⚠ SAM2: 找不到能初始化的幀")
        return [{} for _ in frames]

    print(f"  SAM2 init at frame {init_idx} with {len(init_boxes)} player bboxes")

    # SAM2 需要影格目錄
    tmp_dir = Path(tempfile.mkdtemp(prefix='sam2_frames_'))
    try:
        for i, frame in enumerate(frames):
            cv2.imwrite(str(tmp_dir / f'{i:05d}.jpg'), frame)

        inference_state = predictor.init_state(video_path=str(tmp_dir))
        predictor.reset_state(inference_state)
        autocast_dtype = torch.bfloat16 if device == 'cuda' else torch.float32

        # 共用：把 propagate 結果寫到 raw_tracks
        def _absorb(raw_tracks, frame_idx, object_ids, masks):
            for obj_id, mask in zip(object_ids, masks):
                mask_np = (mask[0].cpu().numpy() > 0)
                if mask_np.any() and 0 <= frame_idx < len(raw_tracks):
                    ys, xs = np.where(mask_np)
                    raw_tracks[frame_idx][int(obj_id)] = {
                        'bbox': [float(xs.min()), float(ys.min()),
                                 float(xs.max()), float(ys.max())]
                    }

        def _full_propagate(raw_tracks, extra_prompt_frames=None):
            """
            從所有 prompts 開始 propagate。
            為了讓「中途加的 obj」也能倒推到 frame 0，對每個 prompt frame 都做一次 reverse。
            """
            # Forward：從最早 prompt 往後傳播
            with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
                for f_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
                    _absorb(raw_tracks, f_idx, obj_ids, masks)

            # Reverse：對每個 prompt frame 個別 reverse propagate
            # 確保「frame K 加的 new obj」能倒推回 frame 0 ~ K-1
            prompt_frames_to_reverse = set()
            if init_idx > 0:
                prompt_frames_to_reverse.add(init_idx)
            if extra_prompt_frames:
                prompt_frames_to_reverse.update(f for f in extra_prompt_frames if f > 0)

            with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
                for start_f in sorted(prompt_frames_to_reverse, reverse=True):
                    for f_idx, obj_ids, masks in predictor.propagate_in_video(
                            inference_state, reverse=True, start_frame_idx=start_f):
                        _absorb(raw_tracks, f_idx, obj_ids, masks)

        # === Pass 1：基礎 init + propagate ===
        next_obj_id = 1
        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            for xyxy in init_boxes:
                predictor.add_new_points_or_box(
                    inference_state, frame_idx=init_idx, obj_id=next_obj_id,
                    box=xyxy.astype(np.float32))
                next_obj_id += 1

        raw_tracks = [{} for _ in frames]
        for i, xyxy in enumerate(init_boxes):
            raw_tracks[init_idx][i + 1] = {'bbox': xyxy.tolist()}
        _full_propagate(raw_tracks)   # Pass 1 只有 init_idx 這個 prompt frame

        # === Pass 2：整段均勻掃描，找新球員 ===
        scan_indices = list(range(0, len(frames), scan_interval))
        new_prompts = []   # (frame_idx, bbox)
        n_init = next_obj_id - 1
        for scan_idx in scan_indices:
            if scan_idx == init_idx:
                continue   # init frame 已偵測過
            # YOLO 偵測這幀的所有人
            r = player_model.predict(frames[scan_idx], conf=0.4, verbose=False)[0]
            raw_boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
            yolo_boxes = _filter_player_like_bboxes(raw_boxes)
            if len(yolo_boxes) == 0:
                continue

            # 比對：當前 SAM2 在 scan_idx 的 bboxes
            current_sam2_bboxes = [
                info['bbox'] for info in raw_tracks[scan_idx].values()
                if info.get('bbox') is not None
            ]

            # 對每個 YOLO bbox，找最佳 IOU 配對
            # 兩個條件都嚴：IOU 配對 + 跟既有「new_prompts」也不能重複
            new_at_this_frame = 0
            for ybox in yolo_boxes:
                # 1. 跟「當前 SAM2 在這幀的 bbox」配對（看是不是現有球員）
                best_iou = 0.0
                for sbox in current_sam2_bboxes:
                    v = iou(ybox, sbox)
                    if v > best_iou:
                        best_iou = v
                if best_iou >= iou_match_threshold:
                    continue  # 跟現有球員配上了，跳過

                # 2. 跟「已收集的 new_prompts」也要 dedupe（避免不同 scan 加同個人）
                #    取每個 new_prompt 的 bbox，跟 ybox 比 IOU
                dup_with_new = False
                for prev_f, prev_b in new_prompts:
                    if abs(prev_f - scan_idx) <= scan_interval * 2:
                        # 鄰近時段的 prompt 才需要去重
                        if iou(ybox, prev_b) >= 0.3:
                            dup_with_new = True
                            break
                if dup_with_new:
                    continue

                # 3. 真正的新球員
                new_prompts.append((scan_idx, ybox))
                new_at_this_frame += 1

                # 防爆炸：總數上限 + 單幀上限
                if (n_init + len(new_prompts)) >= max_total_obj:
                    print(f"  ⚠ 達到 max_total_obj={max_total_obj}，停止加新球員")
                    break
                if new_at_this_frame >= max_new_per_scan:
                    break  # 單幀上限

            if (n_init + len(new_prompts)) >= max_total_obj:
                break  # 整個 scan 也跳出

        if not new_prompts:
            print(f"  Pass 2 掃描 {len(scan_indices)} 幀，沒發現新球員")
            predictor.reset_state(inference_state)
            return raw_tracks

        print(f"  Pass 2 掃描 {len(scan_indices)} 幀，發現 {len(new_prompts)} 個新球員")

        # === Pass 3：加新 prompts + 重新 propagate ===
        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            for frame_idx, ybox in new_prompts:
                predictor.add_new_points_or_box(
                    inference_state, frame_idx=frame_idx, obj_id=next_obj_id,
                    box=ybox.astype(np.float32))
                next_obj_id += 1

        # 重新 propagate，這次會包含新 obj_ids
        raw_tracks = [{} for _ in frames]
        # 重填 init prompts 的 bbox
        for i, xyxy in enumerate(init_boxes):
            raw_tracks[init_idx][i + 1] = {'bbox': xyxy.tolist()}
        # 重填新 prompts 的 bbox
        for offset, (f_idx, ybox) in enumerate(new_prompts):
            raw_tracks[f_idx][n_init + 1 + offset] = {'bbox': ybox.tolist()}
        # 對「init_idx + 每個 new prompt frame」都明確 reverse propagate
        # 確保中途加的 obj_id 也能倒推到 frame 0
        new_prompt_frames = [f for f, _ in new_prompts]
        _full_propagate(raw_tracks, extra_prompt_frames=new_prompt_frames)

        print(f"  最終 SAM2 obj_ids: {next_obj_id - 1} 個")
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
                   threshold=0.70, debug=False, class_votes=None,
                   defender_ratio_weight=1.0, fake_handler_weight=1.5):
    """
    對每張 frame 預測掩護。回傳 list[None | dict]。

    Pool 選擇邏輯（取法 3：Siamese + YOLO 加權）：
      - defense_score = defense_confidence (Siamese 說異隊的幀數)
                      + w1 * defender_ratio (handler_detector 說是 defender 的比例)
                      + w2 * fake_handler_ratio (handler_detector 說是 handler 但實際不是 → 貼身防守)
      - defense_pids = top N_DEFENSE by defense_score
      - offense_pids = 剩餘 candidates 中 top N_OFFENSE-1 by offense_confidence

    class_votes: dict[pid: Counter({class_id: count})] from handler_detector 4-class voting.
                 class 0=handler, 1=screener, 2=defender, 3=others
    """
    if class_votes is None:
        class_votes = {}
    num_frames = len(court_coords)
    per_frame = [None] * num_frames

    # pair (0, 1) 的索引：枚舉順序 (i=0,j=1) 是第一對 → index 0
    pair_idx_01 = 0

    roles_arr = np.array([0] + [1] * (N_OFFENSE - 1) + [2] * N_DEFENSE, dtype=np.int64)
    roles_t = torch.tensor(roles_arr).unsqueeze(0).to(device)  # (1, 10)

    # smoothed 過的真 handler 集合 → 用來判斷「假 handler」（貼身防守被誤標）
    smoothed_handlers = set(p for p in ball_acquisition if p is not None and p != -1)
    n_frames = len(player_assignment)

    # 印每個 pid 的 YOLO class 票數分佈（一次性）
    if debug and class_votes:
        print(f"  [DEBUG] Per-pid YOLO class votes "
              f"(h=handler, s=screener, d=defender, o=others):")
        for pid in sorted(class_votes.keys()):
            c = class_votes[pid]
            total = sum(c.values())
            if total == 0:
                continue
            tag = " (smoothed handler)" if pid in smoothed_handlers else ""
            print(f"    pid={pid}: h={c.get(0, 0)}, s={c.get(1, 0)}, "
                  f"d={c.get(2, 0)}, o={c.get(3, 0)}, total={total}{tag}")

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
        candidates = set()
        for f_idx in window:
            candidates.update(court_coords[f_idx].keys())
        candidates.discard(handler_id)

        if not candidates or handler_id not in court_coords[window[-1]]:
            continue

        # ── 兩個 cross-frame confidence ──
        # offense_confidence[pid] = 跨幀被 Siamese 分到 handler_team 的次數
        # defense_confidence[pid] = 跨幀被 Siamese 分到「非 handler_team」的次數
        offense_confidence = {}
        defense_confidence = {}
        for pid in candidates:
            same = sum(1 for f in player_assignment if f.get(pid) == handler_team)
            diff = sum(1 for f in player_assignment
                       if f.get(pid) is not None and f.get(pid) != handler_team)
            offense_confidence[pid] = same
            defense_confidence[pid] = diff

        # ── 加上 YOLO defender 訊號算 defense_score ──
        # 兩個 bonus：
        #  1. defender_ratio: handler_detector 直接投他為 defender 的比例
        #  2. fake_handler_ratio: 被投為 handler 但實際不是 smoothed handler → 貼身防守
        defense_score = {}
        yolo_dbg = {}   # for debug: (defender_bonus, fake_handler_bonus)
        for pid in candidates:
            counter = class_votes.get(pid, Counter())
            total = sum(counter.values())
            if total > 0:
                defender_ratio = counter.get(2, 0) / total
                handler_ratio = counter.get(0, 0) / total
            else:
                defender_ratio = 0.0
                handler_ratio = 0.0
            fake_handler_ratio = handler_ratio if pid not in smoothed_handlers else 0.0

            defender_bonus = defender_ratio_weight * defender_ratio * n_frames
            fake_handler_bonus = fake_handler_weight * fake_handler_ratio * n_frames

            defense_score[pid] = (
                defense_confidence[pid] + defender_bonus + fake_handler_bonus
            )
            yolo_dbg[pid] = (defender_bonus, fake_handler_bonus)

        # ── 硬性取 top N_DEFENSE 個 defense ──
        defense_pids = sorted(candidates, key=lambda p: -defense_score[p])[:N_DEFENSE]
        defense_set = set(defense_pids)

        # ── offense 從剩下的取 top N_OFFENSE-1 by offense_confidence ──
        offense_pool = [p for p in candidates if p not in defense_set]
        offense_pids = sorted(
            offense_pool, key=lambda p: -offense_confidence[p]
        )[:N_OFFENSE - 1]

        if not offense_pids:
            continue

        if debug:
            def_str = ', '.join(
                f'{p}(D={defense_score[p]:.0f}=siam{defense_confidence[p]}'
                f'+d{yolo_dbg[p][0]:.0f}+fh{yolo_dbg[p][1]:.0f})'
                for p in defense_pids
            )
            off_str = ', '.join(f'{p}(O={offense_confidence[p]})' for p in offense_pids)
            print(f"  [DEBUG] window={start:4d}  handler={handler_id}(team={handler_team})"
                  f"\n    offense: {off_str}"
                  f"\n    defense: {def_str}")

        handler_pos = positions_for(handler_id, court_coords, window)

        # defense 已是 top N_DEFENSE，slot 位置按離 handler 距離排序
        defense_sorted = sorted(
            defense_pids,
            key=lambda p: np.linalg.norm(positions_for(p, court_coords, window)[-1] - handler_pos[-1])
        )

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

        # 多 screener 模式：所有 score >= threshold 的 candidate 都當 screener
        # 不再只挑 best，把所有夠 confidence 的人都收下
        screener_picks = [(pid, s) for pid, s in candidate_scores.items() if s >= threshold]
        screener_picks.sort(key=lambda x: -x[1])  # 高分在前

        if debug and screener_picks:
            scores_str = ', '.join(f'{p}={s:.2f}' for p, s in
                                   sorted(candidate_scores.items(), key=lambda x: -x[1]))
            picks_str = ', '.join(f'{p}({s:.2f})' for p, s in screener_picks)
            print(f"    → scores: {scores_str}  → PICKS: {picks_str}")

        if screener_picks:
            # 掩護結果套用到視窗的中間幾幀
            mid_start = start + SEQ_LEN // 3
            mid_end = start + 2 * SEQ_LEN // 3
            screener_ids = [pid for pid, _ in screener_picks]
            scores_list = [float(s) for _, s in screener_picks]
            max_score = scores_list[0]
            for f_idx in range(mid_start, mid_end + 1):
                # 已有的話保留 max_score 較高的
                if per_frame[f_idx] is None or per_frame[f_idx].get('max_score', 0) < max_score:
                    per_frame[f_idx] = {
                        'handler_id': handler_id,
                        'screener_ids': screener_ids,
                        'scores': scores_list,
                        'max_score': max_score,
                        # 為了相容舊版 draw / event 計算，留個 screener_id = 第一個
                        'screener_id': screener_ids[0],
                        'score': max_score,
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
        # 支援多 screener：optional 'screener_ids' list（v2），fallback 到 'screener_id'（v1）
        if ev is not None:
            screener_ids = ev.get('screener_ids') or (
                [ev['screener_id']] if ev.get('screener_id') is not None else []
            )
            scores_list = ev.get('scores') or [ev.get('score', 0.0)] * len(screener_ids)
        else:
            screener_ids = []
            scores_list = []
        screener_set = set(screener_ids)

        # 1. 所有球員：細灰框 + ID（drifted 的不畫）
        for pid, info in tracks.items():
            bbox = info.get('bbox')
            if bbox is None:
                continue
            if pid == handler_id or pid in screener_set:
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

        # 3. Screener(s)：粗紅框 + 信心分數，多 screener 都畫
        for sid, score in zip(screener_ids, scores_list):
            if sid not in tracks:
                continue
            bbox = tracks[sid].get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, f"SCREENER #{sid} {score:.2f}",
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
    parser.add_argument('--siamese_team_model', default=str(PROJECT_ROOT / 'models' / 'team_siamese.pt'),
                        help='訓練好的 Siamese team classifier，空字串 = 用舊 sports.TeamClassifier')
    parser.add_argument('--siamese_threshold', type=float, default=0.5,
                        help='Siamese 同隊判斷閾值')
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

    # 1. 真球員：整段「至少有 1 票」被 handler_detector 認為是任一 class
    min_votes_for_real_player = 1
    yolo_real_player_pids = {
        pid for pid, counter in class_votes.items()
        if sum(counter.values()) >= min_votes_for_real_player
    }

    # 2. 整段所有 SAM2 追蹤的 pid（含可能是球 / 雜訊的）
    all_tracked_pids = set()
    for ft in player_tracks:
        all_tracked_pids.update(ft.keys())

    # 3. 計算每個 pid 平均 bbox 大小與形狀（用來判斷是不是球）
    #    球：通常 h/w ≈ 1.0、面積小（< 5000 px²）
    #    真球員：h/w > 1.5、面積大（> 5000 px²）
    pid_avg_shape = {}    # pid -> (avg_h, avg_w, avg_aspect, avg_area)
    pid_frame_count = {}  # pid -> 出現幀數
    for ft in player_tracks:
        for pid, info in ft.items():
            bbox = info.get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            if pid not in pid_avg_shape:
                pid_avg_shape[pid] = [0.0, 0.0, 0.0, 0.0]
                pid_frame_count[pid] = 0
            pid_avg_shape[pid][0] += h
            pid_avg_shape[pid][1] += w
            pid_avg_shape[pid][2] += h / w
            pid_avg_shape[pid][3] += h * w
            pid_frame_count[pid] += 1
    for pid in pid_avg_shape:
        n = pid_frame_count[pid]
        if n > 0:
            pid_avg_shape[pid] = [v / n for v in pid_avg_shape[pid]]

    # 4. 非球員 = (0 YOLO 票) AND (平均 bbox 像球：細小 或 太方)
    #    真球員即使 0 票（罕見遠景球員）也會保留，避免誤殺
    suspected_non_players = all_tracked_pids - yolo_real_player_pids
    non_player_pids = set()
    for pid in suspected_non_players:
        if pid not in pid_avg_shape:
            non_player_pids.add(pid)   # 沒任何 bbox → 應該清
            continue
        avg_h, avg_w, avg_aspect, avg_area = pid_avg_shape[pid]
        is_ball_like = (avg_aspect < 1.5 or avg_area < 5000)
        if is_ball_like:
            non_player_pids.add(pid)

    # 4. 統計每個 pid 的主要 class（僅診斷用，實際 offense/defense pool 選擇改到
    #    detect_screens 裡用 defense_score = Siamese conf + defender_ratio + fake_handler_ratio）
    yolo_top_class = {}
    for pid, counter in class_votes.items():
        if not counter:
            continue
        top_class, _ = counter.most_common(1)[0]
        yolo_top_class[pid] = top_class

    print(f"  真球員 (>=1 YOLO 票): {sorted(yolo_real_player_pids)}")
    print(f"  非球員 (0 票 + 像球: 整段清掉): {sorted(non_player_pids)}")
    # 印每個被清的 pid 的票數 + 形狀，方便 debug
    for pid in sorted(non_player_pids):
        votes = sum(class_votes.get(pid, {}).values())
        if pid in pid_avg_shape:
            avg_h, avg_w, avg_aspect, avg_area = pid_avg_shape[pid]
            print(f"    pid={pid}: votes={votes}, "
                  f"avg_h={avg_h:.0f}, avg_w={avg_w:.0f}, "
                  f"h/w={avg_aspect:.2f}, area={avg_area:.0f}")
        else:
            print(f"    pid={pid}: votes={votes}, no bbox")

    # 印「0 票但保留」的 pid，提示這可能是 handler_detector 漏抓的真球員
    kept_zero_vote = (all_tracked_pids - yolo_real_player_pids) - non_player_pids
    if kept_zero_vote:
        print(f"  保留的 0 票 pid (像真球員，免於清掉): {sorted(kept_zero_vote)}")
        for pid in sorted(kept_zero_vote):
            avg_h, avg_w, avg_aspect, avg_area = pid_avg_shape[pid]
            print(f"    pid={pid}: avg_h={avg_h:.0f}, avg_w={avg_w:.0f}, "
                  f"h/w={avg_aspect:.2f}, area={avg_area:.0f}")
    _CLASS_NAMES = {0: 'handler', 1: 'screener', 2: 'defender', 3: 'others'}
    top_by_class = {c: [] for c in _CLASS_NAMES}
    for pid, tc in yolo_top_class.items():
        top_by_class.setdefault(tc, []).append(pid)
    for c, name in _CLASS_NAMES.items():
        print(f"  YOLO top-class = {name}: {sorted(top_by_class.get(c, []))}")

    # 5. 把非球員的 bbox 整段清成 None → 下游(team_assigner/court_coords/draw)自動跳過
    n_cleared = 0
    for ft in player_tracks:
        for pid in non_player_pids:
            if pid in ft and ft[pid].get('bbox') is not None:
                ft[pid]['bbox'] = None
                ft[pid]['drifted'] = True
                n_cleared += 1
    print(f"  清掉 {n_cleared} 個非球員的 player-frame bbox")

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
    if args.siamese_team_model and os.path.exists(args.siamese_team_model):
        # 用訓練好的 Siamese 模型（pairwise 比較）
        from nba_data_pipeline.team_classifier.inference import TeamAssignerSiamese
        team_assigner = TeamAssignerSiamese(
            ckpt_path=args.siamese_team_model,
            device=str(device),
            threshold=args.siamese_threshold,
            debug=args.debug,
        )
        player_assignment = team_assigner.get_player_teams_across_frames(
            frames, player_tracks, ball_acquisition,
            read_from_stub=use_stub,
            stub_path=os.path.join(args.stub_dir, 'player_assignment_stub.pkl'),
        )
    else:
        # 舊版 sports.TeamClassifier（K-means + SigLIP）
        team_assigner = TeamAssigner(fit_stride=10)
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
        class_votes=class_votes,
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
