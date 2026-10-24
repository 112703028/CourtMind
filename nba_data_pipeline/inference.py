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
import time
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
from nba_data_pipeline.model import ScreenNetPairwise, N_OFFENSE, N_DEFENSE, N_PLAYERS, to_handler_relative

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
        _t = time.time()
        for i, frame in enumerate(frames):
            cv2.imwrite(str(tmp_dir / f'{i:05d}.jpg'), frame)
        print(f"  [TIMING] write {len(frames)} jpg frames: {time.time()-_t:.1f}s")

        _t = time.time()
        inference_state = predictor.init_state(video_path=str(tmp_dir))
        predictor.reset_state(inference_state)
        print(f"  [TIMING] SAM2 init_state: {time.time()-_t:.1f}s")
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
            _t = time.time()
            with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
                for f_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
                    _absorb(raw_tracks, f_idx, obj_ids, masks)
            print(f"  [TIMING]   forward propagate: {time.time()-_t:.1f}s")

            # Reverse：對每個 prompt frame 個別 reverse propagate
            # 確保「frame K 加的 new obj」能倒推回 frame 0 ~ K-1
            prompt_frames_to_reverse = set()
            if init_idx > 0:
                prompt_frames_to_reverse.add(init_idx)
            if extra_prompt_frames:
                prompt_frames_to_reverse.update(f for f in extra_prompt_frames if f > 0)

            with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
                for start_f in sorted(prompt_frames_to_reverse, reverse=True):
                    _t = time.time()
                    for f_idx, obj_ids, masks in predictor.propagate_in_video(
                            inference_state, reverse=True, start_frame_idx=start_f):
                        _absorb(raw_tracks, f_idx, obj_ids, masks)
                    print(f"  [TIMING]   reverse propagate from frame {start_f}: "
                          f"{time.time()-_t:.1f}s")
            if not prompt_frames_to_reverse:
                print(f"  [TIMING]   reverse propagate: (0 趟)")

        # === Pass 1：基礎 init + propagate ===
        _t_pass1 = time.time()
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
        print(f"  [TIMING] Pass 1 (init + propagate): {time.time()-_t_pass1:.1f}s")

        # === Pass 2：整段均勻掃描，找新球員 ===
        _t_pass2 = time.time()
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
            print(f"  [TIMING] Pass 2 (YOLO scan, 無新球員): {time.time()-_t_pass2:.1f}s")
            predictor.reset_state(inference_state)
            return raw_tracks

        print(f"  Pass 2 掃描 {len(scan_indices)} 幀，發現 {len(new_prompts)} 個新球員")
        print(f"  [TIMING] Pass 2 (YOLO scan): {time.time()-_t_pass2:.1f}s")

        # === Pass 3：加新 prompts + 重新 propagate ===
        _t_pass3 = time.time()
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
        print(f"  [TIMING] Pass 3 (re-propagate, {len(new_prompts)} 新球員): "
              f"{time.time()-_t_pass3:.1f}s")

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
                             conf=0.4, iou_threshold=0.3, debug=False):
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

def smooth_ball_acquisition(ball_acquisition, min_run=4, fill_gap=8):
    """
    對 ball_acquisition 做「連續段」平滑，保留傳球/換手。

    舊版用「整段總幀數 < min_total_frames 就清掉」+「對稱窗口多數決」，會把傳球
    抹平——持球最久的人票數輾壓、短暫接球者被整段清掉，結果整段只剩 1 個 handler。
    新版改成連續段邏輯，區分「單幀抖動」vs「真實傳球」：

      1. 壓成連續持球段（run），略過 -1/None 的空隙。
      2. 清掉「短且孤立」的段（run 長度 < min_run = 單幀誤判抖動）→ 設 -1。
      3. 保留夠長的段（>= min_run），不管該 pid 整段總共持球多久 → 傳球者接球
         持球一小段也留得住。
      4. 填補同一持球者段內的 -1 空洞（handler_detector 只 ~78% 幀抓到，中間會漏）：
         相鄰兩個「同 pid」的段之間、間隔 <= fill_gap 的 -1 一律填成該 pid；
         跨不同 pid 的空隙不填（可能是傳球途中/無人持球）。

    參數：
      min_run  : 連續段最短保留幀數（< 此視為噪音抖動）
      fill_gap : 同持球者兩段之間，最多補幾幀 -1 空洞
    """
    n = len(ball_acquisition)
    if n == 0:
        return ball_acquisition

    # 1. 壓成連續段 [(pid, start, end), ...]，把 -1/None 當空隙略過
    runs = []
    for f, p in enumerate(ball_acquisition):
        if p is None or p == -1:
            continue
        if runs and runs[-1][0] == p and f - runs[-1][2] == 1:
            runs[-1] = (p, runs[-1][1], f)   # 延續同 pid 的連續段
        else:
            runs.append((p, f, f))

    # 2. 只保留夠長的段（>= min_run），清掉短抖動
    kept = [(p, s, e) for (p, s, e) in runs if (e - s + 1) >= min_run]

    smoothed = [-1] * n
    for p, s, e in kept:
        for f in range(s, e + 1):
            smoothed[f] = p

    # 4. 填補同一持球者相鄰段之間的短空洞
    for i in range(len(kept) - 1):
        p0, _, e0 = kept[i]
        p1, s1, _ = kept[i + 1]
        if p0 == p1 and (s1 - e0 - 1) <= fill_gap:
            for f in range(e0 + 1, s1):
                smoothed[f] = p0

    valid = sorted(set(p for p, _, _ in kept))
    n_changed = sum(1 for a, b in zip(ball_acquisition, smoothed) if a != b)
    print(f"  smoothed ball_acquisition: {n_changed}/{n} frames changed, "
          f"valid handlers={valid}  (min_run={min_run}, {len(kept)} runs)")
    return smoothed


# ── Sliding window inference ───────────────────────────────────────────────────

class ProbAverage(torch.nn.Module):
    """多個 ScreenNet 的機率平均（不同訓練種子）。輸出轉回 logit，呼叫端照常取 sigmoid 就是平均機率。"""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, trajectories, roles):
        p = torch.stack([torch.sigmoid(m(trajectories, roles)) for m in self.models]).mean(0)
        return torch.logit(p.clamp(1e-6, 1 - 1e-6))


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
                   threshold=0.70, debug=False, yolo_defender_pids=None,
                   score_log=None):
    """
    對每張 frame 預測掩護。回傳 list[None | dict]。
    debug=True 時，對每個 score >= threshold 的視窗印出 offense/defense 分配與所有候選人分數。
    yolo_defender_pids: set of player_id，YOLO 主要把他們認成 defender 的 id，會從 offense 候選剔除。
    score_log: 給一個 list 就會 append 每個視窗「未經門檻」的全部候選分數
               {start, handler_id, scores:{pid:prob}}，用來畫 ScreenNet 分數曲線
               （診斷連續發射：輸出是平台還是尖峰）。
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

        # 強制 offense 限制為 N_OFFENSE-1=4 個候選（加 handler 自己 = 5）
        # 用「跨幀被分到 handler_team 的次數」當信心度排序，避免 team_assigner
        # 短暫誤分 defender 進 offense 污染候選池
        offense_confidence = {}
        for pid in offense_pids:
            same_team_count = sum(
                1 for f_assign in player_assignment
                if f_assign.get(pid) == handler_team
            )
            offense_confidence[pid] = same_team_count

        # 留信心度最高的 N_OFFENSE-1 個
        n_offense_keep = N_OFFENSE - 1
        offense_pids_full = list(offense_pids)   # 保留完整列表用於 debug
        offense_pids = sorted(
            offense_pids,
            key=lambda p: -offense_confidence[p],   # 高 → 低
        )[:n_offense_keep]

        if debug and len(offense_pids_full) > n_offense_keep:
            kept = set(offense_pids)
            dropped = [(p, offense_confidence[p]) for p in offense_pids_full if p not in kept]
            kept_with_conf = [(p, offense_confidence[p]) for p in offense_pids]
            kept_str = ', '.join(f'{p}({c})' for p, c in kept_with_conf)
            drop_str = ', '.join(f'{p}({c})' for p, c in dropped)
            print(f"  [DEBUG] offense conf filter (window={start}): "
                  f"keep [{kept_str}], drop [{drop_str}]")

        # 對每個進攻候選人 c：把 c 放到 slot 1，其他進攻補 slot 2~4
        best_score = -1.0
        best_screener = None
        candidate_scores = {}  # for debug: {pid: score}

        offense_position_cache = {p: positions_for(p, court_coords, window) for p in offense_pids}

        # 把每個 candidate 的 traj 疊成一個 batch，一次 forward（取代逐一 forward，
        # 省下 (candidate 數量 - 1) 次 kernel launch；數學完全相同）。
        traj_batch = np.zeros((len(offense_pids), N_PLAYERS, SEQ_LEN, 2),
                              dtype=np.float32)
        for b, candidate in enumerate(offense_pids):
            others = [p for p in offense_pids if p != candidate]
            # slot 2~4：按與 handler 的距離排序，最近的排前面
            others_sorted = sorted(
                others,
                key=lambda p: np.linalg.norm(offense_position_cache[p][-1] - handler_pos[-1])
            )[:N_OFFENSE - 2]

            traj_batch[b, 0] = handler_pos
            traj_batch[b, 1] = offense_position_cache[candidate]
            for i, p in enumerate(others_sorted):
                traj_batch[b, 2 + i] = offense_position_cache[p]
            for i in range(N_DEFENSE):
                traj_batch[b, N_OFFENSE + i] = defense_positions[i]

        traj_t = torch.from_numpy(traj_batch).to(device)          # (C, 10, T, 2)
        roles_batch = roles_t.expand(len(offense_pids), -1)       # (C, 10)

        with torch.no_grad():
            logits = model(traj_t, roles_batch)                   # (C, 20)
        probs = torch.sigmoid(logits)[:, pair_idx_01]             # (C,)

        for b, candidate in enumerate(offense_pids):
            prob = probs[b].item()
            candidate_scores[candidate] = prob
            if prob > best_score:
                best_score = prob
                best_screener = candidate

        if score_log is not None:
            score_log.append({'start': start, 'handler_id': handler_id,
                              'scores': dict(candidate_scores)})

        # 多 screener 模式：所有 score >= threshold 的 candidate 都當 screener
        # 不再只挑 best，把所有夠 confidence 的人都收下
        screener_picks = [(pid, s) for pid, s in candidate_scores.items() if s >= threshold]
        screener_picks.sort(key=lambda x: -x[1])  # 高分在前

        if debug and screener_picks:
            scores_str = ', '.join(f'{p}={s:.2f}' for p, s in
                                   sorted(candidate_scores.items(), key=lambda x: -x[1]))
            picks_str = ', '.join(f'{p}({s:.2f})' for p, s in screener_picks)
            print(f"  [DEBUG] window start={start:4d}  "
                  f"handler={handler_id}(team={handler_team})  "
                  f"offense={offense_pids}  defense={defense_sorted}  "
                  f"→ scores: {scores_str}  "
                  f"→ PICKS: {picks_str}")

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

def detect_screen_peaks(model, device, court_coords, player_assignment, ball_acquisition,
                        fps, anchor=3, threshold=0.9, nms=3, dense=False,
                        yolo_defender_pids=None):
    """
    事件定位版擋拆偵測（配 convert_contact.py 訓練的模型，例 finetune_contact_v2_pw5_best.pt）。

    跟訓練 / eval_contact.py 同一套做法：
      1. 原片抽成 5fps（訓練影格就是 5fps，10 張窗 = 2 秒）
      2. 每個窗：每個進攻候選人輪流放 slot 1，其他進攻按離 handler 距離排，
         防守取離 handler 最近的 5 個 → pair(handler, slot1) 分數記在窗口第 anchor 格
      3. 每組 (handler, 候選人) 的分數曲線找尖峰（>= threshold、彼此間隔 >= nms 張）
         → 每個尖峰 = 一次擋拆，只有一個時間點（撞上那一刻）
    dense=True：從每個起始偏移都抽一次（30fps 時 6 次），曲線密 6 倍、時間更精細，但慢 6 倍。

    回 (events, curves)
      events：[{frame, handler_id, screener_id, score}]（frame = 原片幀號）
      curves：{(handler, 候選人): {原片幀號: 分數}}
    """
    yolo_defender_pids = yolo_defender_pids or set()
    n = len(court_coords)
    step = fps / 5.0
    roles = torch.tensor([0] + [1] * (N_OFFENSE - 1) + [2] * N_DEFENSE,
                         dtype=torch.int64).unsqueeze(0).to(device)

    # 分隊用「整支影片的多數決」（跟訓練資料 convert_contact 一致），不用 2 秒窗內的多數決：
    # 分隊模型偶爾會短暫把防守者判成跟持球者同隊（v127 f84 的 #5/#14），
    # 窗內多數決會讓他們被當成掩護者候選 → 假尖峰。
    clip_team = {}
    for fa in player_assignment:
        for p, t in fa.items():
            clip_team.setdefault(p, Counter())[t] += 1
    clip_team = {p: c.most_common(1)[0][0] for p, c in clip_team.items()}
    curves = {}
    offsets = range(max(1, int(round(step)))) if dense else [0]

    for off0 in offsets:
        sampled = []
        k = 0
        while int(round(off0 + k * step)) < n:
            sampled.append(int(round(off0 + k * step)))
            k += 1
        for s in range(0, len(sampled) - SEQ_LEN + 1):
            window = sampled[s:s + SEQ_LEN]

            # handler：窗內多數決
            hv = Counter(ball_acquisition[f] for f in window
                         if ball_acquisition[f] is not None and ball_acquisition[f] != -1)
            if not hv:
                continue
            H = hv.most_common(1)[0][0]

            # 窗內出現次數；隊伍用整支影片的多數決，handler 那隊 = 進攻
            cnt = Counter()
            for f in window:
                cnt.update(court_coords[f].keys())
            if H not in clip_team or cnt.get(H, 0) == 0:
                continue
            h_team = clip_team[H]
            present = {p for p, c in cnt.items() if c >= SEQ_LEN // 2} | {H}
            off_p, def_p = [H], []
            for p in present - {H}:
                if p not in clip_team:
                    continue
                if clip_team[p] == h_team and p not in yolo_defender_pids:
                    off_p.append(p)
                else:
                    def_p.append(p)
            if len(off_p) < 2:
                continue

            hpos = next((court_coords[f][H] for f in reversed(window) if H in court_coords[f]), None)

            def dist_to_h(p):
                pos = next((court_coords[f][p] for f in reversed(window) if p in court_coords[f]), None)
                if hpos is None:
                    return 0.0
                return float('inf') if pos is None else float(np.hypot(pos[0] - hpos[0], pos[1] - hpos[1]))

            order_def = (sorted(def_p, key=dist_to_h) + [None] * N_DEFENSE)[:N_DEFENSE]
            def_pos = np.zeros((N_DEFENSE, SEQ_LEN, 2), dtype=np.float32)
            for i, p in enumerate(order_def):
                if p is not None:
                    def_pos[i] = positions_for(p, court_coords, window)
            # 防守者幾乎沒有座標（homography 失敗）的窗跳過（訓練資料也濾掉了）
            if int((np.abs(def_pos).sum(axis=(1, 2)) > 0).sum()) < 2:
                continue

            cache = {p: positions_for(p, court_coords, window) for p in off_p}
            cands = [p for p in off_p if p != H]
            batch = np.zeros((len(cands), N_PLAYERS, SEQ_LEN, 2), dtype=np.float32)
            for b, cand in enumerate(cands):
                rest = sorted((p for p in off_p if p not in (H, cand)), key=dist_to_h)
                order_off = ([H, cand] + rest + [None] * N_OFFENSE)[:N_OFFENSE]
                for i, p in enumerate(order_off):
                    if p is not None:
                        batch[b, i] = cache[p]
                batch[b, N_OFFENSE:] = def_pos
            if getattr(model, 'rel_coords', False):       # finetune --rel_coords 訓練的模型
                batch = to_handler_relative(batch)
            with torch.no_grad():
                logits = model(torch.from_numpy(batch).to(device), roles.expand(len(cands), -1))
            probs = torch.sigmoid(logits)[:, 0].cpu().numpy()     # pair(0,1) = 第 0 號
            af = window[anchor]
            for cand, pr in zip(cands, probs):
                cv = curves.setdefault((H, cand), {})
                cv[af] = max(cv.get(af, 0.0), float(pr))

    # 找尖峰：>= threshold 的局部最大，同一組 (handler, 候選人) 彼此間隔 >= nms 張（換算原片幀）
    min_gap = nms * step
    events = []
    for (H, cand), cv in curves.items():
        pts = sorted(cv.items())
        cands = []
        for i, (f, sc) in enumerate(pts):
            if sc < threshold:
                continue
            left = pts[i - 1][1] if i > 0 else -1.0
            right = pts[i + 1][1] if i < len(pts) - 1 else -1.0
            if sc >= left and sc >= right:
                cands.append((sc, f))
        kept = []
        for sc, f in sorted(cands, reverse=True):
            if all(abs(f - g) >= min_gap for _, g in kept):
                kept.append((sc, f))
        for sc, f in kept:
            events.append({'frame': int(f), 'handler_id': H, 'screener_id': cand, 'score': sc})
    events.sort(key=lambda e: e['frame'])
    return events, curves


def draw_predictions(frames, player_tracks, per_frame_screens):
    out = []
    n_total = len(frames)
    for f_idx, frame in enumerate(frames):
        img = frame.copy()
        tracks = player_tracks[f_idx]
        ev = per_frame_screens[f_idx]

        # 左上角幀號（標 GT 用，直接對 pkl 幀，不用換算秒數）：黑底白字好讀
        label = f"frame {f_idx}/{n_total - 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (6, 6), (6 + tw + 8, 6 + th + 12), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 6 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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


# ── fps 偵測 ─────────────────────────────────────────────────────────────────────

def get_video_fps(video_path, fallback=30.0):
    """
    偵測影片 fps。這批影片 fps 混雜（~30 與 ~60 都有），必須逐支偵測。
    優先用 ffprobe（連 av1 都讀得到，cv2 對 av1 抓不到 fps），失敗才 fallback cv2 / 常數。
    """
    import subprocess
    # 1. ffprobe：讀 r_frame_rate（可能是 '30/1' 或 '29893/500' 這種分數）
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=30)
        val = out.stdout.strip()
        if '/' in val:
            num, den = val.split('/')
            fps = float(num) / float(den)
        elif val:
            fps = float(val)
        else:
            fps = 0
        if fps and fps > 0:
            return fps
    except Exception:
        pass
    # 2. cv2 fallback（h264 等 cv2 讀得到的 codec）
    try:
        _cap = cv2.VideoCapture(video_path)
        fps = _cap.get(cv2.CAP_PROP_FPS)
        _cap.release()
        if fps and fps > 0:
            return fps
    except Exception:
        pass
    # 3. 都失敗
    return fallback


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--ckpt', required=True, nargs='+',
                        help='ScreenNet ckpt；給多個（例如 3 個種子的最終模型）→ 機率平均（交叉驗證 F1 +0.04~0.08）')
    parser.add_argument('--output_video', default='output_videos/screens_pred.avi')
    parser.add_argument('--threshold', type=float, default=0.70,
                        help='interval 模式（舊模型）的門檻')
    parser.add_argument('--screen_mode', choices=['auto', 'peak', 'interval'], default='auto',
                        help='peak = 事件定位（每次擋拆一個時間點，配 convert_contact 訓練的模型）；'
                             'interval = 舊做法（超過門檻的幀合併成區間）；'
                             'auto = checkpoint 有記錄 anchor 就用 peak')
    parser.add_argument('--peak_threshold', type=float, default=0.6,
                        help='peak 模式：尖峰至少要多少分。2026-10-23 最終模型（座標清理 + 459 支、3 模型平均）交叉驗證：'
                             '0.6 → 準確率 77%、抓到率 70%、F1 0.732；0.8 → 81% / 61% / 0.693（只準 4%、少抓 9%）')
    parser.add_argument('--peak_nms', type=int, default=15,
                        help='peak 模式：同一組人兩次擋拆至少隔幾張 5fps 影格（15 = 3 秒）。'
                             '5 份交叉驗證（158 個掩護，門檻 0.6）：1 秒 F1 0.54（抓錯 119，37 個是同一次掩護發兩次）'
                             '→ 3 秒 F1 0.60（抓錯 69），抓對只少 6 個')
    parser.add_argument('--peak_dense', action='store_true',
                        help='peak 模式：每個起始偏移都抽一次，曲線更密、時間更精細（慢 fps/5 倍）')
    parser.add_argument('--stub_dir', default='stubs')
    parser.add_argument('--no_stub', action='store_true',
                        help='Disable stub caching (recompute everything)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug info per predicted window')
    parser.add_argument('--ba_min_run', type=int, default=4,
                        help='連續持球段最短保留幀數（< 此視為單幀誤判抖動）')
    parser.add_argument('--ba_fill_gap', type=int, default=8,
                        help='同一持球者相鄰段之間，最多補幾幀 -1 空洞')
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
    parser.add_argument('--fps', type=float, default=None,
                        help='影片 fps（換算秒數用）；不給則從影片自動偵測，失敗 fallback 30')
    parser.add_argument('--sam3_tracks', default=None,
                        help='SAM3 產的 player_tracks pkl 路徑。給定則讀它當追蹤結果，'
                             '跳過 SAM2（用 SAM3 乾淨追蹤取代）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Video ──────────────────────────────────────────────────────────────
    print(f"Reading video: {args.video}")
    frames = read_video(args.video)
    if not frames:
        print("ERROR: cannot read video")
        return

    # fps：優先用 --fps，否則 ffprobe 逐支偵測（這批 30/60fps 混雜，不能寫死）
    fps = args.fps if args.fps else get_video_fps(args.video)
    print(f"  {len(frames)} frames @ {fps:.2f} fps  (時長 {len(frames)/fps:.1f}s)")

    use_stub = not args.no_stub
    os.makedirs(args.stub_dir, exist_ok=True)

    # ── Player tracking ──────────────────────────────────────────────────────
    # SAM3 模式：讀預先產好的 player_tracks pkl（run_sam3_tracks.py 在 sam3_env 產），
    # 跳過 SAM2 + verify。SAM3 從源頭只追球員，非球員污染消失。
    if args.sam3_tracks:
        print(f"讀 SAM3 追蹤結果: {args.sam3_tracks}")
        import pickle as _pkl
        with open(args.sam3_tracks, 'rb') as _fp:
            _sam3 = _pkl.load(_fp)
        player_tracks = _sam3['player_tracks']
        if len(player_tracks) != len(frames):
            print(f"  ⚠ SAM3 幀數({len(player_tracks)}) != 影片幀數({len(frames)})，取較短對齊")
            m = min(len(player_tracks), len(frames))
            player_tracks = player_tracks[:m]
            frames = frames[:m]
        n_tracked = sum(1 for ft in player_tracks if ft)
        print(f"  SAM3 tracked {n_tracked}/{len(frames)} frames（跳過 SAM2 + verify）")
        # SAM3 模式不需要 player_model（只有 SAM2 追蹤/verify 用它）；
        # 下游 handler 偵測用的是 handler_model，另外載入。
    else:
        # SAM2 模式：YOLO 第一幀 + SAM2 跨幀傳播
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
    ball_acquisition = detect_handler_per_frame(frames, player_tracks, handler_model,
                                                debug=args.debug)

    # ── YOLO 整段 class 投票（class 0=handler, 1=screener, 2=defender, 3=other）──
    print("YOLO per-player class voting...")
    class_votes = detect_player_class_votes(frames, player_tracks, handler_model)

    # defender 輔助（兩種追蹤模式都用）：YOLO 主要認成 defender 的 pid，
    # 下游 baseline pool 從 offense 候選剔除。這跟清污染無關。
    yolo_defender_pids = set()
    for pid, counter in class_votes.items():
        if not counter:
            continue
        top_class, _ = counter.most_common(1)[0]
        if top_class == 2:
            yolo_defender_pids.add(pid)
    print(f"  YOLO 認為的 defender (從 offense 剔除): {sorted(yolo_defender_pids)}")

    # ── 清非球員（僅 SAM2 模式需要）──────────────────────────────────────────
    # SAM2 追蹤池會混入非球員（裁判/籃框/球）→ 用「0 YOLO 票 + bbox 像球」清掉。
    # SAM3 從源頭只追球員（concept "basketball player"），沒有非球員污染 → 跳過。
    if not args.sam3_tracks:
        min_votes_for_real_player = 1
        yolo_real_player_pids = {
            pid for pid, counter in class_votes.items()
            if sum(counter.values()) >= min_votes_for_real_player
        }
        all_tracked_pids = set()
        for ft in player_tracks:
            all_tracked_pids.update(ft.keys())

        # 每個 pid 平均 bbox 形狀（球：h/w≈1、面積小；球員：h/w>1.5、面積大）
        pid_avg_shape = {}
        pid_frame_count = {}
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

        # 非球員 = (0 YOLO 票) AND (bbox 像球)。真球員即使 0 票也保留，避免誤殺。
        suspected_non_players = all_tracked_pids - yolo_real_player_pids
        non_player_pids = set()
        for pid in suspected_non_players:
            if pid not in pid_avg_shape:
                non_player_pids.add(pid)
                continue
            avg_h, avg_w, avg_aspect, avg_area = pid_avg_shape[pid]
            if avg_aspect < 1.5 or avg_area < 5000:
                non_player_pids.add(pid)

        print(f"  真球員 (>=1 YOLO 票): {sorted(yolo_real_player_pids)}")
        print(f"  非球員 (0 票 + 像球: 整段清掉): {sorted(non_player_pids)}")
        for pid in sorted(non_player_pids):
            votes = sum(class_votes.get(pid, {}).values())
            if pid in pid_avg_shape:
                avg_h, avg_w, avg_aspect, avg_area = pid_avg_shape[pid]
                print(f"    pid={pid}: votes={votes}, avg_h={avg_h:.0f}, "
                      f"avg_w={avg_w:.0f}, h/w={avg_aspect:.2f}, area={avg_area:.0f}")
            else:
                print(f"    pid={pid}: votes={votes}, no bbox")
        kept_zero_vote = (all_tracked_pids - yolo_real_player_pids) - non_player_pids
        if kept_zero_vote:
            print(f"  保留的 0 票 pid (像真球員，免於清掉): {sorted(kept_zero_vote)}")

        # 把非球員的 bbox 整段清成 None → 下游 team/court/draw 自動跳過
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
    # DEBUG：印平滑「前」的 raw pid 分布 + 非主要持球者出現在哪些幀（查射手接球）
    _raw_dist = Counter(p for p in ball_acquisition if p is not None and p != -1)
    print(f"  [RAW] ball_acquisition pid 分布（平滑前）: {dict(_raw_dist.most_common())}")
    if args.debug and _raw_dist:
        _main = _raw_dist.most_common(1)[0][0]
        for _pid in _raw_dist:
            if _pid == _main:
                continue
            _frames = [f for f, p in enumerate(ball_acquisition) if p == _pid]
            print(f"  [RAW] 非主要持球者 pid{_pid} 出現在幀: {_frames}")
        # 完整逐幀時間軸（每 10 幀一格，看誰持球的變化；. = 無偵測）
        _tl = ''.join('.' if (p is None or p == -1) else
                      (str(p)[-1] if p != _main else '=') for p in ball_acquisition)
        print(f"  [RAW] 逐幀時間軸（=主要持球者{_main}, .=無, 其餘=該pid末碼）:")
        for _i in range(0, len(_tl), 60):
            print(f"    f{_i:3d}: {_tl[_i:_i+60]}")
    ball_acquisition = smooth_ball_acquisition(
        ball_acquisition, min_run=args.ba_min_run, fill_gap=args.ba_fill_gap
    )

    # ── Court coords ───────────────────────────────────────────────────────
    print("Projecting players to court coordinates (feet)...")
    court_coords = player_court_coords(player_tracks, court_kps)
    n_with_coords = sum(1 for fc in court_coords if fc)
    print(f"  {n_with_coords}/{len(frames)} frames have homography")

    # 座標清理（跟訓練 / 交叉驗證同一套 clean_court_tracks）：左右鏡像閃爍（球場對稱 → 某些幀
    # 被投成鏡像）、整張 homography 暫時算壞、單人腳點尖刺。從源頭修好，dump 的 pkl 就是乾淨的。
    # 這裡是原始 fps：「壞影格最多連續幾張」照訓練的 5fps 3 張（0.6 秒）換算
    from tactics_tools import clean_court_tracks
    court_coords, cst = clean_court_tracks(court_coords, max_run=max(3, int(round(3 * fps / 5))))
    print(f"  座標清理：鏡像 {cst['flip']}、壞影格 {cst['bad_frames']}、單人尖刺 {cst['spikes']}"
          f"（共 {len(frames)} 幀）")

    # ── Model ──────────────────────────────────────────────────────────────
    models, ckpts = [], []
    for path in args.ckpt:
        print(f"Loading model: {path}")
        ck = torch.load(path, map_location='cpu')
        d_model, n_heads, n_layers = ck.get('d_model', 64), ck.get('n_heads', 4), ck.get('n_layers', 2)
        m = ScreenNetPairwise(d_model, n_heads, n_layers)
        m.load_state_dict(ck['model'])  # 載入完整權重（含 head）
        m = m.to(device).eval()
        m.rel_coords = bool((ck.get('data_meta') or {}).get('rel_coords'))
        print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}"
              + ("，以持球者為原點的相對座標" if m.rel_coords else ""))
        models.append(m); ckpts.append(ck)
    ckpt = ckpts[0]
    metas = [{k: v for k, v in (c.get('data_meta') or {}).items() if k in ('anchor', 'rel_coords')} for c in ckpts]
    if any(mt != metas[0] for mt in metas):
        raise SystemExit(f"--ckpt 的模型輸入設定不一致（anchor / rel_coords）：{metas}")
    model = models[0] if len(models) == 1 else ProbAverage(models)
    model.rel_coords = models[0].rel_coords
    if len(models) > 1:
        print(f"  {len(models)} 個模型機率平均")

    # ── Inference ──────────────────────────────────────────────────────────
    meta = ckpt.get('data_meta') or {}
    screen_mode = args.screen_mode
    if screen_mode == 'auto':
        screen_mode = 'peak' if 'anchor' in meta else 'interval'
    window_scores, peak_curves = [], {}

    if screen_mode == 'peak':
        # 事件定位：每次擋拆只有一個時間點（撞上那一刻）
        anchor = int(meta.get('anchor', 3))
        print(f"Screen detection: peak 模式（5fps、anchor 第 {anchor} 格、"
              f"門檻 {args.peak_threshold}、間隔 ≥{args.peak_nms} 張）...")
        peaks, curves = detect_screen_peaks(
            model, device, court_coords, player_assignment, ball_acquisition, fps,
            anchor=anchor, threshold=args.peak_threshold, nms=args.peak_nms,
            dense=args.peak_dense, yolo_defender_pids=yolo_defender_pids,
        )
        peak_curves = {f"{h}-{c}": sorted(cv.items()) for (h, c), cv in curves.items()}
        # start/end = 撞上那一刻前後各 0.3 秒：下游 tactics_tools 用 start/end 找接觸、畫框用；
        # 事件本身的時間點是 peak_frame
        half = int(round(0.3 * fps))
        events = []
        per_frame = [None] * len(frames)
        for pk in peaks:
            f, H, S, sc = pk['frame'], pk['handler_id'], pk['screener_id'], pk['score']
            ev = {'start': max(0, f - half), 'end': min(len(frames) - 1, f + half),
                  'peak_frame': f, 'handler_id': H, 'screener_id': S, 'score': sc}
            events.append(ev)
            for i in range(ev['start'], ev['end'] + 1):
                cur = per_frame[i]
                if cur is None:
                    per_frame[i] = {'handler_id': H, 'screener_ids': [S], 'scores': [sc],
                                    'max_score': sc, 'screener_id': S, 'score': sc}
                elif S not in cur['screener_ids']:
                    cur['screener_ids'].append(S)
                    cur['scores'].append(sc)
        print(f"  → {len(events)} 次擋拆（每次一個時間點）")
    else:
        print(f"Sliding-window inference (threshold={args.threshold})...")
        per_frame = detect_screens(
            model, device, court_coords, player_assignment, ball_acquisition,
            threshold=args.threshold, debug=args.debug,
            yolo_defender_pids=yolo_defender_pids,
            score_log=window_scores,
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

    # 注意：擋拆接觸幀(screen_frame)、roll/pop、coverage 都是「分析」，
    # 一律在 tactics_tools / describe_screens 讀 pkl 時算，inference 不碰
    # （感知與分析解耦：改分析邏輯不用重跑 15 分鐘 SAM2）。

    def _fmt(fr):
        """幀 → 'f<幀>(秒)'，秒用 fps 換算。"""
        return f"f{fr}({fr / fps:.1f}s)"

    print(f"\nDetected {len(events)} screen events:")
    for ev in events:
        when = (f"撞上 {_fmt(ev['peak_frame'])}" if 'peak_frame' in ev
                else f"{_fmt(ev['start'])} - {_fmt(ev['end'])}")
        print(f"  {when}  handler={ev['handler_id']}, screener={ev['screener_id']}, "
              f"score={ev['score']:.3f}")

    # ── Dump 事件 + 座標，給 tactics_tools.py 判斷戰術（解耦感知與分析）──────────
    import pickle
    dump_path = os.path.splitext(args.output_video)[0] + '_events.pkl'
    os.makedirs(os.path.dirname(dump_path) or '.', exist_ok=True)
    with open(dump_path, 'wb') as f:
        pickle.dump({
            'events': events,
            'court_coords': court_coords,           # list[dict[pid:(x_ft,y_ft)]]
            'player_assignment': player_assignment,  # list[dict[pid:team]]，team1=進攻
            'court_dims': (COURT_W_FT, COURT_H_FT),  # (91.8, 49.2)
            'fps': fps,                              # 換算秒數用
            # 每個視窗未經門檻的全部候選分數（畫 ScreenNet 分數曲線用）
            'window_scores': window_scores,          # list[{start, handler_id, scores:{pid:prob}}]（interval 模式）
            'seq_len': SEQ_LEN,
            # peak 模式：每組 "handler-候選人" 的分數曲線 [(原片幀號, 分數)]；events 帶 peak_frame
            'screen_mode': screen_mode,
            'peak_curves': peak_curves,
            # 每幀持球者（重算 ScreenNet 分數 / 之後重訓驗證會用到）
            'ball_acquisition': ball_acquisition,
            'yolo_defender_pids': sorted(yolo_defender_pids) if yolo_defender_pids else [],
        }, f)
    print(f"  事件資料 dump → {dump_path}")

    # ── Draw + save ────────────────────────────────────────────────────────
    print(f"\nDrawing predictions...")
    output_frames = draw_predictions(frames, player_tracks, per_frame)

    out_path = args.output_video
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    print(f"Saving → {out_path} @ {fps:.2f} fps")
    save_video(output_frames, out_path, fps=fps)   # 用偵測到的 fps，別吃 save_video 預設 24
    print("Done.")


if __name__ == '__main__':
    main()
