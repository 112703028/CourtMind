"""
inference_single_init.py — 單一 init frame 版本的 inference。

跟 inference.py 一樣，但 SAM2 init 用簡單策略：
  - **掃描整段影片**（不只前 60 幀）
  - 找球員偵測數最多的那一幀做 init
  - 不做 Pass 2 / Pass 3 多階段加新 prompts
  - 適合「相信一定有某幀完整出現 10 人」的場景

優點：
  - 流程簡單，propagate 只跑一次
  - 不會有「中途加新 obj 沒倒推回 0」的問題
  - 比 multi-pass 快約 2-3 倍

缺點：
  - 如果整段影片沒任何一幀同時有完整 10 人，會漏球員

Usage:
  python nba_data_pipeline/inference_single_init.py \\
    --video         input_videos/screen/screen_109.mp4 \\
    --ckpt          nba_data_pipeline/checkpoints/finetune_manual_v2_best.pt \\
    --handler_model models/handler_detector.pt \\
    --siamese_team_model models/team_siamese.pt \\
    --output_video  output_videos/screens_v109_single.avi \\
    --threshold     0.73 \\
    --debug
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 從 inference.py 重用所有 helper / function
import nba_data_pipeline.inference as _inf
from nba_data_pipeline.inference import (
    _filter_player_like_bboxes,
    iou,
    verify_sam2_with_yolo,
    detect_handler_per_frame,
    detect_player_class_votes,
    build_homography_from_keypoints,
    foot_pos_xyxy,
    transform_pt,
    tactical_to_feet,
    player_court_coords,
    smooth_ball_acquisition,
    positions_for,
    detect_screens,
    draw_predictions,
    SEQ_LEN,
    SAM2_CHECKPOINT_DEFAULT,
    SAM2_CONFIG_DEFAULT,
    HANDLER_MODEL_DEFAULT,
    PLAYER_MODEL_DEFAULT,
    COURT_W_PX, COURT_H_PX, COURT_W_FT, COURT_H_FT,
    TARGET_KPS,
)

from utils import read_video, save_video
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ultralytics import YOLO

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None

from nba_data_pipeline.model import ScreenNetPairwise, N_OFFENSE, N_DEFENSE, N_PLAYERS


# ── 重寫的 SAM2 追蹤：單一 init frame，全片掃描 ─────────────────────────────────

def _pick_best_frame_in_video(frames, player_model, conf=0.4, stride=5):
    """
    掃描整段影片（每 stride 幀取一次），找球員偵測數最多的那一幀。
    早停：找到 >= 10 個球員就停。
    """
    best_idx, best_count, best_boxes = 0, 0, None
    n_scan = (len(frames) + stride - 1) // stride
    print(f"  掃描全片找最佳 init frame（{n_scan} 個 scan points, stride={stride}）...")

    for scan_n, i in enumerate(range(0, len(frames), stride)):
        r = player_model.predict(frames[i], conf=conf, verbose=False)[0]
        raw_boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        boxes = _filter_player_like_bboxes(raw_boxes)
        n = len(boxes)
        if n > best_count:
            best_idx, best_count, best_boxes = i, n, boxes
            print(f"    frame {i}: {n} player bboxes (current best)")
            if n >= 10:
                print(f"    達到 10 人，提早停止掃描")
                break

    return best_idx, best_boxes


def run_sam2_tracks_single(frames, player_model, predictor, device='cuda'):
    """
    單一 init frame 版的 SAM2 追蹤。
    完整流程：
      1. 全片掃描找最多人的幀（_pick_best_frame_in_video）
      2. 用那幀的 bbox 一次性 init SAM2
      3. Forward + Reverse propagate 整段
    回傳 list[dict[player_id: {'bbox': [...]}]]
    """
    if not frames:
        return []

    init_idx, init_boxes = _pick_best_frame_in_video(frames, player_model)
    if init_boxes is None or len(init_boxes) == 0:
        print("  ⚠ SAM2: 找不到能初始化的幀")
        return [{} for _ in frames]

    print(f"  SAM2 init at frame {init_idx} with {len(init_boxes)} player bboxes")
    obj_ids = list(range(1, len(init_boxes) + 1))

    tmp_dir = Path(tempfile.mkdtemp(prefix='sam2_frames_'))
    try:
        for i, frame in enumerate(frames):
            cv2.imwrite(str(tmp_dir / f'{i:05d}.jpg'), frame)

        inference_state = predictor.init_state(video_path=str(tmp_dir))
        predictor.reset_state(inference_state)
        autocast_dtype = torch.bfloat16 if device == 'cuda' else torch.float32

        # Add prompts at init_idx
        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            for obj_id, xyxy in zip(obj_ids, init_boxes):
                predictor.add_new_points_or_box(
                    inference_state, frame_idx=init_idx, obj_id=obj_id,
                    box=xyxy.astype(np.float32))

        raw_tracks = [{} for _ in frames]
        for obj_id, xyxy in zip(obj_ids, init_boxes):
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

        # Forward propagate
        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            for f_idx, obj_ids_p, masks in predictor.propagate_in_video(inference_state):
                _absorb(f_idx, obj_ids_p, masks)

        # Reverse propagate (if init_idx > 0)
        if init_idx > 0:
            with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
                for f_idx, obj_ids_p, masks in predictor.propagate_in_video(
                        inference_state, reverse=True):
                    _absorb(f_idx, obj_ids_p, masks)

        predictor.reset_state(inference_state)
        return raw_tracks
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output_video', default='output_videos/screens_pred.avi')
    parser.add_argument('--threshold', type=float, default=0.70)
    parser.add_argument('--stub_dir', default='stubs')
    parser.add_argument('--no_stub', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ba_smooth_window', type=int, default=15)
    parser.add_argument('--ba_min_frames', type=int, default=20)
    parser.add_argument('--handler_model', default=HANDLER_MODEL_DEFAULT)
    parser.add_argument('--player_model', default=PLAYER_MODEL_DEFAULT)
    parser.add_argument('--sam2_ckpt', default=SAM2_CHECKPOINT_DEFAULT)
    parser.add_argument('--sam2_config', default=SAM2_CONFIG_DEFAULT)
    parser.add_argument('--siamese_team_model',
                        default=str(PROJECT_ROOT / 'models' / 'team_siamese.pt'))
    parser.add_argument('--siamese_threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Reading video: {args.video}")
    frames = read_video(args.video)
    if not frames:
        print("ERROR: cannot read video")
        return
    print(f"  {len(frames)} frames")

    use_stub = not args.no_stub
    os.makedirs(args.stub_dir, exist_ok=True)

    if build_sam2_video_predictor is None:
        print("ERROR: sam2 not installed.")
        return

    print(f"Loading YOLO player model: {args.player_model}")
    player_model = YOLO(args.player_model)
    print(f"Loading handler detector: {args.handler_model}")
    handler_model = YOLO(args.handler_model)
    print(f"Loading SAM2 predictor on {device}...")
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_ckpt, device=str(device))

    # === 用單一 init 版本的 SAM2 追蹤 ===
    print("Player tracking via SAM2 (single init)...")
    player_tracks = run_sam2_tracks_single(frames, player_model, predictor, device=str(device))
    n_tracked = sum(1 for ft in player_tracks if ft)
    print(f"  SAM2 tracked {n_tracked}/{len(frames)} frames")

    # YOLO drift verification（重用既有 function）
    player_tracks = verify_sam2_with_yolo(frames, player_tracks, player_model)

    # Handler detection
    ball_acquisition = detect_handler_per_frame(frames, player_tracks, handler_model)

    # YOLO class voting + 雙條件 filter（複製 inference.py 主流程的部分）
    print("YOLO per-player class voting...")
    class_votes = detect_player_class_votes(frames, player_tracks, handler_model)
    yolo_real_player_pids = {
        pid for pid, counter in class_votes.items()
        if sum(counter.values()) >= 1
    }
    all_tracked_pids = set()
    for ft in player_tracks:
        all_tracked_pids.update(ft.keys())

    # 算每個 pid 的平均 bbox 形狀
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

    suspected_non_players = all_tracked_pids - yolo_real_player_pids
    non_player_pids = set()
    for pid in suspected_non_players:
        if pid not in pid_avg_shape:
            non_player_pids.add(pid)
            continue
        avg_h, avg_w, avg_aspect, avg_area = pid_avg_shape[pid]
        if avg_aspect < 1.5 or avg_area < 5000:
            non_player_pids.add(pid)

    yolo_defender_pids = set()
    for pid, counter in class_votes.items():
        if counter and counter.most_common(1)[0][0] == 2:
            yolo_defender_pids.add(pid)

    print(f"  真球員: {sorted(yolo_real_player_pids)}")
    print(f"  非球員 (清掉): {sorted(non_player_pids)}")
    print(f"  defender: {sorted(yolo_defender_pids)}")

    for ft in player_tracks:
        for pid in non_player_pids:
            if pid in ft and ft[pid].get('bbox') is not None:
                ft[pid]['bbox'] = None
                ft[pid]['drifted'] = True

    # Court keypoints + court_coords
    print("Court keypoints...")
    court_detector = CourtKeypointDetector(str(PROJECT_ROOT / 'models' / 'court_keypoint_detector.pt'))
    court_kps = court_detector.get_court_keypoints(
        frames, read_from_stub=use_stub,
        stub_path=os.path.join(args.stub_dir, 'court_key_points_stub.pkl'),
    )

    # Team assignment
    print("Team assignment...")
    if args.siamese_team_model and os.path.exists(args.siamese_team_model):
        from nba_data_pipeline.team_classifier.inference import TeamAssignerSiamese
        team_assigner = TeamAssignerSiamese(
            ckpt_path=args.siamese_team_model,
            device=str(device),
            threshold=args.siamese_threshold,
        )
        player_assignment = team_assigner.get_player_teams_across_frames(
            frames, player_tracks, ball_acquisition,
            read_from_stub=use_stub,
            stub_path=os.path.join(args.stub_dir, 'player_assignment_stub.pkl'),
        )
    else:
        team_assigner = TeamAssigner(fit_stride=10)
        player_assignment = team_assigner.get_player_teams_across_frames(
            frames, player_tracks, read_from_stub=use_stub,
            stub_path=os.path.join(args.stub_dir, 'player_assignment_stub.pkl'),
        )

    ball_acquisition = smooth_ball_acquisition(
        ball_acquisition, window=args.ba_smooth_window, min_total_frames=args.ba_min_frames
    )

    print("Projecting players to court coordinates (feet)...")
    court_coords = player_court_coords(player_tracks, court_kps)
    n_with_coords = sum(1 for fc in court_coords if fc)
    print(f"  {n_with_coords}/{len(frames)} frames have homography")

    # ScreenNet inference
    print(f"Loading model: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    d_model = ckpt.get('d_model', 64)
    n_heads = ckpt.get('n_heads', 4)
    n_layers = ckpt.get('n_layers', 2)
    model = ScreenNetPairwise(d_model, n_heads, n_layers)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    print(f"Sliding-window inference (threshold={args.threshold})...")
    per_frame = detect_screens(
        model, device, court_coords, player_assignment, ball_acquisition,
        threshold=args.threshold, debug=args.debug,
        yolo_defender_pids=yolo_defender_pids,
    )

    n_screen_frames = sum(1 for e in per_frame if e is not None)
    print(f"  → {n_screen_frames}/{len(frames)} frames flagged as screen")

    # 事件分組
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

    # Draw + save
    print(f"\nDrawing predictions...")
    output_frames = draw_predictions(frames, player_tracks, per_frame)
    out_path = args.output_video
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    print(f"Saving → {out_path}")
    save_video(output_frames, out_path)
    print("Done.")


if __name__ == '__main__':
    main()
