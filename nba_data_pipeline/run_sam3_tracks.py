"""
run_sam3_tracks.py — SAM3 concept "basketball player" 追蹤 → player_tracks pkl。
取代 SAM2 的 run_sam2_tracks。跑在 sam3_env（cu13.1）。

產出格式跟 SAM2 的 run_sam2_tracks 一致：
  player_tracks = list[dict[pid: {'bbox': [x1,y1,x2,y2]}]]（每 frame 一個 dict）
下游 inference.py 讀這個 pkl 即可跳過 SAM2。

用法：
  python run_sam3_tracks.py --video input_videos/screen/screen_130.mp4 \
      --out stubs/screen_130_sam3_tracks.pkl --single_gpu
"""
import os
import argparse
import subprocess
import tempfile
import shutil
import pickle

import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor


def masks_to_bbox(mask):
    """binary mask → [x1,y1,x2,y2] 或 None。"""
    m = np.asarray(mask)
    if m.ndim > 2:
        m = m.squeeze()
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_iou(a, b):
    """兩個 bbox [x1,y1,x2,y2] 的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run_concept(predictor, frame_dir, text, n):
    """跑一個 concept → list[dict[obj_id: bbox]]（每 frame 一個 dict）。"""
    resp = predictor.handle_request(request=dict(
        type="start_session", resource_path=frame_dir))
    sid = resp["session_id"]
    predictor.handle_request(request=dict(
        type="add_prompt", session_id=sid, frame_index=0, text=text))
    per_frame = {}
    for r in predictor.handle_stream_request(request=dict(
            type="propagate_in_video", session_id=sid)):
        per_frame[r["frame_index"]] = r["outputs"]
    tracks = []
    for f in range(n):
        fo = per_frame.get(f)
        ft = {}
        if fo is not None and 'out_obj_ids' in fo and 'out_binary_masks' in fo:
            for oid, m in zip(fo['out_obj_ids'], fo['out_binary_masks']):
                mm = m.cpu().numpy() if hasattr(m, 'cpu') else np.asarray(m)
                bbox = masks_to_bbox(mm)
                if bbox is not None:
                    ft[int(oid)] = bbox
        tracks.append(ft)
    return tracks


def get_fps(video):
    """ffprobe 取 fps，失敗 fallback 30。"""
    r = subprocess.run(['ffprobe', '-v', '0', '-of', 'csv=p=0',
                        '-select_streams', 'v:0', '-show_entries',
                        'stream=r_frame_rate', video],
                       capture_output=True, text=True)
    try:
        num, den = r.stdout.strip().split('/')
        return float(num) / float(den)
    except Exception:
        return 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompt', default='basketball player')
    ap.add_argument('--single_gpu', action='store_true')
    ap.add_argument('--exclude_referee', action='store_true', default=True,
                    help='跑第二個 concept "referee" 排掉裁判/教練（預設開）')
    ap.add_argument('--no_exclude_referee', dest='exclude_referee',
                    action='store_false', help='關掉 referee 排除')
    ap.add_argument('--ref_iou', type=float, default=0.5,
                    help='球員 obj 跟裁判 obj 的 bbox IoU >= 此才算重疊')
    ap.add_argument('--ref_frac', type=float, default=0.5,
                    help='球員 obj 整段有 >= 此比例的幀跟裁判高重疊 → 判定為裁判排除'
                         '（保險：只排穩定被當裁判的，避免誤刪偶爾誤標的球員）')
    args = ap.parse_args()

    gpus = [torch.cuda.current_device()] if args.single_gpu \
        else list(range(torch.cuda.device_count()))

    # ffmpeg 抽 jpg 資料夾（container cv2 沒 FFMPEG，SAM3 可吃 jpg 資料夾）
    frame_dir = tempfile.mkdtemp(prefix='sam3t_')
    subprocess.run(['ffmpeg', '-y', '-i', args.video, '-start_number', '0',
                    os.path.join(frame_dir, '%05d.jpg')],
                   capture_output=True, text=True)
    jpgs = sorted(f for f in os.listdir(frame_dir) if f.endswith('.jpg'))
    if not jpgs:
        print("ffmpeg 抽幀失敗")
        shutil.rmtree(frame_dir, ignore_errors=True)
        return
    n = len(jpgs)
    print(f"抽出 {n} 幀")

    predictor = build_sam3_video_predictor(gpus_to_use=gpus)

    # concept 1: 球員
    print(f'concept: "{args.prompt}"')
    player_bboxes = run_concept(predictor, frame_dir, args.prompt, n)

    # concept 2: 裁判（排除用）
    ref_bboxes = None
    if args.exclude_referee:
        print('concept: "referee"（排裁判/教練）')
        ref_bboxes = run_concept(predictor, frame_dir, "referee", n)

    # 對照排除：球員 obj 若「整段 >= ref_frac 比例的幀」跟某裁判 obj 高重疊 → 是裁判
    excluded = set()
    if ref_bboxes is not None:
        # 收集每個球員 obj 出現的幀 + 每幀是否跟任一裁判高重疊
        overlap_cnt = {}   # player_obj_id -> 高重疊幀數
        appear_cnt = {}    # player_obj_id -> 出現幀數
        for f in range(n):
            refs = list(ref_bboxes[f].values())
            for pid, pbox in player_bboxes[f].items():
                appear_cnt[pid] = appear_cnt.get(pid, 0) + 1
                if any(bbox_iou(pbox, rb) >= args.ref_iou for rb in refs):
                    overlap_cnt[pid] = overlap_cnt.get(pid, 0) + 1
        for pid, appear in appear_cnt.items():
            frac = overlap_cnt.get(pid, 0) / appear if appear > 0 else 0
            if frac >= args.ref_frac:
                excluded.add(pid)
        print(f"  裁判排除: {sorted(excluded)}（整段 >= {args.ref_frac:.0%} 幀被當裁判）")

    # 轉成 player_tracks（同 SAM2 格式），排掉裁判 obj
    player_tracks = []
    for f in range(n):
        frame_tracks = {}
        for pid, bbox in player_bboxes[f].items():
            if pid in excluded:
                continue
            frame_tracks[pid] = {'bbox': bbox}
        player_tracks.append(frame_tracks)

    fps = get_fps(args.video)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as fp:
        pickle.dump({'player_tracks': player_tracks, 'n_frames': n, 'fps': fps}, fp)
    n_tracked = sum(1 for ft in player_tracks if ft)
    print(f"SAM3 tracked {n_tracked}/{n} frames → {args.out}"
          f"（排掉 {len(excluded)} 個裁判/教練）")
    shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
