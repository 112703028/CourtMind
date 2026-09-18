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
    resp = predictor.handle_request(request=dict(
        type="start_session", resource_path=frame_dir))
    sid = resp["session_id"]
    predictor.handle_request(request=dict(
        type="add_prompt", session_id=sid, frame_index=0, text=args.prompt))

    # 逐幀收 output
    per_frame = {}
    for r in predictor.handle_stream_request(request=dict(
            type="propagate_in_video", session_id=sid)):
        per_frame[r["frame_index"]] = r["outputs"]

    # 轉成 player_tracks（同 SAM2 格式）
    player_tracks = []
    for f in range(n):
        fo = per_frame.get(f)
        frame_tracks = {}
        if fo is not None and 'out_obj_ids' in fo and 'out_binary_masks' in fo:
            for oid, m in zip(fo['out_obj_ids'], fo['out_binary_masks']):
                mm = m.cpu().numpy() if hasattr(m, 'cpu') else np.asarray(m)
                bbox = masks_to_bbox(mm)
                if bbox is not None:
                    frame_tracks[int(oid)] = {'bbox': bbox}
        player_tracks.append(frame_tracks)

    fps = get_fps(args.video)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as fp:
        pickle.dump({'player_tracks': player_tracks, 'n_frames': n, 'fps': fps}, fp)
    n_tracked = sum(1 for ft in player_tracks if ft)
    print(f"SAM3 tracked {n_tracked}/{n} frames → {args.out}")
    shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
