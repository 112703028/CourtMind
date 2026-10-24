"""
run_sam3_tracks.py — SAM3 concept "basketball player" 追蹤 → player_tracks pkl。
取代 SAM2 的 run_sam2_tracks。跑在 sam3_env（cu13.1）。

產出格式跟 SAM2 的 run_sam2_tracks 一致：
  player_tracks = list[dict[pid: {'bbox': [x1,y1,x2,y2]}]]（每 frame 一個 dict）
下游 inference.py 讀這個 pkl 即可跳過 SAM2。

用法：
  # 單支影片（推論用）：預設只追 5fps 再內插回原片每一幀（--sample_fps 0 = 每一幀都追）
  python run_sam3_tracks.py --video input_videos/screen/screen_130.mp4 \
      --out stubs/screen_130_sam3_tracks.pkl --single_gpu

  # 批次：訓練用的 5fps 影格（ScreenNet 重訓），模型只載一次
  python run_sam3_tracks.py --frames_root input_videos/screen_frames \
      --clips_file nba_data_pipeline/train_clips.txt --out_dir stubs/train_sam3 --single_gpu
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


def pack_mask(mask, bbox):
    """遮罩只存 bbox 範圍內的部分，packbits 壓縮 → (bbox, 高, 寬, bytes)。還原用 unpack_mask。"""
    x1, y1, x2, y2 = bbox
    crop = np.asarray(mask).squeeze()[y1:y2 + 1, x1:x2 + 1] > 0
    return (bbox, crop.shape[0], crop.shape[1], np.packbits(crop).tobytes())


def unpack_mask(packed):
    """pack_mask 的反操作 → (bbox, bbox 範圍內的 bool 遮罩)。"""
    bbox, h, w, data = packed
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:h * w]
    return bbox, bits.reshape(h, w).astype(bool)


def run_concept(predictor, frame_dir, text, n, masks_out=None):
    """跑一個 concept → list[dict[obj_id: bbox]]（每 frame 一個 dict）。
    masks_out 給一個 list 時，順便把每幀每個 obj 的遮罩（pack_mask 格式）存進去。"""
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
        ft, fm = {}, {}
        if fo is not None and 'out_obj_ids' in fo and 'out_binary_masks' in fo:
            for oid, m in zip(fo['out_obj_ids'], fo['out_binary_masks']):
                mm = m.cpu().numpy() if hasattr(m, 'cpu') else np.asarray(m)
                bbox = masks_to_bbox(mm)
                if bbox is not None:
                    ft[int(oid)] = bbox
                    if masks_out is not None:
                        fm[int(oid)] = pack_mask(mm, bbox)
        tracks.append(ft)
        if masks_out is not None:
            masks_out.append(fm)
    # 關 session 釋放 GPU 記憶體（批次跑很多支時不關會越吃越多）
    try:
        predictor.handle_request(request=dict(type="close_session", session_id=sid))
    except Exception:
        pass
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


def track_frame_dir(predictor, frame_dir, n, args, masks_out=None):
    """對一個 %05d.jpg 影格資料夾跑 SAM3 → (player_tracks, excluded 裁判 obj)。
    masks_out 給 list 時存球員遮罩（裁判排除前的全部 obj，下游用 player_tracks 的 pid 挑）。"""
    print(f'concept: "{args.prompt}"')
    player_bboxes = run_concept(predictor, frame_dir, args.prompt, n, masks_out)

    excluded = set()
    if args.exclude_referee:
        print('concept: "referee"（排裁判/教練）')
        ref_bboxes = run_concept(predictor, frame_dir, "referee", n)
        # 球員 obj 若「整段 >= ref_frac 比例的幀」跟某裁判 obj 高重疊 → 是裁判
        overlap_cnt, appear_cnt = {}, {}
        for f in range(n):
            refs = list(ref_bboxes[f].values())
            for pid, pbox in player_bboxes[f].items():
                appear_cnt[pid] = appear_cnt.get(pid, 0) + 1
                if any(bbox_iou(pbox, rb) >= args.ref_iou for rb in refs):
                    overlap_cnt[pid] = overlap_cnt.get(pid, 0) + 1
        for pid, appear in appear_cnt.items():
            if appear > 0 and overlap_cnt.get(pid, 0) / appear >= args.ref_frac:
                excluded.add(pid)
        print(f"  裁判排除: {sorted(excluded)}（整段 >= {args.ref_frac:.0%} 幀被當裁判）")

    player_tracks = [{pid: {'bbox': bbox} for pid, bbox in player_bboxes[f].items()
                      if pid not in excluded} for f in range(n)]
    return player_tracks, excluded


def expand_tracks(sub_tracks, idx, n, step):
    """抽樣影格的追蹤（sub_tracks[i] 對應原片第 idx[i] 幀）→ 原片每一幀。
    兩個抽樣影格之間：兩邊都有的人 bbox 線性內插；只有一邊有的，離那張 ≤ 半個間隔才沿用。"""
    out = [dict() for _ in range(n)]
    for i, f0 in enumerate(idx):
        f1 = idx[i + 1] if i + 1 < len(idx) else n
        a = sub_tracks[i]
        b = sub_tracks[i + 1] if i + 1 < len(idx) else {}
        for f in range(f0, f1):
            t = (f - f0) / max(1, f1 - f0)
            fd = {}
            for pid in set(a) | set(b):
                if pid in a and pid in b:
                    ba, bb = a[pid]['bbox'], b[pid]['bbox']
                    fd[pid] = {'bbox': [int(round(x + (y - x) * t)) for x, y in zip(ba, bb)]}
                elif pid in a and f - f0 <= step / 2:
                    fd[pid] = {'bbox': list(a[pid]['bbox'])}
                elif pid in b and f1 - f <= step / 2:
                    fd[pid] = {'bbox': list(b[pid]['bbox'])}
            out[f] = fd
    return out


def run_batch(predictor, args):
    """批次模式：對 frames_root 下每支 clip 的 5fps 訓練影格（screen_N_XXXX.jpg）跑 SAM3。
    輸出 out_dir/screen_N_sam3_tracks.pkl，player_tracks 第 i 個 = 影格索引 i（與 COCO 標註同編號）。"""
    clips = [c.strip() for c in open(args.clips_file) if c.strip()] if args.clips_file \
        else sorted(os.listdir(args.frames_root), key=lambda s: int(s.split('_')[1]))
    os.makedirs(args.out_dir, exist_ok=True)
    for k, clip in enumerate(clips):
        out = os.path.join(args.out_dir, f'{clip}_sam3_tracks.pkl')
        mask_out = os.path.join(args.out_dir, f'{clip}_sam3_masks.pkl')
        if os.path.exists(out) and (not args.save_masks or os.path.exists(mask_out)):
            print(f"[{k + 1}/{len(clips)}] {clip} 已存在，跳過")
            continue
        src = os.path.join(args.frames_root, clip)
        if not os.path.isdir(src):
            print(f"[{k + 1}/{len(clips)}] {clip} 沒有影格資料夾，跳過")
            continue
        jpgs = sorted((f for f in os.listdir(src) if f.endswith('.jpg')),
                      key=lambda f: int(f.rsplit('_', 1)[1].split('.')[0]))
        if not jpgs:
            print(f"[{k + 1}/{len(clips)}] {clip} 沒有影格，跳過")
            continue
        # SAM3 讀資料夾要 %05d.jpg 連號 → 複製成暫存資料夾（索引 = 原檔名的 XXXX）
        tmp = tempfile.mkdtemp(prefix='sam3b_')
        for i, f in enumerate(jpgs):
            shutil.copy(os.path.join(src, f), os.path.join(tmp, f'{i:05d}.jpg'))
        print(f"[{k + 1}/{len(clips)}] {clip}: {len(jpgs)} 影格")
        try:
            masks = [] if args.save_masks else None
            tracks, excluded = track_frame_dir(predictor, tmp, len(jpgs), args, masks)
            with open(out, 'wb') as fp:
                pickle.dump({'player_tracks': tracks, 'n_frames': len(jpgs), 'fps': 5.0,
                             'frame_files': jpgs, 'excluded_referee': sorted(excluded)}, fp)
            if masks is not None:
                with open(mask_out, 'wb') as fp:
                    pickle.dump({'masks': masks, 'n_frames': len(jpgs)}, fp)
            n_obj = len({p for ft in tracks for p in ft})
            print(f"   → {out}（{n_obj} 個球員 obj，排掉裁判 {len(excluded)}）")
        except Exception as e:
            print(f"   ✗ {clip} 失敗：{e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=None, help='單支影片模式')
    ap.add_argument('--out', default=None, help='單支影片模式的輸出 pkl')
    ap.add_argument('--frames_root', default=None,
                    help='批次模式：影格資料夾根目錄（如 input_videos/screen_frames）')
    ap.add_argument('--clips_file', default=None, help='批次模式：只跑這些 clip（一行一個）')
    ap.add_argument('--out_dir', default='stubs/train_sam3', help='批次模式輸出目錄')
    ap.add_argument('--prompt', default='basketball player')
    ap.add_argument('--single_gpu', action='store_true')
    ap.add_argument('--sample_fps', type=float, default=5,
                    help='單支影片模式：只追蹤抽成這個 fps 的影格（預設 5 = 跟訓練一樣），再內插回原片每一幀；0 = 原片每一幀都追。'
                         '原片 fps 追蹤時，掩護者和防守者重疊容易被併成同一人（screen_121）；'
                         '2026-10-24 端到端 9 支：5fps F1（±1 秒）0.89 vs 原片 0.82，SAM3 只處理 1/6 影格')
    ap.add_argument('--save_masks', action='store_true',
                    help='批次模式：另存 <clip>_sam3_masks.pkl（每幀每個 obj 的遮罩，pack_mask 格式）')
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
    if not args.frames_root and not (args.video and args.out):
        ap.error('要嘛給 --video + --out（單支），要嘛給 --frames_root（批次）')

    gpus = [torch.cuda.current_device()] if args.single_gpu \
        else list(range(torch.cuda.device_count()))
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)

    if args.frames_root:
        run_batch(predictor, args)
        return

    # ── 單支影片模式 ──
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
    fps = get_fps(args.video)

    if args.sample_fps and args.sample_fps < fps:
        # 只追蹤抽出來的影格（跟 inference / 訓練一樣的抽法），再內插回原片每一幀
        step = fps / args.sample_fps
        idx = []
        while int(round(len(idx) * step)) < n:
            idx.append(int(round(len(idx) * step)))
        sub_dir = tempfile.mkdtemp(prefix='sam3s_')
        for i, f in enumerate(idx):
            shutil.copy(os.path.join(frame_dir, jpgs[f]), os.path.join(sub_dir, f'{i:05d}.jpg'))
        print(f"只追蹤 {len(idx)} 張（{args.sample_fps:g}fps），再內插回 {n} 幀")
        sub_tracks, excluded = track_frame_dir(predictor, sub_dir, len(idx), args)
        shutil.rmtree(sub_dir, ignore_errors=True)
        player_tracks = expand_tracks(sub_tracks, idx, n, step)
    else:
        player_tracks, excluded = track_frame_dir(predictor, frame_dir, n, args)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as fp:
        pickle.dump({'player_tracks': player_tracks, 'n_frames': n, 'fps': fps}, fp)
    n_tracked = sum(1 for ft in player_tracks if ft)
    print(f"SAM3 tracked {n_tracked}/{n} frames → {args.out}"
          f"（排掉 {len(excluded)} 個裁判/教練）")
    shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
