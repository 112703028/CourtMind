"""
sam3_ball_train.py — 訓練片段（5fps 影格）的 SAM3 追球遮罩，給「逐窗持球者」用。
球員遮罩已經有（run_sam3_tracks.py --save_masks → stubs/train_sam3_masks/），這裡只跑 concept "basketball"。
影格編號跟 train_sam3_masks / 訓練快取 raw_tracks 一致（第 i 張 = 5fps 影格索引 i）。

跑在 sam3_env（pod：source sam3_env.sh；HF_HUB_OFFLINE=1）：
  python nba_data_pipeline/sam3_ball_train.py --frames_root input_videos/screen_frames \
      --out_dir stubs/train_sam3_ball --single_gpu
輸出：<out_dir>/<clip>_ball.pkl = {'balls': list[dict[oid: bbox]], 'ball_masks': list[dict[oid: pack_mask]], 'n_frames'}
"""
import argparse
import os
import pickle
import shutil
import tempfile

import torch
from sam3.model_builder import build_sam3_video_predictor

from run_sam3_tracks import run_concept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_root', default='input_videos/screen_frames')
    ap.add_argument('--out_dir', default='stubs/train_sam3_ball')
    ap.add_argument('--clips', default='', help='逗號分隔；不給就 frames_root 下全部')
    ap.add_argument('--single_gpu', action='store_true')
    args = ap.parse_args()

    gpus = [torch.cuda.current_device()] if args.single_gpu else list(range(torch.cuda.device_count()))
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)
    os.makedirs(args.out_dir, exist_ok=True)
    clips = [c for c in args.clips.split(',') if c] or \
        sorted(os.listdir(args.frames_root), key=lambda s: int(s.split('_')[1]))
    for k, clip in enumerate(clips):
        out = os.path.join(args.out_dir, f'{clip}_ball.pkl')
        if os.path.exists(out):
            continue
        src = os.path.join(args.frames_root, clip)
        jpgs = sorted((f for f in os.listdir(src) if f.endswith('.jpg')),
                      key=lambda f: int(f.rsplit('_', 1)[1].split('.')[0])) if os.path.isdir(src) else []
        if not jpgs:
            print(f'[{k + 1}/{len(clips)}] {clip} 沒有影格，跳過', flush=True); continue
        tmp = tempfile.mkdtemp(prefix='sam3ball_')
        try:
            for i, f in enumerate(jpgs):
                shutil.copy(os.path.join(src, f), os.path.join(tmp, f'{i:05d}.jpg'))
            masks = []
            balls = run_concept(predictor, tmp, 'basketball', len(jpgs), masks)
            pickle.dump({'balls': balls, 'ball_masks': masks, 'n_frames': len(jpgs)}, open(out, 'wb'))
            print(f'[{k + 1}/{len(clips)}] {clip}: 球有追到 {sum(1 for b in balls if b)}/{len(jpgs)} 張', flush=True)
        except Exception as e:
            print(f'[{k + 1}/{len(clips)}] ✗ {clip} 失敗：{e}', flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
