"""
sam3_shot_masks.py — 投籃判斷用：SAM3 同時追「球員」和「球」的遮罩（延長片段，10fps）。
下游 shot_events.py 用它判斷：出手那刻球的遮罩貼在誰身上（= 出手者，排除封蓋的防守者）、
球離手後先碰到另一個球員（傳球）還是飛向籃框（投籃）。

跑在 sam3_env（pod：source sam3_env.sh；HF_HUB_OFFLINE=1）。
用法：
  python nba_data_pipeline/sam3_shot_masks.py --video_dir input_videos/screen_ext \
      --out_dir stubs/shot_sam3 --fps 10 --single_gpu
輸出：<out_dir>/<clip>_shot_sam3.pkl
  {'fps': 抽樣 fps, 'orig_fps', 'frame_idx': 每張抽樣對應的原片影格,
   'players': list[dict[pid: bbox]], 'player_masks': list[dict[pid: pack_mask]],
   'balls': list[dict[oid: bbox]], 'ball_masks': list[dict[oid: pack_mask]], 'excluded_referee'}
"""
import argparse
import os
import pickle
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import torch
from sam3.model_builder import build_sam3_video_predictor

from run_sam3_tracks import run_concept, track_frame_dir, get_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_dir', default='input_videos/screen_ext')
    ap.add_argument('--clips', default='', help='逗號分隔；不給就整個資料夾')
    ap.add_argument('--out_dir', default='stubs/shot_sam3')
    ap.add_argument('--fps', type=float, default=10)
    ap.add_argument('--single_gpu', action='store_true')
    args = ap.parse_args()

    gpus = [torch.cuda.current_device()] if args.single_gpu else list(range(torch.cuda.device_count()))
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)
    os.makedirs(args.out_dir, exist_ok=True)
    clips = [c for c in args.clips.split(',') if c] or \
        sorted(f[:-4] for f in os.listdir(args.video_dir) if f.endswith('.mp4'))
    targs = SimpleNamespace(prompt='basketball player', exclude_referee=True, ref_iou=0.5, ref_frac=0.5)

    for k, clip in enumerate(clips):
        out = os.path.join(args.out_dir, f'{clip}_shot_sam3.pkl')
        if os.path.exists(out):
            print(f'[{k + 1}/{len(clips)}] {clip} 已存在，跳過'); continue
        video = os.path.join(args.video_dir, f'{clip}.mp4')
        ofps = get_fps(video)
        tmp = tempfile.mkdtemp(prefix='sam3shot_')
        try:
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video, '-vf', f'fps={args.fps:g}',
                            '-q:v', '2', '-start_number', '0', os.path.join(tmp, '%05d.jpg')], check=True)
            n = len([f for f in os.listdir(tmp) if f.endswith('.jpg')])
            print(f'[{k + 1}/{len(clips)}] {clip}: {n} 張（{args.fps:g}fps）', flush=True)
            pmasks, bmasks = [], []
            players, excluded = track_frame_dir(predictor, tmp, n, targs, pmasks)
            print('concept: "basketball"')
            balls = run_concept(predictor, tmp, 'basketball', n, bmasks)
            players = [{p: v['bbox'] for p, v in f.items()} for f in players]
            pmasks = [{p: m for p, m in f.items() if p not in excluded} for f in pmasks]
            pickle.dump(dict(fps=args.fps, orig_fps=ofps, frame_idx=[int(round(i * ofps / args.fps)) for i in range(n)],
                             players=players, player_masks=pmasks, balls=balls, ball_masks=bmasks,
                             excluded_referee=sorted(excluded)), open(out, 'wb'))
            nb = sum(1 for f in balls if f)
            print(f'   → {out}（球員 {len({p for f in players for p in f})} 個，球有追到 {nb}/{n} 張）', flush=True)
        except Exception as e:
            print(f'   ✗ {clip} 失敗：{e}', flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
