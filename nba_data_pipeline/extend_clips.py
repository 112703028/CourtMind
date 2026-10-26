"""
extend_clips.py — 把擋拆片段往後延長（看得到出手 / 進沒進），給投籃偵測用。

起點跟原片段完全一樣（0 號影格相同，原本的追蹤 / 標記可以直接對上），只把結尾往後多切 --extra 秒。
來源：new_clips_<youtube_id>.csv 的時間段 + input_videos/yt_src/<youtube_id>.mp4（只有 185 號以後的片段有）。
輸出：input_videos/screen_ext/screen_N.mp4（不動 input_videos/screen/）

用法：
  python nba_data_pipeline/extend_clips.py --clips screen_285,screen_449 --extra 4
  python nba_data_pipeline/extend_clips.py --n 20 --seed 0          # 隨機挑 20 支
"""
import argparse
import csv
import random
import subprocess
from pathlib import Path

from make_new_clips import ROOT, SRC, to_sec

OUT = ROOT / 'input_videos' / 'screen_ext'


def load_ranges():
    rows = {}
    for f in sorted((ROOT / 'nba_data_pipeline').glob('new_clips_*.csv')):
        for r in csv.DictReader(open(f, encoding='utf-8-sig')):
            rows[r['clip']] = (r['youtube_id'], r['time_range'])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', default='', help='逗號分隔，例 screen_285,screen_449')
    ap.add_argument('--n', type=int, default=0, help='另外隨機挑幾支')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--extra', type=float, default=4.0, help='結尾往後多切幾秒')
    args = ap.parse_args()

    ranges = load_ranges()
    clips = [c for c in args.clips.split(',') if c]
    missing = [c for c in clips if c not in ranges]
    if missing:
        print(f'⚠ 沒有來源時間段（185 號以前，或不在 new_clips_*.csv）：{missing}')
    clips = [c for c in clips if c in ranges]
    if args.n:
        rest = sorted(set(ranges) - set(clips))
        clips += random.Random(args.seed).sample(rest, min(args.n, len(rest)))

    OUT.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        vid, r = ranges[clip]
        src = SRC / f'{vid}.mp4'
        out = OUT / f'{clip}.mp4'
        if out.exists():
            print(f'{clip} 已存在，跳過'); continue
        if not src.exists():
            print(f'⚠ {clip}：來源 {src.name} 不在本機，跳過'); continue
        a, b = (to_sec(x) for x in r.split('-'))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{a:.2f}', '-i', str(src),
                        '-t', f'{b - a + args.extra:.2f}', '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
                        '-an', str(out)], check=True)
        print(f'{clip}：{r} + {args.extra:.0f} 秒 → {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
