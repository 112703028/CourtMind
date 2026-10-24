"""
make_new_clips.py — 從一支 YouTube 影片切出使用者指定的擋拆片段，產生跟既有資料同格式的檔案：
  input_videos/screen/screen_N.mp4（H.264、原 fps）
  input_videos/screen_frames/screen_N/screen_N_0000.jpg …（5fps，給 SAM3 / convert_contact）

用法：
  python nba_data_pipeline/make_new_clips.py --url https://www.youtube.com/watch?v=txo8FuZDPR8 \
      --ranges nba_data_pipeline/new_clips_txo8FuZDPR8.txt --start_id 185
"""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / 'input_videos' / 'yt_src'
CLIPS = ROOT / 'input_videos' / 'screen'
FRAMES = ROOT / 'input_videos' / 'screen_frames'


def to_sec(t):
    s = 0.0
    for part in t.strip().split(':'):
        s = s * 60 + float(part)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--ranges', required=True)
    ap.add_argument('--start_id', type=int, required=True)
    ap.add_argument('--pad', type=float, default=0.0, help='每段前後多留幾秒')
    args = ap.parse_args()

    vid = args.url.split('v=')[-1].split('&')[0]
    for d in (SRC, CLIPS, FRAMES):                  # 新 pod 上這些資料夾都還不存在
        d.mkdir(parents=True, exist_ok=True)
    src = SRC / f'{vid}.mp4'
    if not src.exists():
        subprocess.run(['yt-dlp', '-f', 'bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]',
                        '--merge-output-format', 'mp4', '-o', str(src), args.url], check=True)

    ranges = [l.strip() for l in open(args.ranges, encoding='utf-8')
              if l.strip() and not l.startswith('#')]
    dur = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(src)],
                         capture_output=True, text=True).stdout.strip()
    dur = float(dur) if dur else float('inf')
    print(f'來源影片長度 {dur / 60:.1f} 分鐘')
    log, skipped = [], []
    for k, r in enumerate(ranges):
        a, b = (to_sec(x) for x in r.split('-'))
        a, b = max(0.0, a - args.pad), b + args.pad
        cid = args.start_id + k                       # 編號照時間段順序固定，跳過的也保留編號
        if a >= dur - 0.5:
            skipped.append(f'screen_{cid}（{r}）')
            continue
        b = min(b, dur)
        clip = CLIPS / f'screen_{cid}.mp4'
        fdir = FRAMES / f'screen_{cid}'
        if clip.exists() and fdir.exists() and any(fdir.glob('*.jpg')):
            print(f'screen_{cid} 已存在，跳過（重跑時不重切）')
            log.append(f'screen_{cid},{vid},{r},{len(list(fdir.glob("*.jpg")))}')
            continue
        if clip.exists() or fdir.exists():
            raise SystemExit(f'screen_{cid} 只有影片或只有影格（上次中斷？），請先刪掉再重跑，或換 --start_id')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{a:.2f}', '-i', str(src),
                        '-t', f'{b - a:.2f}', '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
                        '-an', str(clip)], check=True)
        fdir.mkdir(parents=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(clip), '-vf', 'fps=5',
                        '-q:v', '2', '-start_number', '0', str(fdir / f'screen_{cid}_%04d.jpg')], check=True)
        n = len(list(fdir.glob('*.jpg')))
        log.append(f'screen_{cid},{vid},{r},{n}')
        print(f'screen_{cid}：{r}（{b - a:.1f} 秒，{n} 張 5fps 影格）')
    if skipped:
        print(f'⚠ {len(skipped)} 段超過影片長度（{dur:.0f} 秒），沒切：{", ".join(skipped)}'
              f'\n  → 請確認時間段是不是抄錯，或影片沒下載完整（刪掉 {src} 重跑會重新下載）')
    with open(ROOT / 'nba_data_pipeline' / f'new_clips_{vid}.csv', 'w', encoding='utf-8-sig') as fo:
        fo.write('clip,youtube_id,time_range,frames_5fps\n' + '\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
