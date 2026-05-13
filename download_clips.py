"""
download_clips.py

讀取 period.txt，下載 YouTube 影片並切出每個片段。
輸出：input_videos/screen_{N}.mp4（接在現有編號後面）

需要：yt-dlp、ffmpeg
"""

import subprocess, re, sys
from pathlib import Path


def parse_period_file(path):
    entries = []
    current_url = None
    current_segments = []

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_url and current_segments:
                    entries.append((current_url, current_segments))
                    current_url = None
                    current_segments = []
                continue

            if line.startswith('http'):
                if current_url and current_segments:
                    entries.append((current_url, current_segments))
                current_url = line
                current_segments = []
            else:
                line = line.replace(';', ':')   # 修正打錯的分號
                m = re.match(r'(\d+:\d+)-(\d+:\d+)', line)
                if m:
                    current_segments.append((m.group(1), m.group(2)))

    if current_url and current_segments:
        entries.append((current_url, current_segments))

    return entries


def to_seconds(t):
    parts = t.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def main():
    period_file = Path('period.txt')
    output_dir  = Path('input_videos')
    output_dir.mkdir(exist_ok=True)

    entries = parse_period_file(period_file)
    total_segs = sum(len(s) for _, s in entries)
    print(f"Found {len(entries)} videos, {total_segs} segments total\n")

    # 找現有最大 screen 編號（看 screen_frames 目錄）
    frames_dir = Path('input_videos/screen_frames')
    max_n = 0
    if frames_dir.exists():
        for d in frames_dir.iterdir():
            m = re.search(r'screen_(\d+)', d.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    # 也看既有的 mp4
    for f in output_dir.glob('screen_*.mp4'):
        m = re.search(r'screen_(\d+)\.mp4', f.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    clip_n = max_n + 1
    print(f"Starting from screen_{clip_n}.mp4\n")

    for url, segments in entries:
        video_id = re.search(r'v=([^&]+)', url).group(1)
        tmp_path = output_dir / f'_tmp_{video_id}.mp4'

        print(f"Downloading {url}")
        result = subprocess.run([
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',
            '-o', str(tmp_path),
            '--no-playlist',
            url
        ])

        if result.returncode != 0 or not tmp_path.exists():
            print(f"  ERROR: download failed, skipping\n")
            continue

        print(f"  Downloaded. Cutting {len(segments)} segments...")

        for start, end in segments:
            start_sec = to_seconds(start)
            end_sec   = to_seconds(end)
            duration  = end_sec - start_sec
            out_path  = output_dir / f'screen_{clip_n}.mp4'

            result = subprocess.run([
                'ffmpeg', '-y',
                '-ss', str(start_sec),
                '-i', str(tmp_path),
                '-t', str(duration),
                '-c:v', 'libx264', '-crf', '18',
                '-c:a', 'aac',
                '-loglevel', 'error',
                str(out_path)
            ])

            if result.returncode == 0:
                print(f"  OK screen_{clip_n}.mp4  ({start} - {end})")
            else:
                print(f"  FAIL screen_{clip_n}.mp4  ({start} - {end})")

            clip_n += 1

        tmp_path.unlink(missing_ok=True)
        print()

    print(f"Done. {clip_n - max_n - 1} clips saved to {output_dir}/")


if __name__ == '__main__':
    main()
