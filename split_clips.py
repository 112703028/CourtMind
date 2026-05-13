"""
split_clips.py

將指定 clip 的特定 frame 範圍切成新的子 clip。
格式：clip_id: [(hi, lo), ...] → 複製 frame lo~hi 到新的 screen_{N} 資料夾
"""

import re, shutil
from pathlib import Path

FRAMES_DIR = Path('input_videos/screen_frames')

SPLITS = {
    166: [(45, 35), (29, 24), (17, 10), (8, 2)],
    165: [(40, 21), (9, 2)],
    160: [(20, 10), (9, 0)],
    158: [(19, 10), (8, 2)],
    154: [(49, 43), (21, 10)],
    148: [(37, 21), (15, 5)],
    140: [(34, 23), (18, 0)],
    139: [(14, 6), (31, 18)],
}

# 找現有最大 clip 號碼
max_n = max(
    int(re.search(r'screen_(\d+)', d.name).group(1))
    for d in FRAMES_DIR.iterdir()
    if d.is_dir() and re.match(r'screen_\d+', d.name)
)
new_id = max_n + 1
print(f"Max existing clip: screen_{max_n}, new clips start from screen_{new_id}\n")

for clip_id, ranges in SPLITS.items():
    src_dir = FRAMES_DIR / f'screen_{clip_id}'
    if not src_dir.exists():
        print(f"WARNING: {src_dir} not found, skipping")
        continue

    for a, b in ranges:
        lo, hi = min(a, b), max(a, b)
        frame_files = sorted(
            [f for f in src_dir.glob('*.jpg')
             if lo <= int(re.search(r'_(\d{4})\.jpg$', f.name).group(1)) <= hi],
            key=lambda f: int(re.search(r'_(\d{4})\.jpg$', f.name).group(1))
        )

        if not frame_files:
            print(f"  screen_{clip_id} frames {lo}-{hi}: no frames found, skipping")
            continue

        dst_dir = FRAMES_DIR / f'screen_{new_id}'
        dst_dir.mkdir(exist_ok=True)

        for i, src_file in enumerate(frame_files):
            shutil.copy2(src_file, dst_dir / f'screen_{new_id}_{i:04d}.jpg')

        print(f"  screen_{clip_id} frames {lo:04d}-{hi:04d} -> screen_{new_id} ({len(frame_files)} frames)")
        new_id += 1

print(f"\nDone. Created clips screen_{max_n+1} ~ screen_{new_id-1}")
