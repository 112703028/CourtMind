"""
ids_framenum_video.py — 原片 + 每個球員的框 / 編號（批次推論的 player_tracks，跟傳球 / 持球者偵測同一套編號）+ 左上角影格號碼與秒數。
顏色：進攻紅、防守藍、持球者金（ball_acquisition）、沒分隊灰。給人工標傳球 / 追蹤錯誤用。

用法：python nba_data_pipeline/ids_framenum_video.py [--only ext] → output_videos/ids_frames/<名稱>.mp4
"""
import sys
import glob
import pickle
import argparse
import collections
from pathlib import Path

import cv2

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]


def render(pkl, video, out, label=True, src_size=None):
    d = pickle.load(open(pkl, 'rb'))
    tracks, pa, ba, fps = d['player_tracks'], d['player_assignment'], d.get('ball_acquisition') or [], d.get('fps', 30.0)
    cnt = collections.defaultdict(collections.Counter)            # 整段多數決分隊（1 = 進攻）
    for fd in pa:
        for p, t in fd.items():
            cnt[p][t] += 1
    team = {p: c.most_common(1)[0][0] for p, c in cnt.items()}
    cap = cv2.VideoCapture(str(video))
    sx = sy = 1.0
    if src_size is not None:                     # 追蹤框是原片解析度（例 1920×1080），畫在縮小過的影片上要等比例縮
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH); h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        sx, sy = w / src_size[0], h / src_size[1]
    frames = []
    f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        lw = max(2, im.shape[0] // 360)
        H = ba[f] if f < len(ba) and ba[f] not in (None, -1) else None
        for p, bb in (tracks[f].items() if f < len(tracks) else []):
            col = (0, 200, 255) if p == H else (60, 60, 220) if team.get(p) == 1 else (200, 130, 40) if team.get(p) == 2 else (150, 150, 150)
            x1, y1, x2, y2 = int(bb[0] * sx), int(bb[1] * sy), int(bb[2] * sx), int(bb[3] * sy)
            cv2.rectangle(im, (x1, y1), (x2, y2), col, lw)
            t = f'#{p}'
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * lw, lw)
            cv2.rectangle(im, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
            cv2.putText(im, t, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * lw, (255, 255, 255), lw, cv2.LINE_AA)
        if label:
            txt = f'frame {f}   {f / fps:.2f}s'
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * lw, lw)
            cv2.rectangle(im, (8, 8), (8 + tw + 16, 8 + th + 18), (0, 0, 0), -1)
            cv2.putText(im, txt, (16, 8 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * lw, (255, 255, 255), lw, cv2.LINE_AA)
        frames.append(im)
        f += 1
    from utils import save_video
    save_video(frames, str(out), fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['ext', 'orig', 'all'], default='ext')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--framenum', action='store_true',
                    help='畫在 output_videos/framenum/<片段>_fn.mp4 上（已經印好影格號碼，不再加字）→ output_videos/framenum_boxes/')
    args = ap.parse_args()
    if args.framenum:
        out = Path(args.out_dir or ROOT / 'output_videos' / 'framenum_boxes'); out.mkdir(parents=True, exist_ok=True)
        for v in sorted((ROOT / 'output_videos' / 'framenum').glob('screen_*_fn.mp4'), key=lambda q: int(q.stem.split('_')[1])):
            clip = v.stem.replace('_fn', '')
            pk = ROOT / 'output_videos' / 'batch' / f'{clip}_events.pkl'
            orig = ROOT / 'input_videos' / 'screen' / f'{clip}.mp4'
            if pk.exists() and orig.exists() and not (out / f'{clip}.mp4').exists():
                oc = cv2.VideoCapture(str(orig))
                size = (oc.get(cv2.CAP_PROP_FRAME_WIDTH), oc.get(cv2.CAP_PROP_FRAME_HEIGHT)); oc.release()
                render(pk, v, out / f'{clip}.mp4', label=False, src_size=size)
        print(f'→ {out}（{len(list(out.glob("*.mp4")))} 支）')
        return
    args.out_dir = args.out_dir or str(ROOT / 'output_videos' / 'ids_frames')
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for pk in sorted(glob.glob(str(ROOT / 'output_videos' / 'batch' / '*_events.pkl'))):
        name = Path(pk).stem.replace('_events', '')
        ext = name.startswith('ext_')
        if (args.only == 'ext' and not ext) or (args.only == 'orig' and ext):
            continue
        clip = name.replace('ext_', '')
        video = ROOT / 'input_videos' / ('screen_ext' if ext else 'screen') / f'{clip}.mp4'
        if video.exists() and not (out / f'{name}.mp4').exists():
            render(pk, video, out / f'{name}.mp4')
    print(f'→ {out}（{len(list(out.glob("*.mp4")))} 支）')


if __name__ == '__main__':
    main()
