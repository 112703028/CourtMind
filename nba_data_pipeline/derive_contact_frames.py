"""
derive_contact_frames.py — 從現有 COCO 標註自動算每個掩護的「接觸幀」（ScreenNet 重訓成事件定位用）。

背景：現在的訓練標籤是「窗內任一幀有 screener 就算正樣本」，而 screener 是整段擋拆都框著，
模型學成「是否處在擋拆中」→ 分數是平台。重訓要改成以接觸幀為中心的標籤，這支負責找出接觸幀。

影格：input_videos/screen_frames/screen_N/screen_N_XXXX.jpg（5fps 抽樣，XXXX = 抽樣索引）。
接觸幀以「抽樣索引」記錄（訓練資料就是這個時間尺度），另附秒數 = 索引 / 5。

兩種接觸訊號（同一個 screener 軌跡內各自取一幀）：
  defender：screener 腳底離最近 defender 腳底的距離 / 平均身高 最小的那幀（直接量接觸）
  screen  ：人工標的 `screen` 框（把 screener 和被擋者框在一起）面積 / screener 框面積 最小的那幀
            （兩人貼越近，包起來的框越小）
兩種都有時以 defender 為主，並記錄兩者差幾幀當可信度檢查。

用法：
  python nba_data_pipeline/derive_contact_frames.py                 # 算全部 → contact_frames.json
  python nba_data_pipeline/derive_contact_frames.py --check 20      # 另畫 20 張抽查圖
"""
import os
import re
import json
import random
import argparse
import collections
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).parent.parent
COCO_V2 = ROOT / 'screen_data' / 'screen_v2.coco' / 'train' / '_annotations.coco.json'
COCO_FD = ROOT / 'screen_data' / 'screen_full_defender' / 'train' / '_annotations.coco.json'
FRAMES_DIR = ROOT / 'input_videos' / 'screen_frames'
SAMPLE_FPS = 5.0


def load_coco(path):
    """→ {clip: {frame_idx: {category: [bbox xywh, ...]}}}"""
    c = json.load(open(path, encoding='utf-8'))
    cats = {x['id']: x['name'] for x in c['categories']}
    img = {i['id']: i['file_name'] for i in c['images']}
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for a in c['annotations']:
        m = re.match(r'(screen_\d+)_(\d+)_', img[a['image_id']])
        if not m:
            continue
        out[m.group(1)][int(m.group(2))][cats[a['category_id']]].append(
            [float(v) for v in a['bbox']])
    return out


def foot(b):
    x, y, w, h = b
    return np.array([x + w / 2, y + h])


def link_screeners(frames, max_gap=3, max_dist=0.8):
    """把各幀的 screener 框串成軌跡（沒有 id，用腳底距離/身高 貪婪配對）。
    回 list[track]，track = {frame_idx: bbox}。"""
    tracks = []
    for f in sorted(frames):
        boxes = frames[f].get('screener', [])
        used = set()
        # 先配已存在的軌跡（由近到遠）
        cands = []
        for ti, tr in enumerate(tracks):
            lf = max(tr)
            if f - lf > max_gap or lf == f:
                continue
            lb = tr[lf]
            for bi, b in enumerate(boxes):
                d = np.linalg.norm(foot(b) - foot(lb)) / max(1.0, (b[3] + lb[3]) / 2)
                if d <= max_dist * (f - lf):
                    cands.append((d, ti, bi))
        cands.sort()
        took_t = set()
        for d, ti, bi in cands:
            if ti in took_t or bi in used:
                continue
            tracks[ti][f] = boxes[bi]
            took_t.add(ti)
            used.add(bi)
        for bi, b in enumerate(boxes):
            if bi not in used:
                tracks.append({f: b})
    return tracks


def contact_by_defender(track, frames):
    """screener 腳底離最近 defender 的距離 / 平均身高，取最小那幀。"""
    best = None
    for f, sb in track.items():
        for db in frames[f].get('defender', []):
            d = np.linalg.norm(foot(sb) - foot(db)) / max(1.0, (sb[3] + db[3]) / 2)
            if best is None or d < best[1]:
                best = (f, d, db)
    return best   # (frame, score, defender_box) or None


def contact_by_screen_box(track, frames):
    """包住這個 screener 的 `screen` 框，面積 / screener 面積 最小那幀。"""
    best = None
    for f, sb in track.items():
        sx, sy, sw, sh = sb
        for cb in frames[f].get('screen', []):
            cx, cy, cw, ch = cb
            ix = max(0, min(sx + sw, cx + cw) - max(sx, cx))
            iy = max(0, min(sy + sh, cy + ch) - max(sy, cy))
            if ix * iy < 0.5 * sw * sh:          # 這個 screen 框不是框這個 screener
                continue
            ratio = (cw * ch) / max(1.0, sw * sh)
            if best is None or ratio < best[1]:
                best = (f, ratio, cb)
    return best


def derive(min_track_frames=2):
    v2, fd = load_coco(COCO_V2), load_coco(COCO_FD)
    result = {}
    for clip in sorted(set(v2) | set(fd), key=lambda s: int(s.split('_')[1])):
        frames = fd[clip] if clip in fd else v2[clip]
        # 早期 clip 的 screen 框只在 v2：補進來
        if clip in v2:
            for f, d in v2[clip].items():
                if 'screen' in d and 'screen' not in frames.get(f, {}):
                    frames[f]['screen'] = d['screen']
        screens = []
        for ti, tr in enumerate(link_screeners(frames)):
            if len(tr) < min_track_frames:
                continue
            bd = contact_by_defender(tr, frames)
            bs = contact_by_screen_box(tr, frames)
            if bd is None and bs is None:
                continue
            main = bd if bd is not None else bs
            entry = {
                'track': ti,
                'span': [min(tr), max(tr)],
                'contact_idx': int(main[0]),
                'contact_sec': round(main[0] / SAMPLE_FPS, 2),
                'method': 'defender' if bd is not None else 'screen',
                'score': round(float(main[1]), 3),
                'screener_box': tr[main[0]],
                'partner_box': main[2],
            }
            if bd is not None and bs is not None:
                entry['screen_box_idx'] = int(bs[0])
                entry['agree_diff'] = abs(int(bd[0]) - int(bs[0]))
            screens.append(entry)
        if screens:
            result[clip] = screens
    return result, v2, fd


def draw_check(clip, scr, frames, out_path):
    """抽查圖：接觸幀前一張 / 接觸幀 / 後一張，畫 screener(紅)、接觸對象(藍)、handler(綠)。"""
    c = scr['contact_idx']
    panels = []
    for f in (c - 1, c, c + 1):
        p = FRAMES_DIR / clip / f'{clip}_{f:04d}.jpg'
        im = cv2.imread(str(p))
        if im is None:
            im = np.zeros((720, 1280, 3), np.uint8)
        d = frames.get(f, {})
        for b in d.get('ball_handler', []):
            x, y, w, h = map(int, b); cv2.rectangle(im, (x, y), (x + w, y + h), (0, 200, 0), 3)
        for b in d.get('defender', []):
            x, y, w, h = map(int, b); cv2.rectangle(im, (x, y), (x + w, y + h), (200, 120, 0), 2)
        for b in d.get('screener', []):
            x, y, w, h = map(int, b); cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 230), 3)
        if f == c:
            x, y, w, h = map(int, scr['partner_box'])
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 4)
            x, y, w, h = map(int, scr['screener_box'])
            cv2.rectangle(im, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 255, 255), 2)
        tag = 'CONTACT' if f == c else ('before' if f < c else 'after')
        cv2.rectangle(im, (0, 0), (1280, 50), (0, 0, 0), -1)
        cv2.putText(im, f'{tag}  idx {f}  ({f / SAMPLE_FPS:.1f}s)', (15, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(im, (640, 360)))
    sheet = np.hstack(panels)
    head = np.full((44, sheet.shape[1], 3), 255, np.uint8)
    info = (f"{clip}  track{scr['track']}  span idx {scr['span'][0]}-{scr['span'][1]}  "
            f"contact idx {c}  method={scr['method']}  score={scr['score']}")
    if 'agree_diff' in scr:
        info += f"  (screen-box idx {scr['screen_box_idx']}, diff {scr['agree_diff']})"
    cv2.putText(head, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([head, sheet]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).parent / 'contact_frames.json'))
    ap.add_argument('--check', type=int, default=0, help='畫幾張抽查圖')
    ap.add_argument('--check_dir', default=str(ROOT / 'output_videos' / 'contact_check'))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    result, v2, fd = derive()
    json.dump(result, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    n_scr = sum(len(v) for v in result.values())
    by_m = collections.Counter(s['method'] for v in result.values() for s in v)
    diffs = [s['agree_diff'] for v in result.values() for s in v if 'agree_diff' in s]
    print(f"{len(result)} 支 clip、{n_scr} 個掩護 → {args.out}")
    print(f"  方法：{dict(by_m)}")
    if diffs:
        diffs = np.array(diffs)
        print(f"  兩法都有 {len(diffs)} 個：差 0 幀 {np.mean(diffs == 0):.0%}、"
              f"≤1 幀 {np.mean(diffs <= 1):.0%}、≤2 幀 {np.mean(diffs <= 2):.0%}、"
              f"中位數 {np.median(diffs):.0f}")
    per_clip = collections.Counter(len(v) for v in result.values())
    print(f"  每支 clip 掩護數分布：{dict(sorted(per_clip.items()))}")

    if args.check:
        os.makedirs(args.check_dir, exist_ok=True)
        rng = random.Random(args.seed)
        pool = [(clip, s) for clip, v in result.items() for s in v]
        # 兩種方法都抽到：一半 defender、一半 screen
        dfs = [p for p in pool if p[1]['method'] == 'defender']
        scs = [p for p in pool if p[1]['method'] == 'screen']
        pick = rng.sample(dfs, min(len(dfs), args.check - args.check // 3)) + \
               rng.sample(scs, min(len(scs), args.check // 3))
        for clip, s in pick:
            frames = fd[clip] if clip in fd else v2[clip]
            draw_check(clip, s, frames,
                       Path(args.check_dir) / f"{clip}_trk{s['track']}_{s['method']}.jpg")
        print(f"  抽查圖 {len(pick)} 張 → {args.check_dir}")


if __name__ == '__main__':
    main()
