"""
offball_candidates.py — 無球掩護標記累積：規則（offball_screens.py）找候選 → 使用者確認 → offball_labels.json。

  python nba_data_pipeline/offball_candidates.py make  --out output_videos/offball_check2   # 產生一批候選（排除已判斷 / 已標記的）
  python nba_data_pipeline/offball_candidates.py merge                                      # 把所有 offball_check*/offball_check.csv 的答案併進標記檔

候選圖：撞上前 2 張 / 撞上 / 後 2 張（5fps），紅 = 掩護者、黃 = 切入者（被掩護的人）、藍 = 被擋的防守者、綠 = 持球者。
表格「是無球掩護嗎」填 t / f / ?；紅黃標反、時間不對等寫在備註。

offball_labels.json：{clip: [{idx, screener, cutter, defender, verdict: t/f/?, note, src}]}
  verdict=t 的是無球掩護正例；f 是規則抓錯的負例（之後訓練模型可以當困難負例）；
  另外收舊的「無球檔」標記（只有時間，screener=None）和 2026-11-04 從持球擋拆標記拿掉的 5 個。
"""
import sys
import csv
import glob
import json
import random
import pickle
import argparse
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from convert_contact import CACHE_DIR, FRAMES_DIR
import offball_screens as O

LABELS = HERE / 'offball_labels.json'
CHECKS = str(ROOT / 'output_videos' / 'offball_check*' / 'offball_check.csv')
VERDICT_COL = '是無球掩護嗎(是/否/不確定)'


def read_checks():
    """所有已填的候選表 → [(clip, idx, S, C, D, verdict, note, 檔案)]。"""
    out = []
    for p in sorted(glob.glob(CHECKS)):
        for r in csv.DictReader(open(p, encoding='utf-8-sig')):
            v = (r.get(VERDICT_COL) or '').strip().lower()
            v = {'是': 't', '否': 'f', '不確定': '?', 'y': 't', 'n': 'f'}.get(v, v)
            num = lambda k: int(r[k].lstrip('#')) if r.get(k, '').lstrip('#').isdigit() else None
            out.append((r['clip'], int(r['撞上5fps影格']), num('掩護者(紅)'), num('切入者(黃)'), num('被擋的防守者(藍)'),
                        v, r.get('備註', ''), Path(p).parent.name))
    return out


def make(out_dir, n, seed):
    import cv2
    import numpy as np
    import finetune_seq
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen = {(c, i) for c, i, *_ in read_checks()}
    lab = O.eval_set()
    dets = []
    for p in sorted(glob.glob(str(CACHE_DIR / 'screen_*.pkl'))):
        clip = Path(p).stem
        if int(clip.split('_')[1]) > 503:
            continue
        c = pickle.load(open(p, 'rb'))
        if c.get('pending'):
            continue
        ct = finetune_seq.clip_coords(c)
        for e in O.keep(O.detect(ct, c['track_team'], c['handler_tid'])):
            if e['onball'] or any(abs(e['t'] - i) <= 5 for i, _, _ in lab.get(clip, [])) or \
                    any(cl == clip and abs(e['t'] - i) <= 2 for cl, i in seen):
                continue
            dets.append((clip, e, c['handler_tid']))
    random.Random(seed).shuffle(dets)
    dets = dets[:n] if n else dets
    rows = []
    for k, (clip, e, H) in enumerate(dets, 1):
        c = pickle.load(open(CACHE_DIR / f'{clip}.pkl', 'rb'))
        tiles = []
        for f in (e['t'] - 2, e['t'], e['t'] + 2):
            f = min(max(f, 0), c['n'] - 1)
            im = cv2.imread(str(FRAMES_DIR / clip / f'{clip}_{f:04d}.jpg'))
            for pid, bb in c['raw_tracks'][f].items():
                col = {H: (0, 255, 0), e['defender']: (255, 128, 0), e['cutter']: (0, 255, 255), e['screener']: (0, 0, 255)}.get(pid)
                if col is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bb]
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 4)
                cv2.putText(im, f'#{pid}', (x1, max(25, y1 - 8)), 0, 1.1, col, 3)
            cv2.putText(im, f'{clip} f{f}', (10, 45), 0, 1.4, (255, 255, 255), 4)
            tiles.append(cv2.resize(im, (640, 360)))
        name = f'B{k:03d}_{clip}_f{e["t"]}.jpg'
        cv2.imwrite(str(out / name), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append([f'B{k:03d}', clip, e['t'], f"#{e['screener']}", f"#{e['cutter']}", f"#{e['defender']}", f'#{H}', name, '', ''])
    with open(out / 'offball_check.csv', 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['編號', 'clip', '撞上5fps影格', '掩護者(紅)', '切入者(黃)', '被擋的防守者(藍)', '持球者(綠)', '圖檔', VERDICT_COL, '備註'])
        w.writerows(rows)
    print(f'候選 {len(rows)} 個 → {out}')


def merge():
    labels = {}

    def add(clip, item):
        lst = labels.setdefault(clip, [])
        for x in lst:
            if abs(x['idx'] - item['idx']) <= 2 and (x['screener'] in (None, item['screener']) or item['screener'] is None):
                if x['verdict'] != 't' and item['verdict'] == 't' or x['screener'] is None:
                    x.update({k: v for k, v in item.items() if v is not None})
                return
        lst.append(item)
    for clip, items in O.eval_set().items():
        for idx, S, src in items:
            add(clip, dict(idx=idx, screener=S, cutter=None, defender=None, verdict='t', note='', src=src))
    for clip, idx, S, C, D, v, note, src in read_checks():
        if v:
            add(clip, dict(idx=idx, screener=S, cutter=C, defender=D, verdict=v, note=note, src=src))
    json.dump(labels, open(LABELS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    vs = [x['verdict'] for lst in labels.values() for x in lst]
    print(f'{LABELS.name}：無球掩護 {vs.count("t")} 個、規則抓錯（負例）{vs.count("f")} 個、不確定 {vs.count("?")} 個（{len(labels)} 支）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['make', 'merge'])
    ap.add_argument('--out', default=str(ROOT / 'output_videos' / 'offball_check2'))
    ap.add_argument('--n', type=int, default=0, help='最多幾個（0 = 全部）')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    make(args.out, args.n, args.seed) if args.cmd == 'make' else merge()


if __name__ == '__main__':
    main()
