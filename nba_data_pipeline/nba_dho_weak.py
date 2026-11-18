"""
nba_dho_weak.py — 用 NBA 公開追蹤資料（nba-movement-data，SportVU 25fps，有球的位置）產生手遞手弱標籤（2026-11-18）。

照 NETS（Hauri & Vucetic 2022，Supplementary A）的規則：
  持球者：離球最近、≤ 5 呎、球高 ≤ 10 呎、球速 ≤ 25 呎/秒、連續 ≥ 5 格（沿用 extract_sequences.find_ball_handler）
  手遞手：持球在兩名同隊進攻球員之間換手，換手時兩人 < 6.5 呎（NETS 人工看 200 個，90.5% 是真的）
另外要求球沒人拿的時間 ≤ max_air 秒（NETS 正文：short time of ball transition）、兩人都持球 ≥ min_hold 秒、1 秒內沒換回去
（不加的話每場 104 個，多是擋拆時持球者在兩個靠很近的隊友之間閃爍；NETS 632 場 15,251 段 ≈ 每場 24）。
同一場比賽的 events 會重疊（同一段 moments 出現在好幾個 event）→ 用（節, 比賽時間）去重。

輸出 stubs/nba_dho/<比賽>.json：[{period, game_clock, shot_clock, event_id, idx, giver, receiver, gap_s, dist_ft, team}]
之後 render 抽 n 個畫成戰術板動畫給人看（output_videos/nba_dho_check/）。

用法：python nba_data_pipeline/nba_dho_weak.py extract [--games 50] [--workers 8]
     python nba_data_pipeline/nba_dho_weak.py stats
     python nba_data_pipeline/nba_dho_weak.py render [--n 30]
"""
import os
import sys
import json
import glob
import random
import argparse
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from extract_sequences import parse_moment, find_ball_handler, dist, FPS

DATA = ROOT / 'nba-movement-data' / 'data'
OUT = ROOT / 'stubs' / 'nba_dho'
CHECK = ROOT / 'output_videos' / 'nba_dho_check'
HANDOFF_DIST = 6.5        # NETS：換手時兩人 < 6.5 呎（NBA 平均臂展約 6'6"）


def load_game(name):
    """→ game json（已解壓的資料夾優先，否則讀 .7z）。"""
    d = DATA / name
    js = list(d.glob('*.json')) if d.is_dir() else []
    if js:
        return json.load(open(js[0]))
    import py7zr
    with tempfile.TemporaryDirectory() as tmp:
        with py7zr.SevenZipFile(DATA / f'{name}.7z', mode='r') as z:
            z.extractall(tmp)
        f = [x for x in os.listdir(tmp) if x.endswith('.json')][0]
        return json.load(open(os.path.join(tmp, f)))


def handoffs_in_event(ev, max_air=0.5, min_hold=0.4, pingpong=1.0):
    moms = [m for m in ev.get('moments', []) if len(m[5]) == 11]       # 球 + 10 人都在
    if len(moms) < 20:
        return []
    pms = [parse_moment(m) for m in moms]
    home, vis = ev['home']['teamid'], ev['visitor']['teamid']
    hs = find_ball_handler(pms, home, vis)
    # 持球段
    runs, cur = [], None
    for i, h in enumerate(hs):
        if h is not None and cur is not None and cur[0] == h and i - cur[2] <= 1:
            cur[2] = i
        elif h is not None:
            if cur:
                runs.append(cur)
            cur = [h, i, i]
    if cur:
        runs.append(cur)
    out = []
    for j, ((A, a1, b1), (B, a2, b2)) in enumerate(zip(runs, runs[1:])):
        if A[0] == B[0] or A[1] != B[1]:
            continue                                  # 同一人 / 不同隊（抄截）
        # 兩人都要真的拿住球（擋拆時掩護者站在持球者旁邊，「離球最近」會短暫跳過去 → 假換手；試跑 2 場每場 104 個、持球 10% 只有 0.12 秒）
        if (b1 - a1 + 1) < min_hold * FPS or (b2 - a2 + 1) < min_hold * FPS:
            continue
        if j + 2 < len(runs) and runs[j + 2][0] == A and (runs[j + 2][1] - b2) / FPS <= pingpong:
            continue                                  # 1 秒內又回到 A = 閃爍
        gap = (a2 - b1 - 1) / FPS
        if gap > max_air:
            continue
        # 換手那幾格（A 最後一格 ~ B 第一格，B 的第一格往回推 4 格 = 連續 5 格才算持球的起點）
        lo, hi = max(0, b1 - 2), min(len(pms) - 1, a2 + 2)
        ds = [dist(pms[i]['players'][A[0]]['pos'], pms[i]['players'][B[0]]['pos']) for i in range(lo, hi + 1)
              if A[0] in pms[i]['players'] and B[0] in pms[i]['players']]
        if not ds or min(ds) >= HANDOFF_DIST:
            continue
        k = (b1 + a2) // 2
        m = moms[k]
        out.append(dict(period=m[0], game_clock=round(m[2], 2), shot_clock=m[3], event_id=ev['eventId'], idx=k,
                        giver=A[0], receiver=B[0], team=A[1], gap_s=round(max(gap, 0), 2), dist_ft=round(min(ds), 1),
                        hold_a_s=round((b1 - a1 + 1) / FPS, 2), hold_b_s=round((b2 - a2 + 1) / FPS, 2)))
    return out


def process(name):
    o = OUT / f'{name}.json'
    if o.exists():
        return name, None
    try:
        g = load_game(name)
    except Exception as e:
        return name, f'讀取失敗 {e}'
    seen, res = set(), []
    for ev in g['events']:
        for h in handoffs_in_event(ev):
            key = (h['period'], round(h['game_clock']))
            if key not in seen:
                seen.add(key); res.append(h)
    json.dump({'gameid': g.get('gameid'), 'handoffs': res}, open(o, 'w'))
    return name, len(res)


def games():
    names = sorted({p.stem for p in DATA.glob('*.7z')} | {p.name for p in DATA.iterdir() if p.is_dir()})
    return names


def extract(n_games, workers):
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    gs = games()
    random.Random(0).shuffle(gs)
    gs = gs[:n_games]
    with Pool(workers) as pool:
        for k, (name, r) in enumerate(pool.imap_unordered(process, gs), 1):
            print(f'{k}/{len(gs)} {name} {r}', flush=True)


def stats():
    tot, per, gaps, ds, ha, hb = 0, [], [], [], [], []
    for f in OUT.glob('*.json'):
        h = json.load(open(f))['handoffs']
        per.append(len(h)); tot += len(h)
        gaps += [x['gap_s'] for x in h]; ds += [x['dist_ft'] for x in h]
        ha += [x['hold_a_s'] for x in h]; hb += [x['hold_b_s'] for x in h]
    q = lambda v: np.percentile(v, [10, 50, 90]).round(2).tolist() if v else []
    print(f'{len(per)} 場、手遞手弱標籤 {tot} 個（每場 {np.mean(per):.1f}，{min(per)}~{max(per)}）')
    print(f'  球沒人拿（秒）10/50/90%：{q(gaps)}；兩人最近（呎）：{q(ds)}；給球者持球（秒）：{q(ha)}；接球者持球（秒）：{q(hb)}')


def render(n=30, seed=0, win_s=2.0):
    """抽 n 個畫成戰術板動畫（換手前後 win_s 秒）：進攻紅、防守藍、球黑；給球者 A、接球者 B 標大字。"""
    import cv2
    CHECK.mkdir(parents=True, exist_ok=True)
    pool = [(f.stem, h) for f in sorted(OUT.glob('*.json')) for h in json.load(open(f))['handoffs']]
    pick = random.Random(seed).sample(pool, min(n, len(pool)))
    W, H, S = 940, 500, 10                         # 1 呎 = 10 像素
    rows = []
    cache = {}
    for k, (name, h) in enumerate(pick, 1):
        if name not in cache:
            cache.clear(); cache[name] = load_game(name)
        ev = next(e for e in cache[name]['events'] if e['eventId'] == h['event_id'])
        moms = [m for m in ev['moments'] if len(m[5]) == 11]
        a, b = max(0, h['idx'] - int(win_s * FPS)), min(len(moms), h['idx'] + int(win_s * FPS))
        vw = cv2.VideoWriter(str(CHECK / f'H{k:02d}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H + 40))
        for i in range(a, b):
            im = np.full((H + 40, W, 3), 235, np.uint8)
            cv2.rectangle(im, (0, 0), (W - 1, H - 1), (60, 60, 60), 2)
            cv2.line(im, (W // 2, 0), (W // 2, H), (60, 60, 60), 1)
            for bx in (52, W - 52):                # 籃框（離底線 5.25 呎）
                cv2.circle(im, (bx, H // 2), 9, (0, 120, 255), 2)
            for side in (0, W):                    # 三分線（簡化成 23.75 呎半徑）
                cv2.ellipse(im, (abs(side - 52), H // 2), (237, 237), 0, -90 if side == 0 else 90, 90 if side == 0 else 270, (120, 120, 120), 1)
            pm = parse_moment(moms[i])
            for pid, p in pm['players'].items():
                x, y = int(p['pos'][0] * S), int(p['pos'][1] * S)
                col = (40, 40, 220) if p['team'] == h['team'] else (220, 120, 40)
                cv2.circle(im, (x, y), 11, col, -1)
                tag = 'A' if pid == h['giver'] else ('B' if pid == h['receiver'] else '')
                if tag:
                    cv2.circle(im, (x, y), 15, (0, 0, 0), 2)
                    cv2.putText(im, tag, (x - 8, y - 18), 0, 0.9, (0, 0, 0), 2)
            bx, by = int(pm['ball']['pos'][0] * S), int(pm['ball']['pos'][1] * S)
            cv2.circle(im, (bx, by), 6, (0, 0, 0), -1)
            t = (i - h['idx']) / FPS
            cv2.putText(im, f'H{k:02d}  {t:+.1f}s {"<-- handoff" if abs(t) < 0.2 else ""}', (10, H + 28), 0, 0.7, (0, 0, 0), 2)
            vw.write(im)
        vw.release()
        rows.append([f'H{k:02d}', name, h['period'], h['game_clock'], h['gap_s'], h['dist_ft'], '', ''])
        print(f'{k}/{len(pick)}', flush=True)
    import csv
    with open(CHECK / 'nba_dho_check.csv', 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['編號', '比賽', '節', '比賽時間(秒)', '球沒人拿(秒)', '兩人最近(呎)', '手遞手嗎(t/f/?)', '備註'])
        w.writerows(rows)
    print(f'{len(rows)} 個 → {CHECK}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['extract', 'stats', 'render'])
    ap.add_argument('--games', type=int, default=50)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    {'extract': lambda: extract(a.games, a.workers), 'stats': stats, 'render': lambda: render(a.n)}[a.cmd]()
