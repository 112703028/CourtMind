"""
handler_audit.py — 批次推論（output_videos/batch/*_events.pkl）的持球者檢查：抓錯人 + 一直跳。

每支片段：
  ① 抓錯人：在你標的每個擋拆撞上那刻（±0.5 秒多數決），推論的持球者（ball_acquisition）是不是你標的持球者。
     兩次 SAM3 編號不同 → 先用框重疊（同一張原片影格 IoU ≥ 0.5，整段多數決）把訓練快取的編號對到批次的編號。
  ② 一直跳：持球者換人次數、短段（< 0.5 秒、前後都是別人）次數、A→B→A 來回（1 秒內又換回來）次數。
     真的傳球通常會拿一陣子；半秒內換來換去幾乎都是偵測在跳。
  ③ 沒持球者的影格比例（handler_detector 抓不到）。
輸出：output_videos/batch/handler_audit.csv + 統計。

用法：python nba_data_pipeline/handler_audit.py
"""
import sys
import csv
import glob
import pickle
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from convert_contact import CACHE_DIR, iou

BATCH = ROOT / 'output_videos' / 'batch'
SHORT_S, FLICKER_S = 0.5, 1.0


def runs(seq):
    """[(pid, 起, 迄)]，pid None = 沒持球者。"""
    out = []
    for f, p in enumerate(seq):
        p = None if p in (None, -1) else p
        if out and out[-1][0] == p:
            out[-1][2] = f
        else:
            out.append([p, f, f])
    return out


def jump_stats(ba, fps):
    rs = [r for r in runs(ba) if r[0] is not None]
    switches = sum(1 for a, b in zip(rs, rs[1:]) if a[0] != b[0])
    short = sum(1 for i, (p, a, b) in enumerate(rs) if (b - a + 1) < SHORT_S * fps and 0 < i < len(rs) - 1
                and rs[i - 1][0] != p and rs[i + 1][0] != p)
    flick = sum(1 for x, y, z in zip(rs, rs[1:], rs[2:]) if x[0] == z[0] != y[0] and (z[1] - x[2]) < FLICKER_S * fps)
    none = sum(1 for p in ba if p in (None, -1)) / max(1, len(ba))
    return switches, short, flick, none


def map_ids(cache, tracks, fps):
    """訓練快取編號 → 批次編號（同一張影格 IoU ≥ 0.5 多數決）。"""
    votes = collections.defaultdict(collections.Counter)
    step = fps / 5.0
    for i, fd in enumerate(cache['raw_tracks']):
        f = int(round(i * step))
        if f >= len(tracks):
            break
        for p, bb in fd.items():
            best = max(((iou(bb, b2), q) for q, b2 in tracks[f].items()), default=(0, None))
            if best[0] >= 0.5:
                votes[p][best[1]] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


def main():
    rows = []
    for pk in sorted(glob.glob(str(BATCH / '*_events.pkl'))):
        name = Path(pk).stem.replace('_events', '')
        clip = name.replace('ext_', '')
        d = pickle.load(open(pk, 'rb'))
        fps, ba = d.get('fps', 30.0), d.get('ball_acquisition') or []
        sw, short, flick, none = jump_stats(ba, fps)
        row = dict(name=name, frames=len(ba), switches=sw, short_runs=short, flicker=flick, no_handler=round(none, 2),
                   label_H='', mapped_H='', screens=0, wrong_at_screen=0, none_at_screen=0, other_at_screen='', unmapped='')
        cp = CACHE_DIR / f'{clip}.pkl'
        if cp.exists() and d.get('player_tracks'):
            c = pickle.load(open(cp, 'rb'))
            if not c.get('pending') and c.get('handler_tid') is not None:
                m = map_ids(c, d['player_tracks'], fps)
                H = m.get(c['handler_tid'])
                row.update(label_H=c['handler_tid'], mapped_H='' if H is None else H, unmapped='' if H is not None else 'y')
                step = fps / 5.0
                for e in c['events']:
                    if e['tid'] is None:
                        continue
                    f = int(round(e['idx'] * step))
                    w = [ba[g] for g in range(max(0, f - int(SHORT_S * fps)), min(len(ba), f + int(SHORT_S * fps) + 1))
                         if ba[g] not in (None, -1)]
                    got = collections.Counter(w).most_common(1)[0][0] if w else None
                    row['screens'] += 1
                    row['wrong_at_screen'] += (H is None or got != H)
                    if H is not None and got is None:
                        row['none_at_screen'] += 1                  # 撞上前後半秒都沒抓到持球者
                    elif H is not None and got != H:
                        row['other_at_screen'] += f'{got} '         # 抓成別人
        rows.append(row)
    with open(BATCH / 'handler_audit.csv', 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    lab = [r for r in rows if r['screens'] and not r['unmapped']]
    wrong = [r for r in lab if r['wrong_at_screen']]
    jumpy = [r for r in rows if r['short_runs'] >= 2 or r['flicker'] >= 1]
    print(f'片段 {len(rows)} 支（延長 {sum(r["name"].startswith("ext_") for r in rows)}、原片 {sum(not r["name"].startswith("ext_") for r in rows)}）')
    print(f'\n① 抓錯人（有標記、對得到編號的 {len(lab)} 支；對不到編號 {sum(1 for r in rows if r["unmapped"])} 支）：')
    ns = sum(r['screens'] for r in lab); nw = sum(r['wrong_at_screen'] for r in lab)
    nn = sum(r['none_at_screen'] for r in lab)
    print(f'   擋拆撞上那刻持球者不對：{nw}/{ns} 個擋拆（{nw / max(1, ns):.0%}）、{len(wrong)} 支片段'
          f'——其中抓成別人 {nw - nn}、前後半秒完全沒抓到持球者 {nn}')
    print(f'\n② 一直跳（短段 ≥ 2 或 1 秒內來回 ≥ 1）：{len(jumpy)}/{len(rows)} 支（{len(jumpy) / len(rows):.0%}）')
    print(f'   換人次數 中位數 {np.median([r["switches"] for r in rows]):.0f}、短段 中位數 {np.median([r["short_runs"] for r in rows]):.0f}、'
          f'來回 中位數 {np.median([r["flicker"] for r in rows]):.0f}；最多跳的：'
          + '、'.join(f"{r['name']}（換 {r['switches']}、來回 {r['flicker']}）" for r in sorted(rows, key=lambda r: -r['flicker'])[:8]))
    print(f'\n③ 沒持球者影格比例 中位數 {np.median([r["no_handler"] for r in rows]):.0%}；超過一半沒持球者 {sum(r["no_handler"] > 0.5 for r in rows)} 支')
    both = [r for r in wrong if r in jumpy]
    print(f'\n抓錯人且一直跳 {len(both)} 支；只抓錯 {len(wrong) - len(both)}；只跳 {len([r for r in jumpy if r not in wrong])}')
    print(f'→ {BATCH / "handler_audit.csv"}')


if __name__ == '__main__':
    main()
