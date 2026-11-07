"""
dho.py — 手遞手（DHO，dribble hand-off）候選（2026-11-06）。

pose_pass.passes() 故意丟掉「A 放球時 B 的手已經在球上」的換手（sep），手遞手正好長這樣 → 這裡反過來撿：
用同一套手腕持球段（10fps），A 段 → B 段（A ≠ B、同隊、B 持球 ≥ hold 張），而且交接很近：
  球在空中 ≤ max_air 秒，交接那幾張兩人框中心距離 ≤ max_gap 倍身高。
每個候選附特徵（兩手同時在球上的程度、A / B 速度、A 先持球多久）給人工檢查表用，門檻之後再用標記調。

用法：python nba_data_pipeline/dho.py               # 種子（使用者備註的 3 個）有沒有抓到 + 候選數
     python nba_data_pipeline/dho.py sheet 26 166  # → output_videos/framenum_boxes/dho_check.csv
"""
import sys
import csv
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from shot_detect import map_ids
from possession_sam3 import clip_teams
import pose_pass as PP

# 使用者在 pass_check_pose.csv 備註寫到的手遞手（片段、大約影格、給球者、接球者；編號 = framenum_boxes 框上的）
SEEDS = [(155, 8, 7, 10), (160, 60, 1, 10), (162, 130, 1, 8)]


def _speed(cc, pid, f, fps, half=0.5):
    """場上速度（呎 / 秒）：f 前後 half 秒的位移；沒座標 → None。"""
    a, b = int(f - half * fps), int(f + half * fps)
    a, b = max(a, 0), min(b, len(cc) - 1)
    pa = next((cc[i][pid] for i in range(a, f + 1) if pid in cc[i]), None)
    pb = next((cc[i][pid] for i in range(b, f - 1, -1) if pid in cc[i]), None)
    if pa is None or pb is None or b <= a:
        return None
    return float(np.hypot(pb[0] - pa[0], pb[1] - pa[1])) / ((b - a) / fps)


def candidates(c, r_hand=0.2, hold=3, max_air=0.4, max_gap=1.5):
    d = pickle.load(open(ROOT / f'output_videos/batch/screen_{c}_events.pkl', 'rb'))
    s = pickle.load(open(ROOT / f'stubs/ball_batch/screen_{c}_ball.pkl', 'rb'))['shot_sam3']
    _, _, tch = pickle.load(open(ROOT / f'stubs/ball_batch/screen_{c}_touch.pkl', 'rb'))
    pose = pickle.load(open(ROOT / f'stubs/pose_hands/screen_{c}.pkl', 'rb'))
    H = float(np.median([b[3] - b[1] for fd in s['players'] for b in fd.values()]))
    seq = PP.smooth_flicker(PP.holders(pose, tch, H, r_hand))
    segs, cur = [], None
    for i, q in enumerate(seq):
        if q is not None and cur and cur[0] == q and i - cur[2] <= 2:
            cur[2] = i
        elif q is not None:
            if cur: segs.append(cur)
            cur = [q, i, i]
    if cur: segs.append(cur)
    # A 段：持球 ≥ 2 張，或從片段開頭就拿著（片段常從交接前一點點開始）；B 段：≥ hold 張（交出去要真的拿住）
    segs = [g for g in segs if g[2] - g[1] + 1 >= 2 or g[1] == 0]
    idmap = map_ids(s, d['player_tracks'])
    team = clip_teams(d['player_assignment'])
    fi, fps, cc = s['frame_idx'], s['orig_fps'], d['court_coords']

    def box(i, q):
        return s['players'][i].get(q) if i < len(s['players']) else None

    def ctr_gap(i, A, B):
        a, b = box(i, A), box(i, B)
        if a is None or b is None:
            return None
        return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)) / H

    out = []
    for (A, a1, b1), (B, a2, b2) in zip(segs, segs[1:]):
        if A == B or b2 - a2 + 1 < hold:
            continue
        air = (fi[a2] - fi[b1]) / fps
        if air > max_air:
            continue
        a, b = idmap.get(A, A), idmap.get(B, B)
        if team.get(a) is None or team.get(a) != team.get(b):
            continue
        win = range(max(b1 - 1, 0), min(a2 + 2, len(fi)))
        gaps = [g for g in (ctr_gap(i, A, B) for i in win) if g is not None]
        if not gaps or min(gaps) > max_gap:
            continue
        # 兩手同時在球上：交接那幾張「兩人手腕離球比較遠的那個」最小值（越小越像兩人一起握）
        both = []
        for i in win:
            cen = tch[i][1]
            if cen is None or i >= len(pose) or A not in pose[i] or B not in pose[i]:
                continue
            da, db = PP.hand_dist(pose[i][A], cen, H), PP.hand_dist(pose[i][B], cen, H)
            if da is not None and db is not None:
                both.append(max(da, db))
        if any(fi[b1] - 0.3 * fps <= sh['release_frame'] <= fi[a2] + 0.3 * fps for sh in d.get('shots') or []):
            continue
        fx = (fi[b1] + fi[a2]) // 2
        out.append(dict(clip=c, frm=a, to=b, release=int(fi[b1]), catch=int(fi[a2]), air=round(air, 2),
                        gap=round(min(gaps), 2), both=None if not both else round(min(both), 2),
                        a_hold=round((fi[b1] - fi[a1]) / fps, 1), b_hold=round((fi[b2] - fi[a2]) / fps, 1),
                        v_a=_speed(cc, a, fx, d['fps']), v_b=_speed(cc, b, fx, d['fps'])))
    return out


def _have(c):
    return all((ROOT / p).exists() for p in (f'stubs/pose_hands/screen_{c}.pkl', f'stubs/ball_batch/screen_{c}_touch.pkl',
                                              f'output_videos/batch/screen_{c}_events.pkl'))


def _fmt(p):
    v = lambda x: '—' if x is None else f'{x:.1f}'
    return (f"screen_{p['clip']} #{p['frm']}→#{p['to']} 影格 {p['release']}-{p['catch']} 空中 {p['air']}s 框距 {p['gap']} "
            f"兩手 {p['both']} A先拿 {p['a_hold']}s B拿 {p['b_hold']}s 速度 A {v(p['v_a'])} B {v(p['v_b'])} 呎/秒")


def main():
    for c, f, A, B in SEEDS:
        ps = candidates(c)
        hit = [p for p in ps if p['frm'] == A and p['to'] == B and p['release'] - 40 <= f <= p['catch'] + 40]
        print(f'種子 screen_{c} ~{f} #{A}→#{B}：{"抓到" if hit else "沒抓到"}')
        for p in ps:
            print('   ', '★' if p in hit else ' ', _fmt(p))
    n, tot = 0, 0
    for c in range(26, 167):
        if _have(c):
            n += 1; tot += len(candidates(c))
    print(f'\n{n} 支片段、候選 {tot} 個')


def sheet(clips, out_csv):
    rows, k = [], 0
    for c in clips:
        if not _have(c):
            continue
        for p in candidates(c):
            k += 1
            v = lambda x: '' if x is None else f'{x:.1f}'
            rows.append([f'D{k:03d}', f"screen_{p['clip']}", p['release'], p['catch'], f"#{p['frm']}", f"#{p['to']}",
                         p['air'], p['gap'], '' if p['both'] is None else p['both'], v(p['v_a']), v(p['v_b']), '', ''])
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['編號', 'clip', 'A放球影格', 'B拿到影格', '給球者', '接球者', '空中秒', '框距(身高)', '兩手同時(身高)',
                    'A速度(呎/秒)', 'B速度(呎/秒)', '手遞手嗎(t/f/?)', '備註（例：一般短傳、掩護時換手、其實沒換人）'])
        w.writerows(rows)
    print(f'{len(rows)} 個候選 → {out_csv}')


if __name__ == '__main__':
    if sys.argv[1:2] == ['sheet']:
        sheet(range(int(sys.argv[2]), int(sys.argv[3]) + 1), ROOT / 'output_videos' / 'framenum_boxes' / 'dho_check.csv')
    else:
        main()
