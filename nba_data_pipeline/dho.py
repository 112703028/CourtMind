"""
dho.py — 手遞手（DHO，dribble hand-off）第 2 版（2026-11-15）：用持球者（SAM3 追球 + Viterbi）的換手找，不另外當一種事件，跟擋拆連起來。

第 1 版（2026-11-06，手腕持球段）141 支 36 個候選、使用者看了多是一般擋拆 → 停掉；根因 = 手遞手本來就是「交球 + 擋」，
系統沒有「這次擋拆有沒有交球」的欄位。使用者備註的 3 個：160 ScreenNet 當擋拆抓到（掩護者 = 給球者、持球者 = 接球者），155 / 162 沒抓到，
但持球者換手都很乾淨（A 持球 → 0~6 張沒人 → B 持球）。

候選：持球者 A 一段 → B 一段（同隊、A ≠ B）、中間沒人持球 ≤ max_air 秒、A 持球 ≥ hold_a 秒（片段一開始就在 A 手上不算）、B 持球 ≥ hold_b 秒、
      交接前後 ±0.5 秒內兩人最近 ≤ max_ft 呎（160 換手那張 6~7 呎、0.3 秒後才貼到 3 呎；162 最近 5.2 呎）。
分類：
  'screen_event' = 交接 ±1.5 秒內有擋拆事件、掩護者 = A、持球者 = B → 手遞手掩護（擋拆事件標 dho）
  'screens_def'  = 沒有擋拆事件，但交接 ±1 秒內 A 貼到 B 的防守者 ≤ 4 呎 → 手遞手掩護（ScreenNet 漏的）
  'handoff'      = 只有近距離交球、沒擋到人

判定（is_dho）：接球者靠過去 + 給球者慢 + 接球者持球夠久 + 不是先擋再傳 → 42 個人工標記上 11/13 對（85%）、真的 15 個抓到 11。

用法：python nba_data_pipeline/dho.py                 # 種子有沒有抓到 + 476 支候選數
     python nba_data_pipeline/dho.py score           # 跟人工標記比
     python nba_data_pipeline/dho.py add <events.pkl> <sam3.pkl> <touch.pkl> <pose.pkl>   # 手腕證據 → dho_hands（run_pipeline.sh 第 3 步）
     python nba_data_pipeline/dho.py add_batch       # 批次 476 支
     python nba_data_pipeline/dho.py sheet           # → output_videos/dho_check/dho_check.csv + 每支片段的編號 / 影格影片
"""
import sys
import csv
import glob
import pickle
import collections
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from tactics_tools import clip_teams, court_teams, _own_defender, _pos_med, _dist

BATCH = ROOT / 'output_videos' / 'batch'
OUT = ROOT / 'output_videos' / 'dho_check'
# 使用者在 pass_check_pose.csv 備註寫到的手遞手（片段、大約影格、給球者、接球者）
SEEDS = [(155, 8, 7, 10), (160, 60, 1, 10), (162, 130, 1, 8)]
KIND = {'screen_event': '手遞手掩護（擋拆事件）', 'screens_def': '手遞手掩護（沒有擋拆事件）', 'handoff': '只有近距離交球'}


def _runs(ba):
    out, cur, s = [], None, 0
    for i, p in enumerate(list(ba) + [None]):
        if p != cur:
            if cur is not None and cur != -1:
                out.append((cur, s, i - 1))
            cur, s = p, i
    return out


def candidates(d, max_air=0.5, hold_a=0.3, hold_b=0.5, max_ft=6.0, near_s=0.5, def_ft=4.0):
    fps, cc, ba = d['fps'], d['court_coords'], d['ball_acquisition']
    tm, ct = clip_teams(d['player_assignment']), court_teams(d)
    R = _runs(ba)
    out = []
    for (a, s1, e1), (b, s2, e2) in zip(R, R[1:]):
        if a == b or tm.get(a) is None or tm.get(a) != tm.get(b) or s2 - e1 - 1 > max_air * fps:
            continue
        if (e1 - s1 + 1 < hold_a * fps and s1 > 2) or e2 - s2 + 1 < hold_b * fps:
            continue
        lo, hi = max(0, e1 - int(near_s * fps)), min(len(cc), s2 + int(near_s * fps) + 1)
        ds = [(_dist(pa, pb), f) for f in range(lo, hi) for pa, pb in [(_pos_med(cc, a, f, 1), _pos_med(cc, b, f, 1))]
              if pa is not None and pb is not None]
        if not ds or min(ds)[0] > max_ft:
            continue
        dmin, fmin = min(ds)
        f = (e1 + s2) // 2
        ev = [e for e in d.get('events') or [] if abs(e['peak_frame'] - f) <= 1.5 * fps
              and e['screener_id'] == a and e['handler_id'] == b]
        rec = {'frame': f, 'giver': a, 'receiver': b, 'air_s': round((s2 - e1 - 1) / fps, 2), 'min_ft': round(dmin, 1),
               'hold_a_s': round((e1 - s1 + 1) / fps, 1), 'hold_b_s': round((e2 - s2 + 1) / fps, 1)}
        if ev:
            rec['kind'], rec['event_frame'] = 'screen_event', ev[0]['peak_frame']
        else:
            defs = [q for q, t in ct.items() if t != tm[a]]
            D = _own_defender(cc, b, defs, f - int(1.5 * fps), f - int(0.3 * fps))
            if D is None:                                  # 交接在片頭（155 第 5 張）→ 看交接後 0.5 秒
                D = _own_defender(cc, b, defs, f, f + int(0.5 * fps))
            hit = None
            if D is not None:
                dd = [(_dist(pa, pd), g) for g in range(max(0, f - int(fps)), min(len(cc), f + int(fps) + 1))
                      for pa, pd in [(_pos_med(cc, a, g, 1), _pos_med(cc, D, g, 1))] if pa is not None and pd is not None]
                if dd and min(dd)[0] <= def_ft:
                    hit = (D, round(min(dd)[0], 1))
                    rec['def_frame'] = min(dd)[1]
            rec['kind'] = 'screens_def' if hit else 'handoff'
            if hit:
                rec['defender'], rec['def_ft'] = hit
        # 手遞手的樣子（使用者標 42 個候選）：給球者慢（站著或小步）、接球者往他靠過去、接到後持球
        def pos(q, g):
            for k in range(int(fps / 2)):
                for h in (g - k, g + k):
                    v = _pos_med(cc, q, min(max(h, 0), len(cc) - 1), 1)
                    if v is not None:
                        return v
        pg0, pg1, ph = pos(a, f - int(fps)), pos(a, f), pos(a, f - int(fps / 2))
        pr0, pr1 = pos(b, f - int(fps)), pos(b, f)
        rec['v_giver'] = round(_dist(ph, pg1) * 2, 1) if ph is not None and pg1 is not None else None
        if None in (pg0, pr0):          # 交接前 1 秒沒座標（160 / 463：片頭、追蹤斷掉）→ 用交接前 1 ~ 0.3 秒裡最早有兩人座標的那張
            g = next((g for g in range(max(0, f - int(fps)), f - int(0.3 * fps))
                      if _pos_med(cc, a, g, 1) is not None and _pos_med(cc, b, g, 1) is not None), None)
            pg0, pr0 = (_pos_med(cc, a, g, 1), _pos_med(cc, b, g, 1)) if g is not None else (None, None)
        rec['approach_ft'] = round(_dist(pg0, pr0) - _dist(pg1, pr1), 1) if None not in (pg0, pg1, pr0, pr1) else None
        out.append(rec)
    return out


def is_dho(p, a_min=0.0, vg_max=12.0, hb_min=0.8, hands=None, hands_yes=0.15, hands_no=0.75):
    """接球者交接前 1 秒有靠近給球者（≥ a_min 呎）、給球者交接前 0.5 秒 ≤ vg_max 呎/秒、接球者持球 ≥ hb_min 秒、
    有擋拆事件的話撞上不能在交接前（378 / 401：先擋再傳 = 一般擋拆）。
    hands = 交接 ±0.4 秒內「兩人手腕離球比較遠的那個」最小值（身高倍數，hands_feature）：≤ hands_yes 兩人手同時在球上 → 不看靠近 / 速度
    （148 接球者從旁邊繞來）；> hands_no 手離球很遠 → 不是（413）。42 個標記上沒有假的 ≤ 0.15、沒有真的（有判到的）> 0.75。"""
    if p.get('event_frame', p['frame']) < p['frame'] or p['hold_b_s'] < hb_min:
        return False
    if hands is not None and hands > hands_no:
        return False
    if hands is not None and hands <= hands_yes:
        return True
    return p['approach_ft'] is not None and p['approach_ft'] >= a_min and p['v_giver'] is not None and p['v_giver'] <= vg_max


def detect(d, **prm):
    """→ 判定是手遞手的候選（tactics_agent / tactics_tools 用）；d['dho_hands']（hands_feature 寫進去的）有就用。"""
    hd = d.get('dho_hands') or {}
    return [p for p in candidates(d) if is_dho(p, hands=hd.get((p['giver'], p['receiver'], p['frame'])), **prm)]


def hands_feature(d, s, tch, pose, win_s=0.4):
    """每個候選：交接 ±win_s 秒內兩人手腕離球比較遠的那個的最小值（身高倍數）→ {(giver, receiver, frame): 值 or None}。
    s = sam3 pkl 的 shot_sam3、tch = possession_sam3 prep 的碰球、pose = pose_hands 的手腕骨架（編號是 shot_sam3 的，用 map_ids 對回來）。"""
    import numpy as np
    import pose_pass as PP
    from shot_detect import map_ids
    H = float(np.median([b[3] - b[1] for fd in s['players'] for b in fd.values()]))
    inv = {}
    for q, v in map_ids(s, d['player_tracks']).items():
        inv.setdefault(v, q)
    fi, fps = s['frame_idx'], s['orig_fps']
    out = {}
    for p in candidates(d):
        A, B, f = inv.get(p['giver']), inv.get(p['receiver']), p['frame']
        both = []
        for i in range(min(len(fi), len(pose), len(tch))):
            if not (f - win_s * fps <= fi[i] <= f + win_s * fps):
                continue
            cen = tch[i][1]
            if cen is None or A not in pose[i] or B not in pose[i]:
                continue
            da, db = PP.hand_dist(pose[i][A], cen, H), PP.hand_dist(pose[i][B], cen, H)
            if da is not None and db is not None:
                both.append(max(da, db))
        out[(p['giver'], p['receiver'], f)] = round(min(both), 2) if both else None
    return out


def add(pkl, sam3_pkl, touch_pkl, pose_pkl):
    """把手腕證據寫進 events.pkl 的 dho_hands（run_pipeline.sh 第 3 步傳球之後）。"""
    d = pickle.load(open(pkl, 'rb'))
    s = pickle.load(open(sam3_pkl, 'rb'))['shot_sam3']
    _, _, tch = pickle.load(open(touch_pkl, 'rb'))
    d['dho_hands'] = hands_feature(d, s, tch, pickle.load(open(pose_pkl, 'rb')))
    pickle.dump(d, open(pkl, 'wb'))
    print(f'手遞手：候選 {len(d["dho_hands"])} 個、判定 {len(detect(d))} 個')


def add_batch():
    n = 0
    for pk in _clips():
        c = Path(pk).stem.replace('_events', '')
        need = [ROOT / 'stubs' / 'ball_batch' / f'{c}_ball.pkl', ROOT / 'stubs' / 'ball_batch' / f'{c}_touch.pkl',
                ROOT / 'stubs' / 'pose_hands' / f'{c}.pkl']
        if all(x.exists() for x in need):
            d = pickle.load(open(pk, 'rb'))
            if candidates(d):
                d['dho_hands'] = hands_feature(d, pickle.load(open(need[0], 'rb'))['shot_sam3'],
                                               pickle.load(open(need[1], 'rb'))[2], pickle.load(open(need[2], 'rb')))
                pickle.dump(d, open(pk, 'wb')); n += 1
    print(f'{n} 支寫入 dho_hands')


def with_dho(d):
    """→ 新的 events：判定是手遞手的，對到的擋拆事件標 dho=True；沒有擋拆事件、但給球者擋到接球者的防守者（ScreenNet 漏的）補一個 dho=True 的事件
    （掩護者 = 給球者、持球者 = 接球者、撞上 = 擋到防守者那張），後面 find_screen_instances / 分析照常算。
    沒有持球者 / 座標資料 → 原樣回傳。"""
    events = [dict(e) for e in d.get('events') or []]
    try:
        found = detect(d)
    except (KeyError, TypeError, ValueError):
        return events
    fps = d['fps']
    for p in found:
        ev = [e for e in events if e['screener_id'] == p['giver'] and e['handler_id'] == p['receiver']
              and abs(e['peak_frame'] - p['frame']) <= 1.5 * fps]
        if ev:
            for e in ev:
                e['dho'], e['dho_frame'] = True, p['frame']
        elif p['kind'] == 'screens_def':        # 沒擋到人的單純換手（162 / 423 防守者 10~17 呎外）不補擋拆事件
            pf = p['def_frame']
            events.append({'start': pf - int(0.3 * fps), 'end': pf + int(0.3 * fps), 'peak_frame': pf,
                           'handler_id': p['receiver'], 'screener_id': p['giver'], 'score': 0.0, 'dho': True, 'dho_frame': p['frame'], 'from_dho': True})
    events.sort(key=lambda e: e['peak_frame'])
    return events


def is_dho_screen(events, S, H, frame, fps):
    """這次擋（掩護者 S、持球者 H、撞上 frame）是不是手遞手：撞上在交接前 0.3 秒 ~ 交接後 1.5 秒
    （真的手遞手撞上比交接晚 0~1.4 秒；155 第 51 張是交接後 1.5 秒的另一次擋）。"""
    return any(e.get('dho') and e['screener_id'] == S and e['handler_id'] == H
               and -0.3 * fps <= frame - e['dho_frame'] < 1.5 * fps for e in events)


def score():
    lab = {(r['clip'], int(r['交接影格'])): r['手遞手嗎(t/f/?)'].strip() for r in csv.DictReader(open(OUT / 'dho_check.csv', encoding='utf-8-sig'))}
    sel = tp = 0
    for c in sorted({c for c, _ in lab}):
        d = pickle.load(open(BATCH / f'{c}_events.pkl', 'rb'))
        hd = d.get('dho_hands') or {}
        for p in candidates(d):
            t = lab.get((c, p['frame']))
            if t and is_dho(p, hands=hd.get((p['giver'], p['receiver'], p['frame']))):
                sel += 1; tp += t == 't'
    n = sum(v == 't' for v in lab.values())
    print(f'標記 {len(lab)} 個（真的 {n}）：判定手遞手 {sel}、對 {tp}（{tp / max(1, sel):.0%}）、抓到 {tp}/{n}')


def _clips():
    return sorted(glob.glob(str(BATCH / 'screen_*_events.pkl')), key=lambda x: int(Path(x).stem.split('_')[1]))


def main():
    for c, f, A, B in SEEDS:
        ps = candidates(pickle.load(open(BATCH / f'screen_{c}_events.pkl', 'rb')))
        hit = [p for p in ps if p['giver'] == A and p['receiver'] == B and abs(p['frame'] - f) <= 40]
        print(f'種子 screen_{c} ~{f} #{A}→#{B}：{KIND[hit[0]["kind"]] if hit else "沒抓到"}  {hit or ps}')
    cnt, nclip = collections.Counter(), 0
    for pk in _clips():
        ps = candidates(pickle.load(open(pk, 'rb')))
        nclip += bool(ps)
        for p in ps:
            cnt[p['kind']] += 1
    print(f'\n476 支：候選 {sum(cnt.values())} 個（{nclip} 支）', {KIND[k]: v for k, v in cnt.items()})


def sheet():
    import ids_framenum_video as IV
    OUT.mkdir(parents=True, exist_ok=True)
    rows, k = [], 0
    for pk in _clips():
        c = Path(pk).stem.replace('_events', '')
        d = pickle.load(open(pk, 'rb'))
        ps = candidates(d)
        for p in ps:
            k += 1
            extra = (f"擋拆事件 @{p['event_frame']}" if p['kind'] == 'screen_event'
                     else (f"擋到 #{p['defender']}（{p['def_ft']} 呎）" if p['kind'] == 'screens_def' else ''))
            rows.append([f'H{k:02d}', c, p['frame'], round(p['frame'] / d['fps'], 1), f"#{p['giver']}", f"#{p['receiver']}",
                         KIND[p['kind']], extra, p['min_ft'], p['air_s'], '', ''])
        if ps and not (OUT / f'{c}.mp4').exists():
            IV.render(Path(pk), ROOT / 'input_videos' / 'screen' / f'{c}.mp4', OUT / f'{c}.mp4')
    with open(OUT / 'dho_check.csv', 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['編號', 'clip', '交接影格', '秒', '給球者', '接球者', '系統判斷', '依據', '兩人最近(呎)', '球沒人拿(秒)',
                    '手遞手嗎(t/f/?)', '備註（例：一般短傳、只是換手沒擋、其實沒換人）'])
        w.writerows(rows)
    print(f'{len(rows)} 個候選 → {OUT / "dho_check.csv"}（影片 {OUT}）')


if __name__ == '__main__':
    if sys.argv[1:2] == ['sheet']:
        sheet()
    elif sys.argv[1:2] == ['score']:
        score()
    elif sys.argv[1:2] == ['add'] and len(sys.argv) == 6:
        add(*sys.argv[2:6])
    elif sys.argv[1:2] == ['add_batch']:
        add_batch()
    else:
        main()
