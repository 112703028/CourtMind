"""
offball_label.py — 無球掩護標記表（2026-11-17）：挑 100 支訓練片段讓使用者「完整」標無球掩護（影格、掩護者、被掩護的人）。

無球掩護還偵測不準 → 不能只給偵測到的地方；給整支片段，弱線索只決定看的順序：
  ① ScreenNet 擋拆裡被擋的防守者原本守的不是持球者（39 個真的無球掩護有 11 個被當成持球擋拆）
  ② 規則版無球撞上（offball_v2.contacts：不是持球者在用、不在擋拆事件裡、被掩護的人有經過 + 在跑）
  ③ 主播事實的無球掩護（tactics_tools.offball_moves 有 screen）
  ④ 舊 COCO 有標 target（被掩護的人）的片段
70 支照分數高到低 + 30 支隨機（估一般片段的比例、檢查排序有沒有用）。開發答案（offball_v2.dev_set）的片段不放（留著評估）。

用法：python nba_data_pipeline/offball_label.py score     # 476 支算線索（快取 stubs/offball_signals.json）
     python nba_data_pipeline/offball_label.py sheet     # 挑 100 支 → output_videos/offball_label/offball_label.csv + 影片
"""
import sys
import csv
import json
import glob
import math
import random
import pickle
import collections
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from convert_contact import CACHE_DIR

BATCH = ROOT / 'output_videos' / 'batch'
SIG = ROOT / 'stubs' / 'offball_signals.json'
OUT = ROOT / 'output_videos' / 'offball_label'


def coco_target_clips():
    import re
    out = set()
    for f in glob.glob(str(ROOT / 'screen_data' / '*' / 'train' / '_annotations.coco.json')):
        j = json.load(open(f, encoding='utf-8'))
        tid = {c['id'] for c in j['categories'] if c['name'] == 'target'}
        img = {im['id']: im['file_name'] for im in j['images']}
        for a in j['annotations']:
            if a['category_id'] in tid:
                m = re.match(r'(screen_\d+)', img[a['image_id']])
                if m:
                    out.add(m.group(1))
    return out


def signals(d):
    """→ dict(s1, s2, s3, hints=[(影格, '#S>#C', 線索)])"""
    import offball_v2 as O
    from tactics_tools import (find_screen_instances, classify_roll_pop, infer_target_basket, offball_moves,
                               court_teams, _pos_med, clip_teams)
    cc, fps, ba = d['court_coords'], d['fps'], d['ball_acquisition']
    off, dfn = O._teams(d)
    hints = []
    # ① 擋拆裡被擋的人原本守的不是持球者
    s1 = 0
    inst = find_screen_instances(cc, d['player_assignment'], d['events'])
    for it in inst:
        sf, S, D = it['frame'], it['screener_id'], it['defender_id']
        if D is None:
            continue
        cnt = collections.Counter()
        for f in range(max(0, sf - int(1.2 * fps)), max(0, sf - int(0.3 * fps)), 3):
            pd = _pos_med(cc, D, f, 2)
            if pd is None:
                continue
            near = [(math.dist(pd, pp), q) for q in off - {S} for pp in [_pos_med(cc, q, f, 2)] if pp is not None]
            if near:
                cnt[min(near)[1]] += 1
        G = cnt.most_common(1)[0][0] if cnt else None
        hc = collections.Counter(ba[f] for f in range(max(0, sf - int(0.5 * fps)), min(len(ba), sf + int(0.3 * fps)))
                                 if ba[f] not in (None, -1))
        H = hc.most_common(1)[0][0] if hc else it['handler_id']
        if G is not None and G != H:
            s1 += 1
            hints.append((sf, f'#{S}>#{G}', '①'))
    # ② 規則版無球撞上
    s2 = 0
    for c in O.contacts(d):
        if (not c['onball'] and not c['in_event'] and c['pass_'] is not None and c['pass_'] <= 6
                and c['v_cut'] is not None and c['v_cut'] >= 12 and c['approach'] is not None and c['approach'] >= 3):
            if all(abs(c['t'] - h[0]) > fps for h in hints if h[2] == '②'):
                s2 += 1
                hints.append((c['t'], f"#{c['S']}>#{c['C']}", '②'))
    # ③ 主播事實的無球掩護
    s3 = 0
    teams = court_teams(d)
    tm = clip_teams(d['player_assignment'])
    for it in inst:
        sf, S, H = it['frame'], it['screener_id'], it['handler_id']
        basket = infer_target_basket(cc, max(0, sf - 60), min(len(cc) - 1, sf + 60))
        action, _ = classify_roll_pop(cc, S, sf, basket, fps)
        for o in offball_moves(d, H, S, sf, basket, fps, tm.get(H, 1), teams, action):
            if o.get('screen'):
                s3 += 1
                g = o['screen'].get('defender_was_guarding')
                hints.append((sf + int(o['screen']['after_s'] * fps), f"#{o['player']}>#{g}" if g is not None else f"#{o['player']}>?", '③'))
    return dict(s1=s1, s2=s2, s3=s3, hints=sorted(hints))


def score():
    sig = json.load(open(SIG, encoding='utf-8')) if SIG.exists() else {}
    clips = sorted(glob.glob(str(BATCH / 'screen_*_events.pkl')), key=lambda x: int(Path(x).stem.split('_')[1]))
    for i, pk in enumerate(clips):
        c = Path(pk).stem.replace('_events', '')
        if c in sig:
            continue
        try:
            sig[c] = signals(pickle.load(open(pk, 'rb')))
        except Exception as e:                      # 少數片段資料不全
            sig[c] = dict(s1=0, s2=0, s3=0, hints=[], error=str(e))
        if i % 20 == 0:
            json.dump(sig, open(SIG, 'w', encoding='utf-8'), ensure_ascii=False)
            print(f'{i}/{len(clips)}', flush=True)
    json.dump(sig, open(SIG, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'{len(sig)} 支線索完成')


def sheet(n_top=70, n_rand=30, seed=17):
    import offball_v2 as O
    import ids_framenum_video as IV
    sig = json.load(open(SIG, encoding='utf-8'))
    dev = {x['clip'] for x in O.dev_set()}
    coco = coco_target_clips()
    ok = []
    for c in sig:
        cp = CACHE_DIR / f'{c}.pkl'
        if c in dev or not cp.exists():
            continue
        lab = pickle.load(open(cp, 'rb'))
        if lab.get('pending'):
            continue
        ok.append(c)
    sc = {c: 2.0 * (c in coco) + 1.5 * sig[c]['s1'] + 1.0 * sig[c]['s3'] + 0.5 * min(sig[c]['s2'], 4) for c in ok}
    top = sorted(ok, key=lambda c: -sc[c])[:n_top]
    rest = [c for c in ok if c not in top]
    rnd = random.Random(seed).sample(rest, min(n_rand, len(rest)))
    pick = [(c, '高分') for c in top] + [(c, '隨機') for c in rnd]
    random.Random(seed + 1).shuffle(pick)                   # 打散，標的人不知道哪支是高分
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (c, src) in enumerate(pick, 1):
        d = pickle.load(open(BATCH / f'{c}_events.pkl', 'rb'))
        hints = '；'.join(f'{f}:{p}' for f, p, _ in sig[c]['hints'][:6])
        rows.append([f'U{k:03d}', c, len(d['court_coords']), hints, '', ''])
        if not (OUT / f'{c}.mp4').exists():
            IV.render(BATCH / f'{c}_events.pkl', ROOT / 'input_videos' / 'screen' / f'{c}.mp4', OUT / f'{c}.mp4')
        print(f'{k}/{len(pick)} {c}', flush=True)
    with open(OUT / 'offball_label.csv', 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['編號', 'clip', '影格數', '可能的時間點（參考，要整支看完）',
                    '無球掩護（影格:掩護者>被掩護的人；多個用；隔開；沒有填「無」）', '備註'])
        w.writerows(rows)
    json.dump({c: s for c, s in pick}, open(OUT / 'pick_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{len(rows)} 支 → {OUT / "offball_label.csv"}（高分 {n_top}、隨機 {len(rnd)}；來源在 pick_source.json，先別看）')


if __name__ == '__main__':
    if sys.argv[1:2] == ['score']:
        score()
    elif sys.argv[1:2] == ['sheet']:
        sheet()
    else:
        print(__doc__)
