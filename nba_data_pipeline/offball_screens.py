"""
offball_screens.py — 規則版無球掩護偵測（不用訓練），在已標記的 29 個無球掩護上量準確率。

無球掩護 = 掩護者撞上「守沒拿球隊友的防守者」，讓那個隊友（切入者）跑出空檔。5fps、第 2 版座標 + 清理（同 ScreenNet）：
  每張 t、每個進攻球員 S（掩護者候選）：
  ① 接觸：某防守者 D 離 S ≤ CONTACT ft，且 t 是前後 ±2 張裡兩人最近的那張
  ② D 在守誰：撞上前 1 秒（t−5 ~ t−1）D 平均離誰最近的進攻球員 C（≠ S），平均 ≤ GUARD ft
     C 是持球者 → 持球擋拆（不是無球）；C 不是持球者 → 無球掩護候選
  ③ 掩護者站定：S 在 t 前後的速度 ≤ STILL ft/張（預設關掉）
  ④ 切入者從旁邊經過：t ± 3 張內 C 離 S 最近 ≤ PASS ft，且 C 有在動（速度 ≥ MOVE ft/張）
  ⑤ 擋到：撞上後 D 離 C 變遠（t+3 比 t−2 多 ≥ SEP ft）（預設關掉）
  同一組（掩護者, 切入者）3 秒內只留接觸最緊的一次。

標記（eval_set）：
  舊 coverage_labels.csv 寫「無球檔」的 22 列（只有時間，沒有掩護者編號 → 只比時間）、
  新片段表格寫「無球檔」的 2 列、2026-11-04 cache_label_fixes 拿掉的 5 個（有掩護者編號）。

用法：python nba_data_pipeline/offball_screens.py [--contact 3.5 --still 2.0 ...]
"""
import sys
import csv
import glob
import json
import math
import pickle
import argparse
import collections
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from convert_contact import CACHE_DIR

# 2026-11-04 在 27 個標記上掃過：站定、拉開兩條拿掉（防守者常硬擠過去 / 5fps 速度太吵）、守人距離放寬到 10 ft（無球防守常放很開）
# → ±1 秒抓到 22/27（±0.4 秒 14/27）、458 支共偵測 444 次、持球擋拆標記被判成無球 22/508。原本嚴格版 10/27。
P = dict(contact=3.0, guard=10.0, still=99.0, pass_=12.0, move=1.5, sep=-99.0, gap=15)


def _d(a, b):
    return math.dist(a, b) if a is not None and b is not None else None


def detect(ct, team, H, prm=None):
    """ct：5fps 清理後座標 list[{pid: (x, y)}]；team：{pid: offense/defense}；H：持球者。
    回 [dict(t, screener, cutter, defender, onball, contact, ...)]（含持球擋拆，onball=True）。"""
    p = dict(P, **(prm or {}))
    n = len(ct)
    pos = lambda q, f: ct[f].get(q) if 0 <= f < n else None
    off = {q for q, v in team.items() if v == 'offense'}
    dfn = {q for q, v in team.items() if v == 'defense'}

    def speed(q, f):
        a, b = pos(q, f - 1), pos(q, f + 1)
        return _d(a, b) / 2 if a is not None and b is not None else None

    out = []
    for t in range(n):
        for S in off - {H}:                       # 持球者不會掩護（持球擋拆時他撞上的是掩護者的防守者，會被誤讀成他在掩護）
            ps = pos(S, t)
            if ps is None:
                continue
            for D in dfn:
                dsd = _d(ps, pos(D, t))
                if dsd is None or dsd > p['contact']:
                    continue
                near = [_d(pos(S, u), pos(D, u)) for u in range(t - 2, t + 3)]
                if any(x is not None and x < dsd for x in near):
                    continue                                       # 不是最近那張
                guard = {}
                for C in off - {S}:
                    ds = [_d(pos(D, u), pos(C, u)) for u in range(t - 5, t)]
                    ds = [x for x in ds if x is not None]
                    if len(ds) >= 3:
                        guard[C] = sum(ds) / len(ds)
                if not guard:
                    continue
                C = min(guard, key=guard.get)
                if guard[C] > p['guard']:
                    continue
                vs = speed(S, t)
                pas = [_d(pos(C, u), pos(S, u)) for u in range(t - 3, t + 4)]
                pas = [x for x in pas if x is not None]
                vc = max([x for x in (speed(C, u) for u in range(t - 2, t + 3)) if x is not None], default=None)
                s0, s1 = _d(pos(C, t - 2), pos(D, t - 2)), _d(pos(C, t + 3), pos(D, t + 3))
                out.append(dict(t=t, screener=S, cutter=C, defender=D, onball=(C == H), contact=dsd, guard=guard[C],
                                still=vs, pass_=min(pas) if pas else None, move=vc,
                                sep=(s1 - s0) if s0 is not None and s1 is not None else None))
    return out


def keep(ev, prm=None):
    """規則 ③④⑤ + 同一掩護者 3 秒去重。"""
    p = dict(P, **(prm or {}))
    ok = [e for e in ev if e['still'] is not None and e['still'] <= p['still']
          and e['pass_'] is not None and e['pass_'] <= p['pass_']
          and e['move'] is not None and e['move'] >= p['move']
          and e['sep'] is not None and e['sep'] >= p['sep']]
    best = []
    for e in sorted(ok, key=lambda e: e['contact']):
        if all(not (b['screener'] == e['screener'] and b['cutter'] == e['cutter'] and abs(b['t'] - e['t']) < p['gap'])
               for b in best):                             # 同一組（掩護者, 切入者）才去重：持球 / 無球不互相吃掉
            best.append(e)
    return sorted(best, key=lambda e: e['t'])


def eval_set():
    """→ {clip: [(5fps 影格, 掩護者 or None, 來源)]}（同一支 ±2 張內的重複併掉）。"""
    lab = collections.defaultdict(list)

    def add(clip, idx, S, src):
        if all(abs(i - idx) > 2 for i, _, _ in lab[clip]):
            lab[clip].append((idx, S, src))
    for r in csv.DictReader(open(HERE / 'coverage_labels.csv', encoding='utf-8-sig')):
        if '無球' in (r.get('防守方式(換防/包夾/退防/其他)') or ''):
            p = CACHE_DIR / f"{r['clip']}.pkl"
            if p.exists():
                c = pickle.load(open(p, 'rb'))
                add(r['clip'], round(int(r['接觸幀(原片)']) / (c['fps'] / 5)), None, '舊標記')
    for p in sorted(glob.glob(str(ROOT / 'output_videos' / 'new_clip_check' / 'handler_labels_*.csv'))):
        for r in csv.reader(open(p, encoding='utf-8-sig')):
            if len(r) > 5 and '無球' in r[5] and r[3].strip().isdigit():
                for f in r[4].replace('，', ',').split(','):
                    if f.strip().isdigit():
                        add(r[0], int(f), int(r[3]), '新片段表格')
    fx = json.load(open(HERE / 'cache_label_fixes.json', encoding='utf-8'))
    for clip, v in fx.items():
        for a in (v.get('remove_events') or []) if isinstance(v, dict) else []:
            if '無球' in a.get('src', ''):
                add(clip, a['idx'], a['tid'], '2026-11-04 拿掉')
    return lab


def main():
    import finetune_seq
    ap = argparse.ArgumentParser()
    for k, v in P.items():
        ap.add_argument(f'--{k.rstrip("_")}', type=float, default=v)
    args = ap.parse_args()
    prm = {k: getattr(args, k.rstrip('_')) for k in P}
    lab = eval_set()
    print(f'無球掩護標記 {sum(map(len, lab.values()))} 個（{len(lab)} 支）；參數 {prm}')
    hit = {2: 0, 5: 0}
    who = 0
    n_det = n_on = on_lab = on_as_off = 0
    rows = []
    for p in sorted(glob.glob(str(CACHE_DIR / 'screen_*.pkl')), key=lambda q: int(Path(q).stem.split('_')[1])):
        clip = Path(p).stem
        if int(clip.split('_')[1]) > 503:
            continue
        c = pickle.load(open(p, 'rb'))
        if c.get('pending') and not c.get('excluded_reason'):
            continue
        ct = finetune_seq.clip_coords(c)
        ev = keep(detect(ct, c['track_team'], c['handler_tid']), prm)
        offb = [e for e in ev if not e['onball']]
        n_det += len(offb); n_on += len(ev) - len(offb)
        for e in c['events']:                                      # 持球擋拆標記被判成無球的比例
            if e['tid'] is None:
                continue
            on_lab += 1
            if any(x['screener'] == e['tid'] and abs(x['t'] - e['idx']) <= 2 for x in offb) and \
                    not any(x['screener'] == e['tid'] and abs(x['t'] - e['idx']) <= 2 for x in ev if x['onball']):
                on_as_off += 1
        for idx, S, src in lab.get(clip, []):
            cand = [e for e in offb if (S is None or e['screener'] == S)]
            best = min(cand, key=lambda e: abs(e['t'] - idx), default=None)
            dt = abs(best['t'] - idx) if best else None
            for tol in hit:
                hit[tol] += dt is not None and dt <= tol
            any_e = min(ev, key=lambda e: abs(e['t'] - idx), default=None)
            rows.append((clip, idx, S, src, best and (best['t'], best['screener'], best['cutter']),
                         any_e and (any_e['t'], any_e['screener'], 'on' if any_e['onball'] else 'off')))
    N = len(rows)
    print(f'抓到（±0.4 秒）{hit[2]}/{N}、（±1 秒）{hit[5]}/{N}')
    print(f'全部 {len(set(r[0] for r in rows)) and ""}片段共偵測到無球掩護 {n_det} 次、持球擋拆 {n_on} 次；'
          f'持球擋拆標記 {on_lab} 個裡被判成無球 {on_as_off} 個')
    print('\n逐個：片段 影格 標記掩護者 來源 → 最近的無球偵測（影格, 掩護者, 切入者）｜最近的任何偵測')
    for r in rows:
        print('  ', *r)


if __name__ == '__main__':
    main()
