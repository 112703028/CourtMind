"""
offball_zeroshot.py — 不訓練，直接拿持球擋拆的 ScreenNet 判無球掩護：slot 0 改放切入者（原本放持球者），slot 1 放掩護者。

每個標記好的候選（offball_labels.json，t = 無球掩護、f = 規則抓錯）算兩個分數（撞上 ±2 張的最高分）：
  無球分數：slot 0 = 切入者 C、slot 1 = 掩護者 S、防守者取離 C 最近的 5 個
  持球分數：slot 0 = 持球者 H、slot 1 = 掩護者 S（現在的正常用法）→ 高 = 其實是幫持球者擋（持球掩護 / 雙擋）
模型：court2fix2flip 交叉驗證模型，每支片段只用「沒看過它」的那一份（5 個種子平均）→ 不偷看答案。
沒有掩護者 / 切入者編號的標記（舊標記）用規則偵測對上的那組；對不上的跳過。

用法（pod）：python nba_data_pipeline/offball_zeroshot.py [--tag court2fix2flip]
"""
import sys
import math
import json
import pickle
import argparse
import collections
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from convert_contact import CACHE_DIR, SEQ_LEN, N_OFF, N_DEF, pair_index
from eval_contact import load_model, window_traj, ROLES
import offball_screens as O
import finetune_seq

ANCHOR = 3
CK = HERE / 'checkpoints'


def pair_score(models, c, ct, P0, S, t):
    """slot 0 = P0、slot 1 = S，撞上 t ±2 張（窗口 anchor 第 3 格）的最高分（模型機率平均）。"""
    raw, team = c['raw_tracks'], c['track_team']
    best = None
    for center in range(t - 2, t + 3):
        start = center - ANCHOR
        if start < 0 or start + SEQ_LEN > c['n']:
            continue
        wr, wc = raw[start:start + SEQ_LEN], ct[start:start + SEQ_LEN]
        cnt = collections.Counter(p for fd in wr for p in fd)
        present = {p for p, k in cnt.items() if k >= SEQ_LEN // 2} | {p for p in (P0, S) if cnt.get(p, 0) >= 1}
        if P0 not in present or S not in present:
            continue
        off = [p for p in present if team.get(p) == 'offense' or p in (P0, S)]
        dfn = [p for p in present if team.get(p) == 'defense' and p not in (P0, S)]
        ref = next((wc[k][P0] for k in range(SEQ_LEN - 1, -1, -1) if P0 in wc[k]), None)

        def dist(p):
            if ref is None:
                return 0.0
            q = next((wc[k][p] for k in range(SEQ_LEN - 1, -1, -1) if p in wc[k]), None)
            return float('inf') if q is None else math.dist(q, ref)
        rest = sorted((p for p in off if p not in (P0, S)), key=dist)
        order = ([P0, S] + rest + [None] * N_OFF)[:N_OFF] + (sorted(dfn, key=dist) + [None] * N_DEF)[:N_DEF]
        tr = torch.from_numpy(window_traj(wc, order)[None])
        with torch.no_grad():
            p = float(np.mean([torch.sigmoid(m(tr, ROLES))[0, pair_index(0, 1)].item() for m in models]))
        best = p if best is None else max(best, p)
    return best


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='court2fix2flip')
    args = ap.parse_args()
    labels = json.load(open(HERE / 'offball_labels.json', encoding='utf-8'))
    # 每支片段在第幾份是考題 → 用那一份的 5 個種子模型
    fold_of, models = {}, {}
    if args.tag.startswith('final'):                     # 粗估用：全部資料訓練的正式模型（看過這些片段 → 持球分數會偏樂觀）
        models[0] = [load_model(CK / f'finetune_contact_{args.tag}_s{s}_best.pt')[0] for s in (1, 2, 3)]
        fold_of = collections.defaultdict(int, {int(Path(q).stem.split('_')[1]): 0 for q in CACHE_DIR.glob('screen_*.pkl')})
    for k in (range(5) if not args.tag.startswith('final') else []):
        models[k] = []
        for s in range(1, 6):
            m, ck = load_model(CK / f'finetune_contact_{args.tag}_s{s}_cv{k}_best.pt')
            models[k].append(m)
            for cid in ck['val_clips']:
                fold_of[cid] = k
    rows = []
    for clip, items in sorted(labels.items()):
        cid = int(clip.split('_')[1])
        p = CACHE_DIR / f'{clip}.pkl'
        if cid not in fold_of or not p.exists():
            continue
        c = pickle.load(open(p, 'rb'))
        ct = finetune_seq.clip_coords(c)
        dets = None
        for it in items:
            S, C, t = it['screener'], it['cutter'], it['idx']
            if C is None:                                    # 沒有切入者編號 → 用規則偵測對上的那組
                if dets is None:
                    dets = [e for e in O.keep(O.detect(ct, c['track_team'], c['handler_tid'])) if not e['onball']]
                m = [e for e in dets if abs(e['t'] - t) <= 5 and (S is None or e['screener'] == S)]
                if not m:
                    rows.append(dict(clip=clip, t=t, v=it['verdict'], skip=True))
                    continue
                e = min(m, key=lambda e: abs(e['t'] - t))
                S, C, t = e['screener'], e['cutter'], e['t']
            ms = models[fold_of[cid]]
            rows.append(dict(clip=clip, t=t, S=S, C=C, v=it['verdict'], note=it.get('note', ''), skip=False,
                             off=pair_score(ms, c, ct, C, S, t), on=pair_score(ms, c, ct, c['handler_tid'], S, t)))
    ok = [r for r in rows if not r['skip'] and r['off'] is not None]
    pos = [r for r in ok if r['v'] == 't']
    neg = [r for r in ok if r['v'] == 'f']
    print(f'標記 {len(rows)} 個，能算分 {len(ok)}（無球掩護 {len(pos)}、規則抓錯 {len(neg)}），對不上規則偵測而跳過 {sum(r["skip"] for r in rows)}')
    on0 = lambda r: r['on'] or 0.0
    print(f"無球分數：無球掩護 中位數 {np.median([r['off'] for r in pos]):.2f}、規則抓錯 {np.median([r['off'] for r in neg]):.2f}；"
          f"AUC {auc([r['off'] for r in pos], [r['off'] for r in neg]):.2f}")
    print(f"持球分數：無球掩護 中位數 {np.median([on0(r) for r in pos]):.2f}、規則抓錯 {np.median([on0(r) for r in neg]):.2f}；"
          f"AUC（越低越好） {auc([on0(r) for r in pos], [on0(r) for r in neg]):.2f}")
    print(f"無球 − 持球：AUC {auc([r['off'] - on0(r) for r in pos], [r['off'] - on0(r) for r in neg]):.2f}")
    print('\n準確率（候選裡真的是無球掩護的比例）在不同篩選下：')
    for th_off, th_on in ((0, 1.01), (0.3, 1.01), (0.5, 1.01), (0, 0.5), (0.3, 0.5), (0.5, 0.5), (0.3, 0.3)):
        keep_ = [r for r in ok if r['off'] >= th_off and on0(r) < th_on]
        tp = sum(r['v'] == 't' for r in keep_)
        print(f'  無球 ≥ {th_off}、持球 < {th_on}：留下 {len(keep_)}，真的 {tp}（準確率 {tp / max(1, len(keep_)):.0%}），'
              f'無球掩護保留 {tp}/{len(pos)}')
    print('\n逐個（答案 片段 影格 掩護者→切入者 無球分數 持球分數 備註）：')
    for r in sorted(ok, key=lambda r: (r['v'], -r['off'])):
        print(f"  {r['v']} {r['clip']} {r['t']} #{r['S']}→#{r['C']} {r['off']:.2f} {on0(r):.2f} {r['note'][:12]}")
    pickle.dump(rows, open(ROOT / 'stubs' / 'offball_zeroshot.pkl', 'wb'))


if __name__ == '__main__':
    main()
