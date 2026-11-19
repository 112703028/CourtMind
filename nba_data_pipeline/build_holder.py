"""
build_holder.py — 轉播訓練窗加「誰拿球」通道 + 手遞手目標（2026-11-19，配 nba_dho_data.py 的 NBA 手遞手弱標籤一起訓練）。

同 build_court2.py（訓練快取 + 第 2 版球場座標 + 清座標），多了：
  - c['holder']：批次 events.pkl 的持球者（SAM3 追球 + Viterbi，撞上那刻 89% 對）→ 用框重疊把批次編號換成快取 tid、對到 5fps
  - 手遞手：output_videos/dho_check/dho_check.csv 使用者判「是」的（批次編號 + 交接影格）→ 對到快取裡同一個掩護者、
    交接前 0.6 ~ 後 2.1 秒的擋拆事件 → dho=True（其他擋拆 = 不是手遞手）
    （訓練標記的撞上時間跟交接差：157 早 0.47 秒、149 / 154 晚 1.6~2 秒；160 / 139 根本沒標這次手遞手 → 對不到）
  - build_windows(holder_ch=True, with_dho=True) → trajectories (N, 10, 10, 3)、pairwise_dho
--no_holder：同一批片段、不加通道也不加手遞手（對照組，2 通道）

用法：python nba_data_pipeline/build_holder.py output_contact_court2holder_503
     python nba_data_pipeline/build_holder.py output_contact_court2base_503 --no_holder
"""
import sys
import csv
import glob
import pickle
import argparse
import collections
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path[:0] = [str(ROOT / 'nba_data_pipeline'), str(ROOT)]
from convert_contact import CACHE_DIR, build_windows, N_OFF, N_DEF
import court_v2

BATCH = ROOT / 'output_videos' / 'batch'


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def batch_to_cache(c, d):
    """→ {批次編號: 快取 tid}（整支片段框重疊加總最多的一對一配對）。"""
    step = c['fps'] / 5
    score = collections.Counter()
    for i, fd in enumerate(c['raw_tracks']):
        f = int(round(i * step))
        if f >= len(d['player_tracks']):
            break
        for t, bx in fd.items():
            if bx is None:
                continue
            for q, v in d['player_tracks'][f].items():
                bb = v.get('bbox') if isinstance(v, dict) else v
                if bb is not None:
                    s = _iou(bx, bb)
                    if s > 0.3:
                        score[(q, t)] += s
    mp, used = {}, set()
    for (q, t), s in score.most_common():
        if q not in mp and t not in used and s >= 2.0:
            mp[q] = t; used.add(t)
    return mp


def holder_track(c, d, mp):
    """快取 5fps 每張的持球者 tid（該張前後半個 5fps 間隔裡批次持球者的多數）。"""
    step = c['fps'] / 5
    ba = d.get('ball_acquisition') or []
    out = []
    for i in range(c['n']):
        a, b = int(round((i - 0.5) * step)), int(round((i + 0.5) * step))
        cnt = collections.Counter(mp.get(ba[f]) for f in range(max(0, a), min(len(ba), b)) if ba[f] not in (None, -1))
        cnt.pop(None, None)
        out.append(cnt.most_common(1)[0][0] if cnt else None)
    return out


def dho_labels():
    """→ {clip: [(交接影格, 給球者批次編號, 接球者批次編號)]}（使用者判是的）。"""
    out = collections.defaultdict(list)
    p = ROOT / 'output_videos' / 'dho_check' / 'dho_check.csv'
    for r in csv.DictReader(open(p, encoding='utf-8-sig')):
        if r['手遞手嗎(t/f/?)'].strip() == 't':
            out[r['clip']].append((int(r['交接影格']), int(r['給球者'].lstrip('#')), int(r['接球者'].lstrip('#'))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--no_holder', action='store_true')
    args = ap.parse_args()
    roles = np.array([0] + [1] * (N_OFF - 1) + [2] * N_DEF, dtype=np.int64)
    dl = dho_labels()
    T, P, PD, C, CTR = [], [], [], [], []
    stat = collections.Counter()
    dho_hit, dho_miss = [], []
    for p in sorted(glob.glob(str(CACHE_DIR / 'screen_*.pkl')), key=lambda q: int(Path(q).stem.split('_')[1])):
        cid = int(Path(p).stem.split('_')[1])
        if cid > 503:
            continue
        c = pickle.load(open(p, 'rb'))
        if c.get('pending'):
            continue
        if cid < 185:                                       # 同 build_court2：分隊不可靠的舊片段不用
            app = collections.Counter(q for fd in c['raw_tracks'] for q in fd)
            main_ = [q for q in c['track_team'] if app[q] >= c['n'] / 3]
            off = sum(1 for q in main_ if c['track_team'][q] == 'offense')
            if off < 4 or len(main_) - off < 4:
                continue
        ct2 = court_v2.court_tracks(c)
        if ct2 is not None:
            c['court_tracks'] = ct2
        if not args.no_holder:
            bp = BATCH / f"{c['clip']}_events.pkl"
            pre = ROOT / 'stubs' / 'backup_events_pretrim' / f"{c['clip']}_events.pkl"
            if bp.exists():
                d = pickle.load(open(bp, 'rb'))
                mp = batch_to_cache(c, d)
                c['holder'] = holder_track(c, d, mp)
                stat['有持球者的張'] += sum(h is not None for h in c['holder']); stat['張'] += c['n']
                step = c['fps'] / 5
                c['events'] = [dict(e) for e in c['events']]
                for f, A, B in dl.get(c['clip'], []):
                    gt = mp.get(A)
                    ev = [e for e in c['events'] if e['tid'] == gt and gt is not None and -0.6 * c['fps'] <= e['idx'] * step - f <= 2.1 * c['fps']]
                    if ev:
                        for e in ev:
                            e['dho'] = True
                        dho_hit.append(f"{c['clip']}@{f}")
                    else:
                        dho_miss.append(f"{c['clip']}@{f}")
            else:
                c['holder'] = [None] * c['n']
                stat['沒有批次資料'] += 1
        ws = build_windows(c, 1.0, 1.5, 0.01, 3, clean=True, holder_ch=not args.no_holder, with_dho=not args.no_holder)
        for w in ws:
            if args.no_holder:
                traj, pw, center = w; pwd = np.zeros(20, np.float32)
            else:
                traj, pw, pwd, center = w
            T.append(traj); P.append(pw); PD.append(pwd); C.append(cid); CTR.append(center)
    T, P, PD = np.stack(T), np.stack(P), np.stack(PD)
    C, CTR = np.array(C, np.int32), np.array(CTR, np.int32)
    lab = (P.max(axis=1) >= 0.5).astype(np.float32)
    out = ROOT / 'nba_data_pipeline' / args.out
    out.mkdir(parents=True, exist_ok=True)
    extra = {} if args.no_holder else {'pairwise_dho': None, 'holder_ch': np.int32(1)}
    for name, m in (('positive_sequences.npz', lab == 1), ('negative_sequences.npz', lab == 0)):
        kw = dict(trajectories=T[m], roles=np.repeat(roles[None], m.sum(), 0), labels=lab[m], pairwise_labels=P[m],
                  clip_ids=C[m], centers=CTR[m], anchor=np.int32(3), sigma=np.float32(1.0), clean_coords=np.float32(1.0),
                  court_v2=np.float32(1.0), court_feet=np.float32(0.0))
        if not args.no_holder:
            kw.update(pairwise_dho=PD[m], holder_ch=np.int32(1))
        np.savez_compressed(str(out / name), **kw)
    print(f'{out.name}：{len(set(C.tolist()))} 支、{len(T)} 窗、正 {int(lab.sum())}、形狀 {T.shape}')
    if not args.no_holder:
        print(f"  持球者有值的張 {stat['有持球者的張']}/{stat['張']}（{stat['有持球者的張'] / max(1, stat['張']):.0%}）、沒有批次資料 {stat['沒有批次資料']} 支")
        print(f"  手遞手對到擋拆事件 {len(dho_hit)} 個 {dho_hit}；對不到 {dho_miss}；手遞手正窗 {int((PD.max(axis=1) >= 0.5).sum())}")


if __name__ == '__main__':
    main()
