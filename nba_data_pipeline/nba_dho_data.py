"""
nba_dho_data.py — NBA 追蹤資料 → ScreenNet 訓練窗（含手遞手標籤 + 「誰拿球」通道），2026-11-19。

跟 nba_contact_data.py 同一套（5fps、10 張窗、絕對球場座標、接觸幀高斯軟標籤、候選人輪流 slot 1、轉播式雜訊），多了：
  - 手遞手事件（nba_dho_weak.handoffs_from_parsed：NETS 規則 + 持球 ≥ 0.4 秒 + 離籃框 ≥ 15 呎 + 給球者擋到人；人工抽查 17/20）
    → 事件 tid = 給球者、handler = 接球者、dho=True（我們自己的標法：掩護者 = 給球者、被掩護的人 = 接球者）
  - 同一對（給球者 → 接球者）1 秒內的 NETS 擋拆 = 同一個手遞手 → 不另算一般擋拆
  - c['holder']：逐張持球者（SportVU 球的位置，準）→ build_windows(holder_ch=True) 每人多一個通道
  - pairwise_dho：手遞手那組的目標（build_windows(with_dho=True)）
片段：同一個被掩護的人（持球者 / 接球者）、3 秒內的擋拆 / 手遞手併一段，前後各 3 秒。
clip_ids 負數（finetune.py 看到負數一律放訓練集）。

用法：python nba_data_pipeline/nba_dho_data.py --games 700 --workers 8        # 全部比賽；手遞手片段全收、一般擋拆片段抽到同樣窗數
輸出：nba_data_pipeline/output_contact_nbadho/{positive,negative}_sequences.npz（trajectories (N, 10, 10, 3)、pairwise_labels、pairwise_dho）
"""
import os
import sys
import json
import random
import argparse
import tempfile
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from extract_sequences import parse_moment, find_ball_handler, dist
from nba_coverage_probe import segments, find_screens, FPS, SX, SY
from convert_contact import build_windows, N_OFF, N_DEF
from nba_dho_weak import handoffs_from_parsed, games, load_game

OUT = HERE / 'output_contact_nbadho'
STEP = FPS // 5                         # 25fps → 5fps
PAD = 3 * 5                             # 片段前後各 3 秒（5fps 張數）


def timeline(g):
    """game json → 去重、排好的 moments（同 nba_coverage_probe.load_timeline，但吃已讀進來的 dict）。"""
    seen, frames = set(), []
    for ev in g['events']:
        for m in ev['moments']:
            key = (m[0], round(m[2], 2))
            if key in seen or len(m[5]) != 11:
                continue
            seen.add(key); frames.append(m)
    frames.sort(key=lambda m: (m[0], -m[2]))
    return frames


def chunks_from_segment(parsed, handlers):
    """一段連續追蹤 → [c 快取格式 dict]：擋拆（NETS 三角形、持球者離籃框 ≥ 16 呎）+ 手遞手，同一個被掩護的人 3 秒內併一段。"""
    evs = []
    for s in find_screens(parsed, handlers):
        hx, hy = parsed[s['cf']]['players'][s['H']]['pos']
        rim = (88.75, 25.0) if hx >= 47 else (5.25, 25.0)
        if dist((hx, hy), rim) >= 16:
            evs.append(dict(H=s['H'], S=s['S'], team=s['team'], cf=s['cf'], dho=False))
    for h in handoffs_from_parsed(parsed, handlers):
        evs.append(dict(H=h['receiver'], S=h['giver'], team=h['team'], cf=h['idx'], dho=True))
    # 同一對 1 秒內：手遞手優先（擋拆規則也會在交接時觸發）
    dh = [e for e in evs if e['dho']]
    evs = [e for e in evs if e['dho'] or not any(d['H'] == e['H'] and d['S'] == e['S'] and abs(d['cf'] - e['cf']) <= FPS for d in dh)]
    groups = []
    for e in sorted(evs, key=lambda x: x['cf']):
        g = groups[-1] if groups else None
        if g and g[-1]['H'] == e['H'] and e['cf'] - g[-1]['cf'] <= 3 * FPS:
            g.append(e)
        else:
            groups.append([e])
    out = []
    for g in groups:
        f0, f1 = g[0]['cf'] - PAD * STEP, g[-1]['cf'] + PAD * STEP
        if f0 < 0 or f1 >= len(parsed):
            continue
        frames = list(range(f0, f1 + 1, STEP))
        ct = [{p: (i['pos'][0] * SX, i['pos'][1] * SY) for p, i in parsed[f]['players'].items()} for f in frames]
        team = {p: ('offense' if i['team'] == g[0]['team'] else 'defense') for p, i in parsed[g[0]['cf']]['players'].items()}
        holder = [(handlers[f][0] if handlers[f] is not None else None) for f in frames]
        events = [{'idx': int(round((e['cf'] - f0) / STEP)), 'tid': e['S'], 'contact': True, 'dho': e['dho']} for e in g]
        out.append({'n': len(ct), 'court_tracks': ct, 'raw_tracks': ct, 'track_team': team, 'holder': holder,
                    'handler_tid': g[0]['H'], 'screener_tids': sorted({e['S'] for e in g}),
                    'defender_tid': None, 'events': events, 'valid_start_idx': 0})
    return out


def game_chunks(name):
    try:
        g = load_game(name)
    except Exception:
        return []
    frames = timeline(g)
    out = []
    for seg in segments(frames) if frames else []:
        if len(seg) < 2 * PAD * STEP + 10:
            continue
        parsed = [parse_moment(m) for m in seg]
        out += chunks_from_segment(parsed, find_ball_handler(parsed, None, None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=50)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--anchor', type=int, default=3)
    ap.add_argument('--min_target', type=float, default=0.05)
    ap.add_argument('--max_windows', type=int, default=150000)
    ap.add_argument('--other_ratio', type=float, default=1.0, help='只有一般擋拆的片段，窗數最多是手遞手片段窗數的幾倍（手遞手片段全收）')
    ap.add_argument('--holder_flip', type=float, default=0.1, help='持球通道雜訊：每張以這個機率換成 None / 隔壁的人（轉播持球者約 89% 對）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    import sportvu_noise
    from multiprocessing import Pool
    gs = games()
    random.Random(args.seed).shuffle(gs)
    chunks = []
    with Pool(args.workers) as pool:
        for k, ch in enumerate(pool.imap_unordered(game_chunks, gs[:args.games]), 1):
            chunks += ch
            if k % 10 == 0:
                print(f'{k}/{args.games} 場、{len(chunks)} 段', flush=True)
    rng = np.random.RandomState(args.seed)
    roles = np.array([0] + [1] * (N_OFF - 1) + [2] * N_DEF, dtype=np.int64)
    T, P, PD, C, CTR = [], [], [], [], []
    n_dho_ev = n_scr_ev = 0
    # 手遞手片段全收（先排），只有擋拆的片段抽到窗數 ≈ other_ratio × 手遞手片段窗數
    has_dho = [k for k in range(len(chunks)) if any(e['dho'] for e in chunks[k]['events'])]
    dho_set = set(has_dho)
    others = [k for k in range(len(chunks)) if k not in dho_set]
    order = list(rng.permutation(has_dho)) + list(rng.permutation(others))
    n_dho_win = None
    for k in order:
        if n_dho_win is None and k not in dho_set:
            n_dho_win = len(T)                        # 手遞手片段做完了
        if n_dho_win is not None and len(T) - n_dho_win >= args.other_ratio * n_dho_win:
            break
        c = sportvu_noise.apply_noise(chunks[k], rng, sportvu_noise.BROADCAST)
        hd = list(c['holder'])
        for i in range(len(hd)):                    # 持球通道加雜訊（轉播的持球者不完美）
            if hd[i] is not None and rng.rand() < args.holder_flip:
                hd[i] = None if rng.rand() < 0.5 else rng.choice([p for p in c['court_tracks'][i]] or [None])
        c['holder'] = hd
        n_dho_ev += sum(e['dho'] for e in c['events']); n_scr_ev += sum(not e['dho'] for e in c['events'])
        for traj, pw, pwd, center in build_windows(c, args.sigma, args.sigma * 1.5, args.min_target, args.anchor,
                                                   holder_ch=True, with_dho=True):
            T.append(traj); P.append(pw); PD.append(pwd); C.append(-(int(k) + 1)); CTR.append(center)
        if len(T) >= args.max_windows:
            break
    T, P, PD = np.stack(T), np.stack(P), np.stack(PD)
    C, CTR = np.array(C, np.int32), np.array(CTR, np.int32)
    lab = (P.max(axis=1) >= 0.5).astype(np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, m in (('positive_sequences.npz', lab == 1), ('negative_sequences.npz', lab == 0)):
        np.savez_compressed(str(out / name), trajectories=T[m], roles=np.repeat(roles[None], m.sum(), 0),
                            labels=lab[m], pairwise_labels=P[m], pairwise_dho=PD[m], clip_ids=C[m], centers=CTR[m],
                            anchor=np.int32(args.anchor), sigma=np.float32(args.sigma), holder_ch=np.int32(1))
    print(f'{len(chunks)} 段（用了 {len(set(C.tolist()))} 段：擋拆事件 {n_scr_ev}、手遞手事件 {n_dho_ev}）→ {len(T)} 個窗：'
          f'掩護正 {int(lab.sum())}、其中手遞手正（≥ 0.5）{int((PD.max(axis=1) >= 0.5).sum())}、負 {int((1 - lab).sum())} → {out}')


if __name__ == '__main__':
    main()
