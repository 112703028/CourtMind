"""
eval_contact.py — 事件級驗證：ScreenNet 抓到的擋拆「時間點 + 掩護者」準不準。

窗口級的 AUC/F1 會把「只差 0.4 秒」也算錯，看不出事件定位的好壞。這支照推論的做法：
  1. 每個窗、每個進攻候選人輪流放到 slot 1（跟 inference.detect_screens 一樣），
     算 pair(handler, slot1) 的分數 → 每個候選人一條分數曲線
  2. 曲線找尖峰（>= 門檻、間隔 >= nms 張）→ 每個尖峰 = 一次預測的擋拆
  3. 跟人工接觸幀比：尖峰在接觸幀 ±tol 張內、且是同一個掩護者 → 抓對

只用 checkpoint 記錄的 val_clips（沒訓練過的 clip），資料來自 stubs/train_contact_cache。
舊模型（finetune_manual_best.pt）也可用同樣流程比較。

用法：
  python nba_data_pipeline/eval_contact.py --ckpt nba_data_pipeline/checkpoints/finetune_contact_v1_best.pt
  python nba_data_pipeline/eval_contact.py --ckpt 新.pt --compare 舊1.pt --compare 舊2.pt
"""
import sys
import math
import pickle
import argparse
import collections
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from nba_data_pipeline.model import ScreenNetPairwise, to_handler_relative
from convert_contact import CACHE_DIR, SEQ_LEN, N_OFF, N_DEF, N_PLAYERS, pair_index

ROLES = torch.tensor([0] + [1] * (N_OFF - 1) + [2] * N_DEF).unsqueeze(0)


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = ScreenNetPairwise(ck.get('d_model', 64), ck.get('n_heads', 4), ck.get('n_layers', 2))
    m.load_state_dict(ck['model'])
    m.eval()
    m.rel_coords = bool((ck.get('data_meta') or {}).get('rel_coords'))   # finetune --rel_coords 訓練的
    m.clean_coords = bool((ck.get('data_meta') or {}).get('clean_coords'))   # 訓練資料有清座標 → 評估也清
    return m, ck


def window_traj(wc, order):
    """跟 convert_contact.build_windows 同樣的軌跡組法（缺幀沿用最後已知位置）。"""
    traj = np.zeros((N_PLAYERS, SEQ_LEN, 2), dtype=np.float32)
    last = [None] * N_PLAYERS
    for k in range(SEQ_LEN):
        for pi, p in enumerate(order):
            if p is not None and p in wc[k]:
                traj[pi, k] = wc[k][p]
                last[pi] = wc[k][p]
            elif last[pi] is not None:
                traj[pi, k] = last[pi]
    return traj


@torch.no_grad()
def score_curves(model, c, anchor):
    """→ {候選人 tid: {anchor 影格: 分數}}（每個進攻候選人輪流當 slot 1）。"""
    n, ct, raw, team, H = c['n'], c['court_tracks'], c['raw_tracks'], c['track_team'], c['handler_tid']
    if getattr(model, 'clean_coords', False):
        from tactics_tools import clean_court_tracks
        ct = clean_court_tracks(ct)[0]
    curves = collections.defaultdict(dict)
    pidx = pair_index(0, 1)
    for start in range(c['valid_start_idx'], n - SEQ_LEN + 1):
        wr, wc = raw[start:start + SEQ_LEN], ct[start:start + SEQ_LEN]
        cnt = collections.Counter(p for fd in wr for p in fd)
        present = {p for p, k in cnt.items() if k >= SEQ_LEN // 2}
        if H is not None and cnt.get(H, 0) >= 1:
            present.add(H)
        off = [p for p in present if team.get(p) == 'offense']
        dfn = [p for p in present if team.get(p) == 'defense']
        if H is None or H not in off or len(off) < 2:
            continue
        hpos = next((wc[k][H] for k in range(SEQ_LEN - 1, -1, -1) if H in wc[k]), None)

        def dist_to_h(p):
            if hpos is None:
                return 0.0
            pos = next((wc[k][p] for k in range(SEQ_LEN - 1, -1, -1) if p in wc[k]), None)
            return float('inf') if pos is None else math.dist(pos, hpos)

        order_def = (sorted(dfn, key=dist_to_h) + [None] * N_DEF)[:N_DEF]
        cands = [p for p in off if p != H]
        batch = []
        for cand in cands:
            others = sorted((p for p in off if p not in (H, cand)), key=dist_to_h)
            order_off = ([H, cand] + others + [None] * N_OFF)[:N_OFF]
            batch.append(window_traj(wc, order_off + order_def))
        traj = np.stack(batch)
        if getattr(model, 'rel_coords', False):
            traj = to_handler_relative(traj)
        traj = torch.from_numpy(traj)
        prob = torch.sigmoid(model(traj, ROLES.expand(len(batch), -1)))[:, pidx].numpy()
        for cand, pr in zip(cands, prob):
            curves[cand][start + anchor] = float(pr)
    return curves


def find_peaks(curve, thr, nms):
    """曲線（{影格: 分數}）找尖峰：>= thr 的局部最大，彼此間隔 >= nms 張（高的優先）。"""
    pts = sorted(curve.items())
    cands = []
    for i, (f, s) in enumerate(pts):
        if s < thr:
            continue
        left = pts[i - 1][1] if i > 0 else -1
        right = pts[i + 1][1] if i < len(pts) - 1 else -1
        if s >= left and s >= right:
            cands.append((s, f))
    peaks = []
    for s, f in sorted(cands, reverse=True):
        if all(abs(f - g) >= nms for _, g in peaks):
            peaks.append((s, f))
    return [(f, s) for s, f in peaks]


def plateau_width(curve, thr):
    """>= thr 的最長連續段長度（看還是不是平台）。"""
    best = run = 0
    prev = None
    for f, s in sorted(curve.items()):
        if s >= thr and prev is not None and f == prev + 1 and run:
            run += 1
        else:
            run = 1 if s >= thr else 0
        best = max(best, run)
        prev = f
    return best


def evaluate(model, anchor, clips, thr, tol, nms):
    tp = fp = fn = time_only = 0
    errs, widths = [], []
    for clip in clips:
        c = pickle.load(open(CACHE_DIR / f'{clip}.pkl', 'rb'))
        curves = score_curves(model, c, anchor)
        preds = [(f, cand, s) for cand, cv in curves.items() for f, s in find_peaks(cv, thr, nms)]
        widths += [plateau_width(cv, thr) for cv in curves.values()]
        gts = [(e['idx'], e['tid']) for e in c['events'] if e['tid'] is not None]
        used = set()
        for gi, gt in gts:
            m = [(abs(f - gi), k) for k, (f, cand, s) in enumerate(preds)
                 if k not in used and cand == gt and abs(f - gi) <= tol]
            if m:
                d, k = min(m)
                used.add(k)
                tp += 1
                errs.append(d)
            else:
                fn += 1
                if any(abs(f - gi) <= tol for f, _, _ in preds):
                    time_only += 1      # 時間對了但掩護者認錯人
        fp += len(preds) - len(used)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'prec': prec, 'rec': rec,
            'f1': 2 * prec * rec / max(1e-9, prec + rec),
            'err': float(np.mean(errs)) if errs else float('nan'),
            'time_only': time_only,
            'width': float(np.mean([w for w in widths if w > 0])) if any(widths) else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='新模型（提供 val_clips 與 anchor）')
    ap.add_argument('--compare', action='append', default=[],
                    help='比較用的其他模型（可給多次）；用 --ckpt 的 val_clips 一起考')
    ap.add_argument('--tol', type=int, default=2, help='尖峰離接觸幀幾張內算對（5fps，2 = 0.4 秒）')
    ap.add_argument('--nms', type=int, default=15, help='同一人兩個尖峰至少隔幾張（15 = 3 秒，同 inference 預設）')
    args = ap.parse_args()

    model, ck = load_model(args.ckpt)
    clips = [f'screen_{c}' for c in ck.get('val_clips', [])]
    anchor = int(ck.get('data_meta', {}).get('anchor', 3))
    if not clips:
        print('checkpoint 沒有 val_clips（不是用 convert_contact 資料訓練的？）')
        return
    print(f"驗證 clip {len(clips)} 支：{clips}\nanchor 第 {anchor} 格、±{args.tol} 張內算對、尖峰間隔 ≥{args.nms} 張\n")
    models = [(Path(args.ckpt).stem, model)]
    for cp in args.compare:
        models.append((Path(cp).stem, load_model(cp)[0]))
    for name, m in models:
        print(f"── {name} ──")
        print("  門檻 | 抓對 抓錯 漏抓 | 準確率 抓到率  F1  | 平均時間誤差 | 時間對但認錯人 | 超過門檻的平均連續長度")
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            r = evaluate(m, anchor, clips, thr, args.tol, args.nms)
            print(f"  {thr:.1f} | {r['tp']:4d} {r['fp']:4d} {r['fn']:4d} | {r['prec']:5.2f}  {r['rec']:5.2f}  {r['f1']:4.2f} |"
                  f"   {r['err']:.2f} 張    |      {r['time_only']:3d}      |   {r['width']:.1f} 張")
        print()


if __name__ == '__main__':
    main()
