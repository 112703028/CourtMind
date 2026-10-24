"""
ingest_new_labels.py — 使用者標好的新片段（handler_labels_<起>_<迄>.csv）→ 更新快取 → 產生訓練窗。

使用者只標了：持球者（錯才改）、掩護者#、撞上5fps影格（多個用逗號，例 8,11 / 5,11）、防守方式。
分隊沒標 → 這裡自動重做，而且用上使用者標的角色：
  - 持球者、掩護者一定是進攻（固定）
  - 出現最多的 10 個 id = 場上球員 → 強制 5 對 5
  - 兩個依據一起：TeamSiamese 兩兩同隊機率（整段平均）＋ 球衣顏色（上半身 Lab 中位數距離）
  - 其他 id（替補、碎片）：跟哪一邊比較像就歸哪邊
不放進訓練：防守方式寫「無球檔」的掩護（只標幫持球者擋的，同舊規則）、整支片段在 SKIP_CLIPS。

輸出：
  stubs/train_contact_cache/<clip>.pkl   pending=False、events = 使用者標的掩護、track_team = 新分隊
  nba_data_pipeline/output_contact_new/{positive,negative}_sequences.npz
      跟 convert_contact.build 同一套 build_windows；clip_ids 用負數 → finetune --extra_data_dir 一律放訓練，
      考題（舊 141 支）的 5 份切法不變，可以直接跟之前的 F1 比
  兩兩機率快取：stubs/team_pair_cache/<clip>.pkl

用法：
  python nba_data_pipeline/ingest_new_labels.py output_videos/new_clip_check/handler_labels_185_282.csv
"""
import sys
import csv
import glob
import pickle
import argparse
import itertools
import collections
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from convert_contact import FRAMES_DIR, CACHE_DIR, build_windows, N_OFF, N_DEF

PAIR_CACHE = ROOT / 'stubs' / 'team_pair_cache'
OUT = HERE / 'output_contact_new'

# 使用者在對話裡補充的（2026-10-20）：表格空白的兩支
FIXES = {'screen_243': {'screeners': [1], 'frames': [8]},
         'screen_217': {'screeners': [7], 'frames': [11]}}
# 192：持球者 #13 → #18（中途換人）、且是無球擋 → 不放進訓練
SKIP_CLIPS = {'screen_192'}
OFFBALL = {'無球檔', '無球擋'}


def ids(s):
    return [int(x) for x in str(s).replace('#', '').replace('，', ',').split(',') if x.strip().isdigit()]


def parse_labels(path):
    rows = list(csv.reader(open(path, encoding='utf-8-sig')))
    head, rows = rows[0], rows[1:]
    out = {}
    for r in rows:
        r = (r + [''] * 7)[:7]
        clip, h_model, h_fix, scr, frm, cov, note = [x.strip() for x in r]
        if not clip:
            continue
        d = out.setdefault(clip, {'handler': None, 'screens': [], 'raw': []})
        h = ids(h_fix) or ids(h_model)
        if h and d['handler'] is None:
            d['handler'] = h[0]
        fx = FIXES.get(clip)
        d['raw'].append((scr, frm, cov))
        if fx and not scr:                          # 表格空白、在對話裡補的
            s_ids, f_ids = fx['screeners'], fx['frames']
        else:
            if cov in OFFBALL or scr in ('無', ''):
                continue
            s_ids, f_ids = ids(scr), ids(frm)
        if len(f_ids) == 1 and len(s_ids) > 1:
            f_ids = f_ids * len(s_ids)
        for s, f in zip(s_ids, f_ids):
            d['screens'].append({'tid': s, 'idx': f, 'coverage': cov})
    return out


# ── 分隊 ──────────────────────────────────────────────────────────────────────

def color_feat(frame, bb):
    x1, y1, x2, y2 = [int(v) for v in bb]
    h, w = y2 - y1, x2 - x1
    crop = frame[max(0, y1 + h // 6): max(0, y1 + h // 2), max(0, x1 + w // 4): max(0, x2 - w // 4)]
    if crop.size < 30:
        return None
    return np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32), axis=0)


def pair_stats(clip, c, ta):
    p = PAIR_CACHE / f'{clip}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    import torch
    from nba_data_pipeline.team_classifier.inference import _crop_pil
    jpgs = sorted(glob.glob(str(FRAMES_DIR / clip / '*.jpg')), key=lambda q: int(Path(q).stem.rsplit('_', 1)[1]))
    S, C, cols = collections.defaultdict(float), collections.defaultdict(int), collections.defaultdict(list)
    with torch.no_grad():
        for f, q in enumerate(jpgs):
            fr = cv2.imread(q)
            pids, crops = [], []
            for pid, bb in c['raw_tracks'][f].items():
                cf = color_feat(fr, bb)
                if cf is not None:
                    cols[pid].append(cf)
                cr = _crop_pil(fr, bb)
                if cr is not None:
                    pids.append(pid); crops.append(cr)
            if len(pids) < 2:
                continue
            E = ta._encode_crops(crops)
            ii, jj = zip(*itertools.combinations(range(len(pids)), 2))
            pr = torch.sigmoid(ta.model.compare(E[list(ii)], E[list(jj)])).cpu().numpy()
            for a, b, v in zip(ii, jj, pr):
                k = tuple(sorted((pids[a], pids[b]))); S[k] += float(v); C[k] += 1
    d = {'siam': {k: S[k] / C[k] for k in S}, 'color': {q: np.median(v, axis=0) for q, v in cols.items()}}
    PAIR_CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(d, open(p, 'wb'))
    return d


COURT_W, COURT_H = 91.8, 49.2


def on_court_ratio(c):
    """每個 id 是不是場上球員（回傳 1.0 = 是、0.0 = 不是，給 assign_teams 用 ≥ 0.6 判斷）：
      - 「這張有 homography」的影格裡，至少 30% 有球場座標（板凳 / 觀眾 / 頭像多半完全沒有）
      - 有座標的影格裡，至少 70% 腳在球場界內（觀眾、頭像會在界外 3~4 ft）
    遠端邊線附近的真球員常因座標略超出邊線而缺幀（座標覆蓋 40~60%），但有座標時都在界內 → 保留。"""
    seen, has, inside = collections.Counter(), collections.Counter(), collections.Counter()
    for raw, ct in zip(c['raw_tracks'], c['court_tracks']):
        if not ct:                                   # 這張 homography 失敗，不算
            continue
        for p in raw:
            seen[p] += 1
            xy = ct.get(p)
            if xy is not None:
                has[p] += 1
                if 0 <= xy[0] <= COURT_W and 0 <= xy[1] <= COURT_H:
                    inside[p] += 1
    return {p: float(has[p] / seen[p] >= 0.3 and inside[p] / max(1, has[p]) >= 0.7) for p in seen}


def clean_colors(clip, c):
    """球衣顏色只取「這個人的框沒跟別人重疊」的影格（重疊時會混到別人的顏色）。快取在 team_color_cache。"""
    cache = ROOT / 'stubs' / 'team_color_cache' / f'{clip}.pkl'
    if cache.exists():
        return pickle.load(open(cache, 'rb'))
    from convert_contact import iou
    jpgs = sorted(glob.glob(str(FRAMES_DIR / clip / '*.jpg')), key=lambda q: int(Path(q).stem.rsplit('_', 1)[1]))
    cols = collections.defaultdict(list)
    for f, q in enumerate(jpgs):
        boxes = c['raw_tracks'][f]
        fr = None
        for pid, bb in boxes.items():
            if any(iou(bb, ob) > 0.05 for op, ob in boxes.items() if op != pid):
                continue
            if fr is None:
                fr = cv2.imread(q)
            cf = color_feat(fr, bb)
            if cf is not None:
                cols[pid].append(cf)
    out = {p: np.median(v, axis=0) for p, v in cols.items() if len(v) >= 2}
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(cache, 'wb'))
    return out


def assign_teams(c, d, offense_fixed, defense_fixed=(), colors=None):
    """
    2026-10-22 改版（整體分隊錯誤分析後）：
      1. 先過濾非球員：出現 ≥ 1/3 影格、且腳在球場界內的比例 ≥ 60% 才算場上球員（主要名單最多 10 人）
         → 轉播小頭像、觀眾、板凳不再佔名額；場外 id 不分隊（不會變成候選人）
      2. 場上人數不足 10 → 不強制 5 對 5，每隊 ≤ 5 人即可
      3. 球衣顏色只用沒重疊的影格（clean_colors）
      4. 分隊模型的權重看它有多「果斷」：同隊機率都在 0.5 附近（分不出來）→ 權重低，主要看顏色
    """
    app = collections.Counter(p for fd in c['raw_tracks'] for p in fd)
    ratio = on_court_ratio(c)
    valid = [p for p, k in app.most_common() if k >= c['n'] / 3 and ratio.get(p, 0) >= 0.6]
    main = valid[:10]
    for p in list(offense_fixed) + list(defense_fixed):     # 使用者標的角色一定算進場上球員
        if p in app and p not in main:
            main.append(p)
    fixed = [p for p in offense_fixed if p in main]
    dfix = [p for p in defense_fixed if p in main and p not in fixed]
    others = [p for p in main if p not in fixed and p not in dfix]
    k = len(main)
    sizes = {k // 2, (k + 1) // 2} if k >= 10 else {s for s in range(max(1, k - 5), min(5, k) + 1)}
    col = colors if colors is not None else d['color']
    dist = {}
    cm = [p for p in main if p in col]
    for a, b in itertools.combinations(cm, 2):
        dist[(a, b)] = dist[(b, a)] = float(np.linalg.norm(col[a] - col[b]))
    t = float(np.median(list(dist.values()))) if dist else 1.0
    s = max(1.0, 0.25 * t)

    def ps(a, b):
        return d['siam'].get(tuple(sorted((a, b))), 0.5)

    def pc(a, b):
        return 1 / (1 + np.exp((dist[(a, b)] - t) / s)) if (a, b) in dist else 0.5

    pairs = list(itertools.combinations(main, 2))
    decisive = np.mean([abs(ps(a, b) - 0.5) for a, b in pairs]) if pairs else 0.0
    w_siam = float(min(1.0, decisive / 0.25))        # 0.25 以上 = 很果斷，全權；都在 0.5 附近 → 接近 0

    def ll(A):
        v = 0.0
        for a, b in pairs:
            same = (a in A) == (b in A)
            for f, w in ((ps, w_siam), (pc, 1.0)):
                q = min(max(f(a, b), 1e-3), 1 - 1e-3)
                v += w * np.log(q if same else 1 - q)
        return v

    best = None
    for size in sizes:
        need = size - len(fixed)
        if need < 0 or need > len(others):
            continue
        for extra in itertools.combinations(others, need):
            A = set(fixed) | set(extra)
            v = ll(A)
            if best is None or v > best[0]:
                best = (v, A)
    if best is None:                                # 標的進攻人數超過一半（不太可能）→ 只固定角色
        A = set(fixed)
    else:
        A = best[1]
    B = [p for p in main if p not in A]
    team = {p: 'offense' if p in A else 'defense' for p in main}

    def pc_any(a, b):
        if a in col and b in col:
            return 1 / (1 + np.exp((float(np.linalg.norm(col[a] - col[b])) - t) / s))
        return 0.5

    for p in app:                                   # 碎片 id（在場上、但出現時間短）：跟哪邊比較像就歸哪邊
        if p in team or ratio.get(p, 0) < 0.6:      # 場外的（板凳 / 觀眾 / 頭像）不分隊
            continue
        sa = np.mean([(w_siam * ps(p, q) + pc_any(p, q)) / (1 + w_siam) for q in A]) if A else 0.5
        sb = np.mean([(w_siam * ps(p, q) + pc_any(p, q)) / (1 + w_siam) for q in B]) if B else 0.5
        team[p] = 'offense' if sa >= sb else 'defense'
    for p in offense_fixed:
        team[p] = 'offense'
    for p in defense_fixed:
        if p not in offense_fixed:
            team[p] = 'defense'
    return team


def contacted_defenders(c):
    """撞上那一刻（前後 1 張）離掩護者最近的人（持球者、掩護者除外）＝被擋的防守者。舊資料裡這條 99% 成立。"""
    import math
    out = []
    off = {c['handler_tid']} | set(c['screener_tids'])
    for e in c['events']:
        S, f = e['tid'], e['idx']
        best = None
        for g in range(max(0, f - 1), min(c['n'], f + 2)):
            pos = c['court_tracks'][g]
            if S not in pos:
                continue
            for q, xy in pos.items():
                if q in off:
                    continue
                dd = math.dist(pos[S], xy)
                if best is None or dd < best[0]:
                    best = (dd, q)
        if best is not None:
            out.append(best[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='+', help='handler_labels_*.csv（可多個）')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    labels = {}
    for p in args.labels:
        labels.update(parse_labels(p))
    from nba_data_pipeline.team_classifier.inference import TeamAssignerSiamese
    ta = TeamAssignerSiamese(ckpt_path=str(ROOT / 'models' / 'team_siamese.pt'), device=args.device)

    roles = np.array([0] + [1] * (N_OFF - 1) + [2] * N_DEF, dtype=np.int64)
    T, P, Cid, CTR = [], [], [], []
    n_scr, skipped, changed_h = 0, [], 0
    for k, (clip, lab) in enumerate(sorted(labels.items(), key=lambda kv: int(kv[0].split('_')[1]))):
        cp = CACHE_DIR / f'{clip}.pkl'
        if clip in SKIP_CLIPS or not cp.exists() or lab['handler'] is None:
            skipped.append(clip)
            continue
        c = pickle.load(open(cp, 'rb'))
        if c['handler_tid'] != lab['handler']:
            changed_h += 1
        c['handler_tid'] = lab['handler']
        scr = [s for s in lab['screens'] if 0 <= s['idx'] < c['n']]
        step = c['fps'] / 5.0
        c['events'] = [{'frame': round(s['idx'] * step), 'idx': s['idx'], 'contact': True, 'tid': s['tid'],
                        'tid_method': 'user', 'coverage': s['coverage']} for s in scr]
        c['screener_tids'] = sorted({s['tid'] for s in scr})
        d = pair_stats(clip, c, ta)
        c['track_team'] = assign_teams(c, d, [c['handler_tid']] + c['screener_tids'], contacted_defenders(c),
                                       colors=clean_colors(clip, c))
        c['team_method'] = 'oncourt+cleancolor+weighted_siam+user_roles+contact_def'
        c['pending'] = False
        c['source'] = 'new_clip_labeled'
        pickle.dump(c, open(cp, 'wb'))
        n_scr += len(scr)
        cid = int(clip.split('_')[1])
        for traj, pw, center in build_windows(c, 1.0, 1.5, 0.01, 3):
            T.append(traj); P.append(pw); Cid.append(-cid); CTR.append(center)
        if (k + 1) % 10 == 0:
            print(f'{k + 1}/{len(labels)}', flush=True)

    T, P = np.stack(T), np.stack(P)
    Cid, CTR = np.array(Cid, np.int32), np.array(CTR, np.int32)
    lab = (P.max(axis=1) >= 0.5).astype(np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, m in (('positive_sequences.npz', lab == 1), ('negative_sequences.npz', lab == 0)):
        np.savez_compressed(str(out / name), trajectories=T[m], roles=np.repeat(roles[None], m.sum(), 0),
                            labels=lab[m], pairwise_labels=P[m], clip_ids=Cid[m], centers=CTR[m],
                            anchor=np.int32(3), sigma=np.float32(1.0))
    print(f'{len(labels) - len(skipped)} 支片段、{n_scr} 個掩護（持球者改了 {changed_h} 支）→ {len(T)} 個窗：'
          f'正 {int(lab.sum())}、負 {int((1 - lab).sum())} → {out}')
    print(f'跳過：{skipped}')


if __name__ == '__main__':
    main()
