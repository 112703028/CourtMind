"""
team_fix.py — 推論分隊的「塌掉偵測 + 換方法」：TeamSiamese 以持球者為錨逐張判同隊，約 14% 片段會塌成「幾乎全部跟持球者同隊」。

偵測（推論時拿得到）：出現最多的 10 人裡，除了持球者以外有 ≥ COLLAPSE_N 人被判成跟持球者同隊（整段多數決）→ 塌掉。
換方法：跟訓練資料同一套 ingest_new_labels.assign_teams —— 抽 5fps，全部兩兩 Siamese 同隊機率（整段平均）+ 球衣顏色
（只用框沒重疊的影格）+ 5 對 5 + Siamese 分不開時自動降權，持球者固定進攻。結果整段固定（每張 1 = 進攻、2 = 防守）。

2026-11-05 在 462 支訓練片段比（舊片段 145 支的分隊來自 COCO 人工角色 = 獨立答案）：
  現在做法 95.9% / 89.5%（舊 / 全部）、全部換 assign_teams 88.5% / 93.9%、**混合（同隊 ≥ 6 才換）97.3% / 94.7%**（換 77 支）。
"""
import itertools
import collections

import numpy as np

COLLAPSE_N = 6


def _main_players(player_tracks, k=10):
    app = collections.Counter(p for fd in player_tracks for p in fd)
    return [p for p, _ in app.most_common(k)]


def clip_handler(ball_acquisition):
    c = collections.Counter(p for p in ball_acquisition if p is not None and p != -1)
    return c.most_common(1)[0][0] if c else None


def n_same_as_handler(player_assignment, player_tracks, H):
    """出現最多的 10 人裡（不含持球者），整段多數決被判成跟持球者同隊（team 1）的人數。"""
    n = 0
    for p in _main_players(player_tracks):
        if p == H:
            continue
        v = collections.Counter(pa.get(p) for pa in player_assignment if pa.get(p) is not None)
        n += bool(v) and v.most_common(1)[0][0] == 1
    return n


def _bbox(v):
    return v.get('bbox') if isinstance(v, dict) else v


def reassign(frames, player_tracks, court_coords, H, siamese, fps):
    """抽 5fps → pair_stats（兩兩 Siamese + 顏色）→ assign_teams。回 {pid: 'offense' | 'defense'}。"""
    import torch
    import ingest_new_labels as IG
    from convert_contact import iou
    from nba_data_pipeline.team_classifier.inference import _crop_pil
    idx = sorted({min(len(frames) - 1, int(round(k * fps / 5.0))) for k in range(int(len(frames) * 5.0 / fps) + 1)})
    raw = [{p: _bbox(v) for p, v in player_tracks[f].items() if _bbox(v) is not None} for f in idx]
    S, C = collections.defaultdict(float), collections.defaultdict(int)
    cols, clean = collections.defaultdict(list), collections.defaultdict(list)
    with torch.no_grad():
        for f, boxes in zip(idx, raw):
            fr = frames[f]
            pids, crops = [], []
            for pid, bb in boxes.items():
                cf = IG.color_feat(fr, bb)
                if cf is not None:
                    cols[pid].append(cf)
                    if not any(iou(bb, ob) > 0.05 for op, ob in boxes.items() if op != pid):
                        clean[pid].append(cf)
                cr = _crop_pil(fr, bb)
                if cr is not None:
                    pids.append(pid); crops.append(cr)
            if len(pids) < 2:
                continue
            E = siamese._encode_crops(crops)
            ii, jj = zip(*itertools.combinations(range(len(pids)), 2))
            pr = torch.sigmoid(siamese.model.compare(E[list(ii)], E[list(jj)])).cpu().numpy()
            for a, b, v in zip(ii, jj, pr):
                k = tuple(sorted((pids[a], pids[b]))); S[k] += float(v); C[k] += 1
    d = {'siam': {k: S[k] / C[k] for k in S}, 'color': {q: np.median(v, axis=0) for q, v in cols.items()}}
    colors = {p: np.median(v, axis=0) for p, v in clean.items() if len(v) >= 2}
    c = {'raw_tracks': raw, 'court_tracks': [court_coords[f] for f in idx], 'n': len(idx)}
    return IG.assign_teams(c, d, [H], colors=colors)


def fix_collapsed(frames, player_tracks, player_assignment, ball_acquisition, court_coords, siamese, fps,
                  collapse_n=COLLAPSE_N):
    """塌掉才換；回 (player_assignment, 說明字串)。"""
    H = clip_handler(ball_acquisition)
    if H is None or siamese is None:
        return player_assignment, '沒有持球者 / 不是 Siamese 分隊，略過'
    n = n_same_as_handler(player_assignment, player_tracks, H)
    if n < collapse_n:
        return player_assignment, f'沒塌（持球者 #{H} 以外 {n} 人同隊）'
    team = reassign(frames, player_tracks, court_coords, H, siamese, fps)
    new = [{p: (1 if team.get(p) == 'offense' else 2) if p in team else t for p, t in pa.items()}
           for pa in player_assignment]
    off = sorted(p for p, t in team.items() if t == 'offense')
    return new, f'分隊塌掉（持球者 #{H} 以外 {n} 人同隊）→ 改用 assign_teams：進攻 {off}'



def inherit_uncertain_teams(player_assignment, prior_assignment, lo=0.35, hi=0.65):
    """重新追蹤（片頭剪掉重跑、編號已換回舊的）時：這次分隊不確定（判成進攻的比例在 lo~hi）的人，沿用舊結果的整段分隊；
    確定的照新結果。剪片頭只是要去掉假球員，不是要改分隊（screen_366：#7 這次進攻 45% → 多數決翻成防守 → 使用者確認過的擋拆消失）。
    「兩隊各 5 人」硬平衡試過：分隊準確 92.8 → 93.0%，但有換人的 70 支擋拆 ±1 秒 F1 0.827 → 0.784 ❌（前 10 人常混進場邊的人）。
    回 (新的 player_assignment, {pid: 沿用的隊})。"""
    def votes(pa):
        v = collections.defaultdict(collections.Counter)
        for fd in pa:
            for p, tm in fd.items():
                v[p][tm] += 1
        return v
    new_v, old_v = votes(player_assignment), votes(prior_assignment)
    keep = {}
    for p, c in new_v.items():
        share = c[1] / max(1, sum(c.values()))
        if lo <= share <= hi and p in old_v:
            old_team = old_v[p].most_common(1)[0][0]
            if old_team != (1 if share >= 0.5 else 2):
                keep[p] = old_team
    if not keep:
        return player_assignment, {}
    return [{p: (keep[p] if p in keep else tm) for p, tm in fd.items()} for fd in player_assignment], keep
