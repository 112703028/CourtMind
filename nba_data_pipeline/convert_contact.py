"""
convert_contact.py — 用 SAM3 軌跡 + 人工接觸幀，產生 ScreenNet「事件定位」訓練資料。

取代 convert_manual.py 的標籤定義：
  舊：窗內任一幀有 screener 標註 → 整窗正樣本（模型學成「是否處在擋拆中」→ 分數是平台）
  新：以人工標的接觸幀為中心的高斯軟標籤，每個 (handler, screener) 各自一條曲線
      → 模型要學「擋拆發生在哪一刻、是誰在擋」

兩段（分開是因為 TeamSiamese ckpt 與 GPU 只在 vipl-spark）：
  prep  ：每支 clip → 球場座標、TeamSiamese 分隊（以 COCO 投票出的 handler 當 anchor，
          與 inference.py 同一套分隊）、COCO 角色投票、接觸幀 → SAM3 軌跡 id
          → 快取 stubs/train_contact_cache/<clip>.pkl
  build ：快取 → 10 幀滑窗（5fps，= 2 秒）+ 軟標籤 → output_contact/*.npz（本地就能跑）

輸入：
  input_videos/screen_frames/<clip>/        5fps 訓練影格
  stubs/train_sam3/<clip>_sam3_tracks.pkl    run_sam3_tracks.py 批次模式產出
  nba_data_pipeline/contact_labels.json      build_contact_labels.py 產出（人工接觸幀）
  screen_data/*/train/_annotations.coco.json 角色標註（handler/screener/defender）

用法：
  # vipl-spark（basketball_analysis container）
  python nba_data_pipeline/convert_contact.py prep
  # 本地或 spark
  python nba_data_pipeline/convert_contact.py build --sigma 1.0 --sigma_nocontact 1.5
"""
import os
import sys
import math
import glob
import json
import pickle
import argparse
import collections
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

FRAMES_DIR = ROOT / 'input_videos' / 'screen_frames'
TRACKS_DIR = ROOT / 'stubs' / 'train_sam3'
CACHE_DIR = ROOT / 'stubs' / 'train_contact_cache'
OUT_DIR = HERE / 'output_contact'
LABELS = HERE / 'contact_labels.json'
SAMPLE_FPS = 5.0

SEQ_LEN = 10          # 與 convert_manual / inference 一致（5fps 下 = 2 秒）
N_OFF, N_DEF = 5, 5
N_PLAYERS = N_OFF + N_DEF


def pair_index(i, j):
    """與 convert_manual.pair_index 相同：pair (i,j) = j 幫 i 擋 的線性索引。
    （本地 build 不 import convert_manual，因為它 import 了本地沒裝的 supervision）"""
    idx = 0
    for ii in range(N_OFF):
        for jj in range(N_OFF):
            if ii == jj:
                continue
            if ii == i and jj == j:
                return idx
            idx += 1
    return -1


def xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def foot(b):
    return np.array([(b[0] + b[2]) / 2, b[3]])


def target_handler(coco_frames, raw):
    """沒標 ball_handler 時，用 COCO `target`（掩護要幫忙甩開的人）投票出軌跡 id。"""
    v = collections.Counter()
    for idx, anns in coco_frames.items():
        if idx >= len(raw):
            continue
        for b in anns.get('target', []):
            bb = xywh_to_xyxy(b)
            best = max(((iou(bb, xy), pid) for pid, xy in raw[idx].items()), default=(0, None))
            if best[1] is not None and best[0] >= 0.2:
                v[best[1]] += 1
    return v.most_common(1)[0][0] if v else None


def team_by_color(frames, raw, handler_tid, screener_tids, defender_tid):
    """球衣顏色 2 群分隊（TeamSiamese 在某些比賽塌成「全部同隊」時的退路）。
    每個主要 id（出現 >= 1/3 影格）取上半身中段的 Lab 中位數 → k-means(k=2)，
    handler（沒有就用 screener）所在的群 = 進攻；碎片 id 歸到最近的群心。"""
    n = len(raw)
    app = collections.Counter(p for fd in raw for p in fd)
    feats = {}
    for pid in app:
        cols = []
        fidx = [f for f in range(n) if pid in raw[f]]
        for f in fidx[:: max(1, len(fidx) // 8)][:8]:
            x1, y1, x2, y2 = [int(v) for v in raw[f][pid]]
            h, w = y2 - y1, x2 - x1
            crop = frames[f][max(0, y1 + h // 6): max(0, y1 + h // 2),
                             max(0, x1 + w // 4): max(0, x2 - w // 4)]
            if crop.size >= 30:
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
                cols.append(np.median(lab, axis=0))
        if cols:
            feats[pid] = np.median(cols, axis=0).astype(np.float32)
    main = [p for p in feats if app[p] >= n / 3]
    if len(main) < 2:
        return None
    X = np.stack([feats[p] for p in main])
    _, lbl, centers = cv2.kmeans(X, 2, None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5),
                                 10, cv2.KMEANS_PP_CENTERS)
    cl = {p: int(l) for p, l in zip(main, lbl.flatten())}
    for p in feats:
        if p not in cl:
            cl[p] = int(np.argmin(np.linalg.norm(centers - feats[p], axis=1)))
    anchor = handler_tid if handler_tid in cl else next((s for s in screener_tids if s in cl), None)
    if anchor is None:
        return None
    off_c = cl[anchor]
    return {p: ('offense' if c == off_c else 'defense') for p, c in cl.items()}


def team_collapsed(c):
    """分隊是否塌掉：主要 id 裡防守或進攻少於 3 人。"""
    app = collections.Counter(p for fd in c['raw_tracks'] for p in fd)
    main = [p for p in c['track_team'] if app[p] >= c['n'] / 3]
    off = sum(1 for p in main if c['track_team'][p] == 'offense')
    return c['handler_tid'] is None or off < 3 or len(main) - off < 3


def apply_roles(track_team, handler_tid, screener_tids, defender_tid):
    """COCO 已知角色強制覆蓋分隊結果（screener/handler = 進攻、defender = 防守）。"""
    track_team = dict(track_team)
    for t in screener_tids:
        track_team[t] = 'offense'
    if handler_tid is not None:
        track_team[handler_tid] = 'offense'
    if defender_tid is not None:
        track_team[defender_tid] = 'defense'
    return track_team


def map_events(info, raw, track_team, screener_tids, handler_tid):
    """人工接觸幀 → SAM3 軌跡 id：先用確認過的 screener 框（IoU），
    沒框到 / 對不上就在接觸幀附近找候選 screener 裡腳離防守方最近的。"""
    n = len(raw)
    events = []
    for e in info['events']:
        idx = min(n - 1, e['idx'])
        tid, how = None, None
        if e.get('tid_fixed') is not None:
            tid, how = e['tid_fixed'], 'user'          # 使用者直接指定（contact_overrides add_events）
        elif e.get('screener_box') is not None and e.get('coco_idx') is not None and e['coco_idx'] < n:
            bb = xywh_to_xyxy(e['screener_box'])
            best = max(((iou(bb, xy), pid) for pid, xy in raw[e['coco_idx']].items()), default=(0, None))
            if best[1] is not None and best[0] >= 0.2:
                tid, how = best[1], 'box'
        if tid is None:
            cands = screener_tids or [p for p, t in track_team.items()
                                      if t == 'offense' and p != handler_tid]
            best = None
            for f in range(max(0, idx - 2), min(n, idx + 3)):
                defs = [xy for p, xy in raw[f].items() if track_team.get(p) == 'defense']
                for c in cands:
                    if c not in raw[f] or not defs:
                        continue
                    sb = raw[f][c]
                    h = max(1.0, sb[3] - sb[1])
                    d = min(np.linalg.norm(foot(sb) - foot(db)) for db in defs) / h
                    if best is None or d < best[0]:
                        best = (d, c)
            if best is not None:
                tid, how = best[1], 'nearest_defender'
        events.append({'frame': e['frame'], 'idx': idx, 'contact': e['contact'],
                       'tid': tid, 'tid_method': how})
    return events


# ── prep：每支 clip 的追蹤 / 分隊 / 角色 / 接觸幀對應 ─────────────────────────────

def prep_clip(clip, info, coco_frames, court_model, team_assigner, cm, device='cuda'):
    jpgs = sorted(glob.glob(str(FRAMES_DIR / clip / '*.jpg')),
                  key=lambda p: int(Path(p).stem.rsplit('_', 1)[1]))
    frames = [cv2.imread(p) for p in jpgs]
    tr = pickle.load(open(TRACKS_DIR / f'{clip}_sam3_tracks.pkl', 'rb'))
    player_tracks = tr['player_tracks']            # list[dict[pid:{'bbox':xyxy}]]
    n = len(frames)
    assert len(player_tracks) == n, f'{clip}: 軌跡 {len(player_tracks)} 幀 ≠ 影格 {n}'
    raw = [{pid: v['bbox'] for pid, v in ft.items()} for ft in player_tracks]

    # 1. COCO 角色投票（IoU 配對，同一幀不重複指派）
    votes = {k: collections.Counter() for k in ('screener', 'defender', 'ball_handler')}
    max_scr = 0
    for idx, anns in coco_frames.items():
        if idx >= n or not raw[idx]:
            continue
        max_scr = max(max_scr, len(anns.get('screener', [])))
        used = set()
        for cat in ('ball_handler', 'screener', 'defender'):
            for b in anns.get(cat, []):
                bb = xywh_to_xyxy(b)
                best = max(((iou(bb, xy), pid) for pid, xy in raw[idx].items() if pid not in used),
                           default=(0, None))
                if best[1] is not None and best[0] >= 0.2:
                    used.add(best[1])
                    votes[cat][best[1]] += 1
    handler_tid = votes['ball_handler'].most_common(1)[0][0] if votes['ball_handler'] else None
    if handler_tid is None:
        # 無球掩護的 clip（如 screen_164 底線發球 backscreen）沒標 ball_handler，
        # 掩護是為 `target`（被幫忙甩開防守的無球球員）設的 → 用它當 slot 0
        handler_tid = target_handler(coco_frames, raw)
    screener_tids = [t for t, _ in votes['screener'].most_common(max(1, max_scr))]
    defender_tid = votes['defender'].most_common(1)[0][0] if votes['defender'] else None

    # 2. TeamSiamese 分隊（同 inference.py：每幀以 handler 為 anchor 兩兩比對）→ 整段多數決
    ball_acq = [handler_tid if (handler_tid is not None and handler_tid in raw[f]) else -1
                for f in range(n)]
    per_frame = team_assigner.get_player_teams_across_frames(frames, player_tracks, ball_acq)
    tv = collections.defaultdict(collections.Counter)
    for fa in per_frame:
        for pid, t in fa.items():
            tv[pid][t] += 1
    track_team = {pid: ('offense' if c.most_common(1)[0][0] == 1 else 'defense')
                  for pid, c in tv.items()}
    track_team = apply_roles(track_team, handler_tid, screener_tids, defender_tid)
    team_method = 'siamese'
    if team_collapsed({'raw_tracks': raw, 'n': n, 'track_team': track_team,
                       'handler_tid': handler_tid}):
        # TeamSiamese 在沒看過異隊樣本的早期比賽會塌成「全部同隊」→ 退回球衣顏色分群
        color = team_by_color(frames, raw, handler_tid, screener_tids, defender_tid)
        if color:
            track_team = apply_roles(color, handler_tid, screener_tids, defender_tid)
            team_method = 'color'

    # 3. 球場座標（同 convert_manual）。分批跑：整支 clip 一次丟進 YOLO 會 GPU OOM
    kps = []
    for i in range(0, n, 8):
        kps.extend(court_model.predict(frames[i:i + 8], conf=0.5, verbose=False,
                                       device=device))
    court_tracks = []
    for fd, kp in zip(raw, kps):
        H = cm.build_homography(kp)
        ct = {}
        if H is not None:
            for pid, xyxy in fd.items():
                pt = cm.transform(H, *cm.foot_pos_xyxy(xyxy))
                if pt is not None:
                    tx, ty = cm.tactical_to_feet(*pt)
                    if 0 <= tx <= cm.COURT_W_FT + 5 and 0 <= ty <= cm.COURT_H_FT + 5:
                        ct[pid] = (tx, ty)
        court_tracks.append(ct)

    # 4. 接觸幀 → SAM3 軌跡 id
    step = info['fps'] / SAMPLE_FPS
    events = map_events(info, raw, track_team, screener_tids, handler_tid)

    return {
        'clip': clip, 'n': n, 'fps': info['fps'],
        'valid_start_idx': int(math.ceil(info['valid_start'] / step)),
        'pending': info.get('pending', False),
        'court_tracks': court_tracks, 'raw_tracks': raw, 'track_team': track_team,
        'handler_tid': handler_tid, 'screener_tids': screener_tids, 'defender_tid': defender_tid,
        'team_method': team_method, 'events': events,
    }


def run_prep(args):
    from ultralytics import YOLO
    import convert_manual as cm
    from build_contact_labels import load_coco, COCO_V2, COCO_FD
    from nba_data_pipeline.team_classifier.inference import TeamAssignerSiamese

    labels = json.load(open(LABELS, encoding='utf-8'))
    v2, fd = load_coco(COCO_V2), load_coco(COCO_FD)
    court_model = YOLO(str(ROOT / 'models' / 'court_keypoint_detector.pt'))
    team_assigner = TeamAssignerSiamese(ckpt_path=str(ROOT / 'models' / 'team_siamese.pt'),
                                        device=args.device)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clips = args.clips.split(',') if args.clips else sorted(labels, key=lambda s: int(s.split('_')[1]))
    for k, clip in enumerate(clips):
        out = CACHE_DIR / f'{clip}.pkl'
        if out.exists() and not args.force:
            continue
        coco_frames = fd[clip] if clip in fd else v2[clip]
        try:
            c = prep_clip(clip, labels[clip], coco_frames, court_model, team_assigner, cm,
                          device=args.device)
            pickle.dump(c, open(out, 'wb'))
            if args.device != 'cpu':
                import torch
                torch.cuda.empty_cache()
            miss = sum(1 for e in c['events'] if e['tid'] is None)
            print(f"[{k + 1}/{len(clips)}] {clip}: handler={c['handler_tid']} "
                  f"screeners={c['screener_tids']}  事件 {len(c['events'])}（對不到軌跡 {miss}）")
        except Exception as ex:
            print(f"[{k + 1}/{len(clips)}] {clip} 失敗：{ex}")


def run_relabel(args):
    """標記（contact_labels.json）改了之後，依新標記重新對應每支快取的掩護（本地跑，幾秒）。
    例如從標記拿掉無球掩護、補上漏標的掩護後，不用回 spark 重跑 prep。"""
    labels = json.load(open(LABELS, encoding='utf-8'))
    changed = 0
    for p in sorted(glob.glob(str(CACHE_DIR / '*.pkl')), key=lambda q: int(Path(q).stem.split('_')[1])):
        c = pickle.load(open(p, 'rb'))
        if c['clip'] not in labels:
            continue
        old = [(e['idx'], e['tid']) for e in c['events']]
        c['events'] = map_events(labels[c['clip']], c['raw_tracks'], c['track_team'],
                                 c['screener_tids'], c['handler_tid'])
        c['valid_start_idx'] = int(math.ceil(labels[c['clip']]['valid_start'] / (c['fps'] / SAMPLE_FPS)))
        if old != [(e['idx'], e['tid']) for e in c['events']]:
            changed += 1
            pickle.dump(c, open(p, 'wb'))
    print(f"更新了 {changed} 支快取的掩護")


def run_fixteam(args):
    """修已經跑好的快取（本地就能跑，不用 GPU / TeamSiamese）：
    沒有 handler 的改用 COCO target；分隊塌掉的改用球衣顏色分群，並重新對應接觸幀。"""
    from build_contact_labels import load_coco, COCO_V2, COCO_FD
    labels = json.load(open(LABELS, encoding='utf-8'))
    v2, fd = load_coco(COCO_V2), load_coco(COCO_FD)
    for p in sorted(glob.glob(str(CACHE_DIR / '*.pkl')), key=lambda q: int(Path(q).stem.split('_')[1])):
        c = pickle.load(open(p, 'rb'))
        clip, raw = c['clip'], c['raw_tracks']
        coco = fd[clip] if clip in fd else v2[clip]
        changed = []
        if c['handler_tid'] is None:
            h = target_handler(coco, raw)
            if h is not None:
                c['handler_tid'] = h
                c['track_team'] = apply_roles(c['track_team'], h, c['screener_tids'], c['defender_tid'])
                changed.append(f'handler←target #{h}')
        if team_collapsed(c):
            jpgs = sorted(glob.glob(str(FRAMES_DIR / clip / '*.jpg')),
                          key=lambda q: int(Path(q).stem.rsplit('_', 1)[1]))
            frames = [cv2.imread(q) for q in jpgs]
            color = team_by_color(frames, raw, c['handler_tid'], c['screener_tids'], c['defender_tid'])
            if color:
                c['track_team'] = apply_roles(color, c['handler_tid'], c['screener_tids'],
                                              c['defender_tid'])
                c['team_method'] = 'color'
                changed.append('分隊改用球衣顏色')
            else:
                changed.append('顏色分群也失敗')
        c.setdefault('team_method', 'siamese')
        if changed:
            c['events'] = map_events(labels[clip], raw, c['track_team'],
                                     c['screener_tids'], c['handler_tid'])
            pickle.dump(c, open(p, 'wb'))
            app = collections.Counter(q for fdict in raw for q in fdict)
            main = [q for q in c['track_team'] if app[q] >= c['n'] / 3]
            off = sum(1 for q in main if c['track_team'][q] == 'offense')
            print(f"{clip}: {'、'.join(changed)} → 主要 id 進攻 {off}、防守 {len(main) - off}"
                  f"；事件 tid {[e['tid'] for e in c['events']]}")


# ── build：快取 → 滑窗 + 軟標籤 ────────────────────────────────────────────────

def build_windows(c, sigma, sigma_nc, min_target, anchor=3, rotate=True, clean=False, pose=False, poss=None):
    """anchor：標籤記在窗口第幾格（0~9）。3 = 接觸前看 3 張（0.6 秒）、接觸後看 6 張（1.2 秒）。
    推論時要用同一個 anchor 把窗口分數對回影格。
    clean：先用 tactics_tools.clean_court_tracks 清座標（鏡像閃爍 / 壞影格 / 單人尖刺）；
           clean=2 再加「個人短缺口（≤6 張 = 1.2 秒）前後內插」。
    pose：每人每張在 x, y 後面接 6 個姿態 / 遮罩通道（pose_features.channels）→ traj (10, 10, 8)。
    poss：逐張持球者（possession.holders，快取 pid）。窗內拿球最多的進攻球員 ≠ handler_tid 時，
          多產生一組「以他為持球者」的排法（原本的排法保留；候選人不含原 handler，候選範圍不變）。"""
    n, ct, raw, team = c['n'], c['court_tracks'], c['raw_tracks'], c['track_team']
    if clean:
        from tactics_tools import clean_court_tracks
        ct = clean_court_tracks(ct, fill_gaps=6 if clean == 2 else 0)[0]
    pf = None
    if pose:
        import pose_features
        pf = pose_features.load_clip(c)
    H, S_all, D = c['handler_tid'], c['screener_tids'], c['defender_tid']
    ev = [e for e in c['events'] if e['tid'] is not None]
    out = []
    for start in range(c['valid_start_idx'], n - SEQ_LEN + 1):
        center = start + anchor
        wr = raw[start:start + SEQ_LEN]
        wc = ct[start:start + SEQ_LEN]
        cnt = collections.Counter(p for fdict in wr for p in fdict)
        present = {p for p, k in cnt.items() if k >= SEQ_LEN // 2}
        ev_tids = {e['tid'] for e in ev}
        for forced in list(S_all) + list(ev_tids) + [H, D]:
            if forced is not None and cnt.get(forced, 0) >= 1:
                present.add(forced)
        off = [p for p in present if team.get(p) == 'offense']
        dfn = [p for p in present if team.get(p) == 'defense']
        if H is None or H not in off or len(off) < 2:
            continue

        # 每個 screener 的目標 = 它所有接觸幀的高斯取最大（沒撞到的掩護用較寬的 σ）
        tgt = collections.defaultdict(float)
        for e in ev:
            s = sigma if e['contact'] else sigma_nc
            tgt[e['tid']] = max(tgt[e['tid']], math.exp(-((center - e['idx']) ** 2) / (2 * s * s)))

        handlers = [H]
        if poss is not None:
            import possession
            hw = possession.window_handler(poss, start, SEQ_LEN, team, H)
            if hw is not None and hw in off:
                handlers.append(hw)
        for Hh in handlers:
            # 排序跟 inference.detect_screens 一致：以窗內最後一個有 handler 座標的幀為準，
            # 防守取離 handler 最近的 5 個（替補/短暫假軌跡通常離得遠 → 自然被擠出去），
            # 其他進攻也按離 handler 的距離排。
            hpos = next((wc[k][Hh] for k in range(SEQ_LEN - 1, -1, -1) if Hh in wc[k]), None)

            def dist_to_h(p):
                if hpos is None:
                    return 0.0
                pos = next((wc[k][p] for k in range(SEQ_LEN - 1, -1, -1) if p in wc[k]), None)
                return float('inf') if pos is None else math.dist(pos, hpos)

            order_def = (sorted(dfn, key=dist_to_h) + [None] * N_DEF)[:N_DEF]
            if rotate:
                # 跟推論一樣：每個進攻候選人輪流放 slot 1，其他進攻按離 handler 的距離排。
                # 標籤是「這個人此刻」的目標 → 模型會看到「掩護者在 slot 1 但不是接觸時刻 = 低分」
                # 以及「掩護發生時 slot 1 放的是旁人 = 低分」，跟推論時的輪流方式一致。
                layouts = []
                for cand in (p for p in off if p not in (Hh, H)):
                    rest = sorted((p for p in off if p not in (Hh, cand)), key=dist_to_h)
                    layouts.append(([Hh, cand] + rest + [None] * N_OFF)[:N_OFF])
            else:
                # 舊做法：有目標的 screener 優先放前面
                scr_w = sorted((p for p in off if tgt.get(p, 0) >= min_target), key=lambda p: -tgt[p])
                others = sorted((p for p in off if p != Hh and p not in scr_w), key=dist_to_h)
                layouts = [([Hh] + scr_w + others + [None] * N_OFF)[:N_OFF]]

            for order_off in layouts:
                order = order_off + order_def
                traj = np.zeros((N_PLAYERS, SEQ_LEN, 2), dtype=np.float32)
                last = [None] * N_PLAYERS
                for k in range(SEQ_LEN):
                    for pi, p in enumerate(order):
                        if p is not None and p in wc[k]:
                            traj[pi, k] = wc[k][p]
                            last[pi] = wc[k][p]
                        elif last[pi] is not None:
                            traj[pi, k] = last[pi]

                # 防守者幾乎沒有座標（該段 homography 失敗）的窗不要：模型會學到「沒防守者也能判掩護」
                if int((np.abs(traj[N_OFF:]).sum(axis=(1, 2)) > 0).sum()) < 2:
                    continue

                pw = np.zeros(20, dtype=np.float32)
                for slot in range(1, N_OFF):
                    p = order_off[slot]
                    if p is not None and tgt.get(p, 0) >= min_target:
                        pw[pair_index(0, slot)] = tgt[p]
                if pf is not None:
                    import pose_features
                    ch = pose_features.channels(pf, range(start, start + SEQ_LEN), order, Hh, order_off[1])
                    traj = np.concatenate([traj, ch], axis=-1)
                out.append((traj, pw, center))
    return out


def run_build(args):
    roles = np.array([0] + [1] * (N_OFF - 1) + [2] * N_DEF, dtype=np.int64)
    caches = sorted(glob.glob(str(CACHE_DIR / '*.pkl')),
                    key=lambda p: int(Path(p).stem.split('_')[1]))
    if not caches:
        print(f'{CACHE_DIR} 沒有快取，先跑 prep')
        return
    T, P, C, CTR = [], [], [], []
    skipped = []
    for p in caches:
        c = pickle.load(open(p, 'rb'))
        if c['pending'] and not args.include_pending:
            skipped.append(c['clip'])
            continue
        # 分隊不可靠（主要球員裡進攻或防守少於 min_team 人）就不拿來訓練：
        # 防守 slot 會塞錯人，等於教模型錯的東西（screen_32/47 兩隊球衣太像、153 進攻漏追）
        app = collections.Counter(q for fdict in c['raw_tracks'] for q in fdict)
        main = [q for q in c['track_team'] if app[q] >= c['n'] / 3]
        off = sum(1 for q in main if c['track_team'][q] == 'offense')
        if off < args.min_team or len(main) - off < args.min_team:
            skipped.append(c['clip'])
            continue
        for traj, pw, center in build_windows(c, args.sigma, args.sigma_nocontact,
                                              args.min_target, args.anchor,
                                              rotate=not args.no_rotate):
            T.append(traj); P.append(pw)
            C.append(int(c['clip'].split('_')[1])); CTR.append(center)
    T, P = np.stack(T), np.stack(P)
    C, CTR = np.array(C, np.int32), np.array(CTR, np.int32)
    lab = (P.max(axis=1) >= 0.5).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, m in (('positive_sequences.npz', lab == 1), ('negative_sequences.npz', lab == 0)):
        np.savez_compressed(str(OUT_DIR / name), trajectories=T[m],
                            roles=np.repeat(roles[None], m.sum(), 0),
                            labels=lab[m], pairwise_labels=P[m],
                            clip_ids=C[m], centers=CTR[m],
                            anchor=np.int32(args.anchor), sigma=np.float32(args.sigma))
    print(f"{len(caches) - len(skipped)} 支 clip → {len(T)} 個窗："
          f"正 {int(lab.sum())}（目標≥0.5）、負 {int((1 - lab).sum())}")
    print(f"  有任何非零目標的窗 {int((P.max(axis=1) > 0).sum())}；跳過（待確認/分隊不可靠）{skipped}")
    print(f"  σ={args.sigma}（沒撞到 {args.sigma_nocontact}）、anchor 第 {args.anchor} 格 → {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prep')
    a.add_argument('--device', default='cuda')
    a.add_argument('--clips', default=None, help='只跑這些（逗號分隔），測試用')
    a.add_argument('--force', action='store_true')
    sub.add_parser('relabel', help='標記改了之後，依新標記重新對應每支快取的掩護')
    sub.add_parser('fixteam', help='本地修快取：缺 handler 用 target、分隊塌掉改用球衣顏色')
    b = sub.add_parser('build')
    b.add_argument('--sigma', type=float, default=1.0, help='高斯寬度（5fps 影格數，1 = 0.2 秒）')
    b.add_argument('--sigma_nocontact', type=float, default=1.5)
    b.add_argument('--min_target', type=float, default=0.01)
    b.add_argument('--anchor', type=int, default=3,
                   help='標籤記在窗口第幾格（0~9）；3 = 接觸前 0.6 秒、接觸後 1.2 秒')
    b.add_argument('--include_pending', action='store_true')
    b.add_argument('--no_rotate', action='store_true',
                   help='不輪流放候選人（舊做法：只有有目標的 screener 放前面）')
    b.add_argument('--min_team', type=int, default=4,
                   help='主要球員裡進攻、防守都至少這麼多人才拿來訓練（排除分隊不可靠的 clip）')
    args = ap.parse_args()
    {'prep': run_prep, 'relabel': run_relabel, 'fixteam': run_fixteam, 'build': run_build}[args.cmd](args)


if __name__ == '__main__':
    main()
