"""
possession_sam3.py — 持球者改用「SAM3 追球 → 球遮罩碰到誰的遮罩」（方向 B），跟 handler_detector 比 / 融合。

10fps（run_sam3_tracks --ball 的 shot_sam3）：
  1. 每張：球遮罩跟哪個球員遮罩（往外擴）重疊最多 = 碰球的人（shot_owner.touches；球物體用 pick_ball）；只碰 1 張的當雜訊
  2. 補洞：球看不到（被擋）且前後是同一人、≤ 1 秒 → 還是他；球看得到但沒人碰（運球彈地）前後同一人、≤ 0.5 秒 → 還是他；
     更久的「球在空中」（傳球 / 投籃）→ 沒持球者
  3. SAM3 10fps 編號 → 推論的 player_tracks 編號（shot_detect.map_ids，IoU 多數決），展開回原片 fps
融合版：SAM3 有答案就用 SAM3，沒有（球不見太久 / 對不到編號）才用 handler_detector。

用法（本機，用 stubs/shot_sam3 的 84 支延長片段 + output_videos/batch 的 events.pkl 比較）：
  python nba_data_pipeline/possession_sam3.py
"""
import sys
import pickle
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import shot_owner as so
from shot_detect import map_ids

HIDDEN_S, BOUNCE_S = 1.0, 0.5


def holders_10fps(tch, fps):
    """touches（(pid, 球中心)）→ 每張 10fps 持球者（SAM3 編號或 None）。"""
    tch = so.smooth_touches(tch)
    h = [p for p, _ in tch]
    n = len(h)
    i = 0
    while i < n:
        if h[i] is not None:
            i += 1
            continue
        j = i
        while j < n and h[j] is None:
            j += 1
        before, after = (h[i - 1] if i > 0 else None), (h[j] if j < n else None)
        if before is not None and before == after:
            hidden = all(tch[k][1] is None for k in range(i, j))
            if (j - i) <= (HIDDEN_S if hidden else BOUNCE_S) * fps:
                for k in range(i, j):
                    h[k] = before
        i = j
    return h


def holders_full(s, tch, player_tracks, n, with_flight=False):
    """→ 原片每張的持球者（推論編號、-1 = 沒有 / 不知道）；with_flight 另回每張是不是「球在空中」。"""
    h10 = holders_10fps(tch, s['fps'])
    idmap = map_ids(s, player_tracks)
    out, flight = [-1] * n, [False] * n
    fi = list(s['frame_idx']) + [n]
    for i, p in enumerate(h10):
        q = idmap.get(p) if p is not None else None
        for f in range(fi[i], min(fi[i + 1], n)):
            out[f] = q if q is not None else -1
            flight[f] = p is None and tch[i][1] is not None
    return (out, flight) if with_flight else out


def viterbi(sam, old, flight, fps, w_sam=2.0, w_det=1.0, stay=0.0, to_free=8.0, direct=32.0, team=None, def_pen=3.0):
    """持球者當成序列問題（Viterbi 最可能路徑）：狀態 = 每個候選球員持球 / FREE（球在空中、沒人拿）。
    觀測：SAM3 球碰到誰（權重 w_sam）、handler_detector 判誰（w_det）、球在空中 → 偏 FREE。
    轉移：維持不變 0；持球 ↔ FREE 罰 to_free；A 直接跳到 B（中間沒有傳球）罰 direct（物理上球不會瞬移）。
    罰分以 30fps 為準。2026-11-05 在 84 支延長片段掃過：to_free=8、direct=32 → 撞上那刻錯 2/87、一直跳 30/84、沒持球者 31%
    （現在 handler_detector 6/87、32/84、65%）。回原片每張的持球者（-1 = FREE / 沒人）。"""
    n = len(old)
    cands = sorted({p for p in sam + list(old) if p not in (None, -1)})
    if not cands:
        return [-1] * n
    S = cands + [-1]
    k = len(S)
    emit = np.zeros((n, k))
    for f in range(n):
        for j, p in enumerate(S):
            v = 0.0
            if sam[f] != -1:
                v += w_sam if p == sam[f] else (-w_sam if p != -1 else -w_sam / 2)
            elif flight[f]:
                v += w_sam / 2 if p == -1 else -w_sam / 2
            if old[f] not in (None, -1):
                v += w_det if p == old[f] else (-w_det / 2)
            if team is not None and p != -1 and team.get(p) == 2:
                v -= def_pen                # 防守者拿球（抄截）要證據夠強才成立：貼身防守時球遮罩常擦到防守者
            emit[f, j] = v
    scale = 30.0 / fps                      # 罰分以 30fps 為準
    T = np.full((k, k), -direct * scale)
    np.fill_diagonal(T, -stay)
    T[:, -1] = -to_free * scale; T[-1, :] = -to_free * scale; T[-1, -1] = -stay
    score = emit[0].copy(); back = np.zeros((n, k), int)
    for f in range(1, n):
        cand = score[:, None] + T
        back[f] = cand.argmax(0)
        score = cand.max(0) + emit[f]
    path = [int(score.argmax())]
    for f in range(n - 1, 0, -1):
        path.append(back[f, path[-1]])
    return [S[j] for j in path[::-1]]


def clip_teams(pa):
    """整段多數決分隊：{pid: 1（跟持球者同隊 = 進攻）/ 2}。"""
    cnt = collections.defaultdict(collections.Counter)
    for fd in pa:
        for p, t in fd.items():
            cnt[p][t] += 1
    return {p: c.most_common(1)[0][0] for p, c in cnt.items()}


def possession_path(d, s, tch, def_pen=3.0, min_touch=0.2):
    """events.pkl + shot_sam3 + touches → Viterbi 持球者路徑（原片每張，-1 = 沒人拿）。
    min_touch：追到球的影格裡有人碰球的比例 < 這個值 = 選錯球（比分條上的球圖示等；screen_147 追到 110/110 張、只有 6 張有人碰）
    → SAM3 的球不可信，整支退回 handler_detector（d['ball_acquisition']）。476 支中位數 0.75、低於 0.2 的 17 支。"""
    seen = [t_ for t_ in tch if t_[1] is not None]
    if not seen or sum(1 for t_ in seen if t_[0] is not None) / len(seen) < min_touch:
        return list(d['ball_acquisition'])
    n, fps = len(d['ball_acquisition']), d.get('fps', 30.0)
    sam, flight = holders_full(s, tch, d['player_tracks'], n, with_flight=True)
    return viterbi(sam, list(d['ball_acquisition']), flight, fps, team=clip_teams(d['player_assignment']), def_pen=def_pen)


def detect_passes(path, fps, team, shots=(), max_air=1.5, shot_win=0.5, min_air=0.1, min_hold=0.2):
    """持球者路徑上「A 拿 → 球在空中 → B 拿」= 一次傳球。
    同隊 = pass（球幾乎沒離手 = handoff 手遞手）、不同隊 = turnover（抄截）；
    球在空中那段（±0.5 秒）有任何人出手 = 投籃 + 籃板 / 進球後發球，不算；球在空中超過 max_air 秒不算（loose ball / 出界）；
    拿不到 min_hold 秒的當擦到。2026-11-05 screen_189：#0 手遞手給 #2 → #2 出手進 → 對方發球，舊規則誤判成 #0→#10 傳球。
    回 [{kind, frm, to, release（A 最後一張）, catch（B 第一張）, air_s}]。"""
    from handler_audit import runs
    rs = [r for r in runs(path) if r[0] is not None and (r[2] - r[1] + 1) >= min_hold * fps]   # 拿不到 0.2 秒的當擦到
    out = []
    for (A, a1, b1), (B, a2, b2) in zip(rs, rs[1:]):
        if A == B:
            continue
        air = (a2 - b1 - 1) / fps
        if air > max_air:
            continue
        if any(b1 - shot_win * fps <= sh['release_frame'] <= a2 + shot_win * fps for sh in shots or []):
            continue                        # 球在空中那段有人出手（不管是誰）= 投籃 + 籃板 / 進球後發球，不是傳球
        same = team.get(A) is not None and team.get(A) == team.get(B)
        kind = ('pass' if air >= min_air else 'handoff') if same else 'turnover'   # 同隊、球幾乎沒離手 = 手遞手（DHO）
        out.append(dict(kind=kind, frm=A, to=B, release=b1, catch=a2, air_s=round(max(air, 0), 2)))
    return out


def _near(box, pt, h):
    """球中心離球員框多遠（以球員身高為單位；在框內 = 0）。"""
    if box is None or pt is None:
        return 9.0
    dx = max(box[0] - pt[0], 0, pt[0] - box[2]); dy = max(box[1] - pt[1], 0, pt[1] - box[3])
    return float(np.hypot(dx, dy)) / max(h, 1.0)


def detect_passes_flight(s, tch, player_tracks, team, shots=(), min_free=2, min_dist=1.5, near=1.0, look=3, shot_win=0.5):
    """傳球 = 球真的在空中飛（不看「碰球的人」怎麼換，那在掩護 / 貼身防守時會在疊在一起的人之間亂跳）：
      ① 球連續 ≥ min_free 張（10fps → 0.2 秒）沒碰到任何人、而且看得到球
      ② 這段球移動 ≥ min_dist 倍球員身高
      ③ 飛行前最後碰球的是 A、飛行後第一個碰球的是 B（前後各找 look 張），A ≠ B；
         飛行起點離 A、終點離 B 都 ≤ near 倍身高
      ④ 這段（±0.5 秒）沒有偵測到出手 → 不是投籃
    同隊 = pass、不同隊 = turnover。編號對回推論的 player_tracks（IoU）。回 [{kind, frm, to, release, catch, air_s, dist}]。"""
    tch = so.smooth_touches(tch)
    idmap = map_ids(s, player_tracks)
    fi, fps_o = list(s['frame_idx']), s['orig_fps']
    hs = [b[3] - b[1] for fd in s['players'] for b in fd.values()]
    H = float(np.median(hs)) if hs else 100.0
    out = []
    for i, k in so.free_runs(tch, 0, len(tch), min_len=min_free):
        A = next((tch[j][0] for j in range(i - 1, max(-1, i - 1 - look), -1) if tch[j][0] is not None), None)
        B = next((tch[j][0] for j in range(k, min(len(tch), k + look)) if tch[j][0] is not None), None)
        if A is None or B is None or A == B:
            continue
        c0, c1 = tch[i][1], tch[k - 1][1]
        dist = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])) / H
        if dist < min_dist:
            continue
        if _near(s['players'][max(0, i - 1)].get(A), c0, H) > near or _near(s['players'][min(len(tch) - 1, k)].get(B), c1, H) > near:
            continue
        rel, cat = fi[max(0, i - 1)], fi[min(len(fi) - 1, k)]
        if any(rel - shot_win * fps_o <= sh['release_frame'] <= cat + shot_win * fps_o for sh in shots or []):
            continue
        a, b = idmap.get(A, A), idmap.get(B, B)
        same = team.get(a) is not None and team.get(a) == team.get(b)
        out.append(dict(kind='pass' if same else 'turnover', frm=a, to=b, release=rel, catch=cat,
                        air_s=round((cat - rel) / fps_o, 2), dist=round(dist, 1)))
    return out


def detect_passes_traj(s, tch, player_tracks, team, shots=(), v_min=0.35, min_len=2, min_disp=1.2, straight=0.7,
                       look=5, shot_win=0.5):
    """傳球 = 球自己的軌跡「快、方向一致、走一段距離」（不靠「沒碰到人」：球飛過人身邊也會碰到擴大後的遮罩）。
      球中心（10fps，缺 ≤ 2 張內插）→ 每張速度（身高 / 張）；連續 ≥ min_len 張速度 ≥ v_min 的段落；
      淨位移 ≥ min_disp 倍身高、淨位移 / 路徑長 ≥ straight（直直飛，運球上下彈不會過）；
      A = 段落前 look 張內最後碰球的人、B = 段落後 look 張內第一個碰球的人，A ≠ B；段落 ±0.5 秒有出手 → 投籃，不算。"""
    tch = so.smooth_touches(tch)
    idmap = map_ids(s, player_tracks)
    fi, fps_o = list(s['frame_idx']), s['orig_fps']
    hs = [b[3] - b[1] for fd in s['players'] for b in fd.values()]
    H = float(np.median(hs)) if hs else 100.0
    n = len(tch)
    c = [np.array(t[1], float) if t[1] is not None else None for t in tch]
    for i in range(n):                                   # 缺 ≤ 2 張內插
        if c[i] is None:
            a = next((j for j in range(i - 1, max(-1, i - 3), -1) if c[j] is not None), None)
            b = next((j for j in range(i + 1, min(n, i + 3)) if c[j] is not None), None)
            if a is not None and b is not None:
                c[i] = c[a] + (c[b] - c[a]) * (i - a) / (b - a)
    v = [float(np.hypot(*(c[i + 1] - c[i]))) / H if c[i] is not None and c[i + 1] is not None else 0.0 for i in range(n - 1)]
    out, i = [], 0
    while i < n - 1:
        if v[i] < v_min:
            i += 1; continue
        j = i
        while j < n - 1 and v[j] >= v_min:
            j += 1
        if j - i >= min_len:
            disp = float(np.hypot(*(c[j] - c[i]))) / H
            path = sum(v[i:j])
            A = next((tch[q][0] for q in range(i, max(-1, i - look), -1) if tch[q][0] is not None), None)
            B = next((tch[q][0] for q in range(j, min(n, j + look)) if tch[q][0] is not None), None)
            rel, cat = fi[i], fi[min(j, len(fi) - 1)]
            if disp >= min_disp and disp / max(path, 1e-6) >= straight and A is not None and B is not None and A != B and                     not any(rel - shot_win * fps_o <= sh['release_frame'] <= cat + shot_win * fps_o for sh in shots or []):
                a_, b_ = idmap.get(A, A), idmap.get(B, B)
                same = team.get(a_) is not None and team.get(a_) == team.get(b_)
                out.append(dict(kind='pass' if same else 'turnover', frm=a_, to=b_, release=rel, catch=cat,
                                air_s=round((cat - rel) / fps_o, 2), dist=round(disp, 1)))
        i = j
    return out


def detect_passes_rel(s, tch, player_tracks, team, shots=(), have=0.3, sep=0.4, hold=3, max_air=1.5, shot_win=0.5, rel_v=0.25):
    """傳球 = 球從 A 身上移動到「原本有距離」的 B 身上（2026-11-05 從使用者標的 21 個真傳球歸納）：
      每張 10fps：持球者 = 球中心離誰的框 ≤ have 倍身高（多人都貼著 → 用碰球遮罩決定）；持球要連續 ≥ hold 張才算一段。
      持球段 A → 下一段 B（A ≠ B、同隊、間隔 ≤ max_air 秒）且
        ① A 段最後一張時 B 離球 ≥ sep 倍身高（B 原本不在球旁邊 → 排除身體疊在一起的換人）
        ② B 段第一張時 A 離球 ≥ sep 倍身高（球真的離開 A）
        ③ 這段 ±0.5 秒沒有出手
    編號是推論的 player_tracks（SAM3 10fps 的碰球編號用 IoU 對過來）。回 [{kind, frm, to, release, catch, air_s}]。"""
    idmap = map_ids(s, player_tracks)
    fi, fps_o = list(s['frame_idx']), s['orig_fps']
    hs = [b[3] - b[1] for fd in player_tracks for b in fd.values()]
    H = float(np.median(hs)) if hs else 150.0
    n = len(tch)

    def tracks_at(i):
        return player_tracks[min(fi[i], len(player_tracks) - 1)]

    def rel(i, q):
        """球相對 q 框中心的位置（身高單位）。"""
        b, cen = tracks_at(i).get(q), tch[i][1]
        if b is None or cen is None:
            return None
        return np.array([(cen[0] - (b[0] + b[2]) / 2) / H, (cen[1] - (b[1] + b[3]) / 2) / H])

    def moves_with(i, q):
        """球跟著 q 動（拿球 / 運球）：前後一張的相對位置變化都 ≤ rel_v；飛行中的球相對每個人都在動（2D 上穿過路人的框也不算）。"""
        r0 = rel(i, q)
        if r0 is None:
            return False
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                r1 = rel(j, q)
                if r1 is not None and float(np.hypot(*(r1 - r0))) > rel_v:
                    return False
        return True

    hold_seq = []
    for i in range(n):
        p, cen = tch[i]
        if cen is None:
            hold_seq.append(None); continue
        near = {q: _near(b, cen, H) for q, b in tracks_at(i).items()}
        close = [q for q, dq in near.items() if dq <= have and moves_with(i, q)]
        tq = idmap.get(p) if p is not None else None
        hold_seq.append(tq if (tq in close) else (min(close, key=near.get) if close else None))
    segs, cur = [], None                                  # [(pid, 起, 迄)]（中間看不到球 / 沒人 ≤ 1 張不斷開）
    for i, q in enumerate(hold_seq):
        if q is not None and cur and cur[0] == q and i - cur[2] <= 2:
            cur[2] = i
        elif q is not None:
            if cur: segs.append(cur)
            cur = [q, i, i]
    if cur: segs.append(cur)
    major = [g for g in segs if g[2] - g[1] + 1 >= hold]  # 拿夠久的才算；中間短暫擦到路過的人（< hold 張）略過
    out = []
    for (A, a1, b1), (B, a2, b2) in zip(major, major[1:]):
        if A == B or (fi[a2] - fi[b1]) / fps_o > max_air:
            continue
        # 傳球者：球開始離開前最後一張「離球最近」的人（離手瞬間掩護者 / 持球者疊在一起時，碰球遮罩常判成旁邊的人）
        cA = tch[b1][1]
        if cA is not None:
            cand = {q: _near(bx, cA, H) for q, bx in tracks_at(b1).items() if team.get(q) == team.get(A)}
            if cand:
                A = min(cand, key=cand.get)
        if A == B:
            continue
        cB = tch[a2][1]
        dB0 = _near(tracks_at(b1).get(B), cA, H)
        dA1 = _near(tracks_at(a2).get(A), cB, H)
        if dB0 < sep or dA1 < sep:
            continue
        rel, cat = fi[b1], fi[a2]
        if any(rel - shot_win * fps_o <= sh['release_frame'] <= cat + shot_win * fps_o for sh in shots or []):
            continue
        same = team.get(A) is not None and team.get(A) == team.get(B)
        out.append(dict(kind='pass' if same else 'turnover', frm=A, to=B, release=rel, catch=cat,
                        air_s=round((cat - rel) / fps_o, 2)))
    return out


def prep(video, ball_pkl, out_touch):
    """原片 + run_sam3_tracks --ball_only 的 pkl → Roboflow 選球 + 每張碰球的人 → (chosen, bscore, touches) 存檔（同 stubs/shot_sam3/*_touch.pkl）。"""
    from utils import read_video
    from shot_detect import run_rf
    frames = read_video(video)
    s = pickle.load(open(ball_pkl, 'rb'))['shot_sam3']
    rf = run_rf(frames, s['orig_fps'], target_fps=10)
    chosen, bscore = so.pick_ball(s, rf)
    out = (chosen, bscore, so.touches(s, frames[0].shape[:2], chosen))
    pickle.dump(out, open(out_touch, 'wb'))
    return out


def main():
    import glob, csv
    import handler_audit as HA
    from convert_contact import CACHE_DIR
    rows = collections.defaultdict(list)
    for pk in sorted(glob.glob(str(ROOT / 'output_videos' / 'batch' / 'ext_*_events.pkl'))):
        name = Path(pk).stem.replace('_events', '')
        clip = name.replace('ext_', '')
        sp, tp = ROOT / 'stubs' / 'shot_sam3' / f'{clip}_shot_sam3.pkl', ROOT / 'stubs' / 'shot_sam3' / f'{clip}_touch.pkl'
        if not (sp.exists() and tp.exists()):
            continue
        d = pickle.load(open(pk, 'rb'))
        s = pickle.load(open(sp, 'rb'))
        chosen, _, tch = pickle.load(open(tp, 'rb'))
        n, fps = len(d['ball_acquisition']), d.get('fps', 30.0)
        old = list(d['ball_acquisition'])
        sam, flight = holders_full(s, tch, d['player_tracks'], n, with_flight=True)
        fused = [p if p != -1 else o for p, o in zip(sam, old)]
        vit = viterbi(sam, old, flight, fps)
        c = pickle.load(open(CACHE_DIR / f'{clip}.pkl', 'rb')) if (CACHE_DIR / f'{clip}.pkl').exists() else None
        H = None
        if c is not None and not c.get('pending'):
            H = HA.map_ids(c, d['player_tracks'], fps).get(c['handler_tid'])
        for tag, ba in (('A 現在（handler_detector）', old), ('B 只用 SAM3 追球', sam), ('C 融合（SAM3 優先）', fused), ('D 融合 + Viterbi', vit)):
            sw, short, flick, none = HA.jump_stats(ba, fps)
            sc = wr = nn = 0
            if H is not None:
                for e in c['events']:
                    if e['tid'] is None:
                        continue
                    f = int(round(e['idx'] * fps / 5.0))
                    w = [ba[g] for g in range(max(0, f - int(0.5 * fps)), min(n, f + int(0.5 * fps) + 1)) if ba[g] not in (None, -1)]
                    got = collections.Counter(w).most_common(1)[0][0] if w else None
                    sc += 1; wr += got != H; nn += got is None
            rows[tag].append(dict(name=name, sw=sw, short=short, flick=flick, none=none, sc=sc, wr=wr, nn=nn))
    print(f'延長片段 {len(next(iter(rows.values())))} 支（撞上那刻比對你標的持球者）\n')
    print(f'{"":26s} 撞上那刻持球者不對（其中沒抓到）   一直跳的片段   沒持球者影格（中位數）  換人次數（中位數）')
    for tag, R in rows.items():
        sc, wr, nn = sum(r['sc'] for r in R), sum(r['wr'] for r in R), sum(r['nn'] for r in R)
        jumpy = sum(1 for r in R if r['short'] >= 2 or r['flick'] >= 1)
        print(f'{tag:26s} {wr:3d}/{sc}（{wr / max(1, sc):4.0%}，沒抓到 {nn:2d}）       {jumpy:3d}/{len(R)}         '
              f'{np.median([r["none"] for r in R]):5.0%}               {np.median([r["sw"] for r in R]):.0f}')


def pass_report():
    """84 支延長片段：傳球偵測統計。"""
    import glob
    allp, per = [], []
    for pk in sorted(glob.glob(str(ROOT / 'output_videos' / 'batch' / 'ext_*_events.pkl'))):
        clip = Path(pk).stem.replace('_events', '').replace('ext_', '')
        sp, tp = ROOT / 'stubs' / 'shot_sam3' / f'{clip}_shot_sam3.pkl', ROOT / 'stubs' / 'shot_sam3' / f'{clip}_touch.pkl'
        if not (sp.exists() and tp.exists()):
            continue
        d = pickle.load(open(pk, 'rb')); s = pickle.load(open(sp, 'rb')); _, _, tch = pickle.load(open(tp, 'rb'))
        fps = d.get('fps', 30.0)
        ps = detect_passes_flight(s, tch, d['player_tracks'], clip_teams(d['player_assignment']), d.get('shots'))
        per.append((clip, ps)); allp += ps
    k = collections.Counter(p['kind'] for p in allp)
    air = [p['air_s'] for p in allp if p['kind'] == 'pass']
    print(f'{len(per)} 支：傳球 {k["pass"]} 次、手遞手 {k["handoff"]} 次、抄截 / 換隊 {k["turnover"]} 次；每支傳球中位數 '
          f'{np.median([sum(p["kind"] == "pass" for p in ps) for _, ps in per]):.0f}；空中時間 中位數 {np.median(air):.2f} 秒'
          f'（10% {np.percentile(air, 10):.2f}、90% {np.percentile(air, 90):.2f}）')
    for clip, ps in per[:12]:
        print(f'  {clip}：' + '、'.join(f"{dict(pass_='傳', handoff='手遞手', turnover='換隊').get(p['kind'] if p['kind'] != 'pass' else 'pass_')} #{p['frm']}→#{p['to']} @{p['release'] / 30:.1f}s（{p['air_s']}s）" for p in ps))


if __name__ == '__main__':
    if sys.argv[1:2] == ['passes']:
        pass_report()
    elif sys.argv[1:2] == ['prep']:                 # python possession_sam3.py prep <影片> <ball.pkl> <touch.pkl>
        prep(*sys.argv[2:5])
    else:
        main()
