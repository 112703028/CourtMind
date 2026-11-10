"""
shot_v2.py — 投籃候選第 2 版（2026-11-09，實驗中；驗證過才換掉 shot_owner.shot_candidates）。

舊版的錯（傳球當投籃 13/56、出手者認錯）：
  ① 一段鏡頭沒強候選 → 硬挑幅度最大的當投籃（192、136）
  ② 只看畫面上球離籃框近不近 → 路線經過籃框的傳球變強候選（17）
  ③ 出手者 = 1.5 秒內碰最多的人 → 傳完馬上被接走的算成傳球者（249）
  ④ 球在空中飛過別人 → 遮罩重疊算「碰到」→ 空中段被切斷（136）
這版的開關（可以一個一個開來量）：
  wrist     用手腕判誰拿著球（pose_hands）：空中段 = 沒人的手腕在球旁邊；出手者 = 離手前最後連續拿 ≥ 2 張的人
            ❌ 63 支抓對 50 → 38（出手時球舉在頭上，手腕常被擋 / 骨架抓不到）→ 出手者維持用遮罩
  catch_wrist 仲裁時「誰接住球」改用手腕判
  catch     仲裁：離手後 catch_s 秒內就被人的手接住、而且接住前球沒到籃框 → 傳球，不是投籃
  fallback  False = 拿掉保底：弱候選要 Roboflow 判跳投 / 上籃（離手附近）才算
"""
import numpy as np

import shot_owner as so
import pose_pass as PP
from shot_events import analyze


def make_cand_fn(pose, cuts, wrist=False, catch=True, fallback=False, catch_s=1.0, r_hand=0.2, catch_wrist=False, pos_win=1.5, rf_layup=False, layup_hits=2, min_cnt=3, touch_filter=False, far=0.5):
    def cand(s, tch, rf, rng):
        fps = s['fps']; n = len(tch)
        k = max(1, int(round(fps / 10.0)))      # 門檻都是照 10fps 調的「張數」→ 30fps 乘 3，維持同樣秒數
        if k > 1:
            # 碰球只連續 < k 張（10fps 時的 1 張）、前後不是他 → 當沒碰到（同 shot_owner.smooth_touches）
            tch = list(tch); i = 0
            while i < n:
                q = tch[i][0]
                if q is None:
                    i += 1; continue
                j = i
                while j < n and tch[j][0] == q: j += 1
                if j - i < k:
                    for m in range(i, j): tch[m] = (None, tch[m][1])
                i = j
        lo = next((i for i, f in enumerate(s['frame_idx']) if f >= rng[0]), 0)
        hi = next((i for i, f in enumerate(s['frame_idx']) if f >= rng[1]), n)
        H = float(np.median([b[3] - b[1] for fd in s['players'] for b in fd.values()])) if s['players'] else 150.0
        if touch_filter and pose is not None:
            # ④ 球飛過別人時遮罩重疊算成「碰到」→ 空中段被切斷（screen_117：#3 跳投，球飛向籃框時一直疊在 #0 身上）。
            # 只在骨架清楚顯示這個人兩隻手腕都離球很遠（> far 倍身高）時才當沒碰到；骨架抓不到就照舊相信遮罩
            #（手腕判出手者會失敗就是因為出手時手腕常被擋）
            tch = list(tch)
            for i, (q, cen) in enumerate(tch):
                if q is None or cen is None or i >= len(pose) or q not in pose[i]:
                    continue
                dd = PP.hand_dist(pose[i][q], cen, H)
                if dd is not None and dd > far:
                    tch[i] = (None, cen)
        whold = None
        if (wrist or catch_wrist) and pose is not None:
            hs = PP.smooth_flicker(PP.holders(pose, tch, H, r_hand))
            hs += [None] * (n - len(hs))
            whold = [(hs[i], tch[i][1]) for i in range(n)]      # 跟 tch 同格式：(誰拿著, 球中心)
        hold = whold if (wrist and whold is not None) else tch
        chold = whold if (catch_wrist and whold is not None) else hold   # 仲裁看誰接住球用哪一種
        cands = []
        for k0, _ in so.free_runs(hold, lo, hi, min_len=2 * k):
            r = k0 - 1
            while r >= 0 and hold[r][0] is None:
                r -= 1
            if r < lo:
                continue
            if wrist and pose is not None:
                pid = hold[r][0]
                run = sum(1 for j in range(max(0, r - 5 * k), r + 1) if hold[j][0] == pid)   # 離手前最後連續拿著
                if run < 2 * k:
                    continue
            else:
                pid, cnt = so.possessor(tch, r, fps, win=pos_win)
                if pid is None or cnt < k * (min(min_cnt, max(2, int(pos_win * 10 * 0.6))) if min_cnt >= 2 else min_cnt):
                    continue
            h = so.player_h(s, r, pid) or 150
            amp = so.flight(s, hold, r) / h
            if amp < so.AMP_REL or amp > so.AMP_MAX:
                continue
            d, top = so.rim_feat(s, tch, rf, r, h)
            strong = d <= so.RIM_REL or top
            if catch:
                # 離手後第一次有人的手拿到球；那之前球最靠近籃框的距離
                j = next((j for j in range(r + 1, min(n, r + 1 + int(catch_s * fps)))
                          if chold[j][0] is not None and chold[j][0] != pid), None)
                if j is not None:
                    d_before, top_before = so.rim_feat(s, tch, rf, r, h, win=(j - r) / fps)
                    if not (d_before <= so.RIM_REL or top_before):
                        continue                                 # 很快被接住、接住前球沒到籃框 = 傳球
            cands.append(dict(shooter=pid, release=round(r / fps, 2), amp=round(amp, 2),
                              rim=round(d, 2) if d < 1e9 else None, top=top, strong=strong))
        strong_c = [c for c in cands if c['strong']]
        if strong_c:
            shots = strong_c
        elif fallback:
            shots = [max(cands, key=lambda c: c['amp'])] if cands else []
        else:
            ev, _, _ = analyze(None, rf, cuts, rng=rng)            # 弱候選要 Roboflow 判跳投 / 上籃
            shots = [c for c in cands if any(-0.6 <= e['t'] - c['release'] <= 1.2 for e in ev)]
            shots = [max(shots, key=lambda c: c['amp'])] if shots else []
        if rf_layup:
            # ⑤ 上籃 / 灌籃沒有空中段：Roboflow 連續偵測到上籃姿勢（框裡的人手上有球）→ 補一次出手，出手者 = 框跟誰的框最重疊
            ev, _, _ = analyze(None, rf, cuts, rng=rng)
            for e in ev:
                if e['kind'] != 'layup' or not e['with_ball'] or e['hits'] < layup_hits:
                    continue
                if any(abs(c['release'] - e['t']) <= 1.5 for c in shots):
                    continue
                i = min(range(n), key=lambda i: abs(s['frame_idx'][i] - e['frame']))
                bx = e['box']
                def iou(b):
                    ix = max(0, min(b[2], bx[2]) - max(b[0], bx[0])); iy = max(0, min(b[3], bx[3]) - max(b[1], bx[1]))
                    u = (b[2] - b[0]) * (b[3] - b[1]) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - ix * iy
                    return ix * iy / u if u > 0 else 0
                best = max(s['players'][i].items(), key=lambda kv: iou(kv[1]), default=None)
                if best is None or iou(best[1]) < 0.3:
                    continue
                shots.append(dict(shooter=best[0], release=e['t'], amp=0.0, rim=None, top=False, strong=True, layup_rf=True))
        out = []
        for c in sorted(shots, key=lambda c: c['release']):
            if not out or c['release'] - out[-1]['release'] > 1.0:
                out.append(c)
        return out
    return cand


def detect_fn(**opt):
    """給 shot_eval.run_detect 用的 detect(s, rf, cuts, shape, n, tracks, tch, pose)。"""
    from shot_detect import detect_core

    def detect(s, rf, cuts, shape, n, tracks, tch, pose):
        return detect_core(s, rf, cuts, shape, n, tracks, tch, cand_fn=make_cand_fn(pose, cuts, **opt))[0]
    return detect
