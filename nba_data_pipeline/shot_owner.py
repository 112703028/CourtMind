"""
shot_owner.py — 用 SAM3 遮罩（sam3_shot_masks.py）判斷「誰出手」「投籃還是傳球」，補 Roboflow 框的不足
（封蓋的防守者也會被 RF 判成跳投，框疊在一起分不出球在誰手上）。

對 shot_events.analyze 給的每個出手候選（時間 t）：
  1. 每張抽樣影格：球遮罩跟每個球員遮罩（往外擴 DILATE 像素）的重疊像素 → 重疊最多的人 = 此刻碰球的人
  2. 出手者 = t 附近（t-0.8 ~ t+0.5 秒）最後一個碰球的人；離手 = 他最後碰球的那張
  3. 離手後 PASS_WIN 秒內：球先碰到另一個球員 → 傳球；球往上飛（中心 y 變小超過 RISE_PX）→ 投籃
  4. SAM3 編號 → 訓練快取（stubs/train_contact_cache）的編號（bbox IoU 多數決）→ 持球者 / 掩護者 / 進攻 / 防守

用法：python nba_data_pipeline/shot_owner.py [--render]
輸出：output_videos/shot_check/shot_owner.csv、--render 時 output_videos/shot_check/mask_view/<clip>.mp4
"""
import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from shot_events import ROOT, analyze, find_cuts, read_frames

DILATE = 6
PASS_WIN = 1.0
AMP_REL = 0.9          # 投籃動作幅度（出手前最低點 → 空中最高點）≥ 出手者身高 0.9 倍
AMP_MAX = 6.0          # 真投籃幅度最多 ~5 倍身高（63 支答案）
RIM_REL = 1.0          # 球離籃框 ≤ 出手者身高 1.0 倍 = 強候選（63 支答案：門檻 0.7–1.0 / 0.5–2.0 都在 47–50 支全對）
RISE_REL = 0.8          # 球往上 ≥ 出手者身高 0.8 倍 = 投籃（2026-10-25 第二輪 60 支：固定 150px 漏掉遠景的投籃）
RISE_PX = 150          # 24 支人工核對：真投籃球往上 180–420px，假候選（搶籃板 / 運球）29–99px


def unpack_mask(packed):                     # 同 run_sam3_tracks.unpack_mask（那支 import 時要 sam3/torch）
    bbox, h, w, data = packed
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:h * w]
    return bbox, bits.reshape(h, w).astype(bool)


def full_mask(packed, shape):
    (x1, y1, x2, y2), m = unpack_mask(packed)
    out = np.zeros(shape, bool); out[y1:y1 + m.shape[0], x1:x1 + m.shape[1]] = m
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def pick_ball(s, rf):
    """SAM3 的 "basketball" 常追到好幾個物體（比分條上的球圖示、觀眾席的東西）。
    每張選「跟 Roboflow 球偵測位置吻合次數最多」的物體；該物體不在時，選離上一張球最近的其他物體。"""
    rf_at = dict(rf['frames'])
    score = Counter()
    for i, fi in enumerate(s['frame_idx']):
        dets = rf_at.get(fi) or rf_at.get(fi - 1) or rf_at.get(fi + 1) or []
        rb = [x[2] for x in dets if x[0] == 0 and x[1] >= 0.5]
        for o, b in s['balls'][i].items():
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2); sz = max(b[2] - b[0], 10)
            if any(abs(c[0] - (r[0] + r[2]) / 2) < 2 * sz and abs(c[1] - (r[1] + r[3]) / 2) < 2 * sz for r in rb):
                score[o] += 1
    order = [o for o, _ in score.most_common()]
    chosen, prev = [], None
    for f in s['balls']:
        o = next((o for o in order if o in f), None)
        if o is None and f and prev is not None:
            o = min(f, key=lambda k: abs((f[k][0] + f[k][2]) / 2 - prev[0]) + abs((f[k][1] + f[k][3]) / 2 - prev[1]))
        chosen.append(o)
        if o is not None: prev = ((f[o][0] + f[o][2]) / 2, (f[o][1] + f[o][3]) / 2)
    return chosen, dict(score)


def touches(s, shape, chosen):
    """每張抽樣影格：(碰球的球員 pid 或 None, 球中心 或 None)。只用 pick_ball 選出的那顆球。"""
    k = np.ones((2 * DILATE + 1, 2 * DILATE + 1), np.uint8)
    out = []
    for pm, bm, o in zip(s['player_masks'], s['ball_masks'], chosen):
        if o is None or o not in bm:
            out.append((None, None)); continue
        ball = full_mask(bm[o], shape)
        ys, xs = np.nonzero(ball)
        center = (xs.mean(), ys.mean()) if len(xs) else None
        best, bo = None, 0
        for pid, p in pm.items():
            m = cv2.dilate(full_mask(p, shape).astype(np.uint8), k).astype(bool)
            ov = int((m & ball).sum())
            if ov > bo: best, bo = pid, ov
        out.append((best, center))
    return out


def map_to_cache(s, cache):
    """SAM3 pid → 訓練快取 pid（同一原片影格的 bbox IoU ≥ 0.5，多數決）。"""
    if cache is None: return {}
    step = cache['fps'] / 5
    votes = {}
    for i, fi in enumerate(s['frame_idx']):
        k = fi / step
        if abs(k - round(k)) > 1e-6 or round(k) >= len(cache['raw_tracks']): continue
        ref = cache['raw_tracks'][int(round(k))]
        for pid, b in s['players'][i].items():
            cand = max(ref.items(), key=lambda kv: iou(b, kv[1]), default=None)
            if cand and iou(b, cand[1]) >= 0.5:
                votes.setdefault(pid, Counter())[cand[0]] += 1
    return {pid: c.most_common(1)[0][0] for pid, c in votes.items()}


def role(cpid, cache):
    if cache is None or cpid is None: return '?'
    if cpid == cache['handler_tid']: return '持球者'
    if cpid in cache.get('screener_tids', []): return '掩護者'
    team = cache['track_team'].get(cpid)
    return {'offense': '其他進攻', 'defense': '防守者'}.get(team, '?')


def cached_touches(clip, s, rf, shape, d):
    """touches() 要逐張解遮罩算重疊（一支 ~10 秒）→ 存 <clip>_touch.pkl，調規則時不用重算。"""
    p = Path(d) / f'{clip}_touch.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    chosen, bscore = pick_ball(s, rf)
    out = (chosen, bscore, touches(s, shape, chosen))
    pickle.dump(out, open(p, 'wb'))
    return out


def smooth_touches(tch):
    """只碰到 1 張、前後都不是他的 → 當作沒碰到（球飛過封蓋者手邊 / 遮罩邊緣擦到；screen_463 3.8s、414 3.9s）。"""
    out = list(tch)
    for i in range(1, len(tch) - 1):
        p = tch[i][0]
        if p is not None and tch[i - 1][0] != p and tch[i + 1][0] != p:
            out[i] = (None, tch[i][1])
    return out


def possessor(tch, r, fps, win=1.5):
    """離手前 win 秒內碰球最多的人（持球者連續拿好幾張，封蓋的人只碰一兩張；screen_414/500）。"""
    c = Counter(tch[j][0] for j in range(max(0, r - int(win * fps)), r + 1) if tch[j][0] is not None)
    if not c: return None, 0
    pid, k = c.most_common(1)[0]
    return pid, k


def player_h(s, r, pid):
    for d in (0, -1, 1, -2, 2, -3, 3):
        b = s['players'][min(max(r + d, 0), len(s['players']) - 1)].get(pid)
        if b: return b[3] - b[1]
    return None


def flight(s, tch, r, win=1.0, pre=0.8):
    """投籃動作的上升幅度：離手前 pre 秒內球的最低點（收球 / 蹲下）→ 離手後 win 秒內（沒人碰時）的最高點。
    只從離手點量會偏小（球已經舉到頭上；screen_491 146px vs 從胸前量 212px）。"""
    fps = s['fps']
    before = [tch[j][1][1] for j in range(max(0, r - int(pre * fps)), r + 1) if tch[j][1] is not None]
    ys = [tch[j][1][1] for j in range(r + 1, min(len(tch), r + 1 + int(win * fps)))
          if tch[j][1] is not None and tch[j][0] is None]
    return (max(before) - min(ys)) if (ys and before) else 0


def is_shot(rise, h):
    """往上 ≥ 出手者身高 RISE_REL 倍（遠景球員小、像素少；screen_309 143px、456 137px 是真投籃），或 ≥ RISE_PX。"""
    return rise >= RISE_PX or (h is not None and rise >= RISE_REL * h)


def free_runs(tch, lo, hi, min_len=2):
    """[球在空中、沒人碰] 的連續段 → [(開始, 結束)]。"""
    runs, i = [], lo
    while i < hi:
        if tch[i][0] is None and tch[i][1] is not None:
            k = i
            while k < hi and tch[k][0] is None and tch[k][1] is not None: k += 1
            if k - i >= min_len: runs.append((i, k))
            i = k
        else:
            i += 1
    return runs


def release_at(s, tch, k):
    """空中段從 k 開始 → (離手影格 r, 出手者, 碰球張數)。r = k 前最後一張有人碰的。"""
    r = k - 1
    while r >= 0 and tch[r][0] is None: r -= 1
    if r < 0: return None, None, 0
    pid, cnt = possessor(tch, r, s['fps'])
    return r, pid, cnt


def describe(s, tch, r, pid):
    rise = flight(s, tch, r); h = player_h(s, r, pid)
    rel = f'{rise / h:.1f}x 身高' if h else '?'
    return rise, h, (f'投籃（球往上 {rise:.0f}px = {rel}）' if is_shot(rise, h) else f'不是投籃（球只往上 {rise:.0f}px = {rel}）')


def judge(ev_t, s, tch):
    """RF 出手時間附近（t-1.2 ~ t+0.5 秒）找「球開始在空中」的時刻；出手者 = 離手前 1.5 秒碰球最多的人。
    多個空中段時，選往上飛最多的那個。"""
    fps = s['fps']; n = len(tch)
    lo, hi = max(0, int((ev_t - 1.2) * fps)), min(n, int((ev_t + 0.5) * fps) + 1)
    best = None
    for k, _ in free_runs(tch, lo, hi):
        r, pid, cnt = release_at(s, tch, k)
        if pid is None or cnt < 3: continue
        rise = flight(s, tch, r)
        if best is None or rise > best[2]: best = (r, pid, rise)
    if best is None:
        return dict(shooter=None, release=None, verdict='球沒追到 / 沒人持球', rise=0)
    r, pid, _ = best
    rise, h, verdict = describe(s, tch, r, pid)
    return dict(shooter=pid, release=round(r / fps, 2), verdict=verdict, rise=round(rise), h=h)


def traj_candidates(s, tch, rng):
    """只靠 SAM3 球軌跡找出手（補 Roboflow 漏抓的）：球離開持球者 → 空中 ≥ 0.3 秒 → 往上 ≥ 身高 RISE_REL 倍。"""
    fps = s['fps']; n = len(tch); out = []
    lo = next((i for i, f in enumerate(s['frame_idx']) if f >= rng[0]), 0)
    hi = next((i for i, f in enumerate(s['frame_idx']) if f >= rng[1]), n)
    for k, e in free_runs(tch, lo, hi, min_len=max(3, int(0.3 * fps))):
        r, pid, cnt = release_at(s, tch, k)
        if pid is None or cnt < 3 or r < lo: continue
        rise, h, verdict = describe(s, tch, r, pid)
        if is_shot(rise, h):
            out.append(dict(shooter=pid, release=round(r / fps, 2), rise=round(rise), h=h,
                            verdict=verdict.replace('）', '，只靠球軌跡）')))
    return out


def rim_feat(s, tch, rf, r, h, win=1.5):
    """離手後 win 秒內：球（SAM3）跟籃框（Roboflow rim，幾乎每張都有）最近距離 / 出手者身高；球是否從畫面上方飛出。
    84 支人工核對：真投籃 49/57 ≤ 1.0 倍身高；高吊傳球 / 假動作 3.25、沒靠近。"""
    at = dict(rf['frames']); best, top = float('inf'), False
    for j in range(r + 1, min(len(tch), r + 1 + int(win * s['fps']))):
        c = tch[j][1]
        if c is None: continue
        if c[1] < 40: top = True
        fi = s['frame_idx'][j]
        for x in at.get(fi) or at.get(fi - 1) or at.get(fi + 1) or []:
            if x[0] == 9 and x[1] >= 0.4:
                rx, ry = (x[2][0] + x[2][2]) / 2, (x[2][1] + x[2][3]) / 2
                best = min(best, float(np.hypot(c[0] - rx, c[1] - ry)) / h)
    return best, top


def shot_candidates(s, tch, rf, rng):
    """每段「球在空中」= 一個候選：出手者 = 離手前 1.5 秒碰球最多的人（≥3 張）、幅度 ≥ AMP_REL 倍身高。
    強候選 = 球靠近籃框（≤ RIM_REL 倍身高）或從畫面上方飛出。投籃 = 全部強候選；沒有就取幅度最大的一個。"""
    fps = s['fps']; n = len(tch)
    lo = next((i for i, f in enumerate(s['frame_idx']) if f >= rng[0]), 0)
    hi = next((i for i, f in enumerate(s['frame_idx']) if f >= rng[1]), n)
    cands = []
    for k, _ in free_runs(tch, lo, hi):
        r, pid, cnt = release_at(s, tch, k)
        if pid is None or cnt < 3 or r < lo: continue
        h = player_h(s, r, pid) or 150
        amp = flight(s, tch, r) / h
        if amp < AMP_REL or amp > AMP_MAX: continue          # > AMP_MAX = 球物體中途換掉（screen_463 18 倍），不是真的飛那麼高
        d, top = rim_feat(s, tch, rf, r, h)
        cands.append(dict(shooter=pid, release=round(r / fps, 2), amp=round(amp, 2), rim=round(d, 2) if d < 1e9 else None,
                          top=top, strong=d <= RIM_REL or top))
    strong = [c for c in cands if c['strong']]
    shots = strong or ([max(cands, key=lambda c: c['amp'])] if cands else [])
    out = []
    for c in shots:                                   # 同一次出手（離手差 ≤ 1 秒）只留第一個
        if not out or c['release'] - out[-1]['release'] > 1.0: out.append(c)
    return out


def render(clip, frames, s, tch, results, out, chosen=None):
    fps_o = s['orig_fps']; h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps_o, (w, h))
    shooters = {r['shooter'] for r in results if r.get('shooter') is not None}
    layer = {}                                                # 遮罩 10fps，同一張抽樣影格算一次重複用
    for f, img in enumerate(frames):
        i = min(int(round(f * s['fps'] / fps_o)), len(tch) - 1)
        if i not in layer:
            paint, lab = [], []
            for pid, p in s['player_masks'][i].items():
                col = (0, 0, 255) if pid in shooters else ((0, 200, 255) if pid == tch[i][0] else (180, 180, 180))
                paint.append((full_mask(p, (h, w)), col)); b = s['players'][i][pid]; lab.append((f'#{pid}', (b[0], b[1] - 3), col))
            o = chosen[i] if chosen else None
            if o is not None and o in s['ball_masks'][i]: paint.append((full_mask(s['ball_masks'][i][o], (h, w)), (0, 255, 0)))
            layer = {i: (paint, lab)}
        paint, lab = layer[i]
        im = img.copy(); ov = im.copy()
        for m, col in paint: ov[m] = col
        for t, pos, col in lab: cv2.putText(im, t, pos, 0, 0.5, col, 2)
        im = cv2.addWeighted(ov, 0.45, im, 0.55, 0)
        for r in results:                                     # 出手那一刻：粗框 + 出手者頭上大字（RF 候選紅色、只靠球軌跡黃色）
            rel = r.get('release')
            if rel is None or not r['verdict'].startswith('投籃') or not rel - 0.1 <= f / fps_o <= rel + 0.6: continue
            col = (0, 255, 255) if r.get('kind') == 'traj-only' else (0, 0, 255)
            cv2.rectangle(im, (0, 0), (w - 1, h - 1), col, 14)
            b = s['players'][i].get(r['shooter'])
            x, y = (b[0], max(80, b[1] - 12)) if b else (w // 2 - 120, h // 2)
            txt = f"SHOT! #{r['shooter']}"
            cv2.putText(im, txt, (x - 2, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 9)
            cv2.putText(im, txt, (x - 2, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
        cv2.rectangle(im, (0, 0), (w, 34), (0, 0, 0), -1)
        txt = f'{clip} {f / fps_o:.2f}s  ball-touch: #{tch[i][0]}'
        for r in results:
            if r.get('release') is not None and r['t'] - 0.8 <= f / fps_o <= r['t'] + 2:
                en = {'持球者': 'handler', '掩護者': 'screener', '其他進攻': 'offense', '防守者': 'defense'}.get(r['role'], '?')
                txt += f"  SHOOTER #{r['shooter']}({en}) release {r['release']}s {r['verdict_en']}"
        cv2.putText(im, txt, (8, 24), 0, 0.6, (0, 255, 255), 2)
        vw.write(im)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', default='output_videos/shot_check/rf_dets.pkl')
    ap.add_argument('--sam3_dir', default='stubs/shot_sam3')
    ap.add_argument('--video_dir', default='input_videos/screen_ext')
    ap.add_argument('--out_dir', default='output_videos/shot_check')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--clips', default='', help='逗號分隔，只跑這些')
    ap.add_argument('--csv', default='shot_owner.csv', help='平行跑時每份給不同檔名')
    args = ap.parse_args()
    R = pickle.load(open(ROOT / args.dets, 'rb'))
    rows = []
    only = [c for c in args.clips.split(',') if c]
    for clip in sorted(only or R):
        sp = ROOT / args.sam3_dir / f'{clip}_shot_sam3.pkl'
        if not sp.exists():
            print(f'{clip}: 還沒有 SAM3 遮罩，跳過'); continue
        s = pickle.load(open(sp, 'rb'))
        cp = ROOT / 'stubs' / 'train_contact_cache' / f'{clip}.pkl'
        cache = pickle.load(open(cp, 'rb')) if cp.exists() else None
        frames = read_frames(ROOT / args.video_dir / f'{clip}.mp4')
        shape = frames[0].shape[:2]
        events, rng, _ = analyze(clip, R[clip], find_cuts(frames))
        chosen, bscore, raw = cached_touches(clip, s, R[clip], shape, ROOT / args.sam3_dir)
        tch = smooth_touches(raw)
        pmap = map_to_cache(s, cache)
        res = []
        for c in shot_candidates(s, tch, R[clip], rng):
            near = [e for e in events if -0.6 <= e['t'] - c['release'] <= 1.2]     # RF 出手偵測：補類型 / 進球
            e = min(near, key=lambda e: abs(e['t'] - c['release'])) if near else None
            made = any(x['made'] and -0.6 <= x['t'] - c['release'] <= 2.0 for x in events)
            rim = f"籃框 {c['rim']}x" if c['rim'] is not None else '沒看到籃框'
            c.update(t=c['release'], kind=e['kind'] if e else 'traj-only', made=made, cache_pid=pmap.get(c['shooter']),
                     verdict=f"投籃（幅度 {c['amp']}x 身高，{rim}{'，飛出畫面上方' if c['top'] else ''}{'' if c['strong'] else '，弱候選'}）")
            c['role'] = role(c['cache_pid'], cache)
            c['verdict_en'] = 'SHOT' if c['strong'] else 'SHOT?(weak)'
            res.append(c)
            rows.append({'clip': clip, 'RF時間(s)': e['t'] if e else '', 'RF類型': c['kind'], 'RF進球': 'Y' if made else 'N',
                         '離手(s)': c['release'], 'SAM3出手者': c['shooter'], '訓練編號': c['cache_pid'],
                         '角色': c['role'], '判斷': c['verdict']})
        nb = sum(1 for x in tch if x[1] is not None)
        print(f"{clip}（球追到 {nb}/{len(tch)}，球物體吻合 {bscore}）: " + ('; '.join(
            f"{r['t']}s → #{r['shooter']}[{r['role']}] 離手 {r['release']} {r['verdict']}" for r in res) or '沒有候選'))
        if args.render and not (ROOT / args.out_dir / 'mask_view' / f'{clip}.mp4').exists():
            (ROOT / args.out_dir / 'mask_view').mkdir(parents=True, exist_ok=True)
            render(clip, frames, s, tch, res, ROOT / args.out_dir / 'mask_view' / f'{clip}.mp4', chosen)
    out = ROOT / args.out_dir / args.csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8-sig', newline='') as fo:
        wr = csv.DictWriter(fo, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f'→ {out}')


if __name__ == '__main__':
    main()
