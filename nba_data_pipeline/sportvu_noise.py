"""
sportvu_noise.py — 量轉播座標（第 2 版 + clean_court_tracks，跟訓練同一套）的誤差長相，
並把 SportVU（乾淨、10 人全在）加上「長得一樣」的誤差，給 nba_contact_data.py 混進訓練用。

量的統計（5fps，每支片段出現最多的 10 個有分隊的人）：
  miss      ：沒座標的比例
  vis_dx    ：離持球者 x 距離（0-10 / 10-20 / 20-30 / 30-40 / 40+ ft）各段的可見比例 → 鏡頭只拍持球者附近
  miss_run  ：缺座標連續段的平均長度（張）
  jit       ：二階差分 |p_t − (p_{t−1} + p_{t+1}) / 2| 的中位數（ft）→ 逐張抖動
  jump      ：相鄰兩張位移 > 8 ft 的比例

雜訊模型（apply_noise）：
  1. 鏡頭：每張的鏡頭中心 = 持球者 x（平滑）+ 慢漂移，半寬 view_w → 範圍外的人沒座標
  2. 遮擋：每人隨機缺一段（rate occ、平均長度 occ_len）
  3. homography 誤差：同一張所有人一起偏（慢漂移 g_sigma）
  4. 腳點誤差：每人慢漂移 d_sigma + 逐張抖動 j_sigma
  5. 單人腳點錯：spike 比例、偏 5~15 ft、持續 1~spike_len 張（轉播的錯常持續幾張，清理只拿得掉單張的）
  加完再跑 clean_court_tracks（跟轉播一樣）。

用法：
  python nba_data_pipeline/sportvu_noise.py measure        # 印轉播 vs SportVU（不加 / 加雜訊）統計
"""
import sys
import glob
import pickle
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

BINS = (10, 20, 30, 40)
# 2026-11-03 對齊轉播：缺 12.6%（轉播 10.8%）、30-40 / 40+ ft 可見 .88 / .65（同）、抖動 0.69（0.70）ft、跳 >8ft 1.43%（1.34%）
BROADCAST = dict(view_w=46, view_drift=10, occ=0.02, occ_len=8, j_sigma=0.42, spike=0.01, spike_len=5)
DEFAULT = dict(view_w=30.0, view_drift=3.0, occ=0.01, occ_len=4.0, g_sigma=0.6, d_sigma=0.8, j_sigma=0.5, spike=0.004, spike_len=1)


def main_players(c, k=10):
    app = collections.Counter(p for fd in c['raw_tracks'] for p in fd)
    return [p for p, _ in app.most_common() if c['track_team'].get(p) in ('offense', 'defense')][:k]


def stats(clips):
    """clips：list of (court_tracks 清理後, 主要球員, handler)。"""
    miss = vis_n = 0
    dx_vis, dx_all = np.zeros(len(BINS) + 1), np.zeros(len(BINS) + 1)
    runs, jit, jumps, steps = [], [], 0, 0
    for ct, ps, H in clips:
        n = len(ct)
        for p in ps:
            have = [p in ct[t] for t in range(n)]
            miss += n - sum(have); vis_n += n
            r = 0
            for h in have + [True]:
                if not h:
                    r += 1
                elif r:
                    runs.append(r); r = 0
            for t in range(n):
                if H in ct[t] and p != H:
                    hx = ct[t][H][0]
                    # 不可見的人用真實位置算 dx：轉播沒有真實位置 → 用前後最近一次看到的位置近似
                    q = ct[t].get(p)
                    if q is None:
                        near = [ct[u][p] for u in range(max(0, t - 5), min(n, t + 6)) if p in ct[u]]
                        if not near:
                            continue
                        q = near[0]
                    b = int(np.searchsorted(BINS, abs(q[0] - hx)))
                    dx_all[b] += 1; dx_vis[b] += p in ct[t]
            for t in range(1, n):
                if p in ct[t] and p in ct[t - 1]:
                    steps += 1
                    jumps += np.hypot(ct[t][p][0] - ct[t - 1][p][0], ct[t][p][1] - ct[t - 1][p][1]) > 8
                if 1 <= t < n - 1 and all(p in ct[u] for u in (t - 1, t, t + 1)):
                    m = (np.array(ct[t - 1][p]) + np.array(ct[t + 1][p])) / 2
                    jit.append(float(np.hypot(*(np.array(ct[t][p]) - m))))
    return dict(miss=miss / vis_n, vis_dx=np.round(dx_vis / np.maximum(dx_all, 1), 2).tolist(),
                miss_run=float(np.mean(runs)) if runs else 0.0, jit=float(np.median(jit)), jump=jumps / max(1, steps))


def broadcast_clips():
    import court_v2
    from convert_contact import CACHE_DIR
    from tactics_tools import clean_court_tracks
    out = []
    for p in sorted(glob.glob(str(CACHE_DIR / 'screen_*.pkl')), key=lambda q: int(Path(q).stem.split('_')[1])):
        if int(Path(p).stem.split('_')[1]) > 503:
            continue
        c = pickle.load(open(p, 'rb'))
        if c.get('pending'):
            continue
        ct = court_v2.court_tracks(c)
        if ct is None:
            continue
        out.append((clean_court_tracks(ct)[0], main_players(c), c['handler_tid']))
    return out


def apply_noise(c, rng, prm=None):
    """SportVU 片段（c 快取格式、10 人每張都在）→ 加轉播式誤差 + clean_court_tracks。回新的 c。"""
    from tactics_tools import clean_court_tracks
    P = dict(DEFAULT, **(prm or {}))
    ct0, n, H = c['court_tracks'], c['n'], c['handler_tid']
    pids = sorted({p for fd in ct0 for p in fd})
    hx = np.array([fd.get(H, (np.nan,))[0] for fd in ct0], dtype=float)
    hx = np.where(np.isnan(hx), np.nanmean(hx), hx)
    k = np.ones(5) / 5
    cam = np.convolve(np.pad(hx, 2, mode='edge'), k, mode='valid') + np.cumsum(rng.normal(0, P['view_drift'] * 0.3, n))
    g = np.cumsum(rng.normal(0, P['g_sigma'] * 0.3, (n, 2)), axis=0) + rng.normal(0, P['g_sigma'], 2)
    out = [dict() for _ in range(n)]
    for p in pids:
        d = np.cumsum(rng.normal(0, P['d_sigma'] * 0.3, (n, 2)), axis=0) + rng.normal(0, P['d_sigma'], 2)
        occ = np.zeros(n, bool)
        t = 0
        while t < n:
            if rng.rand() < P['occ']:
                L = max(1, int(rng.exponential(P['occ_len'])))
                occ[t:t + L] = True; t += L
            t += 1
        sp_left, sp = 0, np.zeros(2)
        for t, fd in enumerate(ct0):
            if sp_left == 0 and rng.rand() < P['spike']:          # 腳點抓錯一小段（1~spike_len 張同一個偏移）
                a = rng.uniform(0, 2 * np.pi)
                sp, sp_left = rng.uniform(5, 15) * np.array([np.cos(a), np.sin(a)]), int(rng.randint(1, P['spike_len'] + 1))
            off = sp if sp_left > 0 else 0.0
            sp_left = max(0, sp_left - 1)
            if p not in fd or occ[t] or abs(fd[p][0] - cam[t]) > P['view_w']:
                continue
            xy = np.array(fd[p]) + g[t] + d[t] + rng.normal(0, P['j_sigma'], 2) + off
            out[t][p] = (float(xy[0]), float(xy[1]))
    c = dict(c)
    c['raw_tracks'] = [{p: None for p in fd} for fd in out]       # build_windows 只看 key（誰在畫面裡）
    c['court_tracks'] = clean_court_tracks(out)[0]
    return c


def sportvu_chunks(games=4):
    from nba_contact_data import game_jsons, chunks_from_segment, PAD, STEP
    from nba_coverage_probe import load_timeline, segments
    from extract_sequences import parse_moment, find_ball_handler
    chunks = []
    for js in game_jsons(games):
        _, frames, _ = load_timeline(js)
        for seg in segments(frames):
            if len(seg) < 2 * PAD * STEP + 10:
                continue
            parsed = [parse_moment(m) for m in seg]
            chunks += chunks_from_segment(parsed, find_ball_handler(parsed, None, None))
    return chunks


def measure(prm_list=None):
    bc = broadcast_clips()
    print('轉播（第 2 版 + 清理）', stats(bc), f'{len(bc)} 支', flush=True)
    ch = sportvu_chunks()
    print('SportVU 原始', stats([(c['court_tracks'], main_players(c), c['handler_tid']) for c in ch]), f'{len(ch)} 段', flush=True)
    for prm in (prm_list or [{}]):
        rng = np.random.RandomState(0)
        nz = [apply_noise(c, rng, prm) for c in ch]
        print('SportVU + 雜訊', prm or 'DEFAULT', stats([(c['court_tracks'], main_players(c), c['handler_tid']) for c in nz]), flush=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['measure']:
        measure()
