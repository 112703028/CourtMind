"""
nba_contact_data.py — 把 NBA SportVU 擋拆轉成跟 convert_contact.py 完全同格式的接觸幀訓練窗，
混進轉播資料一起微調（只進訓練，不進驗證/考題）。

為什麼：學習曲線顯示擋拆偵測很缺資料；NBA 預訓練有用（拿掉 −0.09），但當初 Step 2 弱標記
用的是 25fps 0.4 秒窗、相對座標、「整窗都標擋拆」（就是連續發射的根因），跟現在的任務格式不同。
這裡改成同一套：5fps、10 張窗（2 秒）、絕對球場座標、接觸幀高斯軟標籤、候選人輪流 slot 1
—— 直接呼叫 convert_contact.build_windows，保證跟轉播資料同一套產生方式。

擋拆來源：nba_coverage_probe.find_screens（NETS 三角形條件、持球者離籃框 ≥16ft、同組 3 秒內去重）
片段：同一持球者、時間相近的擋拆併成一段，取第一次擋拆前 3 秒 ~ 最後一次後 3 秒
雜訊：轉播座標有雜訊（homography + 腳點），NBA 乾淨 → 每人加「緩慢漂移 + 逐張抖動」的位置雜訊
clip_ids：負數（−1, −2, …），finetune.py 看到負數就一律放訓練集

用法：
  python nba_data_pipeline/nba_contact_data.py                 # 已解壓的所有比賽
  python nba_data_pipeline/nba_contact_data.py --games 6 --noise 1.5 --max_windows 16000
輸出：nba_data_pipeline/output_contact_nba/{positive,negative}_sequences.npz
"""
import argparse
import collections
from pathlib import Path

import numpy as np

from extract_sequences import parse_moment, find_ball_handler, dist
from nba_coverage_probe import load_timeline, segments, find_screens, FPS, SX, SY
from convert_contact import build_windows, N_OFF, N_DEF

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'nba-movement-data' / 'data'
OUT = Path(__file__).parent / 'output_contact_nba'
STEP = FPS // 5                         # 25fps → 5fps
PAD = 3 * 5                             # 片段前後各 3 秒（5fps 張數）


def game_jsons(n):
    js = sorted(p for d in DATA.iterdir() if d.is_dir() for p in d.glob('*.json'))
    return js[:n] if n else js


def chunks_from_segment(parsed, handlers, team_of_handler_cache=None):
    """一段連續追蹤 → [(c 快取格式 dict)]：同持球者、時間相近的擋拆併成一個片段。"""
    screens = find_screens(parsed, handlers)
    keep = []
    for s in screens:
        pl = parsed[s['cf']]['players']
        hx, hy = pl[s['H']]['pos']
        rim = (88.75, 25.0) if hx >= 47 else (5.25, 25.0)
        if dist((hx, hy), rim) >= 16:
            keep.append(s)
    groups = []
    for s in sorted(keep, key=lambda x: x['cf']):
        g = groups[-1] if groups else None
        if g and g[-1]['H'] == s['H'] and s['cf'] - g[-1]['cf'] <= 3 * FPS:
            g.append(s)
        else:
            groups.append([s])
    out = []
    for g in groups:
        f0 = g[0]['cf'] - PAD * STEP
        f1 = g[-1]['cf'] + PAD * STEP
        if f0 < 0 or f1 >= len(parsed):
            continue
        frames = list(range(f0, f1 + 1, STEP))
        ct = [{p: (i['pos'][0] * SX, i['pos'][1] * SY) for p, i in parsed[f]['players'].items()} for f in frames]
        team = {p: ('offense' if i['team'] == g[0]['team'] else 'defense')
                for p, i in parsed[g[0]['cf']]['players'].items()}
        events = [{'idx': int(round((s['cf'] - f0) / STEP)), 'tid': s['S'], 'contact': True} for s in g]
        out.append({'n': len(ct), 'court_tracks': ct, 'raw_tracks': ct, 'track_team': team,
                    'handler_tid': g[0]['H'], 'screener_tids': sorted({s['S'] for s in g}),
                    'defender_tid': None, 'events': events, 'valid_start_idx': 0})
    return out


def add_noise(c, sigma, rng):
    """每人：緩慢漂移（隨機遊走，模擬 homography/腳點偏差）+ 逐張抖動。"""
    if sigma <= 0:
        return c
    pids = {p for fd in c['court_tracks'] for p in fd}
    drift = {p: np.cumsum(rng.normal(0, sigma * 0.3, (c['n'], 2)), axis=0) + rng.normal(0, sigma, 2)
             for p in pids}
    ct = []
    for k, fd in enumerate(c['court_tracks']):
        ct.append({p: tuple(np.array(xy) + drift[p][k] + rng.normal(0, sigma * 0.5, 2)) for p, xy in fd.items()})
    c = dict(c)
    c['court_tracks'] = c['raw_tracks'] = ct
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=0, help='用幾場（0 = 所有已解壓的）')
    ap.add_argument('--noise', type=float, default=1.5, help='位置雜訊尺度（英尺）；0 = 不加')
    ap.add_argument('--sigma', type=float, default=1.0, help='接觸幀高斯 σ（同 convert_contact）')
    ap.add_argument('--anchor', type=int, default=3)
    ap.add_argument('--min_target', type=float, default=0.05)
    ap.add_argument('--max_windows', type=int, default=16000, help='最多幾個窗（避免 NBA 淹沒轉播資料）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    chunks = []
    for js in game_jsons(args.games):
        gameid, frames, _ = load_timeline(js)
        n0 = len(chunks)
        for seg in segments(frames):
            if len(seg) < 2 * PAD * STEP + 10:
                continue
            parsed = [parse_moment(m) for m in seg]
            chunks += chunks_from_segment(parsed, find_ball_handler(parsed, None, None))
        ev = sum(len(c['events']) for c in chunks[n0:])
        print(f'比賽 {gameid}：{len(chunks) - n0} 個片段、{ev} 個擋拆')

    roles = np.array([0] + [1] * (N_OFF - 1) + [2] * N_DEF, dtype=np.int64)
    T, P, C, CTR = [], [], [], []
    for k in rng.permutation(len(chunks)):
        c = add_noise(chunks[k], args.noise, rng)
        for traj, pw, center in build_windows(c, args.sigma, args.sigma * 1.5, args.min_target, args.anchor):
            T.append(traj); P.append(pw); C.append(-(int(k) + 1)); CTR.append(center)
        if len(T) >= args.max_windows:
            break
    T, P = np.stack(T), np.stack(P)
    C, CTR = np.array(C, np.int32), np.array(CTR, np.int32)
    lab = (P.max(axis=1) >= 0.5).astype(np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, m in (('positive_sequences.npz', lab == 1), ('negative_sequences.npz', lab == 0)):
        np.savez_compressed(str(out / name), trajectories=T[m], roles=np.repeat(roles[None], m.sum(), 0),
                            labels=lab[m], pairwise_labels=P[m], clip_ids=C[m], centers=CTR[m],
                            anchor=np.int32(args.anchor), sigma=np.float32(args.sigma))
    print(f'{len(set(C.tolist()))} 個片段 → {len(T)} 個窗：正 {int(lab.sum())}、負 {int((1 - lab).sum())}'
          f'（雜訊 {args.noise}ft）→ {out}')


if __name__ == '__main__':
    main()
