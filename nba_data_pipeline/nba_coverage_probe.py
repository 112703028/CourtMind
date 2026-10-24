"""
nba_coverage_probe.py — 上限測試：在乾淨的 NBA SportVU 軌跡（25fps、座標準）上，
人只看俯視動畫的點，能不能分辨防守方式（換防 / 延誤 / 包夾 / 退防）？

  能 → 軌跡裡有這個資訊，轉播版分不出是座標太吵 → 投資感知精度 / NBA 弱監督
  不能 → 光靠軌跡本來就不夠 → 改走姿態 / 畫面

流程：
  1. 讀一場比賽的 SportVU json（events 有重疊 → 用 (period, game_clock) 去重成連續時間軸）
  2. 持球者：同 extract_sequences.find_ball_handler（球近、球低、球慢、連續 5 幀）
  3. 擋拆：NETS 三角形條件（掩護者↔持球者 <6ft、Dh↔持球者 <6ft、Dh↔掩護者 <3ft）連續 ≥3 幀，
     接觸幀 = 這段裡 Dh↔掩護者最近那幀；同一組 3 秒內只留一次；
     撞上時持球者離籃框 ≥16ft（只要外線擋拆）
  4. Dh / Ds：撞上前 1 秒平均距離最近的防守者（同 coverage_ml）
  5. 每個擋拆畫一支俯視動畫（撞上前 2 秒 ~ 後 3 秒，半速播放）+ 一行標記表
  6. 另存 5fps 球場座標（換算成轉播 pipeline 的球場尺寸），之後可直接套 coverage_ml 特徵

用法：
  python nba_data_pipeline/nba_coverage_probe.py --n 40
輸出：
  output_videos/nba_probe_perimeter/NN_*.mp4、label.csv、probe_samples.pkl
"""
import json
import math
import collections
import pickle
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from extract_sequences import parse_moment, find_ball_handler, dist

ROOT = Path(__file__).parent.parent
DEFAULT_JSON = ROOT / 'nba-movement-data' / 'data' / '01.01.2016.CHA.at.TOR' / '0021500492.json'
OUT_DIR = ROOT / 'output_videos' / 'nba_probe_perimeter'
FPS = 25
PRE_S, POST_S = 2.0, 3.0
SX, SY = 91.8 / 94.0, 49.2 / 50.0          # SportVU 94×50 ft → 轉播 pipeline 的 91.8×49.2


# ── 資料 ──────────────────────────────────────────────────────────────────────

def load_timeline(path):
    d = json.load(open(path))
    seen, frames, roster = set(), [], {}
    for ev in d['events']:
        for side in ('home', 'visitor'):
            t = ev[side]
            for p in t['players']:
                roster[p['playerid']] = {'team': t['abbreviation'], 'jersey': p['jersey'],
                                         'name': f"{p['firstname'][:1]}.{p['lastname']}"}
        for m in ev['moments']:
            key = (m[0], round(m[2], 2))
            if key in seen or len(m[5]) != 11:
                continue
            seen.add(key)
            frames.append(m)
    frames.sort(key=lambda m: (m[0], -m[2]))
    return d['gameid'], frames, roster


def segments(frames):
    """依 period 與時間斷點（> 0.2 秒沒資料）切成連續段。"""
    seg, out = [frames[0]], []
    for a, b in zip(frames, frames[1:]):
        if b[0] != a[0] or abs((a[2] - b[2]) - 1 / FPS) > 0.2:
            out.append(seg)
            seg = []
        seg.append(b)
    out.append(seg)
    return out


# ── 擋拆偵測 ──────────────────────────────────────────────────────────────────

def nearest_def_mean(parsed, frames_idx, target, defs):
    def md(d):
        v = [dist(parsed[f]['players'][d]['pos'], parsed[f]['players'][target]['pos'])
             for f in frames_idx if d in parsed[f]['players'] and target in parsed[f]['players']]
        return np.mean(v) if v else np.inf
    return min(defs, key=md)


def find_screens(parsed, handlers):
    n = len(parsed)
    runs = {}                                      # (H, S) → [frames]
    found = []
    for f in range(FPS, n):
        h = handlers[f]
        if h is None:
            continue
        H, team = h
        pl = parsed[f]['players']
        if H not in pl:
            continue
        defs = [p for p, i in pl.items() if i['team'] != team]
        offs = [p for p, i in pl.items() if i['team'] == team and p != H]
        if len(defs) != 5 or len(offs) != 4:
            continue
        hp = pl[H]['pos']
        dh = min(defs, key=lambda d: dist(pl[d]['pos'], hp))
        for S in offs:
            sp = pl[S]['pos']
            ok = dist(sp, hp) < 6 and dist(pl[dh]['pos'], hp) < 6 and dist(pl[dh]['pos'], sp) < 3
            if ok:
                runs.setdefault((H, S, team), []).append(f)
    for (H, S, team), fs in runs.items():
        # 連續段（容許 2 幀斷點）
        grp = [[fs[0]]]
        for a, b in zip(fs, fs[1:]):
            (grp[-1].append(b) if b - a <= 3 else grp.append([b]))
        for g in grp:
            if len(g) < 3:
                continue
            def dd(f):
                pl = parsed[f]['players']
                defs = [p for p, i in pl.items() if i['team'] != team]
                return min(dist(pl[d]['pos'], pl[S]['pos']) for d in defs)
            cf = min(g, key=dd)
            found.append({'H': H, 'S': S, 'team': team, 'cf': cf})
    found.sort(key=lambda x: x['cf'])
    kept = []
    for x in found:
        if all(not (y['H'] == x['H'] and y['S'] == x['S'] and abs(y['cf'] - x['cf']) < 3 * FPS) for y in kept):
            kept.append(x)
    return kept


# ── 畫圖 ──────────────────────────────────────────────────────────────────────

PX = 10                                       # 每英尺幾像素
COL = {'H': (40, 180, 40), 'S': (40, 40, 220), 'Ds': (0, 140, 255), 'Dh': (220, 120, 0)}


def court_xy(p, flip):
    x, y = p
    if flip:                                   # 一律畫成往右攻
        x, y = 94 - x, 50 - y
    return int((x - 41) * PX) + 20, int(y * PX) + 20


def draw_court(flip):
    W, Hh = int(53 * PX) + 40, int(50 * PX) + 40
    img = np.full((Hh, W, 3), (205, 225, 240), np.uint8)
    c = lambda x, y: (int((x - 41) * PX) + 20, int(y * PX) + 20)
    cv2.rectangle(img, c(47, 0), c(94, 50), (60, 60, 60), 2)
    cv2.rectangle(img, c(94 - 19, 17), c(94, 33), (60, 60, 60), 2)        # 禁區
    cv2.circle(img, c(94 - 19, 25), 6 * PX, (60, 60, 60), 2)
    cv2.circle(img, c(88.75, 25), int(0.75 * PX), (0, 90, 230), 2)       # 籃框
    cv2.ellipse(img, c(88.75, 25), (int(23.75 * PX), int(23.75 * PX)), 0, 90, 270, (60, 60, 60), 2)
    cv2.line(img, c(94 - 14, 3), c(94, 3), (60, 60, 60), 2)
    cv2.line(img, c(94 - 14, 47), c(94, 47), (60, 60, 60), 2)
    cv2.line(img, c(47, 0), c(47, 50), (60, 60, 60), 3)
    return img


def render(path, parsed, s, roster, f0, f1):
    pl0 = parsed[s['cf']]['players']
    flip = pl0[s['H']]['pos'][0] < 47           # 進攻往左 → 翻成往右
    base = draw_court(flip)
    role = {s['H']: 'H', s['S']: 'S', s['Ds']: 'Ds', s['Dh']: 'Dh'}
    h, w = base.shape[:2]
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                             '-s', f'{w}x{h}', '-r', str(FPS // 2), '-i', '-', '-c:v', 'libx264',
                             '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)
    trail = {p: [] for p in role}
    for f in range(f0, f1):
        img = base.copy()
        pm = parsed[f]
        for p, r in role.items():
            if p in pm['players']:
                trail[p].append(court_xy(pm['players'][p]['pos'], flip))
                if len(trail[p]) > 1:
                    cv2.polylines(img, [np.array(trail[p][-40:])], False, COL[r], 1)
        for p, info in pm['players'].items():
            q = court_xy(info['pos'], flip)
            off = info['team'] == s['team']
            r = role.get(p)
            col = COL[r] if r else ((90, 90, 90) if off else (150, 150, 150))
            if off:
                cv2.circle(img, q, 11, col, -1)
            else:
                cv2.circle(img, q, 11, col, 3)
            jersey = roster.get(p, {}).get('jersey', '?')
            cv2.putText(img, str(jersey), (q[0] - 9, q[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255) if off else col, 1)
            if r:
                cv2.putText(img, r, (q[0] + 12, q[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        b = court_xy(pm['ball']['pos'], flip)
        cv2.circle(img, b, 5 + int(min(pm['ball']['height'], 15) / 3), (0, 100, 200), -1)
        t = (f - s['cf']) / FPS
        txt = f"{t:+.1f}s" + ("  << CONTACT" if abs(f - s['cf']) <= 3 else "")
        cv2.putText(img, txt, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 220) if abs(f - s['cf']) <= 3 else (30, 30, 30), 2)
        cv2.putText(img, "filled=offense  ring=defense  0.5x speed", (25, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


# ── main ─────────────────────────────────────────────────────────────────────

def aggressive_kind(win, s):
    """
    用乾淨座標粗估「可能是積極防守」：回 'switch' / 'blitz' / 'hedge' / None（只拿來挑候選，不給使用者看）。
      switch：撞後 0.4~2.4 秒，離持球者最近的防守者常是 Ds，且 Dh 比較靠近掩護者
      blitz ：撞後持球者 6ft 內常有 ≥2 個防守者
      hedge ：Ds 撞後 1.2 秒內上前到持球者 5ft 內，最後退開（>9ft），Dh 回到持球者（<7ft）
    """
    cf, H, S, Ds, Dh = s['cf'], s['H'], s['S'], s['Ds'], s['Dh']
    fr = [cf + k for k in range(int(0.4 * FPS), int(2.4 * FPS) + 1, 5) if cf + k < len(win)]
    sw = bl = 0
    for f in fr:
        pl = win[f]['players']
        if not all(p in pl for p in (H, S, Ds, Dh)):
            continue
        hp = pl[H]['pos']
        defs = {p: dist(i['pos'], hp) for p, i in pl.items() if i['team'] != s['team']}
        if min(defs, key=defs.get) == Ds and dist(pl[Dh]['pos'], pl[S]['pos']) < dist(pl[Dh]['pos'], hp):
            sw += 1
        if sum(v <= 6 for v in defs.values()) >= 2:
            bl += 1
    if fr and sw / len(fr) >= 0.4:
        return 'switch'
    if fr and bl / len(fr) >= 0.3:
        return 'blitz'

    def dd(a, b, f):
        pl = win[f]['players']
        return dist(pl[a]['pos'], pl[b]['pos']) if a in pl and b in pl else np.inf
    near = min(dd(Ds, H, cf + k) for k in range(0, int(1.2 * FPS)))
    end = [cf + k for k in range(int(2.0 * FPS), int(2.4 * FPS)) if cf + k < len(win)]
    if near <= 5 and end and np.mean([dd(Ds, H, f) for f in end]) > 9 \
            and np.mean([dd(Dh, H, f) for f in end]) < 7:
        return 'hedge'
    return None


def clip_tracks(win, s):
    """片段 → 5fps 球場座標（換算成轉播 pipeline 尺寸），格式可直接套 coverage_ml.features。"""
    ct = [{p: (i['pos'][0] * SX, i['pos'][1] * SY) for p, i in win[f]['players'].items()}
          for f in range(0, len(win), FPS // 5)]
    team = {p: ('offense' if i['team'] == s['team'] else 'defense') for p, i in win[s['cf']]['players'].items()}
    return {'court_tracks': ct, 'track_team': team, 'handler_tid': s['H'], 'screener': s['S'],
            'contact_idx': s['cf'] // (FPS // 5)}


def drop_prob(c):
    """coverage_model（轉播資料訓練）判「退防/延誤」的機率；算不出來回 1（排到最後）。"""
    import coverage_ml as cm
    import coverage_model as cmod
    M = cmod.load_model()
    f = cm.features(c, {'tid': c['screener'], 'idx': c['contact_idx']})
    if M is None or f is None:
        return 1.0
    p = M['model'].predict_proba(cmod._fill_nan(f[None], M['nan_fill']))[0]
    return float(p[list(M['model'].classes_).index('drop')])


def game_sources(args):
    """--json 一個解壓好的 json；--games N：從 data/*.7z 依序取 N 場（沒解壓就解壓到同名資料夾）。"""
    if not args.games:
        return [Path(args.json)]
    import py7zr
    data = ROOT / 'nba-movement-data' / 'data'
    out = []
    for z in sorted(data.glob('*.7z'))[args.skip_games:args.skip_games + args.games]:
        d = data / z.stem
        js = list(d.glob('*.json')) if d.exists() else []
        if not js:
            d.mkdir(exist_ok=True)
            with py7zr.SevenZipFile(z, 'r') as a:
                a.extractall(d)
            js = list(d.glob('*.json'))
        out += js[:1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=str(DEFAULT_JSON))
    ap.add_argument('--games', type=int, default=0, help='改從 data/*.7z 取幾場（0 = 只用 --json）')
    ap.add_argument('--skip_games', type=int, default=0, help='--games 從第幾場開始（避開已用過的）')
    ap.add_argument('--n', type=int, default=40, help='挑幾個擋拆做動畫')
    ap.add_argument('--select', choices=['random', 'aggressive'], default='random',
                    help='aggressive：只挑規則估計可能是換防/包夾/延誤的候選，三類平均分配')
    ap.add_argument('--out', default=str(OUT_DIR))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out)

    allc = []                                   # (s, 片段影格, roster, gameid)
    for src in game_sources(args):
        gameid, frames, roster = load_timeline(src)
        n0 = len(allc)
        for seg in segments(frames):
            if len(seg) < (PRE_S + POST_S) * FPS + 10:
                continue
            parsed = [parse_moment(m) for m in seg]
            handlers = find_ball_handler(parsed, None, None)
            for s in find_screens(parsed, handlers):
                f0, f1 = s['cf'] - int(PRE_S * FPS), s['cf'] + int(POST_S * FPS)
                if f0 < 0 or f1 > len(parsed):
                    continue
                pl = parsed[s['cf']]['players']
                hx, hy = pl[s['H']]['pos']
                rim = (88.75, 25.0) if hx >= 47 else (5.25, 25.0)
                if dist((hx, hy), rim) < 16:           # 只要外線擋拆（禁區附近多是低位/無球動作）
                    continue
                defs = [p for p, i in pl.items() if i['team'] != s['team']]
                pre = list(range(s['cf'] - FPS, s['cf']))
                s['Dh'] = nearest_def_mean(parsed, pre, s['H'], defs)
                s['Ds'] = nearest_def_mean(parsed, pre, s['S'], [d for d in defs if d != s['Dh']])
                s['period'], s['clock'] = seg[s['cf']][0], seg[s['cf']][2]
                win = parsed[f0:f1]                    # 只留這段，省記憶體
                s['cf'] -= f0
                s['kind'] = aggressive_kind(win, s)
                allc.append((s, win, roster, gameid))
        kinds = collections.Counter(x[0]['kind'] for x in allc[n0:])
        print(f'比賽 {gameid}：{len(frames) / FPS / 60:.1f} 分鐘，外線擋拆 {len(allc) - n0} 個，'
              f'候選 換防 {kinds["switch"]}、包夾 {kinds["blitz"]}、延誤 {kinds["hedge"]}')

    rng = np.random.RandomState(args.seed)
    if args.select == 'random':
        pick = list(rng.choice(len(allc), min(args.n, len(allc)), replace=False))
    else:
        # 用 coverage_model 的「退防機率」排序，挑最低的（第一批 40 支回測：最低 8 個裡 4 個是積極防守，
        # 隨機約 18%；aggressive_kind 的規則反而只有 8%，所以不用規則挑）
        p_drop = np.array([drop_prob(clip_tracks(x[1], x[0])) for x in allc])
        order = np.argsort(p_drop)
        pick = [int(i) for i in order[:args.n]]
        rng.shuffle(pick)                       # 打散，不讓使用者從順序猜
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, samples = [], []
    for k, i in enumerate(pick, 1):
        s, parsed, roster, gameid = allc[i]
        f0, f1 = 0, len(parsed)
        J = lambda p: roster.get(p, {}).get('jersey', '?')
        name = f"{k:02d}_{gameid}_Q{s['period']}_{int(s['clock'] // 60)}m{int(s['clock'] % 60):02d}s.mp4"
        render(out_dir / name, parsed, s, roster, f0, f1)
        rows.append(f"{k:02d},{name},{s['period']},{s['clock']:.1f},{roster.get(s['H'], {}).get('team')},"
                    f"#{J(s['H'])},#{J(s['S'])},#{J(s['Ds'])},#{J(s['Dh'])},,,")
        c = clip_tracks(parsed, s)
        samples.append({'id': k, 'file': name, 'game': gameid, **c,
                        'Ds': s['Ds'], 'Dh': s['Dh'], 'rule_kind': s['kind'], 'p_drop': drop_prob(c)})
    with open(out_dir / 'label.csv', 'w', encoding='utf-8-sig') as fo:
        fo.write('編號,檔名,節,時間(秒),進攻隊,持球者H,掩護者S,Ds守掩護者,Dh守持球者,'
                 '防守方式(換防/延誤/包夾/退防/其他/沒掩護),有把握(t/f),備註\n')
        fo.write('\n'.join(rows) + '\n')
    pickle.dump(samples, open(out_dir / 'probe_samples.pkl', 'wb'))
    print(f'已輸出 {len(rows)} 支動畫 → {out_dir}')


if __name__ == '__main__':
    main()
