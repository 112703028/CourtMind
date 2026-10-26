"""
shot_events.py — 讀 shot_detect_rf.py 的逐幀偵測，判斷「有沒有出手、什麼時候、跳投/上籃、進沒進」。
規則（改這裡不用重跑模型）：
  1. 切鏡頭：相鄰影格色彩直方圖差很大 → 之後的畫面（重播 / 換場）不看
  2. 出手候選：player-jump-shot / player-layup-dunk 信心 ≥ SHOT_CONF；1 秒內的合併成一個事件
  3. 軌跡確認（排除跳起來傳球）：出手後 BALL_WIN 秒內，球要出現在籃框附近（籃框框往外擴、往上延伸）
  4. 進球：出手後 MADE_WIN 秒內有 ball-in-basket，且
       - 連續長度 ≤ MAX_IN_RUN 秒（籃網常被持續誤判成進框）
       - 進框前 0.5 秒內看得到球在籃框附近
不判斷「誰出手」：延長的那段沒有 SAM3 追蹤，先用出手框畫在影片上給人看。

用法：
  python nba_data_pipeline/shot_events.py --dets output_videos/shot_check/rf_dets.pkl \
      --video_dir input_videos/screen_ext --out_dir output_videos/shot_check [--no_video]
輸出：<out_dir>/shot_events_model.csv（含人工核對欄；人工標好的請另存，例 shot_labels_round1.csv）、<out_dir>/<clip>.mp4（標註影片）
"""
import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
BALL, IN_BASKET, PLAYER, POSSESS, JUMP, LAYUP, BLOCK, RIM = 0, 1, 3, 4, 5, 6, 7, 9
NAMES = {BALL: 'ball', IN_BASKET: 'in-basket', JUMP: 'jump-shot', LAYUP: 'layup', BLOCK: 'block', RIM: 'rim'}
COLORS = {BALL: (0, 255, 255), IN_BASKET: (0, 255, 0), JUMP: (0, 0, 255), LAYUP: (255, 0, 255),
          BLOCK: (255, 128, 0), RIM: (255, 255, 255)}

SHOT_CONF = 0.3          # 0.4 會漏掉 334 / 400 這種 0.30–0.34 的真出手
MERGE_S = 1.0
BALL_WIN = 1.5
MADE_WIN = 2.0
MAX_IN_RUN = 0.7
CUT_THRESH = 0.5          # 直方圖相關係數低於此 = 切鏡頭
PIX_CUT = 45              # 64x36 灰階平均像素差；正常鏡頭移動 99% < 36，閃白壞影格 30–38 且馬上回來


def read_frames(path):
    cap = cv2.VideoCapture(str(path)); fs = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fs.append(f)
    return fs


def find_cuts(frames):
    """切鏡頭 = 相鄰兩張差很大，且 3 張後也沒回到原畫面（單張閃白的壞影格會馬上回來，不算）。"""
    hs = []
    for f in frames:
        hsv = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]); cv2.normalize(h, h); hs.append(h)
    cor = lambda a, b: cv2.compareHist(hs[a], hs[b], cv2.HISTCMP_CORREL)
    gs = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32) for f in frames]
    pd = lambda a, b: float(np.abs(gs[a] - gs[b]).mean())
    n = len(hs)
    # 顏色分布大變，或整張像素差大（同球場的精華剪輯接點顏色很像，只看直方圖抓不到；screen_453 5.07s 像素差 47）
    jump = lambda i: (cor(i - 1, i) < CUT_THRESH and cor(max(i - 2, 0), min(i + 3, n - 1)) < CUT_THRESH) or                      (pd(i - 1, i) > PIX_CUT and pd(max(i - 2, 0), min(i + 3, n - 1)) > PIX_CUT)
    cuts = [i for i in range(1, n) if jump(i)]
    return [c for k, c in enumerate(cuts) if k == 0 or c - cuts[k - 1] > 3]    # 轉場連續幾張只算一次


def contact_frame(clip, fps):
    p = ROOT / 'stubs' / 'train_contact_cache' / f'{clip}.pkl'
    if not p.exists(): return 0
    c = pickle.load(open(p, 'rb'))
    idx = [e['idx'] for e in c.get('events', []) if e.get('idx') is not None]
    return int(round(min(idx) * fps / 5)) if idx else 0


def holds_ball(box, balls):
    """球在這個人框的上半部附近 = 球在他手上（封蓋的防守者也會被判成跳投，但手上沒球）。"""
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    for b in balls:
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if x1 - 0.3 * w <= bx <= x2 + 0.3 * w and y1 - 0.3 * h <= by <= y1 + 0.5 * h:
            return True
    return False


def near_rim(ball_box, rims):
    bx, by = (ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2
    for r in rims:
        w = r[2] - r[0]
        if r[0] - 1.0 * w <= bx <= r[2] + 1.0 * w and r[1] - 4.0 * w <= by <= r[3] + 0.5 * w:
            return True
    return False


def analyze(clip, d, cuts, rng=None):
    """rng=(起, 迄) 原片影格：直接指定分析範圍（推論用）；None = 包含訓練快取「擋拆接觸」那段連續鏡頭。"""
    fps = d['fps']
    if rng is None:
        cf = contact_frame(clip, fps)                                    # 只看「包含擋拆接觸」的那段連續鏡頭
        start_f = max([0] + [c for c in cuts if c <= cf])
        end_f = next((c for c in cuts if c > cf), d['n'])
    else:
        start_f, end_f = rng
    fr = [(f, dets) for f, dets in d['frames'] if start_f <= f < end_f]
    t = lambda f: f / fps
    by_cls = lambda dets, c, th=0.4: [x for x in dets if x[0] == c and x[1] >= th]

    cands = []
    for f, dets in fr:
        s = by_cls(dets, JUMP, SHOT_CONF) + by_cls(dets, LAYUP, SHOT_CONF)
        if s:
            balls = [b[2] for b in by_cls(dets, BALL, 0.3)]
            held = [x for x in s if holds_ball(x[2], balls)]
            best = max(held or s, key=lambda x: x[1])              # 優先選手上有球的人
            best = (*best, bool(held))
            if cands and t(f) - t(cands[-1]['last']) <= MERGE_S:
                c = cands[-1]; c['last'] = f; c['hits'].append((f, best))
            else:
                cands.append(dict(first=f, last=f, hits=[(f, best)]))

    ball_near = [f for f, dets in fr
                 if any(near_rim(b[2], [r[2] for r in by_cls(dets, RIM)]) for b in by_cls(dets, BALL, 0.3))]
    in_frames = [f for f, dets in fr if by_cls(dets, IN_BASKET)]
    runs = []                                                            # 進框連續段
    for f in in_frames:
        if runs and f - runs[-1][1] <= 2 * d['stride']: runs[-1][1] = f
        else: runs.append([f, f])

    events = []
    for c in cands:
        kinds = [x[0] for _, x in c['hits']]
        kind = 'layup' if kinds.count(LAYUP) >= kinds.count(JUMP) else 'jump-shot'
        f0 = c['first']
        traj = any(f0 <= f <= f0 + BALL_WIN * fps for f in ball_near)
        made_run = next((r for r in runs if f0 <= r[0] <= f0 + MADE_WIN * fps
                         and (r[1] - r[0]) / fps <= MAX_IN_RUN
                         and any(r[0] - 0.5 * fps <= f <= r[1] for f in ball_near)), None)
        events.append(dict(clip=clip, t=round(t(f0), 2), frame=f0, kind=kind,
                           conf=round(max(x[1] for _, x in c['hits']), 2), hits=len(c['hits']),
                           score=round(sum(x[1] for _, x in c['hits']), 2),
                           with_ball=any(x[3] for _, x in c['hits']),
                           shooter={f: [int(v) for v in x[2]] for f, x in c['hits']},
                           traj=traj, made=made_run is not None,
                           made_t=round(t(made_run[0]), 2) if made_run else None,
                           box=[int(v) for v in c['hits'][0][1][2]]))
    long_in = [round((r[1] - r[0]) / fps, 1) for r in runs if (r[1] - r[0]) / fps > MAX_IN_RUN]
    return events, (start_f, end_f), long_in


def draw(frames, d, events, rng, out):
    fps = d['fps']; h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    dets_at = dict(d['frames']); last = []
    for i, f in enumerate(frames):
        im = f.copy()
        if i in dets_at: last = dets_at[i]
        for c, s, b in last:
            if c in NAMES and s >= (0.3 if c == BALL else 0.4):
                b = [int(v) for v in b]
                cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), COLORS[c], 2)
                if c not in (BALL, RIM): cv2.putText(im, f'{NAMES[c]} {s:.2f}', (b[0], b[1] - 4), 0, 0.5, COLORS[c], 2)
        cv2.rectangle(im, (0, 0), (w, 34), (0, 0, 0), -1)
        txt = f'{out.stem}  f{i}  {i / fps:.2f}s'
        if not rng[0] <= i < rng[1]: txt += '  [other camera - not analyzed]'
        for e in events:
            if e['frame'] <= i <= e['frame'] + 2 * fps:
                txt += f"  SHOT {e['kind']} @{e['t']}s ball={'Y' if e['with_ball'] else 'N'} {'MADE' if e['made'] else 'miss?'}"
        cv2.putText(im, txt, (8, 24), 0, 0.65, (0, 255, 255), 2)
        vw.write(im)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', default='output_videos/shot_check/rf_dets.pkl')
    ap.add_argument('--video_dir', default='input_videos/screen_ext')
    ap.add_argument('--out_dir', default='output_videos/shot_check')
    ap.add_argument('--no_video', action='store_true')
    args = ap.parse_args()
    res = pickle.load(open(ROOT / args.dets, 'rb'))
    out_dir = ROOT / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in sorted(res):
        d = res[clip]
        frames = read_frames(ROOT / args.video_dir / f'{clip}.mp4')
        cuts = find_cuts(frames)
        events, (start_f, end_f), long_in = analyze(clip, d, cuts)
        ok = events                                                   # 軌跡只當參考（RF 追不到飛行中的球，會刪掉真投籃）
        print(f"{clip}: 分析 {start_f / d['fps']:.1f}–{end_f / d['fps']:.1f}s（切鏡頭 {[round(c / d['fps'], 1) for c in cuts]}）"
              f"{' 長進框誤報 ' + str(long_in) + 's' if long_in else ''} | "
              + ('; '.join(f"{e['kind']}@{e['t']}s conf{e['conf']}×{e['hits']} 軌跡{'✓' if e['traj'] else '✗'} {'進' if e['made'] else '沒進/沒看到'}"
                           for e in events) or '沒有出手'))
        if not args.no_video:
            draw(frames, d, events, (start_f, end_f), out_dir / f'{clip}.mp4')
        final = max(ok, key=lambda e: (e['with_ball'], e['score'])) if ok else None
        rows.append({'clip': clip, '模型：出手': 'Y' if final else 'N',
                     '模型：時間(s)': final['t'] if final else '', '模型：類型': final['kind'] if final else '',
                     '模型：進球': ('Y' if final['made'] else 'N') if final else '',
                     '被軌跡排除的候選': '; '.join(f"{e['kind']}@{e['t']}" for e in events if not e['traj']),
                     '分析範圍(s)': f"{start_f / d['fps']:.1f}-{end_f / d['fps']:.1f}",
                     '人工：有出手(Y/N)': '', '人工：時間對(Y/N)': '', '人工：進球(Y/N)': '', '備註': ''})
    out_csv = out_dir / 'shot_events_model.csv'                      # 不寫回人工標記檔（shot_labels_round1.csv）
    with open(out_csv, 'w', encoding='utf-8-sig', newline='') as fo:
        wr = csv.DictWriter(fo, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f'→ {out_csv}')


if __name__ == '__main__':
    main()
