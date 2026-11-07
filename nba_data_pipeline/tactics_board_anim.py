"""
tactics_board_anim.py — 動態戰術板：左邊原影片、右邊俯視半場板（逐格同步）→ mp4（H.264）。

讀 inference.py 的 events.pkl（跟 tactics_board.py 同一套前處理與戰術事實），每一張原片影格畫：
  進攻 = 號碼圓圈（持球者金色、擋拆時掩護者紅色）、防守 = 藍色 × + 號碼，各帶 1 秒漸淡軌跡；
  持球者（ball_acquisition）= 黃色外圈；
  擋拆：撞上前 0.5 秒 ~ 後 1 秒畫掩護短槓 + 字幕（誰幫誰擋、擋到誰 → roll / pop、防守方式）；
  出手：出手那刻起 1.5 秒標出手位置 + 進 / 沒進。
每段鏡頭（cuts）各自決定攻哪個籃框（精華剪輯切點後常是另一球）。

用法：
  python nba_data_pipeline/tactics_board_anim.py output_videos/pipeline_test/screen_358_events.pkl \
      --video input_videos/screen_ext/screen_358.mp4
  → output_videos/board/<名稱>_anim.mp4
"""
import sys
import pickle
import argparse
import collections
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import tactics_tools as T
import tactics_board as B

H_PX = 600                                   # 板子高度（像素）
CAP_PX = 96                                  # 板子上方的字幕列；影片縮成 H_PX + CAP_PX 高
X0, X1 = -2.0, T.COURT_H_FT + 2.0            # 半場板的資料範圍（同 tactics_board.draw_court）
Y0, Y1 = -2.5, B.HALF + 1.5
W_PX = int(round(H_PX * (X1 - X0) / (Y1 - Y0)))
FONT_PATHS = ['C:/Windows/Fonts/msjh.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              str(ROOT / 'fonts' / 'NotoSansCJK-Regular.ttc'),
              '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']
BGR = lambda h: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))
C_OFF, C_HANDLER, C_SCREENER = BGR(B.C_OFF), BGR(B.C_HANDLER), BGR(B.C_SCREENER)
C_DEF, C_DEF_HIT = BGR(B.C_DEF), BGR(B.C_DEF_HIT)


def font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def court_image():
    """matplotlib 畫一次半場（跟靜態板同樣的線）→ BGR 陣列，資料座標 ↔ 像素一對一。"""
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    B.draw_court(ax)
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_aspect('auto')
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
    plt.close(fig)
    return cv2.resize(img, (W_PX, H_PX))


def px(p):
    u, d = p
    return int(round((u - X0) / (X1 - X0) * W_PX)), int(round((d - Y0) / (Y1 - Y0) * H_PX))


def smooth_positions(cc, k=3):
    """每人逐張位置（±k 張移動平均）；缺座標 ≤ 0.5 秒沿用上一個位置，再久就不畫。"""
    n = len(cc)
    pids = {p for fd in cc for p in fd}
    out = [dict() for _ in range(n)]
    for p in pids:
        have = [cc[f].get(p) for f in range(n)]
        for f in range(n):
            win = [have[g] for g in range(max(0, f - k), min(n, f + k + 1)) if have[g] is not None]
            if have[f] is not None and win:
                out[f][p] = tuple(np.mean(win, axis=0))
    return out


def segment_baskets(data, n):
    """每段鏡頭的進攻籃框：該段進攻方平均位置在哪一半。回 list（逐張）。"""
    cuts = [0] + sorted(c for c in (data.get('cuts') or []) if 0 < c < n) + [n]
    out = [None] * n
    for a, b in zip(cuts[:-1], cuts[1:]):
        bk = T.infer_target_basket(data['court_coords'], a, max(a, b - 1))
        for f in range(a, b):
            out[f] = bk
    return out


def segment_teams(pa, n, cuts):
    """每段鏡頭內多數決分隊（單張雜訊）。回 list（逐張）{pid: team}。"""
    cuts = [0] + sorted(c for c in (cuts or []) if 0 < c < n) + [n]
    out = [None] * n
    for a, b in zip(cuts[:-1], cuts[1:]):
        cnt = collections.defaultdict(collections.Counter)
        for f in range(a, b):
            for p, t in (pa[f] if f < len(pa) else {}).items():
                cnt[p][t] += 1
        team = {p: c.most_common(1)[0][0] for p, c in cnt.items()}
        for f in range(a, b):
            out[f] = team
    return out


def build_overlays(data, events, fps, passes=()):
    """擋拆 / 出手 → 逐張要畫的東西：[(f0, f1, dict)]。"""
    ACT = {'roll': '順下', 'pop': '外彈', 'unclear': ''}
    COV = {'switch': '換防', 'blitz': '包夾', 'drop': '退防/延誤', 'hedge': '延誤', 'unclear': ''}
    ov = []
    for i, (g, facts) in enumerate(events, 1):
        H = g[0]['handler_id']
        for s in facts['screens']:
            sf = s['screen_frame']
            d = f"（擋 #{s['screened_defender']}）" if s['screened_defender'] is not None else ''
            tail = '　→　'.join(x for x in (ACT.get(s['action'], ''), ('防守：' + COV[s['coverage']]) if COV.get(s['coverage']) else '') if x)
            ov.append((sf - int(0.5 * fps), sf + int(1.0 * fps),
                       dict(kind='screen', sf=sf, S=s['screener_id'], D=s['screened_defender'], H=H,
                            text=f"擋拆 {i}：#{s['screener_id']} 幫 #{H} 掩護{d}", text2=tail)))
    for p in passes or []:                          # 傳球 / 手遞手 / 抄截（possession_sam3.detect_passes）
        word = {'pass': '傳球', 'handoff': '手遞手', 'turnover': '抄截 / 換隊'}[p['kind']]
        ov.append((p['release'], p['catch'] + int(0.6 * fps),
                   dict(kind='pass', A=p['frm'], B=p['to'], rel=p['release'], cat=p['catch'],
                        text=f"#{p['frm']} → #{p['to']} {word}（空中 {p['air_s']} 秒）")))
    MADE = {True: '進', False: '沒進', None: '進沒進看不出來'}
    for sh in data.get('shots') or []:
        rf = sh['release_frame']
        kind = {'jump-shot': '跳投', 'layup': '上籃'}.get(sh.get('kind'), '出手')
        ov.append((rf, rf + int(1.5 * fps), dict(kind='shot', rf=rf, P=sh.get('shooter'), made=sh.get('made'),
                                                  text=f"#{sh.get('shooter')} {kind}，{MADE[sh.get('made')]}")))
    return ov


def render(data, frames, out_path, fps, tracks=None, passes=()):
    n = min(len(frames), len(data['court_coords']))
    tracks = tracks if tracks is not None else data.get('player_tracks')
    pos = smooth_positions(data['court_coords'][:n])
    baskets = segment_baskets(data, n)
    teams = segment_teams(data['player_assignment'], n, data.get('cuts'))
    ba = data.get('ball_acquisition') or [None] * n
    events = B.build(data)
    ov = build_overlays(data, events, fps, passes)
    court = court_image()
    f_big, f_small, f_tiny = font(26), font(20), font(15)
    vid_w = int(round(frames[0].shape[1] * (H_PX + CAP_PX) / frames[0].shape[0]))
    trail = int(1.0 * fps)
    out = []
    for f in range(n):
        board = court.copy()
        bk = baskets[f]
        team = teams[f]
        H = ba[f] if f < len(ba) and ba[f] not in (None, -1) else None
        th = 1        # 推論 / team_fix 的分隊定義：1 = 跟持球者同隊 = 進攻（沒持球者的影格不能拿「票多的那隊」當進攻，會整個顛倒）
        act = [o for a, b, o in ov if a <= f <= b]
        scr = {o['S'] for o in act if o['kind'] == 'screen'}
        hitd = {o['D'] for o in act if o['kind'] == 'screen' and o['D'] is not None}
        # 軌跡（漸淡）
        for p in pos[f]:
            pts = [B.to_board(pos[g][p], bk) for g in range(max(0, f - trail), f + 1) if p in pos[g] and baskets[g] == bk]
            if len(pts) < 2:
                continue
            off = team.get(p) == th
            col = C_HANDLER if p == H else C_SCREENER if p in scr else C_OFF if off else C_DEF
            for k in range(1, len(pts)):
                a = k / len(pts)
                c = tuple(int(255 - (255 - x) * a * 0.8) for x in col)       # 越舊越淡
                cv2.line(board, px(pts[k - 1]), px(pts[k]), c, 2, cv2.LINE_AA)
        # 掩護短槓
        for o in act:
            if o['kind'] == 'screen' and o['S'] in pos[min(o['sf'], n - 1)]:
                sp = np.array(B.to_board(pos[min(o['sf'], n - 1)][o['S']], bk))
                tgt = pos[min(o['sf'], n - 1)].get(o['D']) if o['D'] is not None else pos[min(o['sf'], n - 1)].get(o['H'])
                v = (np.array(B.to_board(tgt, bk)) - sp) if tgt is not None else np.array([1.0, 0.0])
                v = v / max(np.hypot(*v), 1e-6)
                nrm = np.array([-v[1], v[0]]) * 1.8
                cv2.line(board, px(sp - nrm), px(sp + nrm), C_SCREENER, 7, cv2.LINE_AA)
            if o['kind'] == 'pass':                         # 傳球虛線箭頭：A 離手位置 → B 接球位置
                pa_, pb_ = pos[min(o['rel'], n - 1)].get(o['A']), pos[min(o['cat'], n - 1)].get(o['B'])
                if pa_ is not None and pb_ is not None:
                    a_, b_ = np.array(px(B.to_board(pa_, bk)), float), np.array(px(B.to_board(pb_, bk)), float)
                    L = max(np.hypot(*(b_ - a_)), 1.0)
                    for t0 in np.arange(0, L - 14, 16):         # 虛線
                        p1, p2 = a_ + (b_ - a_) * t0 / L, a_ + (b_ - a_) * min(t0 + 9, L - 14) / L
                        cv2.line(board, tuple(int(v) for v in p1), tuple(int(v) for v in p2), (40, 40, 40), 3, cv2.LINE_AA)
                    cv2.arrowedLine(board, tuple(int(v) for v in b_ - (b_ - a_) * 22 / L), tuple(int(v) for v in b_ - (b_ - a_) * 8 / L),
                                    (40, 40, 40), 3, cv2.LINE_AA, tipLength=0.9)
            if o['kind'] == 'shot' and o['P'] in pos[min(o['rf'], n - 1)]:
                q = px(B.to_board(pos[min(o['rf'], n - 1)][o['P']], bk))
                if o['made'] is True:
                    cv2.circle(board, q, 15, (87, 139, 46), -1, cv2.LINE_AA)
                elif o['made'] is False:
                    cv2.line(board, (q[0] - 13, q[1] - 13), (q[0] + 13, q[1] + 13), (34, 34, 178), 5, cv2.LINE_AA)
                    cv2.line(board, (q[0] - 13, q[1] + 13), (q[0] + 13, q[1] - 13), (34, 34, 178), 5, cv2.LINE_AA)
                else:
                    cv2.circle(board, q, 15, (119, 119, 119), 3, cv2.LINE_AA)
        # 球員
        labels = []
        for p, xy in pos[f].items():
            q = px(B.to_board(xy, bk))
            if not (0 <= q[0] < W_PX and 0 <= q[1] < H_PX):
                continue
            if team.get(p) == th:
                col = C_HANDLER if p == H else C_SCREENER if p in scr else C_OFF
                cv2.circle(board, q, 15, col, -1, cv2.LINE_AA)
                cv2.circle(board, q, 15, (255, 255, 255), 2, cv2.LINE_AA)
                if p == H:
                    cv2.circle(board, q, 21, (0, 215, 255), 3, cv2.LINE_AA)
                labels.append((q, str(p), (255, 255, 255), True))
            elif team.get(p) is not None:
                col = C_DEF_HIT if p in hitd else C_DEF
                cv2.line(board, (q[0] - 10, q[1] - 10), (q[0] + 10, q[1] + 10), col, 4, cv2.LINE_AA)
                cv2.line(board, (q[0] - 10, q[1] + 10), (q[0] + 10, q[1] - 10), col, 4, cv2.LINE_AA)
                labels.append(((q[0] + 13, q[1] - 22), f'#{p}', col[::-1], False))
        # 文字（PIL 才能寫中文）
        im = Image.fromarray(board[:, :, ::-1])
        dr = ImageDraw.Draw(im)
        for q, t, c, center in labels:
            if center:
                w = dr.textlength(t, font=f_tiny)
                dr.text((q[0] - w / 2, q[1] - 10), t, font=f_tiny, fill=c)
            else:
                dr.text(q, t, font=f_tiny, fill=tuple(c))
        dr.text((10, H_PX - 34), f'{f / fps:5.1f} 秒', font=f_small, fill=(60, 60, 60))
        board = np.asarray(im)[:, :, ::-1]
        cap = Image.new('RGB', (W_PX, CAP_PX), (255, 255, 255))     # 上方字幕列（不蓋住球場）
        dc = ImageDraw.Draw(cap)
        y = 6
        for o in sorted(act, key=lambda o: o['kind'])[:2]:
            dc.text((10, y), o['text'], font=f_big if o['kind'] == 'screen' else f_small,
                    fill=(201, 48, 44) if o['kind'] == 'screen' else (40, 40, 40))
            if o.get('text2'):
                dc.text((10, y + 32), o['text2'], font=f_small, fill=(70, 70, 70))
            y += 62 if o.get('text2') else 34
        board = np.vstack([np.asarray(cap)[:, :, ::-1], board])
        vid = frames[f]
        if tracks is not None and f < len(tracks):        # 原片畫框 + 編號（顏色同板子：持球金、掩護紅、進攻紅、防守藍、沒分隊灰）
            vid = vid.copy()
            lw = max(2, vid.shape[0] // 360)
            for p, bb in tracks[f].items():
                if team.get(p) == th:
                    col = C_HANDLER if p == H else C_SCREENER if p in scr else C_OFF
                elif team.get(p) is not None:
                    col = C_DEF_HIT if p in hitd else C_DEF
                else:
                    col = (150, 150, 150)
                x1, y1, x2, y2 = [int(v) for v in bb]
                cv2.rectangle(vid, (x1, y1), (x2, y2), col, lw + (1 if p in (H,) or p in scr else 0))
                t = f'#{p}'
                (tw, tht), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * lw, lw)
                cv2.rectangle(vid, (x1, y1 - tht - 6), (x1 + tw + 4, y1), col, -1)
                cv2.putText(vid, t, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * lw, (255, 255, 255), lw, cv2.LINE_AA)
        vid = cv2.resize(vid, (vid_w, H_PX + CAP_PX))
        out.append(np.hstack([vid, board]))
    from utils import save_video
    save_video(out, str(out_path), fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl')
    ap.add_argument('--video', required=True)
    ap.add_argument('--out_dir', default=str(ROOT / 'output_videos' / 'board'))
    ap.add_argument('--tracks', help='舊 events.pkl 沒存球員框時：run_sam3_tracks 的 pkl（取 player_tracks）')
    ap.add_argument('--no_vlm', action='store_true', help='不呼叫 VLM 判進球（pod 上沒金鑰時）')
    ap.add_argument('--ball', help='10fps 追球（run_sam3_tracks --ball 的 pkl 或 shot_sam3 pkl）→ 持球者改用 SAM3 追球 + Viterbi、畫傳球')
    ap.add_argument('--touch', help='對應的碰球快取（possession_sam3.prep / stubs/shot_sam3/*_touch.pkl）')
    args = ap.parse_args()
    data = pickle.load(open(args.pkl, 'rb'))
    if not args.no_vlm:
        try:
            from shot_detect import apply_vlm_made
            apply_vlm_made(data, args.pkl)
        except Exception as e:
            print(f'  [VLM 進球判斷略過] {e}')
    tracks = None
    if args.tracks:
        tr = pickle.load(open(args.tracks, 'rb'))['player_tracks']
        tracks = [{p: (v.get('bbox') if isinstance(v, dict) else v) for p, v in fd.items()} for fd in tr]
    from utils import read_video
    frames = read_video(args.video)
    fps = data.get('fps', 30.0)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{Path(args.pkl).stem.replace('_events', '')}_anim.mp4"
    passes = ()
    if args.ball and args.touch:
        import possession_sam3 as PS
        bs = pickle.load(open(args.ball, 'rb'))
        bs = bs.get('shot_sam3', bs)
        _, _, tch = pickle.load(open(args.touch, 'rb'))
        path = PS.possession_path(data, bs, tch)
        data['ball_acquisition'] = path                       # 持球者外圈也改用新的
        passes = PS.detect_passes(path, fps, PS.clip_teams(data['player_assignment']), data.get('shots'))
        print(f"  SAM3 追球 + Viterbi：傳球 {sum(p['kind'] == 'pass' for p in passes)}、手遞手 {sum(p['kind'] == 'handoff' for p in passes)}、"
              f"抄截 {sum(p['kind'] == 'turnover' for p in passes)}")
    elif data.get('passes') is not None:                     # run_pipeline 第 3 步寫進 events.pkl 的手腕版傳球
        passes = data['passes']
        print(f"  手腕傳球偵測：傳球 {sum(p['kind'] == 'pass' for p in passes)}、抄截 {sum(p['kind'] == 'turnover' for p in passes)}")
    render(data, frames, p, fps, tracks, passes)
    print(f'→ {p}')


if __name__ == '__main__':
    main()
