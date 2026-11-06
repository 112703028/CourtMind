"""
tactics_board.py — 擋拆戰術板：把 events.pkl 的球場座標 + 戰術分析事實畫成教練式半場圖（每個擋拆事件一張）。

讀 inference.py dump 的 events.pkl，跟 tactics_tools.py main 同一套前處理（鏡像修正、ID 合併、擋拆實例 → 事件分組、
analyze_screen_instance 算 roll/pop、防守方式、出手結果）。不需要原影片（--video 給了會在右下附撞上那張畫面）。

圖例（教練慣例）：
  進攻 = 有號碼的圓圈（持球者金色、掩護者紅色）；防守 = ×（被擋的防守者深藍）
  運球 = 鋸齒箭頭；跑位 = 實線箭頭；掩護 = 線尾端一條垂直短槓；掩護後 roll / pop = 實線箭頭；傳球 = 虛線箭頭
  出手 = ●（進）/ ✕（沒進）/ ◯（看不出來）
半場：籃框在上（從中場看向籃框，左邊就是進攻方的左手邊）。時間窗：撞上前 2 秒 ~ 撞上後 2.5 秒（或出手、鏡頭切走為止）。

用法：
  python nba_data_pipeline/tactics_board.py output_videos/pipeline_test/screen_358_events.pkl [--video input_videos/screen_ext/screen_358.mp4]
  → output_videos/board/<名稱>_event<i>.png
"""
import sys
import math
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import tactics_tools as T

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans CJK TC', 'Noto Sans TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 球場尺寸：pipeline 的球場是 91.8 × 49.2 ft（NBA 94 × 50 等比縮），半場線條照比例縮
SX, SY = T.COURT_W_FT / 94.0, T.COURT_H_FT / 50.0
HALF = T.COURT_W_FT / 2
RIM_D = T.BASKET_INSET                  # 籃框離底線
C_OFF, C_HANDLER, C_SCREENER = '#d9534f', '#f0ad4e', '#c9302c'
C_DEF, C_DEF_HIT = '#6f8fb5', '#1f3f75'
C_LINE, C_BG = '#5a4a3a', '#f4ead5'
ACTION_ZH = {'roll': '順下（roll）', 'pop': '外彈（pop）', 'unclear': '不明顯'}
COV_ZH = {'switch': '換防', 'blitz': '包夾', 'drop': '退防 / 延誤', 'hedge': '延誤', 'unclear': '不明確'}


# ── 座標：球場 → 半場圖（籃框在上）────────────────────────────────────────────

def to_board(p, basket):
    """球場座標 (x 沿長邊, y 沿寬邊) → 半場圖 (u 橫向、d 離底線距離)。面向籃框時的左手邊畫在左邊。"""
    x, y = p
    if basket[0] > T.COURT_W_FT / 2:          # 攻右邊籃框：面向 +x，左手邊是 +y
        return T.COURT_H_FT - y, T.COURT_W_FT - x
    return y, x                                # 攻左邊籃框：面向 −x，左手邊是 −y


def draw_court(ax):
    W = T.COURT_H_FT
    lw = 1.6
    ax.add_patch(Rectangle((0, 0), W, HALF, fill=True, fc=C_BG, ec=C_LINE, lw=lw * 1.3, zorder=0))
    cx = W / 2
    lane_w, ft_d = 16 * SY, 19 * SX
    ax.add_patch(Rectangle((cx - lane_w / 2, 0), lane_w, ft_d, fill=True, fc='#ead9b8', ec=C_LINE, lw=lw, zorder=0.5))
    ax.add_patch(Circle((cx, ft_d), 6 * SY, fill=False, ec=C_LINE, lw=lw, zorder=0.6))
    ax.add_patch(Arc((cx, RIM_D), 8 * SY, 8 * SY, theta1=0, theta2=180, ec=C_LINE, lw=lw, zorder=0.6))   # 禁區半圓
    ax.add_patch(Circle((cx, RIM_D), 0.75, fill=False, ec='#e8702a', lw=2.2, zorder=0.7))              # 籃框
    ax.plot([cx - 3 * SY, cx + 3 * SY], [RIM_D - 1.0, RIM_D - 1.0], color=C_LINE, lw=2.5, zorder=0.7)  # 籃板
    r3 = 23.75 * SX
    corner = 22 * SY
    th = math.degrees(math.acos(min(1.0, corner / r3)))
    ax.add_patch(Arc((cx, RIM_D), 2 * r3, 2 * r3, theta1=th, theta2=180 - th, ec=C_LINE, lw=lw, zorder=0.6))
    cd = RIM_D + r3 * math.sin(math.radians(th))
    for s in (-1, 1):
        ax.plot([cx + s * corner] * 2, [0, cd], color=C_LINE, lw=lw, zorder=0.6)
    ax.add_patch(Arc((cx, HALF), 12 * SY, 12 * SY, theta1=180, theta2=360, ec=C_LINE, lw=lw, zorder=0.6))   # 中圈
    ax.set_xlim(-2, W + 2)
    ax.set_ylim(HALF + 1.5, -2.5)              # 籃框在上
    ax.set_aspect('equal')
    ax.axis('off')


# ── 軌跡 ─────────────────────────────────────────────────────────────────────

def track(cc, pid, f0, f1, basket, smooth=5):
    """pid 在 [f0, f1] 的半場圖軌跡（移動平均平滑）；缺座標超過 0.5 秒就斷開 → list of (N,2) 陣列。"""
    pts = [(f, to_board(cc[f][pid], basket)) for f in range(f0, f1 + 1) if pid in cc[f]]
    if not pts:
        return []
    segs, cur = [], [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if b[0] - a[0] > smooth * 3:
            segs.append(cur); cur = []
        cur.append(b)
    segs.append(cur)
    out = []
    for s in segs:
        arr = np.array([p for _, p in s], float)
        if len(arr) >= smooth:
            k = np.ones(smooth) / smooth
            arr = np.column_stack([np.convolve(np.pad(arr[:, i], smooth // 2, mode='edge'), k, mode='valid')[:len(arr)]
                                   for i in range(2)])
        out.append(arr)
    return out


def resample(arr, step=0.5):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(arr, axis=0).T))]
    if d[-1] < step:
        return arr, d
    s = np.arange(0, d[-1], step)
    return np.column_stack([np.interp(s, d, arr[:, 0]), np.interp(s, d, arr[:, 1])]), s


def arrow_end(ax, arr, color, lw, ls='-', zorder=3):
    if len(arr) < 2:
        return
    a, b = arr[max(0, len(arr) - 4)], arr[-1]
    if np.hypot(*(b - a)) < 1e-6:
        return
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls, mutation_scale=16),
                zorder=zorder)


def draw_path(ax, arr, color, lw=2.0, style='solid', arrow=True, zorder=3):
    """style：solid（跑位）/ dashed（傳球）/ zigzag（運球）。"""
    if len(arr) < 2 or np.hypot(*(arr[-1] - arr[0])) < 1.5:
        return
    if style == 'zigzag':
        r, s = resample(arr, 0.35)
        if len(r) > 3:
            tang = np.gradient(r, axis=0)
            nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
            amp = 0.55 * np.sin(2 * np.pi * s / 1.6)
            amp[-4:] = 0                                   # 箭頭前拉直
            r = r + nrm * amp[:, None]
        ax.plot(r[:, 0], r[:, 1], color=color, lw=lw, zorder=zorder, solid_capstyle='round')
        arr = r
    else:
        ax.plot(arr[:, 0], arr[:, 1], color=color, lw=lw, ls='--' if style == 'dashed' else '-', zorder=zorder)
    if arrow:
        arrow_end(ax, arr, color, lw, '--' if style == 'dashed' else '-', zorder)


def screen_bar(ax, pos, toward, color, length=3.2, zorder=4):
    """掩護符號：在 pos 畫一條垂直於「掩護者 → 被擋防守者」方向的短槓。"""
    v = np.array(toward, float) - np.array(pos, float)
    if np.hypot(*v) < 1e-6:
        v = np.array([1.0, 0.0])
    v /= np.hypot(*v)
    n = np.array([-v[1], v[0]]) * length / 2
    p = np.array(pos, float)
    ax.plot([p[0] - n[0], p[0] + n[0]], [p[1] - n[1], p[1] + n[1]], color=color, lw=4.5, zorder=zorder, solid_capstyle='butt')


def player_dot(ax, pos, label, color, zorder=6, r=1.25):
    ax.add_patch(Circle(pos, r, fc=color, ec='white', lw=1.5, zorder=zorder))
    ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=8.5, color='white', weight='bold', zorder=zorder + 1)


def defender_mark(ax, pos, label, color, zorder=5):
    ax.plot(*pos, marker='x', ms=11, mew=3, color=color, zorder=zorder)
    ax.text(pos[0] + 1.0, pos[1] - 1.0, label, fontsize=7.5, color=color, zorder=zorder)


def pos_at(cc, pid, f, basket, search=10):
    for d in range(search + 1):
        for g in (f - d, f + d):
            if 0 <= g < len(cc) and pid in cc[g]:
                return np.array(to_board(cc[g][pid], basket))
    return None


# ── 一個擋拆事件 → 一張圖 ─────────────────────────────────────────────────────

def render_event(data, idx, group, facts, out_path, frame_img=None):
    cc, pa, fps = data['court_coords'], data['player_assignment'], data.get('fps', 30.0)
    n = len(cc)
    sf0 = min(it['frame'] for it in group)
    sf1 = max(it['frame'] for it in group)
    H = group[0]['handler_id']
    basket = facts['basket']
    cuts = data.get('cuts') or []
    seg_start = max([0] + [c for c in cuts if c <= sf0])
    seg_end = min([n - 1] + [c - 1 for c in cuts if c > sf0])
    f0 = max(seg_start, sf0 - int(2.0 * fps))
    f1 = min(seg_end, sf1 + int(2.5 * fps))
    oc = facts['outcome']
    shot = None
    if oc and oc.get('shot'):
        shot = next((s for s in data.get('shots') or [] if s.get('shooter') == oc.get('shooter_id')
                     and abs(s['release_frame'] - (sf1 + oc['seconds_after_screen'] * fps)) < fps), None)
        if shot:
            f1 = min(f1, max(sf1, shot['release_frame']) + int(0.3 * fps))

    from collections import Counter

    def team_of(p):                                     # 時間窗內多數決（單張分隊有雜訊）
        c = Counter(pa[f].get(p) for f in range(f0, f1 + 1) if f < len(pa) and pa[f].get(p) is not None)
        return c.most_common(1)[0][0] if c else None
    team_h = team_of(H)
    ids = {p for f in range(f0, f1 + 1) for p in cc[f]}
    hit = {s['screened_defender'] for s in facts['screens'] if s['screened_defender'] is not None}
    off = [p for p in ids if team_of(p) == team_h and p not in hit]
    dfn = [p for p in ids if p not in off and (team_of(p) is not None or p in hit)]
    screeners = [s['screener_id'] for s in facts['screens']]
    hit_def = {s['screened_defender'] for s in facts['screens'] if s['screened_defender'] is not None}

    fig = plt.figure(figsize=(13, 8.2), dpi=130)
    ax = fig.add_axes([0.01, 0.05, 0.52, 0.92])
    draw_court(ax)

    # 防守者：淡色軌跡 + 終點 ×（被擋的深藍）
    for p in dfn:
        col = C_DEF_HIT if p in hit_def else C_DEF
        for arr in track(cc, p, f0, f1, basket):
            ax.plot(arr[:, 0], arr[:, 1], color=col, lw=1.2 if p not in hit_def else 1.8, alpha=0.55, zorder=2)
        q = pos_at(cc, p, f1, basket)
        if q is not None:
            defender_mark(ax, q, f'#{p}', col)

    # 傳球：持球者換人 = 虛線（從舊持球者到新持球者，換手那張的位置）
    ba = data.get('ball_acquisition') or []
    holders = [(f, ba[f]) for f in range(f0, min(f1 + 1, len(ba))) if ba[f] not in (None, -1)]
    passes = []
    for (fa, a), (fb, b) in zip(holders, holders[1:]):
        if a != b and a in off and b in off:
            passes.append((fa, a, fb, b))

    # 進攻：持球者鋸齒、掩護者到掩護點 + 短槓 + 掩護後箭頭、其他人實線
    for p in off:
        tr = track(cc, p, f0, f1, basket)
        if not tr:
            continue
        if p == H:
            for arr in tr:
                draw_path(ax, arr, C_HANDLER, lw=2.4, style='zigzag', zorder=4)
        elif p in screeners:
            s = next(s for s in facts['screens'] if s['screener_id'] == p)
            sf = s['screen_frame']
            pre = track(cc, p, f0, sf, basket)
            post = track(cc, p, sf, f1, basket)
            for arr in pre:
                draw_path(ax, arr, C_SCREENER, lw=2.4, arrow=False, zorder=4)
            spot = pos_at(cc, p, sf, basket)
            dpos = pos_at(cc, s['screened_defender'], sf, basket) if s['screened_defender'] is not None else None
            if spot is not None:
                screen_bar(ax, spot, dpos if dpos is not None else (pos_at(cc, H, sf, basket) if pos_at(cc, H, sf, basket) is not None else spot + 1),
                           C_SCREENER)
            for arr in post:
                draw_path(ax, arr, C_SCREENER, lw=2.0, zorder=4)
        else:
            for arr in tr:
                draw_path(ax, arr, C_OFF, lw=1.5, zorder=3)
    for fa, a, fb, b in passes:
        pa_, pb_ = pos_at(cc, a, fa, basket), pos_at(cc, b, fb, basket)
        if pa_ is not None and pb_ is not None:
            draw_path(ax, np.array([pa_, pb_]), '#333333', lw=1.6, style='dashed', zorder=5)

    # 起點位置的號碼圓圈（看得出誰從哪裡開始）+ 終點淡圈
    for p in off:
        q0 = pos_at(cc, p, f0, basket)
        if q0 is None:
            continue
        col = C_HANDLER if p == H else C_SCREENER if p in screeners else C_OFF
        player_dot(ax, q0, str(p), col)

    # 出手位置
    if shot is not None and oc:
        q = pos_at(cc, oc['shooter_id'], shot['release_frame'], basket, search=5)
        if q is not None:
            m = {True: ('o', '#2e8b57', '進'), False: ('X', '#b22222', '沒進'), None: ('o', '#777777', '?')}[oc.get('made')]   # X = 沒進
            ax.plot(*q, marker=m[0], ms=15, mew=2.5, mfc=m[1] if oc.get('made') else 'none', mec=m[1], zorder=8)
            ax.text(q[0] + 1.6, q[1] + 1.6, f"出手（{m[2]}）", fontsize=9, color=m[1], weight='bold', zorder=8)

    # ── 右邊文字 ──
    tx = fig.add_axes([0.55, 0.02, 0.44, 0.94]); tx.axis('off')
    kind = '雙擋' if len(group) >= 2 and max(it['frame'] for it in group) - sf0 <= 10 else \
        ('連續擋（staggered）' if len(group) >= 2 else '單擋 pick-and-roll')
    lines = [(f'事件 {idx}：{kind}', 17, 'bold', '#222'),
             (f'{f0 / fps:.1f} ~ {f1 / fps:.1f} 秒（撞上 {sf0 / fps:.1f} 秒）', 11, 'normal', '#555'),
             ('', 6, 'normal', '#000'),
             (f'持球者 #{H}', 13, 'bold', C_HANDLER)]
    for j, s in enumerate(facts['screens'], 1):
        dtxt = f"，擋到 #{s['screened_defender']}" if s['screened_defender'] is not None else ''
        lines += [(f"掩護 {j}：#{s['screener_id']}{dtxt}　·　{s['region']}", 12.5, 'bold', C_SCREENER),
                  (f"  掩護後：{ACTION_ZH.get(s['action'], s['action'])}", 11.5, 'normal', '#333'),
                  (f"  防守：{COV_ZH.get(s['coverage'], s['coverage'])}（把握 {s['coverage_confidence']}）", 11.5, 'normal', '#333')]
    lines.append(('', 6, 'normal', '#000'))
    if oc is None:
        lines.append(('結果：沒有投籃資料', 12, 'normal', '#555'))
    elif oc.get('shot'):
        made = {True: '進', False: '沒進', None: '進沒進看不出來'}[oc.get('made')]
        role = {'handler': '持球者', 'screener': '掩護者'}.get(oc.get('shooter_role'), oc.get('shooter_role', ''))
        lines.append((f"結果：{oc['seconds_after_screen']} 秒後 #{oc['shooter_id']}（{role}）", 12.5, 'bold', '#222'))
        lines.append((f"  {oc['zone']}{'三分' if oc['three_point'] else ''}{'跳投' if oc['kind'] == 'jump-shot' else '上籃' if oc['kind'] == 'layup' else '出手'}，{made}",
                      12, 'normal', '#222'))
    else:
        lines.append((f"結果：{oc.get('note', '')}", 12, 'normal', '#555'))
    y = 0.98
    for t, fs, w, c in lines:
        tx.text(0.0, y, t, fontsize=fs, weight=w, color=c, va='top', transform=tx.transAxes)
        y -= 0.010 + fs * 0.0033
    fig.text(0.02, 0.012, '圖例：號碼圓圈 = 進攻（金 = 持球者、紅 = 掩護者，位置 = 事件開始時）　× = 防守（深藍 = 被擋的人）　'
             '鋸齒 = 運球　實線 = 跑位　短槓 = 掩護　虛線 = 傳球　綠點 / 紅 × = 出手 進 / 沒進',
             fontsize=8.5, color='#555')
    if frame_img is not None:
        im_ax = fig.add_axes([0.56, 0.06, 0.40, 0.40 * 9 / 16 * 13 / 8.2])
        im_ax.imshow(frame_img[:, :, ::-1]); im_ax.axis('off')
        im_ax.set_title(f'撞上那刻（{sf0 / fps:.1f} 秒）', fontsize=9)
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)


def build(data, contact_thr=3.5, time_gap=25):
    """跟 tactics_tools.main 同一套 → [(group, facts)]。"""
    data['court_coords'], _ = T.deflicker_court_coords(data['court_coords'])
    data['court_coords'], data['player_assignment'], data['events'], _ = \
        T.merge_swapped_ids(data['court_coords'], data['player_assignment'], data['events'])
    inst = T.find_screen_instances(data['court_coords'], data['player_assignment'], data['events'], contact_thr=contact_thr)
    out = []
    for g in T.group_screen_events(inst, time_gap=time_gap, cuts=data.get('cuts')):
        f0, f1 = min(it['frame'] for it in g), max(it['frame'] for it in g)
        H = g[0]['handler_id']
        team = T.get_team(data['player_assignment'], H, f0) or 1
        basket = T.infer_target_basket(data['court_coords'], max(0, f0 - 15), f1 + 15)
        screens = [T.analyze_screen_instance(it, data, team, basket) for it in g]
        out.append((g, dict(basket=basket, screens=screens, outcome=screens[-1]['outcome'])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl')
    ap.add_argument('--video', help='原影片（給了就在圖右下附撞上那張畫面）')
    ap.add_argument('--out_dir', default=str(ROOT / 'output_videos' / 'board'))
    args = ap.parse_args()
    data = pickle.load(open(args.pkl, 'rb'))
    try:
        from shot_detect import apply_vlm_made
        apply_vlm_made(data, args.pkl)
    except Exception as e:
        print(f'  [VLM 進球判斷略過] {e}')
    frames = None
    if args.video:
        from utils import read_video
        frames = read_video(args.video)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = Path(args.pkl).stem.replace('_events', '')
    events = build(data)
    for i, (g, facts) in enumerate(events, 1):
        sf = min(it['frame'] for it in g)
        img = frames[min(sf, len(frames) - 1)] if frames is not None else None
        p = out / f'{name}_event{i}.png'
        render_event(data, i, g, facts, p, img)
        print(f'→ {p}')
    if not events:
        print('沒有擋拆事件')


if __name__ == '__main__':
    main()
