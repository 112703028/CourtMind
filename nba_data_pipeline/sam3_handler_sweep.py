"""
sam3_handler_sweep.py — 讀 SAM3 快取 pkl，掃多組持球判定參數，輸出比較表。

跟 sam3_handler_check.py 用同一份 SAM3 輸出（cache pkl），但這裡只做「持球判定」
（overlap + 遲滯），不碰 GPU、不需要 sam3 套件 → **可在本機跑**，秒級掃參數。

用法（本機，把 <out>_sam3.pkl 拉回本地後）：
  python sam3_handler_sweep.py \
      --cache output_videos/sam3_handler_v130_sam3.pkl \
      --gt 6                      # 該支的 ground truth 持球者（可多個，空白分隔）

輸出：每組 (overlap_thr, confirm_frames) 的持球時間軸 + 主鏈/誤標量，讓你挑最佳。
"""

import os
import argparse
import pickle
from collections import Counter


def contain(ball, player):
    """球框落在球員框內比例（交集 / 球框面積）。"""
    ix1, iy1 = max(ball[0], player[0]), max(ball[1], player[1])
    ix2, iy2 = min(ball[2], player[2]), min(ball[3], player[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    barea = (ball[2]-ball[0]) * (ball[3]-ball[1])
    return inter / barea if barea > 0 else 0.0


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compute_holder(players, balls, n, overlap_thr, confirm_frames):
    """回持球時間軸 list（-1=無人, None=無球, int=持球者）。"""
    # Pass 1：每幀球框落在誰框內最多 + 是否 >= 門檻
    nearest, touched = [], []
    for f in range(n):
        pl = players.get(f, {})
        bl = balls.get(f, {})
        if not bl:
            nearest.append(None); touched.append(None); continue
        bball = list(bl.values())[0][2]
        if not pl:
            nearest.append(-1); touched.append(False); continue
        best_pid, best_ov = None, -1.0
        for pid, (px, py, pbox) in pl.items():
            if iou(pbox, bball) > 0.5:
                continue
            ov = contain(bball, pbox)
            if ov > best_ov:
                best_ov, best_pid = ov, pid
        nearest.append(best_pid if best_pid is not None else -1)
        touched.append(best_pid is not None and best_ov >= overlap_thr)
    # Pass 2：遲滯
    tl = []
    holder, cand, cnt = -1, None, 0
    for f in range(n):
        nr, tc = nearest[f], touched[f]
        if nr is None:
            tl.append(holder); continue
        if not tc:
            cand, cnt = None, 0; tl.append(holder); continue
        if nr == holder:
            cand, cnt = None, 0
        else:
            if nr == cand:
                cnt += 1
            else:
                cand, cnt = nr, 1
            if cnt >= confirm_frames:
                holder = cand; cand, cnt = None, 0
        tl.append(holder)
    return tl


def compute_holder_v2(players, balls, n, overlap_thr, pass_speed, recv_window=8):
    """
    球運動方向/事件版：overlap 判穩定持球 + 球速度事件判換手。

    比 confirm_frames 遲滯好在：不用猜連續幾幀，而是看「完整傳球事件」——
    球高速飛出（速度 > pass_speed）代表傳球中，飛行結束後球落在誰的 overlap 內
    = 接球者。飛球掠過路人（球沒停在他身上、繼續飛）不算接球；斷續接球（球在
    接球者身上時進時出）只要飛行終點落在他身上就抓到，不受連續性影響。

    參數：
      overlap_thr : 球框落球員框內比例 >= 此 → 算「球在他身上」
      pass_speed  : 球速度(像素/幀) > 此 → 判定為傳球飛行中
    """
    # 每幀：球中心、球框對角線（尺度）、overlap 最高球員 + 是否觸球
    bcenter, bdiag, ov_pid, ov_touch = [], [], [], []
    for f in range(n):
        bl = balls.get(f, {})
        pl = players.get(f, {})
        if not bl:
            bcenter.append(None); bdiag.append(None)
            ov_pid.append(None); ov_touch.append(None)
            continue
        bx, by, bball = list(bl.values())[0]
        bcenter.append((bx, by))
        bdiag.append(((bball[2]-bball[0])**2 + (bball[3]-bball[1])**2) ** 0.5)
        best_pid, best_ov = None, -1.0
        for pid, (px, py, pbox) in pl.items():
            if iou(pbox, bball) > 0.5:
                continue
            o = contain(bball, pbox)
            if o > best_ov:
                best_ov, best_pid = o, pid
        ov_pid.append(best_pid if best_pid is not None else -1)
        ov_touch.append(best_pid is not None and best_ov >= overlap_thr)

    def speed(f):
        """相對速度：球每幀移動了幾個「球身」（位移 / 球框對角線）。
        用球框當尺度自動抵消鏡頭遠近，讓 pass_speed 門檻在遠近鏡頭通用。"""
        if f == 0 or bcenter[f] is None or bcenter[f-1] is None:
            return 0.0
        dx = bcenter[f][0]-bcenter[f-1][0]; dy = bcenter[f][1]-bcenter[f-1][1]
        disp = (dx*dx+dy*dy) ** 0.5
        scale = bdiag[f] or bdiag[f-1] or 1.0
        return disp / scale

    tl = []
    from collections import Counter
    holder = -1
    flying = False          # 是否在傳球飛行中
    for f in range(n):
        if bcenter[f] is None:              # 無球：維持
            tl.append(holder); continue
        sp = speed(f)
        if sp > pass_speed:                 # 球高速飛 → 傳球中
            flying = True
            tl.append(holder)               # 球在空中，時間軸仍記上一個持球者
            continue
        if flying:
            # 剛結束飛行：球剛落下那幀可能還沒穩定到接球者身上（傳球飛行在數據上
            # 常只有 1-2 幀，落地後接球者可能馬上運球移動）。改用「落地後 recv_window
            # 幀內 overlap 最高的球員多數決」判接球者，比看落下瞬間那一幀穩。
            votes = Counter()
            for g in range(f, min(n, f + recv_window)):
                if ov_touch[g] and ov_pid[g] is not None and ov_pid[g] >= 0:
                    votes[ov_pid[g]] += 1
            if votes:
                receiver = votes.most_common(1)[0][0]
                if receiver != holder:
                    holder = receiver
            flying = False
        else:
            # 非飛行中：初始 holder 尚未確立時，用當前 overlap 觸球者當第一個持球者
            if holder == -1 and ov_touch[f] and ov_pid[f] is not None and ov_pid[f] >= 0:
                holder = ov_pid[f]
        tl.append(holder)
    return tl


def segs(tl):
    s = []
    for h in tl:
        k = '無球' if h is None else ('無人' if h == -1 else h)
        if s and s[-1][0] == k:
            s[-1][1] += 1
        else:
            s.append([k, 1])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, help='SAM3 快取 pkl（<out>_sam3.pkl）')
    ap.add_argument('--gt', nargs='*', type=int, default=[],
                    help='該支 ground truth 持球者 id（SAM3 id；用來標誤標）')
    ap.add_argument('--overlaps', nargs='*', type=float,
                    default=[0.4, 0.5, 0.6, 0.7])
    ap.add_argument('--confirms', nargs='*', type=int, default=[3, 5, 8, 12])
    ap.add_argument('--v2', action='store_true',
                    help='用 compute_holder_v2（球速度事件版），掃 pass_speed')
    ap.add_argument('--speeds', nargs='*', type=float,
                    default=[1.5, 2.0, 3.0, 4.0],
                    help='v2 相對速度門檻（球身/幀；球每幀移動幾個球框對角線）')
    ap.add_argument('--recv_window', type=int, default=8,
                    help='v2 傳球落地後，用接下來幾幀 overlap 多數決判接球者')
    args = ap.parse_args()

    with open(args.cache, 'rb') as fp:
        c = pickle.load(fp)
    players, balls, jpgs = c['players'], c['balls'], c['jpgs']
    n = len(jpgs)
    gt = set(args.gt)
    mode = 'v2(球速度事件)' if args.v2 else 'v1(遲滯)'
    print(f"快取: {args.cache}  ({n} 幀)  GT={sorted(gt) if gt else '未給'}  模式={mode}\n")

    p2 = 'spd' if args.v2 else 'conf'
    print(f"{'ov':>4} {p2:>4} | {'主鏈幀':>6} {'誤標幀':>6} | 時間軸(連續段>=3)")
    print("-" * 90)
    for ov in args.overlaps:
        params = args.speeds if args.v2 else args.confirms
        for p in params:
            if args.v2:
                tl = compute_holder_v2(players, balls, n, ov, p, args.recv_window)
            else:
                tl = compute_holder(players, balls, n, ov, int(p))
            sg = segs(tl)
            main_frames = sum(c for k, c in sg if isinstance(k, int) and k in gt)
            noise_frames = sum(c for k, c in sg if isinstance(k, int) and k not in gt)
            chain = '  '.join(f"#{k}×{c}" if isinstance(k, int) else f"{k}×{c}"
                              for k, c in sg if c >= 3)
            tag = '' if not gt else ('[OK]' if noise_frames < 15 else
                                     ('[~]' if noise_frames < 40 else '[X]'))
            pstr = f"{p:>4.1f}" if args.v2 else f"{int(p):>4}"
            print(f"{ov:>4} {pstr} | {main_frames:>6} {noise_frames:>6} {tag} | {chain[:120]}")
    print("\n（主鏈=GT持球者幀數，誤標=GT以外持球者幀數。誤標越少越好，別把真接球者切掉）")


if __name__ == '__main__':
    main()
