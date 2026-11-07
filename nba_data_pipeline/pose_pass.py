"""
pose_pass.py — 用手腕找持球者 → 傳球（2026-11-05，規則版 2D 框在人擠 / 球穿過路人框時分不出來）。

每張 10fps：持球者 = 球中心離誰的手腕（信心 ≥ KP_CONF）≤ r_hand 倍身高（取最近）；持球要連續 ≥ hold 張才算一段，
中間短暫別人（< hold 張）略過。傳球 = A 段 → B 段（A ≠ B、同隊、間隔 ≤ max_air 秒、±0.5 秒沒有出手）。
A 段最後一張時 B 的手腕要離球 ≥ sep 倍身高（排除兩人手都在球上的換人）。
資料：stubs/pose_hands/<片段>.pkl（pose_hands.py）、stubs/ball_batch/<片段>_ball.pkl + _touch.pkl、output_videos/batch 的 events.pkl。
評估：pass_gt_probe.GT（使用者標的 21 個真傳球）。

用法：python nba_data_pipeline/pose_pass.py                 # 21 個真傳球評估
     python nba_data_pipeline/pose_pass.py add <events.pkl> <ball.pkl> <touch.pkl> <pose.pkl> <影片>   # 寫進 events.pkl
"""
import sys
import pickle
import itertools
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from shot_detect import map_ids
from possession_sam3 import clip_teams
from pass_gt_probe import GT

KP_CONF, WRISTS = 0.3, (9, 10)


def hand_dist(kp, c, H):
    ws = [kp[w, :2] for w in WRISTS if kp[w, 2] >= KP_CONF]
    return min(float(np.hypot(*(w - np.asarray(c)))) for w in ws) / H if ws else None


def holders(pose, tch, H, r_hand):
    seq = []
    for i, (p, c) in enumerate(tch):
        if c is None or i >= len(pose):
            seq.append(None); continue
        ds = {q: hand_dist(kp, c, H) for q, kp in pose[i].items()}
        ds = {q: v for q, v in ds.items() if v is not None and v <= r_hand}
        seq.append(min(ds, key=ds.get) if ds else None)
    return seq


def smooth_flicker(seq, max_len=1):
    """夾在同一人中間、只閃 ≤ max_len 張的別人（1, 7, 1）→ 當成同一人（掩護 / 短傳時手腕距離互換一下）。"""
    seq = list(seq)
    vals = [(i, q) for i, q in enumerate(seq) if q is not None]
    for k in range(1, len(vals) - 1):
        (ia, a), (ib, b), (ic, cc) = vals[k - 1], vals[k], vals[k + 1]
        if a == cc != b and ic - ia <= max_len + 2:
            seq[ib] = a
    return seq


_DET = None


def rims_at(key, frames_needed, video):
    """原片影格 → Roboflow 籃框框（信心最高、≥ 0.4）；快取 stubs/rim_cache/<片段>.pkl（只跑用得到的影格）。"""
    global _DET
    import cv2
    p = ROOT / 'stubs' / 'rim_cache' / f'{key}.pkl'
    cache = pickle.load(open(p, 'rb')) if p.exists() else {}
    todo = sorted(set(frames_needed) - set(cache))
    if todo:
        if _DET is None:
            sys.path.insert(0, str(HERE))
            from shot_detect_rf import OnnxDetector
            _DET = OnnxDetector(ROOT / 'models' / 'rf_basketball_det_v4.onnx')
        cap = cv2.VideoCapture(str(video))
        for f in todo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            r = [d for d in (_DET.infer(im, conf=0.25, iou=0.9) if ok else []) if d[0] == 9 and d[1] >= 0.4]
            cache[f] = max(r, key=lambda d: d[1])[2] if r else None
        p.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(cache, open(p, 'wb'))
    return cache


def passes(c, **prm):
    """批次片段 screen_<c>（output_videos/batch + stubs/ball_batch + stubs/pose_hands）→ passes_from。"""
    d = pickle.load(open(ROOT / f'output_videos/batch/screen_{c}_events.pkl', 'rb'))
    s = pickle.load(open(ROOT / f'stubs/ball_batch/screen_{c}_ball.pkl', 'rb'))['shot_sam3']
    _, _, tch = pickle.load(open(ROOT / f'stubs/ball_batch/screen_{c}_touch.pkl', 'rb'))
    pose = pickle.load(open(ROOT / f'stubs/pose_hands/screen_{c}.pkl', 'rb'))
    return passes_from(d, s, tch, pose, ROOT / 'input_videos' / 'screen' / f'screen_{c}.mp4', f'screen_{c}', **prm)


def passes_from(d, s, tch, pose, video, key, r_hand=0.2, hold=3, sep=0.3, max_air=1.5, shot_win=0.5, refine_release=True,
                rim_excl=2.5, rel_v=0.25, use3d=False, rim3d=False, body_h=2.8, body_r=1.0, tiebreak=False, def_hold=1.0):
    """events.pkl dict + shot_sam3 + touches + 手腕姿態 → 傳球 / 抄截列表（編號 = events.pkl 的編號，影格 = 原片）。
    video = 原片（籃框偵測用），key = 快取檔名（stubs/rim_cache、cam_cache）。"""
    H = float(np.median([b[3] - b[1] for fd in s['players'] for b in fd.values()]))
    seq = holders(pose, tch, H, r_hand)
    idmap = map_ids(s, d['player_tracks'])
    cams, fps_c = None, None
    if use3d or rim3d:
        import camera_calib as CC
        if (ROOT / 'stubs' / 'court_kp' / f'{key}.pkl').exists():   # 沒有球場關鍵點的片段 → 退回 2D
            cams = CC.clip_cams(key)
        fps_c = s['orig_fps']

    def cam_at(i):
        j = int(round(s['frame_idx'][i] * 5.0 / fps_c))
        return cams[min(max(j, 0), len(cams) - 1)] if cams else None

    def court_m(i, q):
        """球員地板位置：用同一個鏡頭把框底部中間（腳）投到地板 —— 不用推論的 court_coords
        （那是另一組關鍵點算的、又經過鏡像修正，可能跟這個鏡頭左右相反）。"""
        import camera_calib as CC
        b, cam = s['players'][i].get(q), cam_at(i)
        if b is None or cam is None:
            return None
        return CC.floor_point(cam, ((b[0] + b[2]) / 2, b[3]))

    def ok3d(i, q):
        import camera_calib as CC
        cam, xy = cam_at(i), court_m(i, q)
        if cam is None or xy is None or tch[i][1] is None:
            return None                                   # 不能判斷
        dist, hz = CC.ray_at(cam, tch[i][1], xy)
        return dist <= body_r and -0.5 <= hz <= body_h

    if use3d and tiebreak:   # 3D 只當和事佬：手腕範圍內有好幾個人時，挑 3D 上球真的在他身高範圍內的；只有一人照 2D
        for i in range(len(seq)):
            c_ = tch[i][1]
            if c_ is None or i >= len(pose):
                continue
            cand = sorted((v, q) for q, kp in pose[i].items() for v in [hand_dist(kp, c_, H)] if v is not None and v <= r_hand)
            if len(cand) >= 2:
                good = [q for _, q in cand if ok3d(i, q)]
                if good:
                    seq[i] = good[0]
    elif use3d:   # 3D 過濾：球的射線經過持球者位置時要在身高範圍內；從頭頂飛過的不算拿球
        for i, q in enumerate(seq):
            if q is not None and ok3d(i, q) is False:
                seq[i] = None
    seq = smooth_flicker(seq)
    segs, cur = [], None
    for i, q in enumerate(seq):
        if q is not None and cur and cur[0] == q and i - cur[2] <= 2:
            cur[2] = i
        elif q is not None:
            if cur: segs.append(cur)
            cur = [q, i, i]
    if cur: segs.append(cur)
    major = [g for g in segs if g[2] - g[1] + 1 >= hold]
    team = clip_teams(d['player_assignment'])
    fi, fps = s['frame_idx'], s['orig_fps']
    # 防守者「拿球」< def_hold 秒 = 貼身防守時手腕剛好靠近球（2026-11-06 抽查 4/4 假抄截都是這種），當雜訊拿掉 —— 真抄截防守者會一直拿著
    if def_hold:
        is_def = lambda q: team.get(idmap.get(q, q)) == 2
        major = [g for g in major if not is_def(g[0]) or (fi[g[2]] - fi[g[1]]) / fps >= def_hold]
    cuts = d.get('cuts') or []
    out = []
    for (A, a1, b1), (B, a2, b2) in zip(major, major[1:]):
        if A == B or (fi[a2] - fi[b1]) / fps > max_air:
            continue
        if any(fi[b1] < c_ <= fi[a2] for c_ in cuts):
            continue          # 跨鏡頭切換：兩邊的編號不是同一批人（screen_431 特寫重播 → 全場）
        if def_hold and team.get(idmap.get(A, A)) == 2:
            continue          # 防守者 → 進攻：防守方長時間拿球後又回到進攻方（籃板 / 換場），不是這波的傳球或抄截
        # ① 離手時間：A 的手腕常在出手前被擋住 → 持球段提早結束。之後球還在 A 框裡、而且跟著 A 一起動 → 還是 A 拿著
        if refine_release:
            def box_d(i, q):
                b, cc = s['players'][i].get(q), tch[i][1]
                if b is None or cc is None:
                    return 9.0
                dx = max(b[0] - cc[0], 0, cc[0] - b[2]); dy = max(b[1] - cc[1], 0, cc[1] - b[3])
                return float(np.hypot(dx, dy)) / H
            def rel(i, q):
                b, cc = s['players'][i].get(q), tch[i][1]
                return None if b is None or cc is None else np.array([(cc[0] - (b[0] + b[2]) / 2) / H, (cc[1] - (b[1] + b[3]) / 2) / H])
            j = b1
            while j + 1 < a2 and box_d(j + 1, A) <= 0.1:
                r0, r1 = rel(j, A), rel(j + 1, A)
                if r0 is None or r1 is None or float(np.hypot(*(r1 - r0))) > rel_v:
                    break
                j += 1
            b1 = j
        cA = tch[b1][1]
        dB = hand_dist(pose[b1][B], cA, H) if B in pose[b1] and cA is not None else 9.0
        if dB is not None and dB < sep:
            continue
        rel, cat = fi[b1], fi[a2]
        if any(rel - 0.2 * fps <= sh['release_frame'] <= cat - 0.1 * fps for sh in d.get('shots') or []):
            continue          # 球在空中那段（A 離手 → B 接球）有人出手 = 投籃；B 接到才出手 = 接球就投，照算傳球（screen_46）
        # ② 球在空中時靠近籃框（≤ rim_excl 倍籃框寬）= 投籃 / 籃板，不是傳球（原片段推論沒開投籃偵測，screen_140 / 162 搶籃板）
        if rim3d:     # 3D：空中那段球的射線經過籃框位置時 ≥ 2.4 m、水平 ≤ 1.5 m = 球真的在籃框附近（投籃 / 籃板）
            import camera_calib as CC
            hit_rim = False
            for i in range(b1, a2 + 1):
                cam = cam_at(i)
                if cam is None or tch[i][1] is None:
                    continue
                for left in (True, False):
                    dist, hz = CC.ray_at(cam, tch[i][1], CC.rim_point(left)[:2])
                    if dist <= 1.5 and hz >= 2.4:
                        hit_rim = True
            if hit_rim:
                continue
        elif rim_excl:
            air = list(range(b1, a2 + 1))
            rc = rims_at(key, [fi[i] for i in air], video)
            near_rim = False
            for i in air:
                rb, cc = rc.get(fi[i]), tch[i][1]
                if rb is not None and cc is not None:
                    rw = rb[2] - rb[0]
                    if np.hypot(cc[0] - (rb[0] + rb[2]) / 2, cc[1] - (rb[1] + rb[3]) / 2) <= rim_excl * rw:
                        near_rim = True; break
            if near_rim:
                continue
        a, b = idmap.get(A, A), idmap.get(B, B)
        same = team.get(a) is not None and team.get(a) == team.get(b)
        out.append(dict(kind='pass' if same else 'turnover', frm=a, to=b, release=rel, catch=cat))
    return out, d['fps']


def evaluate(**prm):
    hit, extra, wrong_pair, miss = 0, 0, 0, []
    for c in sorted({g[0] for g in GT}):
        if not (ROOT / f'stubs/pose_hands/screen_{c}.pkl').exists():
            continue
        ps, fps = passes(c, **prm)
        ps = [p for p in ps if p['kind'] == 'pass']
        used = set()
        for (_, f0, f1, A, B) in [g for g in GT if g[0] == c]:
            win = [i for i, p in enumerate(ps) if i not in used and f0 - 0.6 * fps <= p['catch'] and p['release'] <= f1 + 0.6 * fps]
            m = [i for i in win if ps[i]['frm'] == A and ps[i]['to'] == B]
            if m:
                used.add(m[0]); hit += 1
            else:
                miss.append((c, A, B, [(ps[i]['frm'], ps[i]['to']) for i in win]))
                if win:
                    wrong_pair += 1; used.add(win[0])
        extra += len(ps) - len(used)
    return hit, wrong_pair, extra, miss


def main():
    n = sum((ROOT / f'stubs/pose_hands/screen_{g[0]}.pkl').exists() for g in GT)
    print(f'有姿態的真傳球 {n}/21')
    for r, h, sp in itertools.product((0.15, 0.2, 0.3), (2, 3), (0.0, 0.3)):
        hit, wp, ex, _ = evaluate(r_hand=r, hold=h, sep=sp)
        print(f'手腕 ≤ {r} 身高、持球 ≥ {h} 張、接球者原本離球 ≥ {sp}：抓對 {hit}/{n}、時間對但人錯 {wp}、多抓 {ex}')
    hit, wp, ex, miss = evaluate(r_hand=0.2, hold=2, sep=0.3)
    print('\n漏掉的（片段、標記 A→B、同時段偵測到的）：')
    for m in miss:
        print('  ', m)


def check_sheet(clips, out_csv, r_hand=0.2, hold=3):
    """沒調過參數的片段 → 傳球檢查表（影格 = 原片影格，跟 output_videos/framenum_boxes 影片左上角號碼一樣；編號 = 框上的編號）。"""
    import csv
    rows, k = [], 0
    for c in clips:
        if not all((ROOT / p).exists() for p in (f'stubs/pose_hands/screen_{c}.pkl', f'stubs/ball_batch/screen_{c}_touch.pkl',
                                                  f'output_videos/batch/screen_{c}_events.pkl')):
            continue
        ps, fps = passes(c, r_hand=r_hand, hold=hold)
        for p in ps:
            if p['kind'] != 'pass':
                continue
            k += 1
            rows.append([f'Q{k:03d}', f'screen_{c}', p['release'], p['catch'], f"#{p['frm']}", f"#{p['to']}", '', ''])
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['編號', 'clip', '離手影格', '接球影格', '傳球者', '接球者', '對嗎(t/f/?)', '備註（例：接球者是#5、其實是運球、時間差很多）'])
        w.writerows(rows)
    print(f'{len(rows)} 個傳球 → {out_csv}')


def annotate(events_pkl, ball_pkl, touch_pkl, pose_pkl, video):
    """單支影片：把傳球寫進 events.pkl 的 'passes'（給 tactics_tools / tactics_agent / 動態板）。
    格式同 possession_sam3.detect_passes：[{kind: pass / turnover, frm, to, release, catch}]，編號 = events.pkl 的編號、影格 = 原片。"""
    d = pickle.load(open(events_pkl, 'rb'))
    s = pickle.load(open(ball_pkl, 'rb'))
    s = s.get('shot_sam3', s)
    _, _, tch = pickle.load(open(touch_pkl, 'rb'))
    pose = pickle.load(open(pose_pkl, 'rb'))
    ps, _ = passes_from(d, s, tch, pose, video, Path(events_pkl).stem.replace('_events', ''))
    d['passes'] = ps
    d['passes_source'] = 'pose_wrist'
    pickle.dump(d, open(events_pkl, 'wb'))
    print(f"傳球 {sum(p['kind'] == 'pass' for p in ps)}、抄截 / 換隊 {sum(p['kind'] == 'turnover' for p in ps)} → {events_pkl}")
    for p in ps:
        print(f"  {p['kind']} #{p['frm']}→#{p['to']}  f{p['release']}→f{p['catch']}")
    return ps


if __name__ == '__main__':
    if sys.argv[1:2] == ['add_batch']:      # python pose_pass.py add_batch → output_videos/batch 裡有姿態 + 碰球的片段全部寫入
        import contextlib, io
        n = 0
        for pk in sorted((ROOT / 'stubs' / 'pose_hands').glob('screen_*.pkl'), key=lambda x: int(x.stem.split('_')[1])):
            name = pk.stem
            ev, bp, tp = (ROOT / 'output_videos' / 'batch' / f'{name}_events.pkl', ROOT / 'stubs' / 'ball_batch' / f'{name}_ball.pkl',
                          ROOT / 'stubs' / 'ball_batch' / f'{name}_touch.pkl')
            if all(x.exists() for x in (ev, bp, tp)):
                with contextlib.redirect_stdout(io.StringIO()):
                    annotate(ev, bp, tp, pk, ROOT / 'input_videos' / 'screen' / f'{name}.mp4')
                n += 1
        print(f'{n} 支片段寫入 passes')
    elif sys.argv[1:2] == ['add']:            # python pose_pass.py add <events.pkl> <ball.pkl> <touch.pkl> <pose.pkl> <video>
        annotate(*sys.argv[2:7])
    elif sys.argv[1:2] == ['sheet']:            # python pose_pass.py sheet 47 166
        a, b = int(sys.argv[2]), int(sys.argv[3])
        check_sheet(range(a, b + 1), ROOT / 'output_videos' / 'framenum_boxes' / 'pass_check_pose.csv')
    else:
        main()
