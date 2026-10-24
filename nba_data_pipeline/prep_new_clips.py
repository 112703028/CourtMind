"""
prep_new_clips.py — 沒有 COCO 標記的新片段（screen_185+）：前處理 → 模型預測擋拆 → 產生確認影片 + 表格。

跟 convert_contact.prep_clip 同一套（SAM3 軌跡、TeamSiamese 分隊、球場座標），差別只在持球者：
  舊片段：COCO ball_handler 投票
  新片段：handler_detector.pt（YOLO class 0）每張 5fps 影格偵測 → IoU 配到 SAM3 id → 整段多數決
快取存 stubs/train_contact_cache/<clip>.pkl，pending=True、events=[]（標完才會被 build 拿去訓練）。

預測：finetune_contact_clean_final_best.pt + eval_contact.score_curves（候選人輪流 slot 1）
      → 尖峰（門檻 0.6、同人間隔 15 張）= 模型建議的擋拆

確認（output_videos/new_clip_check/）：
  <clip>.mp4      5fps 影格半速播放；每人框 + 編號（進攻實心標籤、防守空心）、持球者 H、
                  左上角影格編號；模型認為在掩護的人前後 0.6 秒改紅色粗框「SCREEN? #N」，撞上那張「<< CONTACT」
  labels_<起>_<迄>.csv  一份表格：每個模型預測一列（持球者 / 分隊只在片段第一列填）；
                        模型漏掉的掩護：自己新增一列，填 clip、正確掩護者#、正確5fps影格

用法：
  python nba_data_pipeline/prep_new_clips.py --start 185 --end 282
"""
import sys
import glob
import pickle
import argparse
import collections
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from convert_contact import (FRAMES_DIR, TRACKS_DIR, CACHE_DIR, iou, team_by_color, team_collapsed,
                             apply_roles)

OUT = ROOT / 'output_videos' / 'new_clip_check'
CKPT = HERE / 'checkpoints' / 'finetune_contact_clean_final_best.pt'


def video_fps(clip):
    r = subprocess.run(['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0',
                        '-show_entries', 'stream=r_frame_rate', str(ROOT / 'input_videos' / 'screen' / f'{clip}.mp4')],
                       capture_output=True, text=True)
    try:
        a, b = r.stdout.strip().split('/')
        return float(a) / float(b)
    except Exception:
        return 30.0


def team_partition(frames, raw, team_assigner, handler_tid, def_votes, off_votes):
    """
    不依賴單一 anchor 的分隊：TeamSiamese 算每兩人「同隊機率」（整段平均）→ 窮舉主要 id 的二分法，
    取跟所有機率最一致的那個（log 似然最大）。持球者認錯也只影響「哪隊是進攻」，不會把隊伍本身分錯。
    進攻方 = 持球者那隊；YOLO 判成 defender 的票數明顯偏另一邊時改用 YOLO 的判斷。
    """
    import itertools
    import torch
    from nba_data_pipeline.team_classifier.inference import _crop_pil
    n = len(raw)
    app = collections.Counter(p for fd in raw for p in fd)
    S = collections.defaultdict(float)
    C = collections.defaultdict(int)
    m = team_assigner.model
    with torch.no_grad():
        for f in range(n):
            pids, crops = [], []
            for pid, bb in raw[f].items():
                cr = _crop_pil(frames[f], bb)
                if cr is not None:
                    pids.append(pid)
                    crops.append(cr)
            if len(pids) < 2:
                continue
            E = team_assigner._encode_crops(crops)
            ii, jj = zip(*itertools.combinations(range(len(pids)), 2))
            pr = torch.sigmoid(m.compare(E[list(ii)], E[list(jj)])).cpu().numpy()
            for a, b, p in zip(ii, jj, pr):
                key = tuple(sorted((pids[a], pids[b])))
                S[key] += float(p)
                C[key] += 1
    prob = {k: S[k] / C[k] for k in S}
    main = sorted([p for p in app if app[p] >= n / 3], key=lambda p: -app[p])[:14]
    if len(main) < 4:
        return None, 'too_few'

    def P(a, b):
        return prob.get(tuple(sorted((a, b))), 0.5)

    best = None
    for mask in range(1, 2 ** (len(main) - 1)):           # 第一個人固定在 A 邊，避免重複
        A = [main[0]] + [p for k, p in enumerate(main[1:]) if mask >> k & 1]
        B = [p for p in main if p not in A]
        if len(A) < 3 or len(B) < 3:
            continue
        ll = 0.0
        for a, b in itertools.combinations(main, 2):
            same = (a in A) == (b in A)
            p = min(max(P(a, b), 1e-3), 1 - 1e-3)
            ll += np.log(p if same else 1 - p)
        if best is None or ll > best[0]:
            best = (ll, set(A), set(B))
    _, A, B = best
    side = {p: 'A' for p in A} | {p: 'B' for p in B}
    for p in app:                                           # 碎片 id：跟哪邊平均同隊機率高就歸哪邊
        if p not in side:
            pa = np.mean([P(p, q) for q in A])
            pb = np.mean([P(p, q) for q in B])
            side[p] = 'A' if pa >= pb else 'B'
    # 哪邊是進攻：持球者那邊；YOLO defender 票明顯在持球者那邊時，信 YOLO
    dA = sum(def_votes[p] for p in side if side[p] == 'A') - sum(off_votes[p] for p in side if side[p] == 'A')
    dB = sum(def_votes[p] for p in side if side[p] == 'B') - sum(off_votes[p] for p in side if side[p] == 'B')
    if handler_tid in side:
        off = side[handler_tid]
        yolo_off = 'A' if dA < dB else 'B'
        method = 'pairwise+handler' if yolo_off == off else 'pairwise+handler(yolo不同意)'
    else:
        off = 'A' if dA < dB else 'B'
        method = 'pairwise+yolo'
    return {p: ('offense' if s == off else 'defense') for p, s in side.items()}, method


def prep(clip, handler_model, court_model, team_assigner, cm, device='cpu'):
    jpgs = sorted(glob.glob(str(FRAMES_DIR / clip / '*.jpg')), key=lambda p: int(Path(p).stem.rsplit('_', 1)[1]))
    frames = [cv2.imread(p) for p in jpgs]
    tr = pickle.load(open(TRACKS_DIR / f'{clip}_sam3_tracks.pkl', 'rb'))
    player_tracks = tr['player_tracks']
    n = len(frames)
    assert len(player_tracks) == n, f'{clip}: 軌跡 {len(player_tracks)} ≠ 影格 {n}'
    raw = [{pid: v['bbox'] for pid, v in ft.items()} for ft in player_tracks]

    # 1. 持球者：handler_detector 每張偵測 → IoU 配 SAM3 id → 多數決
    votes = collections.Counter()
    def_votes, off_votes = collections.Counter(), collections.Counter()
    per_frame_h = [-1] * n
    for i in range(0, n, 8):
        for k, r in enumerate(handler_model.predict(frames[i:i + 8], conf=0.4, verbose=False, device=device)):
            f = i + k
            for b, cl, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                best = max(((iou(b, xy), pid) for pid, xy in raw[f].items()), default=(0, None))
                if best[1] is None or best[0] < 0.3:
                    continue
                if int(cl) == 0:                      # ball_handler
                    votes[best[1]] += cf
                    per_frame_h[f] = best[1]
                    off_votes[best[1]] += cf
                elif int(cl) == 1:                    # screener（進攻）
                    off_votes[best[1]] += cf
                elif int(cl) == 2:                    # defender
                    def_votes[best[1]] += cf
    handler_tid = votes.most_common(1)[0][0] if votes else None

    # 2. 分隊：TeamSiamese 兩兩比較 + 窮舉二分法（不依賴持球者當 anchor）；失敗才退回球衣顏色
    track_team, team_method = team_partition(frames, raw, team_assigner, handler_tid, def_votes, off_votes)
    if track_team is None:
        color = team_by_color(frames, raw, handler_tid, [], None)
        track_team, team_method = (color or {}), 'color'
    track_team = apply_roles(track_team, handler_tid, [], None)

    # 3. 球場座標（同 convert_contact）
    kps = []
    for i in range(0, n, 8):
        kps.extend(court_model.predict(frames[i:i + 8], conf=0.5, verbose=False, device=device))
    court_tracks = []
    for fd, kp in zip(raw, kps):
        Hm = cm.build_homography(kp)
        ct = {}
        if Hm is not None:
            for pid, xyxy in fd.items():
                pt = cm.transform(Hm, *cm.foot_pos_xyxy(xyxy))
                if pt is not None:
                    tx, ty = cm.tactical_to_feet(*pt)
                    if 0 <= tx <= cm.COURT_W_FT + 5 and 0 <= ty <= cm.COURT_H_FT + 5:
                        ct[pid] = (tx, ty)
        court_tracks.append(ct)

    return {'clip': clip, 'n': n, 'fps': video_fps(clip), 'valid_start_idx': 0, 'pending': True,
            'court_tracks': court_tracks, 'raw_tracks': raw, 'track_team': track_team,
            'handler_tid': handler_tid, 'handler_per_frame': per_frame_h, 'screener_tids': [],
            'defender_tid': None, 'team_method': team_method, 'events': [], 'source': 'new_clip'}


HANDLER_ONLY = False   # --handler_only：影片只標持球者（不畫分隊顏色、不畫模型預測的掩護）
HL_PAD = 3      # 模型預測的掩護者：撞上前後各 3 張 5fps（±0.6 秒）用紅色粗框標出


def screener_marks(peaks):
    """{5fps 影格: {掩護者 id: (分數, 是否撞上那張)}}，給畫圖用。"""
    hl = collections.defaultdict(dict)
    for f, s, sc in peaks:
        for d in range(-HL_PAD, HL_PAD + 1):
            prev = hl[f + d].get(s)
            if prev is None or d == 0:
                hl[f + d][s] = (sc, d == 0)
    return hl


def draw_boxes(img, boxes, c, f5, marks):
    """畫所有人的框 + 編號（黃 = 進攻、藍 = 防守、綠粗 = 持球者 H），
    模型認為在掩護的人改用紅色粗框 + 「SCREEN? #N」，撞上那張加「<< CONTACT」；左上角 5fps 編號。"""
    for pid, (x1, y1, x2, y2) in boxes.items():
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        t = c['track_team'].get(pid)
        isH = pid == c['handler_tid']
        if HANDLER_ONLY and not isH:
            col, thick, label = (230, 230, 230), 2, f"#{pid}"
        elif pid in marks:
            sc, exact = marks[pid]
            col, thick = (0, 0, 255), 7
            label = f"SCREEN? #{pid}" + ("  << CONTACT" if exact else "")
        else:
            col = (0, 220, 0) if isH else ((0, 200, 255) if t == 'offense' else (255, 150, 0))
            thick = 5 if isH else 2
            label = f"{'H ' if isH else ''}#{pid}"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        if (t == 'offense' and not HANDLER_ONLY) or isH or pid in marks:
            cv2.rectangle(img, (x1, y1 - th - 12), (x1 + tw + 8, y1), col, -1)
            txt_col = (255, 255, 255) if pid in marks else (0, 0, 0)
            cv2.putText(img, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.1, txt_col, 3)
        else:
            cv2.putText(img, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    cv2.rectangle(img, (0, 0), (900, 80), (0, 0, 0), -1)
    cv2.putText(img, f"{c['clip']}  5fps #{f5}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    return img


def _writer(path, w, h, fps):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                             '-s', f'{w}x{h}', '-r', f'{fps:.3f}', '-i', '-', '-c:v', 'libx264',
                             '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)


def render(c, peaks, path):
    """5fps 影格版（半速 = 每秒 2.5 張）。"""
    jpgs = sorted(glob.glob(str(FRAMES_DIR / c['clip'] / '*.jpg')), key=lambda p: int(Path(p).stem.rsplit('_', 1)[1]))
    h, w = cv2.imread(jpgs[0]).shape[:2]
    W, Hh = 1280, int(h * 1280 / w)
    proc = _writer(path, W, Hh, 2.5)
    hl = screener_marks(peaks)
    for f, p in enumerate(jpgs):
        img = draw_boxes(cv2.imread(p), c['raw_tracks'][f], c, f, hl.get(f, {}))
        proc.stdin.write(cv2.resize(img, (W, Hh)).tobytes())
    proc.stdin.close()
    proc.wait()


def render30(c, peaks, path, speed=0.5):
    """原片每一幀都畫：5fps 編號 = round(t × 5)，框在相鄰兩張 5fps 影格間線性內插。"""
    src = ROOT / 'input_videos' / 'screen' / f"{c['clip']}.mp4"
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or c['fps']
    ok, img = cap.read()
    if not ok:
        return render(c, peaks, path)
    h, w = img.shape[:2]
    W, Hh = 1280, int(h * 1280 / w)
    proc = _writer(path, W, Hh, fps * speed)
    hl = screener_marks(peaks)
    raw, n, k = c['raw_tracks'], c['n'], 0
    while ok:
        t5 = k / fps * 5.0
        a = min(n - 1, int(np.floor(t5)))
        b = min(n - 1, a + 1)
        r = t5 - a if b > a else 0.0
        boxes = {}
        for pid, ba in raw[a].items():
            bb = raw[b].get(pid)
            boxes[pid] = ba if bb is None else [ba[i] * (1 - r) + bb[i] * r for i in range(4)]
        f5 = min(n - 1, int(round(t5)))
        proc.stdin.write(cv2.resize(draw_boxes(img, boxes, c, f5, hl.get(f5, {})), (W, Hh)).tobytes())
        ok, img = cap.read()
        k += 1
    cap.release()
    proc.stdin.close()
    proc.wait()


HANDLER_HEADER = ('clip,持球者(模型),正確持球者#(錯才填),掩護者#,撞上5fps影格,防守方式(可不填),備註')


def handler_rows(c):
    """只標持球者版：每支片段一列（模型持球者）；掩護你自己填，同一支多個掩護就複製一列。"""
    return [f"{c['clip']},#{c['handler_tid']},,,,,"]


LABEL_HEADER = ('clip,持球者(模型),正確持球者#(錯才填),分隊錯的人#(錯才填),'
                '掩護者(模型),撞上5fps影格(模型),是真掩護(t/f),正確掩護者#(錯才填),正確5fps影格(差2張以上才填),'
                '防守方式(可不填),備註')


def label_rows(c, peaks):
    """合併表格：每個模型預測一列；持球者 / 分隊只在該片段第一列填。"""
    clip, H = c['clip'], c['handler_tid']
    if not peaks:
        return [f"{clip},#{H},,,,,,,,,模型沒抓到掩護：沒有就在「是真掩護」填 f；有就把掩護者和影格填在「正確…」欄"]
    out = []
    for k, (f, s, sc) in enumerate(peaks):
        head = f"{clip},#{H},," if k == 0 else f"{clip},(同上),,"
        out.append(f"{head},#{s},{f},,,,,")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=185)
    ap.add_argument('--end', type=int, default=282)
    ap.add_argument('--thr', type=float, default=0.6)
    ap.add_argument('--nms', type=int, default=15)
    ap.add_argument('--force', action='store_true', help='快取已存在也重跑前處理')
    ap.add_argument('--device', default='cpu', help='cpu / cuda（持球者、球場、分隊模型）')
    ap.add_argument('--render30', action='store_true',
                    help='確認影片改用原片（約 30fps）＋框在 5fps 影格間內插，左上角仍標 5fps 編號')
    ap.add_argument('--no_render', action='store_true',
                    help='只做前處理（快取），不畫確認影片、不寫表格（例：pod 上用 GPU 前處理，影片回本地畫）')
    ap.add_argument('--handler_only', action='store_true',
                    help='只標持球者：影片不畫分隊、不預測掩護，表格只有持球者 + 你自己填的掩護')
    args = ap.parse_args()
    global HANDLER_ONLY
    HANDLER_ONLY = args.handler_only

    clips = [f'screen_{i}' for i in range(args.start, args.end + 1)
             if (TRACKS_DIR / f'screen_{i}_sam3_tracks.pkl').exists()]
    print(f'有 SAM3 軌跡的新片段 {len(clips)} 支')

    todo = [c for c in clips if args.force or not (CACHE_DIR / f'{c}.pkl').exists()]
    if todo:
        from ultralytics import YOLO
        import convert_manual as cm
        from nba_data_pipeline.team_classifier.inference import TeamAssignerSiamese
        handler_model = YOLO(str(ROOT / 'models' / 'handler_detector.pt'))
        court_model = YOLO(str(ROOT / 'models' / 'court_keypoint_detector.pt'))
        team_assigner = TeamAssignerSiamese(ckpt_path=str(ROOT / 'models' / 'team_siamese.pt'), device=args.device)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for k, clip in enumerate(todo):
            try:
                c = prep(clip, handler_model, court_model, team_assigner, cm, device=args.device)
                pickle.dump(c, open(CACHE_DIR / f'{clip}.pkl', 'wb'))
                off = sum(1 for t in c['track_team'].values() if t == 'offense')
                print(f"[{k + 1}/{len(todo)}] {clip}: 持球者 #{c['handler_tid']}，分隊 {c['team_method']}"
                      f"（進攻 {off}／防守 {len(c['track_team']) - off}）")
            except Exception as ex:
                print(f'[{k + 1}/{len(todo)}] {clip} 失敗：{ex}')

    if args.no_render:
        print('只做前處理（--no_render），完成')
        return
    if not HANDLER_ONLY:                       # 只標持球者時不需要擋拆偵測模型
        from eval_contact import load_model, score_curves, find_peaks
        model, ck = load_model(CKPT)
        anchor = int((ck.get('data_meta') or {}).get('anchor', 3))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in clips:
        p = CACHE_DIR / f'{clip}.pkl'
        if not p.exists():
            continue
        c = pickle.load(open(p, 'rb'))
        peaks = []
        if c['handler_tid'] is not None and not HANDLER_ONLY:
            for cand, cv in score_curves(model, c, anchor).items():
                peaks += [(f, cand, s) for f, s in find_peaks(cv, args.thr, args.nms)]
        peaks.sort()
        (render30 if args.render30 else render)(c, peaks, OUT / f'{clip}.mp4')
        rows += handler_rows(c) if HANDLER_ONLY else label_rows(c, peaks)
        print(f'{clip}: 持球者 #{c["handler_tid"]}' + ('' if HANDLER_ONLY else f'，模型預測 {len(peaks)} 個掩護'))
    tag = f'{args.start}_{args.end}'
    name, header = (f'handler_labels_{tag}.csv', HANDLER_HEADER) if HANDLER_ONLY else (f'labels_{tag}.csv', LABEL_HEADER)
    with open(OUT / name, 'w', encoding='utf-8-sig') as fo:
        fo.write(header + '\n' + '\n'.join(rows) + '\n')
    print(f'→ {OUT}')


if __name__ == '__main__':
    main()
