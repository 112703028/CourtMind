"""
sam3_handler_check.py — 驗證方向 B：SAM3 追球 → 判「球在誰手上」→ 抓傳球/接球。

背景（見 docs/superpowers/specs/2026-09-05-...md「附帶發現：持球偵測」）：
handler_detector 的訓練資料把 ball_handler 標成「持球發起者滿全程」，學到的是
「主角」不是「此刻誰持球」→ 傳球後的接球者抓不到。方向 B：用 SAM3 追球
（concept "basketball"）→ 每幀算「離球最近的球員 = 持球者」→ 補傳球/接球。

本腳本一次驗證兩件事：
  1. SAM3 追球在我方影片準不準（球有沒有被穩定追到）
  2. 「球→最近球員」能不能抓到傳球接球（持球者逐幀變化，傳球時換人）

做法：對同一段影片，SAM3 跑兩個 concept（"basketball player" 拿球員位置、
"basketball" 拿球位置），逐幀算離球最近的球員 id，輸出持球者時間軸 + 標框影片。

⚠ 獨立驗證腳本，跑在 sam3_env（cu13.1）。前置同 sam3_concept_check.py。

跑法：
  python sam3_handler_check.py \
      --video input_videos/screen/screen_117.mp4 \
      --max_frames 60 \
      --out output_videos/sam3_handler_v117.mp4
"""

import os
import argparse
import subprocess
import tempfile
import shutil

import numpy as np
import cv2
import torch

from sam3.model_builder import build_sam3_video_predictor


def masks_to_bbox_center(mask):
    """binary mask → (中心x, 中心y, bbox) 或 None。"""
    m = np.asarray(mask)
    if m.ndim > 2:
        m = m.squeeze()
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return ((x1 + x2) / 2, (y1 + y2) / 2, [x1, y1, x2, y2])


def extract_objs(frame_output):
    """SAM3 單幀 outputs → {obj_id: (cx, cy, bbox)}。取不到回 {}。"""
    if not hasattr(frame_output, 'keys'):
        return {}
    if 'out_obj_ids' not in frame_output or 'out_binary_masks' not in frame_output:
        return {}
    out = {}
    for oid, m in zip(frame_output['out_obj_ids'], frame_output['out_binary_masks']):
        mm = m.cpu().numpy() if hasattr(m, 'cpu') else np.asarray(m)
        c = masks_to_bbox_center(mm)
        if c is not None:
            out[int(oid)] = c
    return out


def run_concept(predictor, video_path, text):
    """對一段影片跑一個 concept，回 {frame_idx: {obj_id: (cx,cy,bbox)}}。"""
    resp = predictor.handle_request(request=dict(
        type="start_session", resource_path=video_path))
    sid = resp["session_id"]
    predictor.handle_request(request=dict(
        type="add_prompt", session_id=sid, frame_index=0, text=text))
    per_frame = {}
    for r in predictor.handle_stream_request(request=dict(
            type="propagate_in_video", session_id=sid)):
        per_frame[r["frame_index"]] = extract_objs(r["outputs"])
    return per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_frames', type=int, default=60)
    ap.add_argument('--single_gpu', action='store_true')
    ap.add_argument('--dist_ratio', type=float, default=1.0,
                    help='球離最近球員 > dist_ratio×球員框高 → 算無人持球（球在空中）')
    args = ap.parse_args()
    dist_ratio = args.dist_ratio

    gpus = [torch.cuda.current_device()] if args.single_gpu \
        else list(range(torch.cuda.device_count()))
    print(f"GPUs: {gpus}")

    # ffmpeg 抽 jpg（container cv2 沒 FFMPEG）
    frame_dir = tempfile.mkdtemp(prefix='sam3h_frames_')
    vf = ['-frames:v', str(args.max_frames)] if args.max_frames > 0 else []
    subprocess.run(['ffmpeg', '-y', '-i', args.video, *vf,
                    '-start_number', '0', os.path.join(frame_dir, '%05d.jpg')],
                   capture_output=True, text=True)
    jpgs = sorted(f for f in os.listdir(frame_dir) if f.endswith('.jpg'))
    if not jpgs:
        print("ffmpeg 抽幀失敗"); shutil.rmtree(frame_dir, ignore_errors=True); return
    print(f"抽出 {len(jpgs)} 幀")

    predictor = build_sam3_video_predictor(gpus_to_use=gpus)

    print('concept 1: "basketball player"')
    players = run_concept(predictor, frame_dir, "basketball player")
    print('concept 2: "basketball"')
    balls = run_concept(predictor, frame_dir, "basketball")

    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    # 每幀：找球 → 離球最近的球員 = 持球者
    n = len(jpgs)
    holder_timeline = []           # 套距離門檻後的最終持球者
    nearest_raw = []               # 門檻前的最近球員（不管多遠）
    dist_ratio_frame = []          # 每幀 best_d / best_h（看門檻鬆緊）
    ball_found = 0
    over_thr = 0                   # 因球太遠被判無人持球的幀數
    for f in range(n):
        pl = players.get(f, {})
        bl = balls.get(f, {})
        if not bl:
            holder_timeline.append(None); nearest_raw.append(None)
            dist_ratio_frame.append(None)
            continue
        ball_found += 1
        bcx, bcy, bball = list(bl.values())[0]
        if not pl:
            holder_timeline.append(-1); nearest_raw.append(-1)
            dist_ratio_frame.append(None)
            continue
        best_pid, best_d, best_h = None, 1e9, 0
        for pid, (px, py, pbox) in pl.items():
            if _iou(pbox, bball) > 0.5:
                continue   # 這個「球員」其實就是球，跳過
            d = ((px - bcx) ** 2 + (py - bcy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_pid, best_h = d, pid, (pbox[3] - pbox[1])
        nearest_raw.append(best_pid if best_pid is not None else -1)
        ratio = best_d / best_h if best_h > 0 else 999
        dist_ratio_frame.append(round(ratio, 2) if best_pid is not None else None)
        # 距離門檻：球離最近球員太遠 = 球在空中飛行（傳球途中），算「無人持球」
        if best_pid is not None and ratio <= dist_ratio:
            holder_timeline.append(best_pid)
        else:
            holder_timeline.append(-1)
            if best_pid is not None:
                over_thr += 1

    # ===== 完整診斷 =====
    from collections import Counter
    print(f"\n{'='*60}")
    print(f"=== 診斷報告（dist_ratio={dist_ratio}）===")
    print(f"{'='*60}")
    print(f"[1] 追球率: {ball_found}/{n} ({ball_found/n*100:.0f}%)")
    print(f"[2] 因球太遠(>門檻)判無人持球: {over_thr} 幀 "
          f"（過多表示門檻太嚴，會切碎主持球者）")

    dist = Counter(h for h in holder_timeline if h is not None and h != -1)
    print(f"[3] 持球者分布（門檻後）: {dict(dist.most_common())}")
    raw_dist = Counter(h for h in nearest_raw if h is not None and h != -1)
    print(f"[4] 最近球員分布（門檻前，供對照）: {dict(raw_dist.most_common())}")

    # 換手點：主鏈上從 A 穩定換成 B（用門檻後、連續段 >= 3）
    def _segs(tl):
        s = []
        for h in tl:
            k = '無球' if h is None else ('無人' if h == -1 else h)
            if s and s[-1][0] == k:
                s[-1][1] += 1
            else:
                s.append([k, 1])
        return s
    segs_full = _segs(holder_timeline)
    print(f"[5] 換手點（持球者連續段 >=3 幀，看傳球接球）:")
    fpos = 0
    chain = []
    for k, c in segs_full:
        if isinstance(k, int) and c >= 3:
            chain.append(f"f{fpos}~{fpos+c-1}:#{k}({c}幀)")
        fpos += c
    print(f"    {' → '.join(chain) if chain else '（無穩定持球段）'}")

    # 每個持球者的框大小（辨真球員 vs 碎片）
    print(f"[6] 持球者 id 框大小（辨真球員 vs 碎片/誤判）:")
    id_boxes = {}
    for f in range(n):
        h = holder_timeline[f]
        if isinstance(h, int) and h >= 0 and h in players.get(f, {}):
            id_boxes.setdefault(h, []).append(players[f][h][2])
    for pid in sorted(id_boxes):
        bs = id_boxes[pid]
        aw = sum(b[2]-b[0] for b in bs)/len(bs)
        ah = sum(b[3]-b[1] for b in bs)/len(bs)
        tag = '像球員' if (ah > 50 and ah/max(aw,1) > 1.2) else '⚠可疑'
        print(f"    #{pid}: {len(bs)}幀, 框 w={aw:.0f} h={ah:.0f} (h/w={ah/max(aw,1):.1f}) {tag}")

    # 時間軸（連續段，完整 id）
    print(f"[7] 持球者時間軸（連續段）:")
    line = '  '.join(f"{'無球' if isinstance(k,str) and k=='無球' else ('無人' if k=='無人' else f'#{k}')}×{c}"
                     for k, c in segs_full if c >= 3)
    print(f"  {line}")
    print("→ 主鏈有沒有從某 pid 穩定換成另一個 pid（=抓到傳球接球）")

    # 標框影片：球員框 + 球框 + 持球者標紅
    frames = [cv2.imread(os.path.join(frame_dir, f)) for f in jpgs]
    out_frames = []
    for f in range(n):
        img = frames[f].copy()
        pl = players.get(f, {}); bl = balls.get(f, {})
        holder = holder_timeline[f]
        for pid, (cx, cy, bbox) in pl.items():
            color = (0, 0, 255) if pid == holder else (0, 200, 0)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"#{pid}" + (" BALL" if pid == holder else ""),
                        (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for _, (cx, cy, bbox) in bl.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)  # 球=橘框
        out_frames.append(img)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    draw_dir = tempfile.mkdtemp(prefix='sam3h_drawn_')
    for i, fr in enumerate(out_frames):
        cv2.imwrite(os.path.join(draw_dir, f'{i:05d}.jpg'), fr)
    subprocess.run(['ffmpeg', '-y', '-framerate', '30',
                    '-i', os.path.join(draw_dir, '%05d.jpg'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', args.out],
                   capture_output=True, text=True)
    print(f"\n標框影片 → {args.out}（紅框=持球者, 橘框=球）")
    shutil.rmtree(draw_dir, ignore_errors=True)
    shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
