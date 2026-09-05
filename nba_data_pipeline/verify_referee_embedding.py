"""
verify_referee_embedding.py — 驗證「裁判在 TeamSiamese embedding 空間是不是 outlier」。

假設：裁判服裝跟兩隊都不同（你確認的辨識依據）→ 裁判的球衣 crop encode 出來的
embedding 應該離「兩隊各自的中心」都比球員遠。若成立，就能用「到最近隊中心距離 >
門檻」這個無監督規則抹掉裁判，不需要重新標記。

做法（每支影片）：
  1. read_video → run_sam2_tracks（拿每個 obj 的像素 bbox，跟正式 pipeline 同一套）
  2. verify_sam2_with_yolo 清 drift（跟正式 pipeline 一致）
  3. 對每個 obj，抽最多 N 幀 crop → TeamSiamese.encode → 平均成該 obj 的 embedding
  4. 對所有 obj 的平均 embedding 跑 KMeans(k=2) 找兩隊中心
  5. 算每個 obj 到「最近隊中心」的距離，印出排名（由遠到近）

輸出：每支印一張表（obj_id | 出現幀 | 到最近隊中心距離 | 到兩中心距離），
並存 CSV。你對照影片認出哪些 obj 是裁判，看裁判是不是都排在「距離大」那端。

⚠ SAM2 每次跑 id 會重新分配，所以這裡的 obj_id 跟你之前標的不一定對應 →
跑完請對照影片重新認裁判 id。

跑法（DGX/Spark，需要 GPU + sam2 + team_siamese.pt）：
  python -X utf8 nba_data_pipeline/verify_referee_embedding.py \
      --videos 116 117 118 119 120 121 125 127 128 129 130 \
      --out_csv output_videos/referee_embed_report.csv
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from utils import read_video, save_video
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor

from inference import (run_sam2_tracks, verify_sam2_with_yolo, get_video_fps,
                       SAM2_CHECKPOINT_DEFAULT, SAM2_CONFIG_DEFAULT,
                       PLAYER_MODEL_DEFAULT)
from team_classifier.model import TeamSiamese
from team_classifier.inference import _crop_pil, _val_transform

from sklearn.cluster import KMeans


def obj_embeddings(frames, player_tracks, model, device, max_crops=20):
    """每個 obj 抽最多 max_crops 幀 crop → encode → 平均。回 {obj_id: (512,) np}。"""
    # 收集每個 obj 有 bbox 的幀
    frames_of = defaultdict(list)
    for f_idx, ft in enumerate(player_tracks):
        for pid, info in ft.items():
            if info.get('bbox') is not None and not info.get('drifted'):
                frames_of[pid].append(f_idx)

    result = {}
    for pid, fidxs in frames_of.items():
        # 均勻取樣 max_crops 幀
        if len(fidxs) > max_crops:
            pick = np.linspace(0, len(fidxs) - 1, max_crops).astype(int)
            fidxs = [fidxs[i] for i in pick]
        crops = []
        for f_idx in fidxs:
            bbox = player_tracks[f_idx][pid]['bbox']
            c = _crop_pil(frames[f_idx], bbox)
            if c is not None:
                crops.append(c)
        if len(crops) < 3:
            continue
        with torch.no_grad():
            t = torch.stack([_val_transform(c) for c in crops]).to(device)
            emb = model.encode(t).cpu().numpy()   # (n, 512)
        result[pid] = (emb.mean(axis=0), len(crops))
    return result


def draw_labeled_video(frames, tracks, dist_by_id, out_path, fps, ref_thr):
    """畫每個 obj 的框 + #id + dist_nearest。距離 >= ref_thr（疑似裁判）畫紅，否則綠。
    讓你對照畫面認出哪些 obj_id 是裁判，並看脚本判斷準不準。"""
    out = []
    for f_idx, frame in enumerate(frames):
        img = frame.copy()
        for pid, info in tracks[f_idx].items():
            bbox = info.get('bbox')
            if bbox is None or info.get('drifted'):
                continue
            d = dist_by_id.get(pid)
            color = (0, 0, 255) if (d is not None and d >= ref_thr) else (0, 200, 0)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"#{pid}" + (f" d={d:.0f}" if d is not None else "")
            cv2.putText(img, label, (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out.append(img)
    save_video(out, out_path, fps=fps)
    print(f"  標框影片 → {out_path}  (紅框=dist>={ref_thr} 疑似裁判)")


def analyze_video(video_num, player_model, predictor, siamese, device,
                  save_video_dir=None, ref_thr=12.0):
    """回 list[dict]：每個 obj 的 id / 出現幀數 / 到兩隊中心距離 / 到最近中心距離。
    save_video_dir 給定時，額外輸出一支標框影片供人工對照認裁判。"""
    vp = f'input_videos/screen/screen_{video_num}.mp4'
    if not os.path.exists(vp):
        print(f"  v{video_num}: 影片不存在 {vp}")
        return []
    frames = read_video(vp)
    if not frames:
        print(f"  v{video_num}: 讀不到影片")
        return []

    tracks = run_sam2_tracks(frames, player_model, predictor, device=str(device))
    tracks = verify_sam2_with_yolo(frames, tracks, player_model)

    embs = obj_embeddings(frames, tracks, siamese.model, device)
    if len(embs) < 3:
        print(f"  v{video_num}: 有效 obj 太少（{len(embs)}）")
        return []

    ids = sorted(embs)
    X = np.stack([embs[i][0] for i in ids])
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    centers = km.cluster_centers_

    rows = []
    dist_by_id = {}
    for i, pid in enumerate(ids):
        v = embs[pid][0]
        d0 = float(np.linalg.norm(v - centers[0]))
        d1 = float(np.linalg.norm(v - centers[1]))
        dn = min(d0, d1)
        dist_by_id[pid] = dn
        rows.append({'video': video_num, 'obj_id': pid,
                     'n_crops': embs[pid][1],
                     'dist_c0': round(d0, 2), 'dist_c1': round(d1, 2),
                     'dist_nearest': round(dn, 2)})
    rows.sort(key=lambda r: -r['dist_nearest'])   # 離最近中心最遠的排前面（最可能裁判）

    if save_video_dir:
        os.makedirs(save_video_dir, exist_ok=True)
        fps = get_video_fps(vp)
        out_path = os.path.join(save_video_dir, f'referee_check_v{video_num}.mp4')
        draw_labeled_video(frames, tracks, dist_by_id, out_path, fps, ref_thr)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', nargs='+', type=int, required=True,
                    help='影片編號，如 116 117 ...')
    ap.add_argument('--siamese', default=str(PROJECT_ROOT / 'models' / 'team_siamese.pt'))
    ap.add_argument('--player_model', default=PLAYER_MODEL_DEFAULT)
    ap.add_argument('--sam2_ckpt', default=SAM2_CHECKPOINT_DEFAULT)
    ap.add_argument('--sam2_config', default=SAM2_CONFIG_DEFAULT)
    ap.add_argument('--out_csv', default='output_videos/referee_embed_report.csv')
    ap.add_argument('--save_video_dir', default='output_videos',
                    help='輸出標框影片的資料夾（每支一個 referee_check_vNNN.mp4）；'
                         '設空字串關閉')
    ap.add_argument('--ref_thr', type=float, default=12.0,
                    help='標框影片裡「疑似裁判」的 dist_nearest 門檻（畫紅框）')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    player_model = YOLO(args.player_model)
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_ckpt, device=str(device))

    # 直接載 TeamSiamese（借 inference 的 loader）
    from team_classifier.inference import TeamAssignerSiamese
    siamese = TeamAssignerSiamese(args.siamese, device=str(device))

    all_rows = []
    save_dir = args.save_video_dir or None
    for v in args.videos:
        print(f"\n===== v{v} =====")
        rows = analyze_video(v, player_model, predictor, siamese, device,
                             save_video_dir=save_dir, ref_thr=args.ref_thr)
        if not rows:
            continue
        # 印表：由遠到近（最可能是裁判的在最上面）
        print(f"  {'obj':>4} {'crops':>5} {'dist→c0':>8} {'dist→c1':>8} {'nearest':>8}")
        for r in rows:
            print(f"  {r['obj_id']:>4} {r['n_crops']:>5} {r['dist_c0']:>8.2f} "
                  f"{r['dist_c1']:>8.2f} {r['dist_nearest']:>8.2f}")
        all_rows.extend(rows)

    if all_rows:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n報告存 → {args.out_csv}")
        print("對照影片認出裁判 obj_id，看裁判是不是都排在各支的『nearest 距離大』那端。")


if __name__ == '__main__':
    main()
