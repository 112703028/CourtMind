"""
test_referee_separability.py — 測「SAM3 乾淨 crop 下，裁判是不是 embedding outlier」。

背景：v129 SAM3 追到 12 個 obj = 10 真球員 + 替補#11 + 裁判#6。想從乾淨 crop 裡
分出裁判。早期用 SAM2 髒資料測 embedding 距離「單峰糊」，這次用 SAM3 乾淨資料重測。

兩種方法：
  A. TeamSiamese embedding → KMeans(k=2) 找兩隊中心 → 每個 obj 到最近中心的距離。
     裁判服「第三色」→ 應該離兩隊都遠 = outlier。
  B. sports.TeamClassifier（SigLIP+UMAP+KMeans，roboflow 用的）→ 同樣看 outlier。

⚠ 替補#11 穿同隊球衣，兩法都分不出（顏色跟球員一樣）→ 本測試只驗「能不能分裁判」。
   替補用出現率（#11 只 40 幀）另外排。

跑在 basketball_analysis container（有 team_siamese.pt + sports 套件 + 影片）。
用法：
  python test_referee_separability.py \
      --tracks stubs/screen_129_sam3_tracks.pkl \
      --video input_videos/screen/screen_129.mp4 \
      --referee 6 --sub 11    # 已知裁判/替補 id（對照用）
"""
import os
import argparse
import subprocess
import tempfile
import shutil
import pickle
from pathlib import Path

import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent / 'team_classifier'))


def extract_crops_per_obj(tracks, frame_dir, jpgs, max_crops=20):
    """每個 obj 均勻抽 max_crops 幀 crop（BGR numpy，縮 0.5 取球衣中央）。"""
    frames_of = {}
    for f_idx, ft in enumerate(tracks):
        for pid, info in ft.items():
            if info.get('bbox') is not None:
                frames_of.setdefault(pid, []).append(f_idx)
    crops_by_obj = {}
    for pid, fidxs in frames_of.items():
        if len(fidxs) > max_crops:
            pick = np.linspace(0, len(fidxs) - 1, max_crops).astype(int)
            fidxs = [fidxs[i] for i in pick]
        crops = []
        for f_idx in fidxs:
            bbox = tracks[f_idx][pid]['bbox']
            img = cv2.imread(os.path.join(frame_dir, jpgs[f_idx]))
            if img is None:
                continue
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            # 縮 0.5 取中央（聚焦球衣）
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            nx1, ny1 = int(cx - w * 0.25), int(cy - h * 0.25)
            nx2, ny2 = int(cx + w * 0.25), int(cy + h * 0.25)
            H, W = img.shape[:2]
            nx1, ny1 = max(0, nx1), max(0, ny1)
            nx2, ny2 = min(W, nx2), min(H, ny2)
            if nx2 - nx1 < 5 or ny2 - ny1 < 5:
                continue
            crops.append(img[ny1:ny2, nx1:nx2])
        if len(crops) >= 3:
            crops_by_obj[pid] = crops
    return crops_by_obj


def method_siamese(crops_by_obj, referee, sub):
    """方法A：TeamSiamese embedding → 到兩隊中心距離，看裁判是不是 outlier。"""
    from model import TeamSiamese
    from PIL import Image
    from torchvision import transforms
    from sklearn.cluster import KMeans

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(str(PROJECT_ROOT / 'models' / 'team_siamese.pt'), map_location=device)
    model = TeamSiamese().to(device).eval()
    model.load_state_dict(ckpt['model'])
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    embs = {}
    for pid, crops in crops_by_obj.items():
        tensors = []
        for c in crops:
            pil = Image.fromarray(c[:, :, ::-1])  # BGR→RGB
            tensors.append(tf(pil))
        with torch.no_grad():
            e = model.encode(torch.stack(tensors).to(device)).cpu().numpy()
        embs[pid] = e.mean(axis=0)

    ids = sorted(embs)
    X = np.stack([embs[i] for i in ids])
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    centers = km.cluster_centers_
    print("\n=== 方法A：TeamSiamese embedding 到兩隊中心距離 ===")
    print(f"{'id':>3} {'到中心0':>8} {'到中心1':>8} {'最近':>8}  標記")
    rows = []
    for i, pid in enumerate(ids):
        d0 = float(np.linalg.norm(embs[pid] - centers[0]))
        d1 = float(np.linalg.norm(embs[pid] - centers[1]))
        dn = min(d0, d1)
        rows.append((pid, dn))
        tag = '← 裁判(GT)' if pid == referee else ('← 替補(GT)' if pid == sub else '')
        print(f"{pid:>3} {d0:>8.2f} {d1:>8.2f} {dn:>8.2f}  {tag}")
    rows.sort(key=lambda x: -x[1])
    print(f"\n離最近中心最遠排名（outlier 在前，裁判應在前）:")
    for pid, dn in rows:
        tag = '裁判' if pid == referee else ('替補' if pid == sub else '球員')
        print(f"  #{pid}: {dn:.2f}  [{tag}]")


def method_sports(crops_by_obj, referee, sub):
    """方法B：sports.TeamClassifier（roboflow 用的）看 outlier。"""
    try:
        from sports import TeamClassifier
    except Exception as ex:
        print(f"\n=== 方法B：sports.TeamClassifier — 載入失敗，跳過 ({ex}) ===")
        return
    print("\n=== 方法B：sports.TeamClassifier（SigLIP+UMAP+KMeans）===")
    # 用所有 crops fit，再看每個 obj 多數決分到哪隊 + 一致度
    all_crops, owner = [], []
    for pid, crops in crops_by_obj.items():
        for c in crops:
            all_crops.append(c); owner.append(pid)
    clf = TeamClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
    clf.fit(all_crops)
    preds = clf.predict(all_crops)
    import collections as _c
    by_obj = _c.defaultdict(_c.Counter)
    for pid, t in zip(owner, preds):
        by_obj[pid][int(t)] += 1
    print(f"{'id':>3} {'team0票':>7} {'team1票':>7} {'一致度':>6}  標記")
    for pid in sorted(by_obj):
        c = by_obj[pid]; total = sum(c.values()); top = c.most_common(1)[0][1]
        tag = '← 裁判(GT)' if pid == referee else ('← 替補(GT)' if pid == sub else '')
        print(f"{pid:>3} {c.get(0,0):>7} {c.get(1,0):>7} {top/total:>5.0%}  {tag}")
    print("（若裁判一致度明顯低 = 服裝不屬任一隊，搖擺 → 可當裁判訊號）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--referee', type=int, default=None, help='已知裁判 id（對照）')
    ap.add_argument('--sub', type=int, default=None, help='已知替補 id（對照）')
    args = ap.parse_args()

    with open(args.tracks, 'rb') as fp:
        d = pickle.load(fp)
    tracks = d['player_tracks']

    frame_dir = tempfile.mkdtemp(prefix='reftest_')
    subprocess.run(['ffmpeg', '-y', '-i', args.video, '-start_number', '0',
                    os.path.join(frame_dir, '%05d.jpg')], capture_output=True, text=True)
    jpgs = sorted(f for f in os.listdir(frame_dir) if f.endswith('.jpg'))
    print(f"抽 {len(jpgs)} 幀")

    crops_by_obj = extract_crops_per_obj(tracks, frame_dir, jpgs)
    print(f"抽到 crop 的 obj: {sorted(crops_by_obj)}  "
          f"(裁判GT=#{args.referee}, 替補GT=#{args.sub})")

    method_siamese(crops_by_obj, args.referee, args.sub)
    method_sports(crops_by_obj, args.referee, args.sub)

    shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
