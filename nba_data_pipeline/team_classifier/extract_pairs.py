"""
extract_pairs.py — 從 COCO 標注抽出 (handler_crop, target_crop, same_team) pairs。

訓練資料 logic（同圖任兩人配對，不限 handler 當 anchor）:
  - 同隊 (label=1): offense↔offense（handler/screener 互相）、defense↔defense
  - 異隊 (label=0): offense↔defender
  → 同樣標注量榨出 2-3 倍 pair，且 crop 每個實例只存一次

輸出:
  team_classifier_data/
    pairs.json                # metadata: [(crop1_path, crop2_path, label), ...]
    crops/
      <image_id>_<class>_<idx>.jpg

執行:
  python nba_data_pipeline/team_classifier/extract_pairs.py
"""

import json
import collections
import itertools
from pathlib import Path

import cv2

# ── 路徑 ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
COCO_PATH    = PROJECT_ROOT / 'screen_data' / 'screen_full_defender' / 'train' / '_annotations.coco.json'
IMG_DIR      = PROJECT_ROOT / 'screen_data' / 'screen_full_defender' / 'train'
OUT_DIR      = PROJECT_ROOT / 'team_classifier_data'
CROP_DIR     = OUT_DIR / 'crops'

# ── 設定 ──────────────────────────────────────────────────────────────────────

# 哪些 COCO category 算「offense」（跟 handler 同隊）
OFFENSE_CLASSES = ['ball_handler', 'screener']
# 哪些算「defense」（跟 handler 異隊）
DEFENSE_CLASSES = ['defender']

# Crop 設定：縮小到 bbox 中央 N% 區域（聚焦球衣，跟 team_assigner 一致）
CROP_SCALE = 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────

def scale_bbox(bbox, factor):
    """COCO bbox [x, y, w, h] → 縮到中央 factor 區域，回 [x1, y1, x2, y2]。"""
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * factor, h * factor
    return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]


def crop_image(img, xyxy):
    """從 numpy image 切出 bbox（含 boundary clamp）。"""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return img[y1:y2, x1:x2]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading COCO: {COCO_PATH}")
    with open(COCO_PATH, encoding='utf-8') as f:
        coco = json.load(f)

    # 建 category id → name 對應
    cat_map = {c['id']: c['name'] for c in coco['categories']}
    # 建反向：name → id (集合，因為 'screen' 有兩個 id)
    name_to_ids = collections.defaultdict(set)
    for cat_id, name in cat_map.items():
        name_to_ids[name].add(cat_id)

    # 設定的 OFFENSE/DEFENSE class 對應的 category id → 隊伍標記 'O' / 'D'
    cat_team = {}   # cat_id -> 'O' | 'D'
    for name in OFFENSE_CLASSES:
        for cid in name_to_ids[name]:
            cat_team[cid] = 'O'
    for name in DEFENSE_CLASSES:
        for cid in name_to_ids[name]:
            cat_team[cid] = 'D'

    print(f"  Categories: {cat_map}")
    print(f"  Offense cat ids: {[c for c,t in cat_team.items() if t=='O']}")
    print(f"  Defense cat ids: {[c for c,t in cat_team.items() if t=='D']}")

    # 建立輸出資料夾
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    # ── 把標注按 image_id 分組 ─────────────────────────────────────────────────
    # ann_by_img[image_id] = [(team, cat_name, bbox), ...]
    ann_by_img = collections.defaultdict(list)
    for ann in coco['annotations']:
        cat_id = ann['category_id']
        team = cat_team.get(cat_id)
        if team is None:
            continue   # 只保留 offense / defense class（略過 ball, others, screen...）
        ann_by_img[ann['image_id']].append((team, cat_map[cat_id], ann['bbox']))

    # 建 image_id → file_name 對應
    img_info = {img['id']: img['file_name'] for img in coco['images']}

    # ── 逐圖：先切好每個實例的 crop（只存一次），再產生同圖所有配對 ──────────────
    pairs = []
    n_pos = 0
    n_neg = 0
    n_too_few = 0     # 相關標注 < 2，無法配對
    n_load_fail = 0
    n_crop_fail = 0
    # 配對類型統計（debug 用）
    pair_type_count = collections.Counter()

    for img_id, anns in ann_by_img.items():
        # 至少要 2 個相關實例才能配對
        if len(anns) < 2:
            n_too_few += 1
            continue

        # 讀圖
        file_name = img_info[img_id]
        img = cv2.imread(str(IMG_DIR / file_name))
        if img is None:
            n_load_fail += 1
            continue

        # 切每個實例的 crop（同圖同實例只存一次）
        # instances: [(crop_name, team, cat_name), ...]
        instances = []
        for idx, (team, cat_name, bbox) in enumerate(anns):
            crop = crop_image(img, scale_bbox(bbox, CROP_SCALE))
            if crop is None:
                n_crop_fail += 1
                continue
            crop_name = f'{img_id}_{cat_name}_{idx}.jpg'
            cv2.imwrite(str(CROP_DIR / crop_name), crop)
            instances.append((crop_name, team, cat_name))

        # 同圖任兩人配對（免費榨出 O↔O、D↔D、O↔D）
        for (c1, t1, cls1), (c2, t2, cls2) in itertools.combinations(instances, 2):
            label = 1 if t1 == t2 else 0   # 同隊=1，異隊=0
            if label == 1:
                n_pos += 1
            else:
                n_neg += 1
            # 記錄配對類型（O-O / D-D / O-D），兩端排序讓 key 一致
            ptype = '-'.join(sorted([t1, t2]))
            pair_type_count[ptype] += 1

            pairs.append({
                'crop1': c1,
                'crop2': c2,
                'label': label,
                'image_id': img_id,
                'pair_type': f'{cls1}+{cls2}',
            })

    # ── 存 pairs metadata ──────────────────────────────────────────────────────
    pairs_path = OUT_DIR / 'pairs.json'
    with open(pairs_path, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"\n── Stats ──")
    print(f"  Total pairs:        {len(pairs)}")
    print(f"    Positive (同隊):  {n_pos}")
    print(f"    Negative (異隊):  {n_neg}")
    print(f"  Pair types:")
    print(f"    O-O (offense 互相, pos):  {pair_type_count.get('O-O', 0)}")
    print(f"    D-D (defense 互相, pos):  {pair_type_count.get('D-D', 0)}")
    print(f"    D-O (異隊, neg):          {pair_type_count.get('D-O', 0)}")
    print(f"  Skipped:")
    print(f"    < 2 relevant anns:  {n_too_few}")
    print(f"    Image load fail:    {n_load_fail}")
    print(f"    Crop fail:          {n_crop_fail}")
    print(f"\n  Crops saved → {CROP_DIR}")
    print(f"  Pairs metadata → {pairs_path}")


if __name__ == '__main__':
    main()
