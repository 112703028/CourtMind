"""
extract_pairs.py — 從 COCO 標注抽出 (handler_crop, target_crop, same_team) pairs。

訓練資料 logic:
  - 同隊 (label=1): ball_handler ↔ screener
  - 異隊 (label=0): ball_handler ↔ defender

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
from pathlib import Path

import cv2

# ── 路徑 ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
COCO_PATH    = PROJECT_ROOT / 'screen_data' / 'screen_v2.coco' / 'train' / '_annotations.coco.json'
IMG_DIR      = PROJECT_ROOT / 'screen_data' / 'screen_v2.coco' / 'train'
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

    # 設定的 OFFENSE/DEFENSE class 對應的 category id
    offense_cat_ids = set()
    for name in OFFENSE_CLASSES:
        offense_cat_ids |= name_to_ids[name]
    defense_cat_ids = set()
    for name in DEFENSE_CLASSES:
        defense_cat_ids |= name_to_ids[name]

    print(f"  Categories: {cat_map}")
    print(f"  Offense cat ids: {offense_cat_ids}")
    print(f"  Defense cat ids: {defense_cat_ids}")

    # 建立輸出資料夾
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    # ── 把標注按 image_id 分組 ─────────────────────────────────────────────────
    # ann_by_img[image_id] = [(category_name, bbox), ...]
    ann_by_img = collections.defaultdict(list)
    for ann in coco['annotations']:
        cat_id = ann['category_id']
        cat_name = cat_map[cat_id]
        # 只保留我們在意的 class
        if cat_id in offense_cat_ids or cat_id in defense_cat_ids:
            ann_by_img[ann['image_id']].append((cat_name, ann['bbox']))

    # 建 image_id → file_name 對應
    img_info = {img['id']: img['file_name'] for img in coco['images']}

    # ── 抽 pairs ───────────────────────────────────────────────────────────────
    pairs = []  # [(crop1_path, crop2_path, label), ...]
    n_pos = 0
    n_neg = 0
    n_no_handler = 0
    n_load_fail = 0
    n_crop_fail = 0

    for img_id, anns in ann_by_img.items():
        # 找 handler（pair 的 anchor）
        handlers = [(c, b) for c, b in anns if c == 'ball_handler']
        if not handlers:
            n_no_handler += 1
            continue

        # 讀圖
        file_name = img_info[img_id]
        img_path = IMG_DIR / file_name
        img = cv2.imread(str(img_path))
        if img is None:
            n_load_fail += 1
            continue

        # 對每個 handler（通常只有 1 個）開始抽 pairs
        for h_idx, (_, h_bbox) in enumerate(handlers):
            h_xyxy = scale_bbox(h_bbox, CROP_SCALE)
            h_crop = crop_image(img, h_xyxy)
            if h_crop is None:
                n_crop_fail += 1
                continue

            # 存 handler crop
            h_crop_name = f'{img_id}_handler_{h_idx}.jpg'
            cv2.imwrite(str(CROP_DIR / h_crop_name), h_crop)

            # 找所有可能的「另一個 player」
            for cls_name, t_bbox in anns:
                if cls_name == 'ball_handler':
                    continue  # 不跟自己配對

                t_xyxy = scale_bbox(t_bbox, CROP_SCALE)
                t_crop = crop_image(img, t_xyxy)
                if t_crop is None:
                    n_crop_fail += 1
                    continue

                # 決定 label
                if cls_name in OFFENSE_CLASSES:
                    label = 1
                    n_pos += 1
                elif cls_name in DEFENSE_CLASSES:
                    label = 0
                    n_neg += 1
                else:
                    continue

                t_crop_name = f'{img_id}_{cls_name}_{len(pairs)}.jpg'
                cv2.imwrite(str(CROP_DIR / t_crop_name), t_crop)

                pairs.append({
                    'crop1': h_crop_name,
                    'crop2': t_crop_name,
                    'label': label,
                    'image_id': img_id,
                    'crop2_class': cls_name,
                })

    # ── 存 pairs metadata ──────────────────────────────────────────────────────
    pairs_path = OUT_DIR / 'pairs.json'
    with open(pairs_path, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"\n── Stats ──")
    print(f"  Total pairs:        {len(pairs)}")
    print(f"    Positive (同隊):  {n_pos}")
    print(f"    Negative (異隊):  {n_neg}")
    print(f"  Skipped:")
    print(f"    No handler:       {n_no_handler}")
    print(f"    Image load fail:  {n_load_fail}")
    print(f"    Crop fail:        {n_crop_fail}")
    print(f"\n  Crops saved → {CROP_DIR}")
    print(f"  Pairs metadata → {pairs_path}")


if __name__ == '__main__':
    main()
