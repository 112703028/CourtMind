"""
remap_coco.py

為 split_clips.py 產生的子 clip（screen_167~184）
從原始 clip 的 COCO 標記複製對應的 annotations。
輸出：screen_data/screen.coco/train/_annotations.coco.json（in-place 更新）
"""

import json, re
from pathlib import Path

COCO_PATH = Path('screen_data/screen_v2.coco/train/_annotations.coco.json')

# split_clips.py 的切割資訊：{新clip_id: (原clip_id, lo_frame, hi_frame)}
# 順序與 split_clips.py 相同（每個 range 依序分配新 ID）
REMAP = {
    167: (166, 35, 45),
    168: (166, 24, 29),
    169: (166, 10, 17),
    170: (166,  2,  8),
    171: (165, 21, 40),
    172: (165,  2,  9),
    173: (160, 10, 20),
    174: (160,  0,  9),
    175: (158, 10, 19),
    176: (158,  2,  8),
    177: (154, 43, 49),
    178: (154, 10, 21),
    179: (148, 21, 37),
    180: (148,  5, 15),
    181: (140, 23, 34),
    182: (140,  0, 18),
    183: (139,  6, 14),
    184: (139, 18, 31),
}

with open(COCO_PATH) as f:
    coco = json.load(f)

# 建立 (clip_id, frame_num) -> image_id 的查詢表
img_lookup = {}
for img in coco['images']:
    m = re.match(r'screen_(\d+)_(\d{4})', img['file_name'])
    if m:
        img_lookup[(int(m.group(1)), int(m.group(2)))] = img['id']

# 建立 image_id -> annotations 的查詢表
ann_by_img = {}
for ann in coco['annotations']:
    ann_by_img.setdefault(ann['image_id'], []).append(ann)

max_img_id = max(img['id'] for img in coco['images'])
max_ann_id = max(ann['id'] for ann in coco['annotations'])

new_images = []
new_anns = []
skipped = []

for new_clip_id, (orig_clip_id, lo, hi) in REMAP.items():
    new_frame_idx = 0
    for orig_frame in range(lo, hi + 1):
        key = (orig_clip_id, orig_frame)
        if key not in img_lookup:
            new_frame_idx += 1
            continue

        orig_img_id = img_lookup[key]
        max_img_id += 1
        new_img_id = max_img_id

        # 建立新 image entry
        orig_img = next(i for i in coco['images'] if i['id'] == orig_img_id)
        new_img = dict(orig_img)
        new_img['id'] = new_img_id
        new_img['file_name'] = f'screen_{new_clip_id}_{new_frame_idx:04d}.jpg'
        if 'extra' in new_img:
            new_img['extra'] = {'name': f'screen_{new_clip_id}_{new_frame_idx:04d}.jpg'}
        new_images.append(new_img)

        # 複製 annotations
        for ann in ann_by_img.get(orig_img_id, []):
            max_ann_id += 1
            new_ann = dict(ann)
            new_ann['id'] = max_ann_id
            new_ann['image_id'] = new_img_id
            new_anns.append(new_ann)

        new_frame_idx += 1

    annotated = sum(1 for f in range(lo, hi+1) if (orig_clip_id, f) in img_lookup)
    print(f"  screen_{new_clip_id} ← screen_{orig_clip_id} [{lo}-{hi}]: "
          f"{annotated}/{hi-lo+1} frames had COCO annotations")

coco['images'].extend(new_images)
coco['annotations'].extend(new_anns)

with open(COCO_PATH, 'w') as f:
    json.dump(coco, f)

print(f"\nDone. Added {len(new_images)} images, {len(new_anns)} annotations to {COCO_PATH}")
