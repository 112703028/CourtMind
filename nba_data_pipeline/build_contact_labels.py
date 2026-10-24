"""
build_contact_labels.py — 把人工標的接觸幀（contact_labels.csv）整理成 ScreenNet 重訓用的標籤。

輸入：nba_data_pipeline/contact_labels.csv（使用者看 output_videos/framenum/ 印幀號影片標的）
  欄位 clip, fps, 總幀數, screener, contact_frame, contact(yes/no), note
  - contact_frame：原片幀號（各片原生 fps）；同一列可寫多個「48, 131」= 同一人擋兩次
  - contact：t/f（f = slip 假擋、防守者繞過去，沒有真的撞到）；多個時「t, f」對應
  - note：「N frame 以前 cut 掉」→ 該片從第 N 幀起才有效（標的幀號是 cut 之前的編號）

處理：
  1. 每個掩護 → 抽樣索引 idx = round(frame / (fps / 5))（訓練影格是 5fps 抽樣）
  2. 對到 COCO 標註裡的 screener：取 idx 附近（±1）有標註的影格，
     選腳底離 defender 最近的 screener 框（沒 defender 就離 handler 最近）；
     同一支多個掩護在同一時間（同時雙擋）就分給不同的 screener
  3. screen_full_defender 的座標是 Roboflow 縮成 512x512 的，要依原圖尺寸縮放回去

輸出：nba_data_pipeline/contact_labels.json
  {clip: {fps, n_frames, valid_start, events:[{frame, idx, contact, screener_box, ...}]}}
  另畫確認圖：有 2 個以上掩護的 clip → output_videos/contact_confirm/

用法：
  python nba_data_pipeline/build_contact_labels.py
"""
import re
import csv
import json
import collections
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).parent.parent
CSV_PATH = Path(__file__).parent / 'contact_labels.csv'
OUT_JSON = Path(__file__).parent / 'contact_labels.json'
COCO_V2 = ROOT / 'screen_data' / 'screen_v2.coco' / 'train' / '_annotations.coco.json'
COCO_FD = ROOT / 'screen_data' / 'screen_full_defender' / 'train' / '_annotations.coco.json'
FRAMES_DIR = ROOT / 'input_videos' / 'screen_frames'
CONFIRM_DIR = ROOT / 'output_videos' / 'contact_confirm'
SAMPLE_FPS = 5.0
OVERRIDES = Path(__file__).parent / 'contact_overrides.json'   # 使用者看確認圖後的修正
PENDING = {'screen_37'}   # 待使用者確認標註意思（cut 沒寫從第幾幀開始）


# ── 讀人工標註 ──────────────────────────────────────────────────────────────────

def parse_csv():
    rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8-sig')))
    clips = {}
    for r in rows:
        clip = r['clip']
        c = clips.setdefault(clip, {'fps': float(r['fps']), 'n_frames': int(r['總幀數']),
                                    'valid_start': 0, 'events': [], 'notes': []})
        note = (r.get('note') or '').strip()
        if note:
            c['notes'].append(note)
            m = re.search(r'(\d+)\s*frame\s*(以前|前)', note)
            if m:
                c['valid_start'] = max(c['valid_start'], int(m.group(1)))
        frames = [x.strip() for x in (r.get('contact_frame') or '').split(',') if x.strip()]
        flags = [x.strip().lower() for x in (r.get('contact(yes/no)') or '').split(',') if x.strip()]
        for i, fr in enumerate(frames):
            flag = flags[i] if i < len(flags) else (flags[-1] if flags else 't')
            c['events'].append({'frame': int(float(fr)),
                                'contact': flag not in ('f', 'no', 'n', 'false')})
    return clips


# ── 讀 COCO（座標換回原圖尺寸）───────────────────────────────────────────────────

def load_coco(path):
    """→ {clip: {idx: {category: [bbox xywh（原圖座標）]}}}"""
    c = json.load(open(path, encoding='utf-8'))
    cats = {x['id']: x['name'] for x in c['categories']}
    imgs = {i['id']: i for i in c['images']}
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    size_cache = {}
    for a in c['annotations']:
        im = imgs[a['image_id']]
        m = re.match(r'(screen_\d+)_(\d+)_', im['file_name'])
        if not m:
            continue
        clip, idx = m.group(1), int(m.group(2))
        if clip not in size_cache:
            f = cv2.imread(str(FRAMES_DIR / clip / f'{clip}_{idx:04d}.jpg'))
            size_cache[clip] = (f.shape[1], f.shape[0]) if f is not None else (im['width'], im['height'])
        W, H = size_cache[clip]
        sx, sy = W / im['width'], H / im['height']
        x, y, w, h = [float(v) for v in a['bbox']]
        out[clip][idx][cats[a['category_id']]].append([x * sx, y * sy, w * sx, h * sy])
    return out


def foot(b):
    return np.array([b[0] + b[2] / 2, b[1] + b[3]])


def nearest_annotated(frames, idx, max_off=1):
    for off in (0, -1, 1, -2, 2)[: 1 + 2 * max_off]:
        if idx + off in frames and frames[idx + off].get('screener'):
            return idx + off
    return None


def screener_candidates(fr):
    """每個 screener 框的接觸分數（越小越像正在擋）：離最近 defender（或 handler）的距離 / 平均身高。"""
    targets = fr.get('defender') or fr.get('ball_handler') or []
    cands = []
    for b in fr.get('screener', []):
        if targets:
            d = min(np.linalg.norm(foot(b) - foot(t)) / max(1.0, (b[3] + t[3]) / 2) for t in targets)
        else:
            d = float('inf')
        cands.append((d, b))
    cands.sort(key=lambda x: x[0])
    return cands


# ── 主流程 ──────────────────────────────────────────────────────────────────────

COVERAGE_CSV = Path(__file__).parent / 'coverage_labels.csv'
NOT_HANDLER_SCREEN = {'無球檔', '沒有檔', '沒有掩護'}


def load_not_handler_screens():
    """使用者在 coverage_labels.csv 標成「不是幫持球者擋」的掩護 → {(clip, 原片幀號): 原因}。
    ScreenNet 只學「誰在幫持球者擋」，無球掩護（含西班牙擋拆的背擋）、其實沒有掩護、
    重複標記都要從標記裡拿掉，不然等於教它錯的例子。"""
    if not COVERAGE_CSV.exists():
        return {}
    rows = list(csv.DictReader(open(COVERAGE_CSV, encoding='utf-8-sig')))
    col = [k for k in rows[0] if k.startswith('防守方式')][0]
    out = {}
    for r in rows:
        v, note = r[col].strip(), r.get('備註', '')
        if v in NOT_HANDLER_SCREEN:
            out[(r['clip'], int(r['接觸幀(原片)']))] = v
        elif '同一個掩護' in note:
            out[(r['clip'], int(r['接觸幀(原片)']))] = '重複標記'
    return out


def build():
    labels = parse_csv()
    ovr = json.load(open(OVERRIDES, encoding='utf-8')) if OVERRIDES.exists() else {}
    drop = load_not_handler_screens()
    v2, fd = load_coco(COCO_V2), load_coco(COCO_FD)
    out = {}
    for clip, c in labels.items():
        frames = fd[clip] if clip in fd else v2[clip]
        step = c['fps'] / SAMPLE_FPS
        o = ovr.get(clip, {})
        raw_events = o['replace_events'] if 'replace_events' in o else c['events']
        # add_events：CSV 沒標到、使用者事後確認的真掩護，直接指定 5fps 追蹤的軌跡編號（tid）
        raw_events = list(raw_events) + list(o.get('add_events', []))
        letter_map = o.get('letters', {})
        used_at = collections.defaultdict(set)    # idx → 已分配過的 screener 框（同時雙擋要分給不同人）
        events = []
        for ev in sorted(raw_events, key=lambda e: e['frame']):
            idx = int(round(ev['frame'] / step))
            ai = nearest_annotated(frames, idx)
            base = {'frame': ev['frame'], 'contact': ev['contact'], 'idx': idx, 'coco_idx': ai,
                    'screener_box': None, 'assign_score': None, 'n_screeners': 0,
                    'source': 'auto'}
            if ai is not None:
                base['n_screeners'] = len(frames[ai].get('screener', []))
            if ev.get('tid') is not None:
                # 使用者直接指定了軌跡編號（stubs/train_sam3 的 5fps 追蹤 id），不用框對
                base['tid_fixed'] = ev['tid']
                base['source'] = 'user_tid'
                events.append(base)
                continue
            letters = ev.get('letters') or letter_map.get(str(ev['frame']))
            if letters:
                # 使用者確認過：照字母取確認圖上那格的 screener 框；多個字母 = 同一幀多人掩護
                for L in letters:
                    e = dict(base, source='user')
                    if L == 'unboxed' or ai is None:
                        e['screener_unboxed'] = True     # 那一刻沒被框到 → 重訓時用軌跡判斷
                    else:
                        e['screener_box'] = frames[ai]['screener']['ABCDE'.index(L)]
                        used_at[ai].add(tuple(round(v) for v in e['screener_box']))
                    events.append(e)
                continue
            e = base
            if ai is not None:
                for d, b in screener_candidates(frames[ai]):
                    key = tuple(round(v) for v in b)
                    if key not in used_at[ai]:
                        used_at[ai].add(key)
                        e['screener_box'] = b
                        e['assign_score'] = None if d == float('inf') else round(d, 3)
                        break
            events.append(e)
        excluded = [dict(e, reason=drop[(clip, e['frame'])]) for e in events if (clip, e['frame']) in drop]
        events = [e for e in events if (clip, e['frame']) not in drop]
        out[clip] = {'fps': c['fps'], 'n_frames': c['n_frames'], 'valid_start': c['valid_start'],
                     'pending': clip in PENDING, 'notes': c['notes'],
                     'user_confirmed': len(events) > 1,   # 多掩護的 clip 使用者已看過確認圖
                     'events': events,
                     'excluded': excluded}               # 不是幫持球者擋（無球掩護等），之後做無球掩護用
    return out, v2, fd


def draw_confirm(clip, info, frames):
    """每個掩護一格：5fps 影格上畫全部 screener（編號 A/B/C），我選的那個加粗黃框。"""
    panels = []
    for k, e in enumerate(info['events']):
        ai = e['coco_idx'] if e['coco_idx'] is not None else e['idx']
        im = cv2.imread(str(FRAMES_DIR / clip / f'{clip}_{ai:04d}.jpg'))
        if im is None:
            im = np.zeros((720, 1280, 3), np.uint8)
        fr = frames.get(ai, {})
        for b in fr.get('ball_handler', []):
            x, y, w, h = map(int, b); cv2.rectangle(im, (x, y), (x + w, y + h), (0, 200, 0), 3)
        for b in fr.get('defender', []):
            x, y, w, h = map(int, b); cv2.rectangle(im, (x, y), (x + w, y + h), (220, 130, 0), 2)
        for j, b in enumerate(fr.get('screener', [])):
            x, y, w, h = map(int, b)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 230), 3)
            cv2.putText(im, 'ABCDE'[j], (x, max(30, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (0, 0, 230), 4, cv2.LINE_AA)
        if e['screener_box'] is not None:
            x, y, w, h = map(int, e['screener_box'])
            cv2.rectangle(im, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 255, 255), 5)
        S = im.shape[1] / 1280
        cv2.rectangle(im, (0, 0), (im.shape[1], int(60 * S)), (0, 0, 0), -1)
        cv2.putText(im, f"#{k + 1}  frame {e['frame']}  {'contact' if e['contact'] else 'NO contact'}",
                    (int(15 * S), int(44 * S)), cv2.FONT_HERSHEY_SIMPLEX, 1.3 * S, (255, 255, 255),
                    max(2, int(3 * S)), cv2.LINE_AA)
        panels.append(cv2.resize(im, (800, 450)))
    cols = min(3, len(panels))
    while len(panels) % cols:
        panels.append(np.full((450, 800, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)])
    head = np.full((40, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"{clip}   yellow = my pick for each screen   red A/B = all screeners   "
                      f"green = handler   blue = defender", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(CONFIRM_DIR / f'{clip}.jpg'), np.vstack([head, grid]))


def main():
    out, v2, fd = build()
    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    evs = [e for c in out.values() for e in c['events']]
    print(f"{len(out)} 支、{len(evs)} 個掩護 → {OUT_JSON}")
    print(f"  有撞到 {sum(e['contact'] for e in evs)}、沒撞到 {sum(not e['contact'] for e in evs)}")
    print(f"  有 cut 的 clip：{ {k: v['valid_start'] for k, v in out.items() if v['valid_start']} }")
    miss = [(k, e['frame']) for k, v in out.items() for e in v['events'] if e['screener_box'] is None]
    print(f"  對不到 COCO screener 框的掩護 {len(miss)} 個：{miss[:15]}")
    bad = [(k, e['frame']) for k, v in out.items() for e in v['events']
           if e['frame'] < v['valid_start'] or e['frame'] >= v['n_frames']]
    print(f"  幀號落在 cut 範圍內或超出影片的：{bad}")

    CONFIRM_DIR.mkdir(parents=True, exist_ok=True)
    multi = [k for k, v in out.items() if len(v['events']) > 1]
    for k in multi:
        draw_confirm(k, out[k], fd[k] if k in fd else v2[k])
    print(f"  確認圖 {len(multi)} 張 → {CONFIRM_DIR}")


if __name__ == '__main__':
    main()
