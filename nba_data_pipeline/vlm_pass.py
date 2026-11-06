"""
vlm_pass.py — 傳球偵測試 VLM：一段時間（8 張連續畫面、球員框 + 編號）給 gpt-5.5 判斷有沒有傳球、誰傳給誰。
規則版（2D 框）在掩護人擠一起 / 球在空中穿過路人的框時分不出來；VLM 看得到球和手（進球判斷 67% → 85% 同一招）。

測試：使用者標的 21 個真傳球（pass_gt_probe.GT）+ 之前標錯的假傳球（output_videos/pass_check，f）。
用法：python nba_data_pipeline/vlm_pass.py [--neg 30]
"""
import sys
import csv
import json
import base64
import hashlib
import pickle
import argparse
import collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

MODEL, EFFORT = 'gpt-5.5', 'medium'
CACHE = ROOT / 'stubs' / 'vlm_pass_cache.jsonl'
IMG = ROOT / 'output_videos' / 'vlm_pass'
SYSTEM = ('你是籃球比賽影像判讀員。給你一張由 8 格組成的圖：同一段轉播的連續畫面，時間由左到右、由上到下（每格左上角有秒數）。'
          '每個球員有框和編號（紅框 = 一隊、藍框 = 另一隊）。判斷這段時間內有沒有「傳球」：球離開一個人的手、在空中飛過去、被另一個人接住。'
          '要分清楚：運球（球一直在同一人身邊上下彈）不是傳球；投籃（球飛向籃框）不是傳球；球只是從某人身體前方飛過（2D 畫面重疊）那人不是接球者；'
          '手遞手（兩人很近、球直接從一人手上交給另一人）要標 handoff=true。'
          '看不清楚就回 null，不要猜。只輸出 JSON：{"pass": true/false/null, "from": 編號或 null, "to": 編號或 null, '
          '"handoff": true/false, "confidence": "high"|"medium"|"low", "reason": "一句話"}')


def make_grid(video, events_pkl, f0, f1, out_path, pad_s=0.3):
    d = pickle.load(open(events_pkl, 'rb'))
    tracks, fps = d['player_tracks'], d.get('fps', 30.0)
    cnt = collections.defaultdict(collections.Counter)
    for fd in d['player_assignment']:
        for p, t in fd.items():
            cnt[p][t] += 1
    team = {p: c.most_common(1)[0][0] for p, c in cnt.items()}
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = max(0, int(f0 - pad_s * fps)), min(n - 1, int(f1 + pad_s * fps))
    picks = np.linspace(lo, hi, 8).astype(int)
    tiles = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if not ok:
            im = np.zeros((360, 640, 3), np.uint8)
        else:
            lw = max(2, im.shape[0] // 400)
            for p, bb in (tracks[f].items() if f < len(tracks) else []):
                col = (60, 60, 230) if team.get(p) == 1 else (220, 140, 40) if team.get(p) == 2 else (160, 160, 160)
                x1, y1, x2, y2 = [int(v) for v in bb]
                cv2.rectangle(im, (x1, y1), (x2, y2), col, lw)
                sc = im.shape[0] / 360 * 0.75                  # 縮成 640×360 後字還看得清楚
                (tw, th), _ = cv2.getTextSize(str(p), cv2.FONT_HERSHEY_SIMPLEX, sc, lw + 1)
                cv2.rectangle(im, (x1, y1 - th - 10), (x1 + tw + 8, y1), col, -1)
                cv2.putText(im, str(p), (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), lw + 1, cv2.LINE_AA)
            im = cv2.resize(im, (640, 360))
            cv2.putText(im, f'{f / fps:.2f}s', (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


def ask(client, img, cache):
    data = open(img, 'rb').read()
    h = hashlib.md5((SYSTEM + MODEL).encode() + data).hexdigest()
    if h in cache:
        return cache[h]
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': [{'type': 'text', 'text': '這段有沒有傳球？只輸出 JSON。'},
                                         {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode(),
                                                                             'detail': 'high'}}]}]
    r = client.chat.completions.create(model=MODEL, reasoning_effort=EFFORT, response_format={'type': 'json_object'}, messages=msgs)
    resp = json.loads(r.choices[0].message.content)
    cache[h] = resp
    with open(CACHE, 'a', encoding='utf-8') as fo:
        fo.write(json.dumps({'h': h, 'img': str(img), 'resp': resp}, ensure_ascii=False) + '\n')
    return resp


def main():
    from coverage_model import _llm_client
    from pass_gt_probe import GT
    ap = argparse.ArgumentParser()
    ap.add_argument('--neg', type=int, default=30)
    args = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    items = []
    for c, f0, f1, A, B in GT:
        img = make_grid(ROOT / f'input_videos/screen/screen_{c}.mp4', ROOT / f'output_videos/batch/screen_{c}_events.pkl',
                        f0, f1, IMG / f'pos_screen_{c}_{f0}.jpg')
        items.append(dict(kind='真傳球', clip=f'screen_{c}', A=A, B=B, img=img))
    R = list(csv.DictReader(open(ROOT / 'output_videos/pass_check/pass_check.csv', encoding='utf-8-sig')))
    k = [x for x in R[0] if x.startswith('對嗎')][0]
    neg = [r for r in R if r[k].strip() == 'f'][:args.neg]
    for r in neg:
        clip = r['clip']
        d = pickle.load(open(ROOT / f'output_videos/batch/ext_{clip}_events.pkl', 'rb'))
        fps = d['fps']
        f0, f1 = int(float(r['離手秒']) * fps), int(float(r['接球秒']) * fps)
        img = make_grid(ROOT / f'input_videos/screen_ext/{clip}.mp4', ROOT / f'output_videos/batch/ext_{clip}_events.pkl',
                        f0, f1, IMG / f"neg_{clip}_{r['離手秒']}.jpg")
        items.append(dict(kind='假傳球', clip=clip, A=int(r['傳球者(紅)'][1:]), B=int(r['接球者(綠)'][1:]), img=img))
    cache = {}
    if CACHE.exists():
        for line in open(CACHE, encoding='utf-8'):
            j = json.loads(line); cache[j['h']] = j['resp']
    client = _llm_client()
    with ThreadPoolExecutor(6) as ex:
        resps = list(ex.map(lambda it: ask(client, it['img'], cache), items))
    pos = [(it, r) for it, r in zip(items, resps) if it['kind'] == '真傳球']
    negs = [(it, r) for it, r in zip(items, resps) if it['kind'] == '假傳球']
    said = sum(r.get('pass') is True for _, r in pos)
    to_ok = sum(r.get('pass') is True and r.get('to') == it['B'] for it, r in pos)
    both = sum(r.get('pass') is True and r.get('to') == it['B'] and r.get('from') == it['A'] for it, r in pos)
    print(f'真傳球 {len(pos)} 個：VLM 說有傳球 {said}；接球者對 {to_ok}；傳球者 + 接球者都對 {both}；說 null {sum(r.get("pass") is None for _, r in pos)}')
    rej = sum(r.get('pass') is False for _, r in negs)
    same = sum(r.get('pass') is True and r.get('from') == it['A'] and r.get('to') == it['B'] for it, r in negs)
    print(f'假傳球 {len(negs)} 個：VLM 說沒傳球 {rej}；說有傳球而且跟規則版同一對 {same}；說 null {sum(r.get("pass") is None for _, r in negs)}')
    print('\n逐個：')
    for it, r in pos + negs:
        print(f"  {it['kind']} {it['clip']} #{it['A']}→#{it['B']} | VLM pass={r.get('pass')} {r.get('from')}→{r.get('to')} "
              f"handoff={r.get('handoff')} {r.get('confidence')} | {r.get('reason', '')[:50]}")


if __name__ == '__main__':
    main()
