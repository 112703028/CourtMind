"""
la_test.py — NVIDIA LocateAnything-3B 試找持球者 / 球（2026-11-25，pod GPU）。

測試畫面：stubs/la_test/frames（新來源 27 個手遞手，交接前後各 4 張；meta.json 有答案 = 交接前給球者、交接後接球者）。
每張問兩題：持球的球員（ground_single）、籃球（ground_single）→ stubs/la_test/la_out.json；本機再對到球員框算準確率。
Worker 照 Hugging Face README（nvidia/LocateAnything-3B）；溫度調低讓結果穩定。

用法：python nba_data_pipeline/la_test.py         # pod：跑模型 → la_out.json
     python nba_data_pipeline/la_test.py eval    # 本機：評分
"""
import re
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

ROOT = Path(__file__).parent.parent
D = ROOT / 'stubs' / 'la_test'
MODEL = 'nvidia/LocateAnything-3B'
MAX_SIDE = 1280
PROMPTS = {'holder': 'Locate a single instance that matches the following description: the basketball player who is holding the basketball.',
           'ball': 'Locate a single instance that matches the following description: the basketball.'}


class Worker:
    def __init__(self, path=MODEL, device='cuda', dtype=torch.bfloat16):
        self.device, self.dtype = device, dtype
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True).to(device).eval()

    @torch.no_grad()
    def ask(self, image, question, temperature=0.1):
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': question}]}]
        text = self.proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = self.proc.process_vision_info(msgs)
        inp = self.proc(text=[text], images=images, videos=videos, return_tensors='pt').to(self.device)
        out = self.model.generate(pixel_values=inp['pixel_values'].to(self.dtype), input_ids=inp['input_ids'],
                                  attention_mask=inp['attention_mask'], image_grid_hws=inp.get('image_grid_hws'),
                                  tokenizer=self.tok, max_new_tokens=256, use_cache=True, generation_mode='hybrid',
                                  temperature=temperature, do_sample=True, top_p=0.9, repetition_penalty=1.1, verbose=False)
        return out[0] if isinstance(out, tuple) else out


def boxes(answer, W, H):
    return [[int(a) / 1000 * W, int(b) / 1000 * H, int(c) / 1000 * W, int(d) / 1000 * H]
            for a, b, c, d in re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', answer)]


def main():
    meta = json.load(open(D / 'meta.json'))
    out_p = D / 'la_out.json'
    done = json.load(open(out_p)) if out_p.exists() else {}
    w = Worker()
    t0 = time.time()
    for k, m in enumerate(meta):
        if m['img'] in done:
            continue
        im = Image.open(D / 'frames' / m['img']).convert('RGB')
        W0, H0 = im.size
        if max(W0, H0) > MAX_SIDE:                     # 1920×1080 原圖：視覺編碼器注意力吃 22 GB → OOM；輸出是 0~1000 相對座標，換回原圖不受影響
            sc = MAX_SIDE / max(W0, H0)
            im = im.resize((round(W0 * sc), round(H0 * sc)), Image.BICUBIC)
        r = {}
        for key, q in PROMPTS.items():
            a = w.ask(im, q)
            r[key] = {'answer': a, 'boxes': boxes(a, W0, H0)}
        done[m['img']] = r
        if k % 10 == 0:
            json.dump(done, open(out_p, 'w'))
            print(f'{k + 1}/{len(meta)}  {(time.time() - t0) / max(1, len(done)):.1f} 秒/張', flush=True)
    json.dump(done, open(out_p, 'w'))
    print('LA_DONE', len(done))


def evaluate():
    """本機：LocateAnything 框到的持球球員 → 框重疊最大的球員（IoU ≥ 0.3）→ 跟答案比；也比現在的持球者、球框是不是在答案球員附近。"""
    import numpy as np
    meta = json.load(open(D / 'meta.json'))
    out = json.load(open(D / 'la_out.json'))

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u > 0 else 0.0
    n = la_ok = la_none = ours_ok = both_ok = la_fix = ball_found = ball_near = 0
    for m in meta:
        r = out.get(m['img'])
        if r is None:
            continue
        n += 1
        bx = r['holder']['boxes']
        pred = None
        if bx:
            sc = sorted(((iou(bx[0], b), int(p)) for p, b in m['boxes'].items()), reverse=True)
            pred = sc[0][1] if sc and sc[0][0] >= 0.3 else None
        la_none += pred is None
        la_ok += pred == m['truth']; ours_ok += m['ours'] == m['truth']
        both_ok += pred == m['truth'] and m['ours'] == m['truth']
        la_fix += pred == m['truth'] and m['ours'] != m['truth']
        if r['ball']['boxes']:
            ball_found += 1
            b = r['ball']['boxes'][0]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            t = m['boxes'].get(str(m['truth']))
            if t:
                h = t[3] - t[1]
                ball_near += (t[0] - 0.3 * h <= cx <= t[2] + 0.3 * h) and (t[1] - 0.2 * h <= cy <= t[3])
    print(f'{n} 張：LocateAnything 持球者對 {la_ok}（{la_ok / n:.0%}，框不到 / 對不到球員 {la_none}）｜現在的持球者對 {ours_ok}（{ours_ok / n:.0%}）')
    print(f'   兩個都對 {both_ok}、只有 LocateAnything 對（能補救）{la_fix}；球有框到 {ball_found}、球在答案球員附近 {ball_near}')


if __name__ == '__main__':
    import sys as _s
    evaluate() if _s.argv[1:2] == ['eval'] else main()
