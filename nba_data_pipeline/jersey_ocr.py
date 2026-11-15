"""
jersey_ocr.py — 背號辨識（2026-11-15）：Roboflow 偵測模型的「number」類找背號框 → 對到球員 → 裁圖 → VLM 讀 → 同隊不重複 → 寫進 events.pkl。

試驗（6 支、55 人人工答案，output_videos/jersey_probe/jersey_check.csv）：每支片段一張「每人一列裁圖」整張丟 gpt-5.5，
高信心 30/31 對（97%）、中 9/16、低 1/3。改成「每人一張大圖（最大最清楚的 12 張）」高信心 36/38；本地 EasyOCR 19/21 只涵蓋 35%；
每秒 10 張反而混進更多別人的背號（36/40、38/44）。
正式 = read_combined：每人大圖高信心 + 整張圖沒讀出不同號碼 → 34/34 對、涵蓋 62%（同一份答案上挑的組合，實際會略低）。
`d['jersey'] = {pid: {'num', 'conf', 'n', 'use'}}`，'use' = 高信心且同隊沒撞號。VLM 只在本機（.env 金鑰），pod 不跑。
背號框中心落在球員框上 70%（胸口 / 背）→ 那個人（多個候選取最小框）；同隊兩人同號 → 留信心高 / 裁圖多的，另一個不用。

用法：
  python nba_data_pipeline/jersey_ocr.py add <events.pkl> <影片>            # 寫 d['jersey']
  python nba_data_pipeline/jersey_ocr.py eval [方法 ...]                    # 跟人工答案比（sheet / dense / player / easyocr）
"""
import sys
import csv
import json
import base64
import pickle
import hashlib
import collections
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
NUMBER = 2
CACHE = ROOT / 'stubs' / 'jersey'
TRUTH = ROOT / 'output_videos' / 'jersey_probe' / 'jersey_check.csv'
VLM_MODEL, VLM_EFFORT = 'gpt-5.5', 'medium'
RANK = {'h': 3, 'm': 2, 'l': 1}

SYSTEM_SHEET = ('你是籃球轉播畫面的背號辨識員。圖片每一列是同一個追蹤球員（左邊白字 #編號）身上裁下來的背號，最多 8 張，'
                '每格左下角黃字是影格編號。有的裁圖可能混到旁邊別人的背號，以多數為準。讀出每個球員的背號（0、00 是不同號碼，照原樣寫）。'
                '只輸出 JSON：{"players": [{"id": 追蹤編號(整數), "number": "背號字串或 null", "confidence": "high/medium/low"}]}。'
                '看不清楚、只看到一半（例如兩位數只露出一位）就給 low 或 null，不要猜。')
SYSTEM_PLAYER = ('你是籃球轉播畫面的背號辨識員。圖片是同一個球員身上裁下來的多張背號（不同影格），有的可能混到旁邊別人的背號，以多數為準。'
                 '讀出背號（0、00 是不同號碼，照原樣寫）。注意兩位數常因側身只露出一位、3 容易看成 1/2/5，要看最清楚的幾張再判斷。'
                 '只輸出 JSON：{"number": "背號字串或 null", "confidence": "high/medium/low", "votes": {"背號": 張數}}。')


def _box(v):
    return v.get('bbox') if isinstance(v, dict) else v


# ── 1. 背號框 → 球員 → 裁圖 ──────────────────────────────────────────────────────
def collect(d, video, det, sample_fps=3.0, conf=0.3, key=None):
    """→ {pid: [(score, frame, h, crop)]}；key 給了就快取在 stubs/jersey/<key>_<fps>.pkl。"""
    cp = CACHE / f'{key}_{sample_fps:g}.pkl' if key else None
    if cp and cp.exists():
        return pickle.load(open(cp, 'rb'))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or d['fps']
    step = max(1, int(round(fps / sample_fps)))
    crops = collections.defaultdict(list)
    f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if f % step == 0 and f < len(d['player_tracks']):
            for _, sc, b in (x for x in det.infer(im, conf=0.25, iou=0.5) if x[0] == NUMBER and x[1] >= conf):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                owner = []
                for p, v in d['player_tracks'][f].items():
                    pb = _box(v)
                    if pb and pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[1] + 0.7 * (pb[3] - pb[1]):
                        owner.append(((pb[2] - pb[0]) * (pb[3] - pb[1]), p))
                if not owner:
                    continue
                h = b[3] - b[1]
                m = 0.2 * max(b[2] - b[0], h)
                x1, y1 = int(max(0, b[0] - m)), int(max(0, b[1] - m))
                x2, y2 = int(min(im.shape[1], b[2] + m)), int(min(im.shape[0], b[3] + m))
                crops[min(owner)[1]].append((float(sc), f, float(h), im[y1:y2, x1:x2].copy()))
        f += 1
    crops = dict(crops)
    if cp:
        CACHE.mkdir(parents=True, exist_ok=True)
        pickle.dump(crops, open(cp, 'wb'))
    return crops


def best_crops(lst, k=8, by='score'):
    """挑 k 張：score = 偵測信心；size = 背號框高 × 信心（大張清楚的優先）。"""
    key = (lambda x: -x[0]) if by == 'score' else (lambda x: -x[2] * x[0])
    return sorted(lst, key=key)[:k]


def _tile(im, f, size):
    t = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.putText(t, f'{f}', (2, size - 4), 0, 0.4, (0, 255, 255), 1)
    return t


def sheet_img(crops, k=8, size=96, by='score'):
    rows = []
    for p in sorted(crops, key=lambda q: -len(crops[q])):
        tiles = [_tile(im, f, size) for _, f, _, im in best_crops(crops[p], k, by)]
        tiles += [np.zeros((size, size, 3), np.uint8)] * (k - len(tiles))
        label = np.zeros((size, 80, 3), np.uint8)
        cv2.putText(label, f'#{p}', (6, size // 2 - 8), 0, 0.9, (255, 255, 255), 2)
        cv2.putText(label, f'n={len(crops[p])}', (6, size - 20), 0, 0.5, (200, 200, 200), 1)
        rows.append(np.hstack([label] + tiles))
    return np.vstack(rows) if rows else None


def player_img(lst, k=12, size=128, by='size', cols=6):
    tiles = [_tile(im, f, size) for _, f, _, im in best_crops(lst, k, by)]
    tiles += [np.zeros((size, size, 3), np.uint8)] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


# ── 2. 讀數字 ───────────────────────────────────────────────────────────────────
def _vlm(img, system, ask, cache=CACHE / 'vlm_cache.jsonl'):
    from coverage_model import _llm_client
    data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    h = hashlib.md5((system + ask + VLM_MODEL).encode() + data).hexdigest()
    if cache.exists():
        for line in open(cache, encoding='utf-8'):
            j = json.loads(line)
            if j['h'] == h:
                return j['resp']
    cl = _llm_client()
    if cl is None:
        raise SystemExit('沒有 LLM 金鑰（.env 的 MYAI168_API_KEY 或 OPENAI_API_KEY）')
    msgs = [{'role': 'system', 'content': system},
            {'role': 'user', 'content': [{'type': 'text', 'text': ask},
                                         {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode(),
                                                                             'detail': 'high'}}]}]
    r = cl.chat.completions.create(model=VLM_MODEL, reasoning_effort=VLM_EFFORT,
                                   response_format={'type': 'json_object'}, messages=msgs)
    resp = json.loads(r.choices[0].message.content)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, 'a', encoding='utf-8') as fo:
        fo.write(json.dumps({'h': h, 'resp': resp}, ensure_ascii=False) + '\n')
    return resp


def _norm(num, cf):
    num = None if num in (None, '', 'null') else str(num).strip()
    return num, (cf or 'low')[0].lower()


def read_sheet(crops, k=8, size=96, by='score'):
    """每支片段一次：→ {pid: (num, 'h'/'m'/'l')}"""
    img = sheet_img(crops, k, size, by)
    if img is None:
        return {}
    r = _vlm(img, SYSTEM_SHEET, '讀出每一列球員的背號，只輸出 JSON。')
    out = {}
    for x in r.get('players', []):
        try:
            out[int(x['id'])] = _norm(x.get('number'), x.get('confidence'))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def read_player(crops, k=12, size=128, by='size'):
    """每人一次（張數多、圖大）：→ {pid: (num, conf)}"""
    out = {}
    for p, lst in crops.items():
        r = _vlm(player_img(lst, k, size, by), SYSTEM_PLAYER, '這個球員的背號是幾號？只輸出 JSON。')
        out[p] = _norm(r.get('number'), r.get('confidence'))
    return out


def read_combined(crops):
    """正式：每人一張大圖（read_player）高信心、而且整張圖（read_sheet）沒讀出不同號碼 → 'h'；其他降成 'm'（不用）。
    6 支 55 人：34/34 對、涵蓋 62%（只用整張圖高信心 30/31、55%；只用每人大圖高信心 36/38、65%）。"""
    sh, pl = read_sheet(crops), read_player(crops)
    out = {}
    for p, (n, cf) in pl.items():
        s_n = sh.get(p, (None, 'l'))[0]
        out[p] = (n, 'h' if n and cf == 'h' and (s_n is None or s_n == n) else ('m' if n else 'l'))
    return out


_READER = None


def read_easyocr(crops, k=16, by='size', min_p=0.3):
    """本地 OCR：每張裁圖放大後讀數字、信心 ≥ min_p 的投票 → {pid: (num, conf)}（conf 依得票比例：≥ 0.6 h、≥ 0.4 m）。"""
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    out = {}
    for p, lst in crops.items():
        votes = collections.Counter()
        for _, _, _, im in best_crops(lst, k, by):
            big = cv2.resize(im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            for _, txt, pr in _READER.readtext(big, allowlist='0123456789', detail=1):
                if pr >= min_p and 1 <= len(txt) <= 2:
                    votes[txt] += pr
        if not votes:
            out[p] = (None, 'l')
            continue
        num, v = votes.most_common(1)[0]
        frac = v / sum(votes.values())
        out[p] = (num, 'h' if frac >= 0.6 and v >= 1.5 else ('m' if frac >= 0.4 else 'l'))
    return out


# ── 3. 同隊不重複 → 寫進 events.pkl ──────────────────────────────────────────────
def clip_team(d):
    c = collections.defaultdict(collections.Counter)
    for fd in d.get('player_assignment') or []:
        for p, t in fd.items():
            c[p][t] += 1
    return {p: v.most_common(1)[0][0] for p, v in c.items()}


def resolve(reads, crops, team, min_conf='h'):
    """→ {pid: {'num', 'conf', 'n', 'use'}}；use = 信心 ≥ min_conf 且同隊沒人同號（撞號留信心高 / 裁圖多的）。"""
    out = {p: {'num': n, 'conf': cf, 'n': len(crops.get(p, [])), 'use': bool(n) and RANK[cf] >= RANK[min_conf]}
           for p, (n, cf) in reads.items()}
    by = collections.defaultdict(list)
    for p, r in out.items():
        if r['num']:
            by[(team.get(p), r['num'])].append(p)
    for (t, _), ps in by.items():
        if t is None or len(ps) < 2:
            continue
        ps.sort(key=lambda p: (RANK[out[p]['conf']], out[p]['n']), reverse=True)
        for p in ps[1:]:
            out[p]['use'] = False
            out[p]['dup_of'] = ps[0]
    return out


def add(pkl, video, det=None):
    from shot_detect_rf import OnnxDetector
    det = det or OnnxDetector(ROOT / 'models' / 'rf_basketball_det_v4.onnx')
    d = pickle.load(open(pkl, 'rb'))
    crops = collect(d, video, det)
    d['jersey'] = resolve(read_combined(crops), crops, clip_team(d))
    pickle.dump(d, open(pkl, 'wb'))
    use = {p: r['num'] for p, r in d['jersey'].items() if r['use']}
    print(f'背號：{len(crops)} 人有背號框、高信心用 {len(use)} 人 {dict(sorted(use.items()))}')
    return d['jersey']


# ── 4. 跟人工答案比 ─────────────────────────────────────────────────────────────
def truth():
    t = collections.defaultdict(dict)
    for r in csv.DictReader(open(TRUTH, encoding='utf-8-sig')):
        v = (r.get('正確背號（看不到填 ?）') or '').strip()
        if v and v != '?':
            t[r['clip']][int(r['追蹤編號'].lstrip('#'))] = v
    return t


def eval_(methods):
    from shot_detect_rf import OnnxDetector
    det = OnnxDetector(ROOT / 'models' / 'rf_basketball_det_v4.onnx')
    T = truth()
    fns = {
        'sheet': lambda cr: read_sheet(cr),
        'sheet_size': lambda cr: read_sheet(cr, by='size'),
        'player': lambda cr: read_player(cr),
        'easyocr': lambda cr: read_easyocr(cr),
        'combined': lambda cr: read_combined(cr),
    }
    for m in methods:
        dense = m.startswith('dense_')
        base = m[6:] if dense else m
        st = collections.Counter(); wrong = []
        for c, tc in T.items():
            pre = ROOT / 'stubs' / 'backup_events_pretrim' / f'{c}_events.pkl'   # 答案是照剪片頭前的編號標的
            d = pickle.load(open(pre if pre.exists() else ROOT / 'output_videos' / 'batch' / f'{c}_events.pkl', 'rb'))
            cr = collect(d, ROOT / 'input_videos' / 'screen' / f'{c}.mp4', det, sample_fps=10.0 if dense else 3.0, key=c)
            res = resolve(fns[base](cr), cr, clip_team(d))
            for p, truth_num in tc.items():
                r = res.get(p)
                cf = r['conf'] if r and r['num'] else '無'
                ok = bool(r and r['num']) and r['num'] == truth_num
                st[cf, ok] += 1
                if r and r['num'] and not ok:
                    wrong.append(f"{c.split('_')[1]}#{p}:{r['num']}≠{truth_num}({cf})")
                if r and r['use']:
                    st['use', ok] += 1
        n = sum(len(v) for v in T.values())
        line = '、'.join(f"{k} {st[k, True]}/{st[k, True] + st[k, False]}" for k in ('h', 'm', 'l'))
        print(f"[{m}] {n} 人：{line}、讀不出 {st['無', False]}；正式採用（高信心 + 不撞號）{st['use', True]}/{st['use', True] + st['use', False]} 對"
              f"（涵蓋 {st['use', True] / n:.0%}）")
        print('   錯：', ' '.join(wrong))


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == 'add':
        add(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == 'eval':
        eval_(sys.argv[2:] or ['sheet'])
    else:
        print(__doc__)
