"""
offball_v2.py — 無球掩護第 2 版（2026-11-15）：在正式批次資料（events.pkl）上跑，用逐張持球者 + 擋拆事件 + 「掩護的效果」。

第 1 版（offball_screens.py，訓練快取上）：標記 27 個 ±1 秒抓到 19、使用者看 24 個標記外的偵測只有 11 個是真的（46%）。
誤報 13：持球掩護 4、雙擋 2、其他 7。根因：
  - 訓練快取整支只有一個持球者（handler_tid）→ 傳球後新持球者的擋拆被當成「無球」（持球掩護 4 個）→ 改用逐張持球者（Viterbi）
  - 雙擋的第二個掩護者撞到的人也不是守持球者的 → 擋拆事件 ±1 秒內的掩護者不算無球掩護
  - 只看撞上、沒看效果 → 加「切入者從掩護者旁經過 + 跟防守者拉開」

開發答案（dev_set，編號用框重疊換成 events.pkl 的）：
  正：第 1 版的 27 個標記（有的只有時間）+ 使用者判斷是的 11 個；負：使用者判斷不是的 13 個（附原因）。

用法：python nba_data_pipeline/offball_v2.py dev      # 開發答案對到批次編號的狀況
     python nba_data_pipeline/offball_v2.py eval     # 偵測 vs 開發答案
"""
import sys
import csv
import math
import pickle
import collections
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from convert_contact import CACHE_DIR

BATCH = ROOT / 'output_videos' / 'batch'
CHECK = ROOT / 'output_videos' / 'offball_check' / 'offball_check.csv'


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def to_batch(lab, d, tid, idx):
    """訓練快取的 tid（5fps 第 idx 張附近）→ events.pkl 的編號（框重疊最多的）；對不到 None。"""
    if tid is None:
        return None
    step = lab['fps'] / 5
    score = collections.Counter()
    for i in range(max(0, idx - 3), min(lab['n'], idx + 4)):
        bx, f = lab['raw_tracks'][i].get(tid), int(round(i * step))
        if bx is None or f >= len(d['player_tracks']):
            continue
        for q, v in d['player_tracks'][f].items():
            score[q] += _iou(bx, v.get('bbox') if isinstance(v, dict) else v)
    return score.most_common(1)[0][0] if score and score.most_common(1)[0][1] > 0.3 else None


def dev_set():
    """→ [dict(clip, frame(原片), S, C, D(批次編號 or None), truth, src, note)]"""
    import offball_screens as OS
    out = []
    labs, ds = {}, {}

    def load(clip):
        if clip not in labs:
            labs[clip] = pickle.load(open(CACHE_DIR / f'{clip}.pkl', 'rb'))
            ds[clip] = pickle.load(open(BATCH / f'{clip}_events.pkl', 'rb'))
        return labs[clip], ds[clip]
    for clip, items in OS.eval_set().items():
        if not (BATCH / f'{clip}_events.pkl').exists():
            continue
        lab, d = load(clip)
        for idx, S, src in items:
            out.append(dict(clip=clip, frame=int(round(idx * lab['fps'] / 5)), S=to_batch(lab, d, S, idx), C=None, D=None,
                            truth=True, src=src, note='', has_S=S is not None))
    for r in csv.DictReader(open(CHECK, encoding='utf-8-sig')):
        clip, idx = r['clip'], int(r['撞上5fps影格'])
        lab, d = load(clip)
        t = r['是無球掩護嗎(是/否/不確定)'].strip()
        if t not in ('t', 'f'):
            continue
        tid = lambda k: int(r[k].lstrip('#')) if r[k].strip() else None
        S, C = tid('掩護者(紅)'), tid('切入者(黃)')
        if '標反' in r['備註']:
            S, C = C, S
        out.append(dict(clip=clip, frame=int(round(idx * lab['fps'] / 5)), S=to_batch(lab, d, S, idx), C=to_batch(lab, d, C, idx),
                        D=to_batch(lab, d, tid('被擋的防守者(藍)'), idx), truth=t == 't', src=r['編號'], note=r['備註'], has_S=True))
    return out


def _teams(d):
    """→ (進攻隊員 set, 防守隊員 set)：court_teams（每隊 5 個在場上的人），進攻 = 持球最多的那隊。"""
    from tactics_tools import court_teams, clip_teams
    ct, tm = court_teams(d), clip_teams(d['player_assignment'])
    cnt = collections.Counter(tm.get(p) for p in d['ball_acquisition'] if p is not None and p != -1)
    off_t = cnt.most_common(1)[0][0] if cnt else 1
    return {q for q, t in ct.items() if t == off_t}, {q for q, t in ct.items() if t != off_t}


def contacts(d, stride=3, contact=3.5):
    """每次「進攻球員 S 撞上不是守他的防守者 D」（局部最近）+ 特徵。不做篩選，篩選在 keep。"""
    from tactics_tools import _pos_med
    cc, fps, ba = d['court_coords'], d['fps'], d['ball_acquisition']
    n = len(cc)
    off, dfn = _teams(d)
    P = lambda q, f: _pos_med(cc, q, f, 2) if 0 <= f < n else None
    dist = lambda a, b: math.dist(a, b) if a is not None and b is not None else None
    sec = lambda x: int(round(x * fps))
    holder = lambda q, a, b: any(0 <= f < n and ba[f] == q for f in range(a, b))
    ev = d.get('events') or []
    out = []
    for t in range(0, n, stride):
        for S in off:
            ps = P(S, t)
            if ps is None or holder(S, t - sec(0.3), t + sec(0.3)):
                continue
            for D in dfn:
                dsd = dist(ps, P(D, t))
                if dsd is None or dsd > contact:
                    continue
                if any((x := dist(P(S, u), P(D, u))) is not None and x < dsd for u in range(t - sec(0.3), t + sec(0.3) + 1, stride)):
                    continue
                guard = {}
                for C in off - {S}:
                    ds = [x for u in range(t - sec(1.2), t - sec(0.3), stride) if (x := dist(P(D, u), P(C, u))) is not None]
                    if len(ds) >= 3:
                        guard[C] = sum(ds) / len(ds)
                if not guard:
                    continue
                C = min(guard, key=guard.get)
                pas = [x for u in range(t - sec(0.5), t + sec(0.5) + 1, stride) if (x := dist(P(C, u), P(S, u))) is not None]
                vc = [dist(P(C, u - sec(0.2)), P(C, u + sec(0.2))) for u in range(t - sec(0.4), t + sec(0.4) + 1, stride)]
                vc = [x / 0.4 for x in vc if x is not None]
                s0, s1 = dist(P(C, t - sec(0.4)), P(D, t - sec(0.4))), dist(P(C, t + sec(0.6)), P(D, t + sec(0.6)))
                sd0 = dist(P(S, t - sec(0.5)), P(D, t - sec(0.5)))
                out.append(dict(t=t, S=S, C=C, D=D, contact=round(dsd, 1), guard=round(guard[C], 1),
                                onball=holder(C, t - sec(0.5), t + sec(0.5)),
                                in_event=any(e['screener_id'] == S and abs(e['peak_frame'] - t) <= sec(1.0) for e in ev),
                                pass_=round(min(pas), 1) if pas else None, v_cut=round(max(vc), 1) if vc else None,
                                sep=round(s1 - s0, 1) if s0 is not None and s1 is not None else None,
                                approach=round(sd0, 1) if sd0 is not None else None))
    return out


def main_dev():
    dv = dev_set()
    pos, neg = [x for x in dv if x['truth']], [x for x in dv if not x['truth']]
    print(f'開發答案：正 {len(pos)}（有掩護者 {sum(x["has_S"] for x in pos)}、對到批次編號 {sum(x["S"] is not None for x in pos)}）、'
          f'負 {len(neg)}（對到 {sum(x["S"] is not None for x in neg)}）')
    for x in dv:
        if x['has_S'] and x['S'] is None:
            print('   對不到：', x)


if __name__ == '__main__':
    if sys.argv[1:2] == ['dev']:
        main_dev()
