"""
possession_batch.py — 持球者改用「SAM3 追球 + Viterbi」（possession_sam3.possession_path）的批次驗證（2026-11-13）。

handler_detector 的持球者標記定義是「這波的主角」不是「此刻誰拿球」→ 撞上那刻錯 7%、一半影格沒持球者。
84 支延長片段驗過 Viterbi：撞上那刻錯 7% → 2%、沒持球者 65% → 約 4 成 —— 這裡在 476 支擋拆片段上量它對擋拆偵測的影響。

  python nba_data_pipeline/possession_batch.py eval   # 新舊持球者各跑一次擋拆（final5bgru + 救回），跟訓練標記比（誰幫誰擋都要對）
  python nba_data_pipeline/possession_batch.py apply  # 寫進 events.pkl：ball_acquisition = Viterbi 路線，原本的留 ball_acquisition_det
"""
import sys
import glob
import pickle
import collections
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
LABELS = ROOT / 'stubs' / 'pod_eval' / 'stubs' / 'train_contact_cache'     # pod 的訓練快取（含第 3 輪修正）


def clips():
    return sorted(glob.glob(str(ROOT / 'output_videos' / 'batch' / 'screen_*_events.pkl')), key=lambda x: int(Path(x).stem.split('_')[1]))


def viterbi_path(d, name):
    """events.pkl dict → Viterbi 持球者（原片每張，-1 = 沒人 / 球在空中）；沒有追球資料回 None。"""
    from possession_sam3 import possession_path
    bp, tp = ROOT / 'stubs' / 'ball_batch' / f'{name}_ball.pkl', ROOT / 'stubs' / 'ball_batch' / f'{name}_touch.pkl'
    if not (bp.exists() and tp.exists()):
        return None
    s = pickle.load(open(bp, 'rb'))['shot_sam3']
    _, _, tch = pickle.load(open(tp, 'rb'))
    dd = dict(d)
    dd['ball_acquisition'] = list(d.get('ball_acquisition_det') or d['ball_acquisition'])
    return possession_path(dd, s, tch)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def label_events(c, d):
    """訓練標記 → [(原片影格, 持球者, 掩護者)]（編號換成 events.pkl 的；用框重疊對應）。"""
    p = LABELS / f'{c}.pkl'
    if not p.exists():
        return None
    lab = pickle.load(open(p, 'rb'))
    if lab.get('pending'):
        return None
    step = lab['fps'] / 5

    def to_batch(tid, idx):
        score = collections.Counter()
        for i in range(max(0, idx - 3), min(lab['n'], idx + 4)):
            bx, f = lab['raw_tracks'][i].get(tid), int(round(i * step))
            if bx is None or f >= len(d['player_tracks']):
                continue
            for q, v in d['player_tracks'][f].items():
                score[q] += _iou(bx, v.get('bbox') if isinstance(v, dict) else v)
        return score.most_common(1)[0][0] if score and score.most_common(1)[0][1] > 0.3 else None

    vs = lab['valid_start_idx']
    return [(int(round(e['idx'] * step)), to_batch(lab['handler_tid'], e['idx']), to_batch(e['tid'], e['idx']))
            for e in lab['events'] if e['tid'] is not None and e['idx'] >= vs]


def detect(d, ba, model):
    import inference as I
    pk, _ = I.detect_screen_peaks(model, 'cpu', d['court_coords'], d['player_assignment'], ba, d['fps'], anchor=3, threshold=0.7,
                                  low=0.5, yolo_defender_pids=d.get('yolo_defender_pids'), rescue_lo=0.55)
    return [(p['frame'], p['handler_id'], p['screener_id']) for p in pk]


def score(preds, labs, fps, tol_s, need_handler):
    tp = fp = fn = 0
    for c in labs:
        g, pk, used = labs[c], preds[c], set()
        for f, H, S in g:
            m = [i for i, (pf, ph, ps) in enumerate(pk) if i not in used and ps == S and (not need_handler or ph == H)
                 and abs(pf - f) <= tol_s * fps[c]]
            if m:
                used.add(m[0])
        tp += len(used); fp += len(pk) - len(used); fn += len(g) - len(used)
    return tp, fp, fn, 2 * tp / max(1, 2 * tp + fp + fn)


def eval_():
    import io, contextlib
    from rescreen_batch import load_model
    model = load_model()
    labs, fps, old, new, stat = {}, {}, {}, {}, collections.Counter()
    for pk in clips():
        c = Path(pk).stem.replace('_events', '')
        d = pickle.load(open(pk, 'rb'))
        g = label_events(c, d)
        path = viterbi_path(d, c)
        if g is None or path is None:
            continue
        det = list(d.get('ball_acquisition_det') or d['ball_acquisition'])
        labs[c], fps[c] = g, d['fps']
        with contextlib.redirect_stdout(io.StringIO()):
            old[c], new[c] = detect(d, det, model), detect(d, path, model)
        stat['影格'] += len(det); stat['舊沒持球者'] += sum(1 for x in det if x in (None, -1)); stat['新沒持球者'] += sum(1 for x in path if x in (None, -1))
        for f, H, S in g:                                    # 撞上那刻的持球者對不對
            for k, ba in (('舊', det), ('新', path)):
                if H is not None and 0 <= f < len(ba):
                    stat[f'{k}撞上那刻有判'] += ba[f] not in (None, -1)
                    stat[f'{k}撞上那刻對'] += ba[f] == H
            stat['標記擋拆'] += H is not None
    print(f'{len(labs)} 支（有標記 + 有追球）；沒持球者的影格：舊 {stat["舊沒持球者"] / stat["影格"]:.0%} → 新 {stat["新沒持球者"] / stat["影格"]:.0%}')
    print(f'撞上那刻持球者對：舊 {stat["舊撞上那刻對"]}/{stat["標記擋拆"]}（有判 {stat["舊撞上那刻有判"]}）→ 新 {stat["新撞上那刻對"]}/{stat["標記擋拆"]}（有判 {stat["新撞上那刻有判"]}）')
    for tol in (0.4, 1.0):
        for nh, tn in ((False, '只看掩護者'), (True, '持球者也要對')):
            a, b = score(old, labs, fps, tol, nh), score(new, labs, fps, tol, nh)
            print(f'±{tol} 秒、{tn}：舊 F1 {a[3]:.3f}（抓對 {a[0]} 誤報 {a[1]} 漏抓 {a[2]}） → 新 {b[3]:.3f}（{b[0]} / {b[1]} / {b[2]}）')


def apply_():
    n = 0
    for pk in clips():
        c = Path(pk).stem.replace('_events', '')
        d = pickle.load(open(pk, 'rb'))
        path = viterbi_path(d, c)
        if path is None:
            continue
        d.setdefault('ball_acquisition_det', list(d['ball_acquisition']))
        d['ball_acquisition'] = path
        d['ball_acquisition_src'] = 'sam3_viterbi'
        pickle.dump(d, open(pk, 'wb'))
        n += 1
    print(f'{n} 支寫入 Viterbi 持球者')


if __name__ == '__main__':
    {'eval': eval_, 'apply': apply_}[sys.argv[1]]()
