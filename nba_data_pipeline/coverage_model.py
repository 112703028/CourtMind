"""
coverage_model.py — 防守方式分類器：隨機森林 + 文字 LLM 混合，訓練與推論共用。

三類：switch（換防）/ blitz（包夾）/ drop（退防，含延誤 hedge）
  延誤併入退防：5fps 球場座標上兩者只差約 2 ft（< homography 誤差），隨機森林分不出來；
  LLM 會回 hedge，結果裡用 llm_label 保留下來給主播講解。

推論（混合，2026-10-23 實驗：三類平均 隨機森林 66% / 文字 LLM 73% / 混合 75%，包夾 37% → 66%）：
  1. 隨機森林算三類機率
  2. 退防機率 ≥ llm_gate（0.7）→ 直接回退防（隨機森林很有把握）
  3. 否則問 LLM（距離時間序列文字 + 每類 2 個訓練範例，llm_coverage 的 v1 prompt）
  4. LLM 回 unclear / 沒有 API key / 呼叫失敗 → 退回隨機森林（最高類機率 < 門檻 → unclear）
  LLM 金鑰：.env 的 MYAI168_API_KEY + MYAI168_BASE_URL（OpenAI 相容轉接），沒有就用 OPENAI_API_KEY。

特徵同 coverage_ml.features：撞上前 1 秒 ~ 撞上後 2.4 秒（5fps），Ds / Dh 到持球者、掩護者、籃框的距離。
  Dh / Ds = 撞上前 1 秒平均距離最近的防守者（人工修正檔有指定就用指定的）。座標一律先 clean_court_tracks。

訓練資料（collect_training）：
  舊：coverage_labels.csv（原片影格）
  新：output_videos/new_clip_check/handler_labels_*.csv 的「防守方式」欄，跟同一列的掩護逐一對應
      （多個掩護只寫一個防守方式、寫「或 / 第幾次」等對不上的不用）
  修正：coverage_label_fixes.json（label 改類別 / either 兩類都算對、不進訓練 / exclude 排除 / dh、ds 指定人）

訓練（存 models/coverage_rf.pkl，含 LLM 範例池）：
  python nba_data_pipeline/coverage_model.py
推論（tactics_tools.classify_coverage_ml 會呼叫）：
  predict_coverage(court_coords, player_assignment, handler, screener, frame, fps)
"""
import os
import re
import csv
import glob
import json
import pickle
import collections
from pathlib import Path

import numpy as np

import coverage_ml as cm

HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_PATH = ROOT / 'models' / 'coverage_rf.pkl'
FIX_PATH = HERE / 'coverage_label_fixes.json'
LABEL_CSVS = str(ROOT / 'output_videos' / 'new_clip_check' / 'handler_labels_*.csv')
SAMPLE_FPS = 5.0
CLASSES = ['switch', 'blitz', 'drop']
NAME = {'switch': '換防', 'blitz': '包夾', 'drop': '退防/延誤', 'unclear': '不確定', 'hedge': '延誤'}
CN = {'換防': 'switch', '包夾': 'blitz', '延誤': 'hedge', '退防': 'drop'}
DEFAULT_THRESHOLD = 0.5
LLM_GATE = 0.7
LLM_MODEL, LLM_EFFORT = 'gpt-5.5', 'medium'

_MODEL = None
_CLIENT = None


def to3(label):
    return 'drop' if label == 'hedge' else label


def _rf(seed=0):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight='balanced', random_state=seed)


def _fill_nan(X, med):
    return np.where(np.isnan(X), np.broadcast_to(med, X.shape), X)


# ── LLM ───────────────────────────────────────────────────────────────────────

def _llm_client():
    """OpenAI 相容 client；沒有金鑰回 None（只用隨機森林）。"""
    global _CLIENT
    if _CLIENT is None:
        try:
            from dotenv import load_dotenv, find_dotenv
            load_dotenv(find_dotenv(usecwd=True))
            from openai import OpenAI
            if os.environ.get('MYAI168_API_KEY'):
                _CLIENT = OpenAI(api_key=os.environ['MYAI168_API_KEY'], base_url=os.environ['MYAI168_BASE_URL'])
            elif os.environ.get('OPENAI_API_KEY'):
                _CLIENT = OpenAI()
            else:
                _CLIENT = False
        except Exception:
            _CLIENT = False
    return _CLIENT or None


def _ask_llm(text, M):
    """→ LLM 回應 dict（coverage / confidence / reason）或 None（不能用）。"""
    cl = _llm_client()
    pool = M.get('llm_pool')
    if cl is None or not pool:
        return None
    import llm_coverage as lc
    shots = lc.pick_shots(pool, 2, np.random.RandomState(0))
    system = lc.PROMPTS['v1'].format(noise=lc.NOISE['broadcast'])
    try:
        return lc.ask(cl, M.get('llm_model', LLM_MODEL), LLM_EFFORT, system, lc.build_prompt({'text': text}, shots, 'broadcast'))
    except Exception as e:
        print(f'  [coverage] LLM 呼叫失敗，改用隨機森林：{str(e)[:120]}')
        return None


# ── 推論 ──────────────────────────────────────────────────────────────────────

def load_model(path=MODEL_PATH):
    global _MODEL
    if _MODEL is None:
        if not Path(path).exists():
            return None
        _MODEL = pickle.load(open(path, 'rb'))
    return _MODEL


def predict_coverage(court_coords, player_assignment, handler_id, screener_id,
                     screen_frame, fps, handler_team=1, threshold=None, use_llm=True):
    """
    court_coords / player_assignment：inference dump 的逐幀資料（原片 fps，已 clean_court_tracks）。
    回 {coverage, confidence, prob, probs, detail, source('rf' / 'llm'), llm_label, llm_reason}；模型不存在回 None。
    """
    M = load_model()
    if M is None:
        return None
    threshold = M['threshold'] if threshold is None else threshold

    # 1. 抽成 5fps（跟訓練一致）
    step = fps / SAMPLE_FPS
    n5 = int(len(court_coords) / step)
    idx = [min(len(court_coords) - 1, int(round(i * step))) for i in range(n5)]
    ct = [court_coords[f] for f in idx]

    # 2. 整支影片多數決分隊（跟訓練的 track_team 一致）
    votes = collections.defaultdict(collections.Counter)
    for fd in player_assignment:
        for pid, t in fd.items():
            votes[pid][t] += 1
    team = {pid: ('offense' if c.most_common(1)[0][0] == handler_team else 'defense')
            for pid, c in votes.items()}

    c = {'court_tracks': ct, 'track_team': team, 'handler_tid': handler_id}
    ev = {'tid': screener_id, 'idx': int(round(screen_frame / step))}
    f = cm.features(c, ev)
    if f is None:
        return {'coverage': 'unclear', 'confidence': 'low', 'prob': None, 'probs': None, 'source': 'rf',
                'detail': '撞上前後的座標或防守者不足，無法判斷'}
    X = _fill_nan(f[None, :], M['nan_fill'])
    p = M['model'].predict_proba(X)[0]
    probs = {cl: float(p[list(M['model'].classes_).index(cl)]) for cl in CLASSES}
    best = max(probs, key=probs.get)
    pb = probs[best]
    shown = ', '.join(f"{NAME[k]} {v:.0%}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1]))

    # 3. 混合：隨機森林沒把握是退防 → 問 LLM
    if use_llm and probs['drop'] < M.get('llm_gate', LLM_GATE) and M.get('llm_pool'):
        import llm_coverage as lc
        from coverage_v2 import pick_ds_dh
        ds, dh = pick_ds_dh(c, ev)
        r = _ask_llm(lc.describe(c, ev, ds, dh), M) if ds is not None else None
        if r and r.get('coverage') in ('switch', 'blitz', 'drop', 'hedge'):
            lab = r['coverage']
            conf_v = float(r.get('confidence') or 0)
            return {'coverage': to3(lab), 'confidence': 'high' if conf_v >= 0.75 else 'medium',
                    'prob': conf_v, 'probs': probs, 'source': 'llm', 'llm_label': lab,
                    'llm_reason': r.get('reason', ''),
                    'detail': f"LLM 判斷 {NAME[lab]}（{r.get('reason', '')}）；隨機森林：{shown}"}

    conf = 'high' if pb >= 0.75 else ('medium' if pb >= threshold else 'low')
    if pb < threshold:
        return {'coverage': 'unclear', 'confidence': 'low', 'prob': pb, 'probs': probs, 'source': 'rf',
                'best_guess': best, 'detail': f'模型沒把握（{shown}）'}
    return {'coverage': best, 'confidence': conf, 'prob': pb, 'probs': probs, 'source': 'rf',
            'detail': f'模型判斷 {NAME[best]}（{shown}）'}


# ── 訓練資料 ──────────────────────────────────────────────────────────────────

def _sample(c, ev, clip, y, src, fix):
    import llm_coverage as lc
    from coverage_v2 import pick_ds_dh
    ds, dh = pick_ds_dh(c, ev)
    if ds is None:
        return None
    dh, ds = fix.get('dh', dh), fix.get('ds', ds)
    f = cm.features(c, ev, ds=ds, dh=dh)
    if f is None:
        return None
    y = fix.get('label', y)
    return {'key': f"{clip}:{ev['tid']}@{ev['idx']}", 'clip': clip, 'src': src, 'y': y, 'X': f,
            'text': lc.describe(c, ev, ds, dh), 'either': fix.get('either'), 'exclude': bool(fix.get('exclude'))}


def collect_training(verbose=True):
    """舊標記 + 新片段表格 + 人工修正 → 樣本 list（座標已清理）。"""
    import coverage_eval as ce
    from tactics_tools import clean_court_tracks
    from ingest_new_labels import parse_labels
    fixes = json.load(open(FIX_PATH, encoding='utf-8')) if FIX_PATH.exists() else {}
    S, skip = [], collections.Counter()

    def load(clip):
        p = ce.CACHE / f'{clip}.pkl'
        if not p.exists():
            return None
        c = pickle.load(open(p, 'rb'))
        if c.get('pending'):
            return None
        c['court_tracks'] = clean_court_tracks(c['court_tracks'])[0]
        return c

    for clip, items in ce.load_labels().items():                  # 舊
        c = load(clip)
        if c is None:
            continue
        step = c['fps'] / ce.FPS
        for frame, ans, _ in items:
            if ans is None:
                continue
            idx = round(frame / step)
            evs = [e for e in c['events'] if e['tid'] is not None and abs(e['idx'] - idx) <= 1]
            if evs:
                s = _sample(c, evs[0], clip, ans, 'old', fixes.get(f"{clip}:{evs[0]['tid']}@{evs[0]['idx']}", {}))
                if s:
                    S.append(s)
    for path in sorted(glob.glob(LABEL_CSVS)):                      # 新
        for clip, d in parse_labels(path).items():
            if not d['screens']:
                continue
            c = load(clip)
            if c is None:
                continue
            rows = collections.OrderedDict()
            for s in d['screens']:
                rows.setdefault(s['coverage'], []).append(s)
            for cov, scr in rows.items():
                toks = [t.strip() for t in re.split('[,，、]', cov) if t.strip()]
                if not toks:
                    skip['沒填'] += len(scr); continue
                if any(t not in CN for t in toks):
                    skip['寫法對不上'] += len(scr); continue
                if len(toks) != len(scr):
                    skip['多個掩護只寫一個 / 數量對不上'] += len(scr); continue
                for s, t in zip(scr, toks):
                    evs = [e for e in c['events'] if e['tid'] == s['tid'] and abs(e['idx'] - s['idx']) <= 1]
                    if not evs:
                        skip['快取裡找不到這個掩護'] += 1; continue
                    smp = _sample(c, evs[0], clip, CN[t], 'new', fixes.get(f"{clip}:{evs[0]['tid']}@{evs[0]['idx']}", {}))
                    if smp:
                        S.append(smp)
                    else:
                        skip['座標不足'] += 1
    if verbose:
        n_fix = sum(1 for s in S if s['key'] in fixes)
        print(f"樣本 {len(S)}（舊 {sum(s['src'] == 'old' for s in S)}、新 {sum(s['src'] == 'new' for s in S)}；"
              f"套用人工修正 {n_fix}）；新標記沒用到：{dict(skip)}")
    return S


def sportvu_blitz(copies=3, noise=1.5, seed=0):
    """使用者在 SportVU 動畫（output_videos/nba_probe*/label.csv）標「包夾」且有把握 = t 的擋拆 → 距離特徵，
    每筆加 copies 份轉播等級的位置雜訊（nba_contact_data.add_noise）。只進訓練、不考。
    2026-10-24：只加這 19 筆包夾，混合法包夾 63% → 68%、三類平均 73% → 75%；加全部類別或 LLM 弱標記反而沒幫助 / 變差。"""
    from nba_contact_data import add_noise
    rng = np.random.RandomState(seed)
    X = []
    for d in sorted((ROOT / 'output_videos').glob('nba_probe*')):
        if not (d / 'label.csv').exists() or not (d / 'probe_samples.pkl').exists():
            continue
        L = list(csv.DictReader(open(d / 'label.csv', encoding='utf-8-sig')))
        P = pickle.load(open(d / 'probe_samples.pkl', 'rb'))
        col = [k for k in L[0] if k.startswith('防守方式')][0]
        for r, s in zip(L, P):
            if CN.get(r[col].strip()) != 'blitz' or r.get('有把握(t/f)', '').strip() != 't':
                continue
            c = {'court_tracks': s['court_tracks'], 'track_team': s['track_team'], 'handler_tid': s['handler_tid'],
                 'n': len(s['court_tracks']), 'raw_tracks': s['court_tracks']}
            for _ in range(copies):
                f = cm.features(add_noise(c, noise, rng), {'tid': s['screener'], 'idx': s['contact_idx']})
                if f is not None:
                    X.append(f)
    return np.array(X)


# ── 訓練 ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from sklearn.model_selection import StratifiedGroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    S = [s for s in collect_training() if not s['exclude']]
    X = np.array([s['X'] for s in S])
    y = np.array([to3(s['y']) for s in S], dtype=object)
    g = np.array([s['clip'] for s in S])
    either = np.array([bool(s['either']) for s in S])
    SV = sportvu_blitz()
    med = np.nan_to_num(np.nanmedian(np.vstack([X, SV]) if len(SV) else X, axis=0))
    X = _fill_nan(X, med)
    SV = _fill_nan(SV, med) if len(SV) else np.zeros((0, X.shape[1]))
    sv_y = np.array(['blitz'] * len(SV), dtype=object)
    print(f"可用 {len(y)}（訓練 {(~either).sum()}，其中 {either.sum()} 筆換防/包夾都算對、只考不訓練）：",
          {NAME[k]: v for k, v in collections.Counter(y).items()})
    print(f"另加 SportVU 人工包夾 {len(SV) // 3} 筆 × 3 份雜訊 = {len(SV)}（只進訓練）")

    # 交叉驗證（隨機森林部分；混合法要呼叫 LLM，見 docs/experiments/2026-10-23-*.md）
    ok_fn = lambda p, i: (p in ('switch', 'blitz')) if either[i] else p == y[i]
    rows = []
    for sd in range(args.seeds):
        P = np.zeros((len(y), len(CLASSES)))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(X, y, g):
            tr = tr[~either[tr]]
            m = _rf(sd).fit(np.vstack([X[tr], SV]), np.concatenate([y[tr], sv_y]))
            pr = m.predict_proba(X[te])
            for j, cl in enumerate(m.classes_):
                P[te, CLASSES.index(cl)] = pr[:, j]
        pred = [CLASSES[i] for i in P.argmax(1)]
        ok = np.array([ok_fn(p, i) for i, p in enumerate(pred)])
        rows.append([ok.mean()] + [ok[y == c].mean() for c in CLASSES])
    r = np.mean(rows, axis=0)
    print(f"\n隨機森林交叉驗證（{args.seeds} 種切法平均）：整體 {r[0]:.0%}   "
          + "  ".join(f"{NAME[c]} {v:.0%}" for c, v in zip(CLASSES, r[1:])) + f"   三類平均 {np.mean(r[1:]):.0%}")

    tr = ~either
    model = _rf(0).fit(np.vstack([X[tr], SV]), np.concatenate([y[tr], sv_y]))
    pool = [{'y': s['y'], 'text': s['text']} for s in S if not s['either']]
    MODEL_PATH.parent.mkdir(exist_ok=True)
    pickle.dump({'model': model, 'nan_fill': med, 'offs': cm.OFFS, 'classes': CLASSES,
                 'threshold': args.threshold, 'n_train': int(tr.sum()),
                 'llm_pool': pool, 'llm_gate': LLM_GATE, 'llm_model': LLM_MODEL}, open(MODEL_PATH, 'wb'))
    print(f"\n已用 {int(tr.sum())} 筆訓練最終模型（含 LLM 範例池 {len(pool)} 筆、退防 ≥{LLM_GATE} 才不問 LLM）→ {MODEL_PATH}")


if __name__ == '__main__':
    main()
