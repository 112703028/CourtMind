"""
llm_coverage.py — 讓 LLM 直接從軌跡判斷防守方式（不是潤飾文字），跟隨機森林比。

參考：HARGPT（原始感測數字 + CoT 讓 GPT-4 零樣本辨識動作）、LMTraj（軌跡寫成文字當問答）。
動機：換防 / 包夾 / 延誤樣本很少，隨機森林只能從十幾筆學；LLM 本來就懂這些戰術定義。

考題：
  broadcast：轉播 129 筆（coverage_labels.csv，5fps 座標有雜訊）；範例只從同一切法的訓練份挑（不偷看）
  nba      ：NBA SportVU 兩批動畫中使用者有把握的標記（座標乾淨）；範例從轉播資料挑
輸入（每個擋拆一段文字）：撞前 1 秒 ~ 撞後 2.4 秒，每 0.2 秒一行：
  Ds / Dh 到持球者、掩護者、籃框的距離；離持球者最近的防守者是誰；持球者 6ft 內幾個防守者
輸出：JSON {coverage: switch|blitz|hedge|drop|unclear, confidence, reason}

回應快取在 stubs/llm_coverage_cache.jsonl（同模型 + 同 prompt + 同樣本不重問）。

用法：
  python nba_data_pipeline/llm_coverage.py --model gpt-5.4-mini --shots 0 --limit 20
  python nba_data_pipeline/llm_coverage.py --model gpt-5.5 --shots 2
"""
import os
import csv
import json
import pickle
import hashlib
import argparse
import collections
import concurrent.futures as cf
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import coverage_eval as ce
import coverage_ml as cm
import coverage_v2 as cv2_
import coverage_model as cmod
from coverage_features import series

ROOT = Path(__file__).parent.parent
CACHE = ROOT / 'stubs' / 'llm_coverage_cache.jsonl'
NBA_BATCHES = [ROOT / 'output_videos' / 'nba_probe_perimeter', ROOT / 'output_videos' / 'nba_probe_batch2']
OFFS = cm.OFFS
CN = {'換防': 'switch', '包夾': 'blitz', '延誤': 'hedge', '退防': 'drop'}
ZH = {'switch': '換防', 'blitz': '包夾', 'hedge': '延誤', 'drop': '退防', 'unclear': '不確定'}

SYSTEM = """你是籃球防守戰術分析師。你會拿到一次擋拆（pick-and-roll）前後的球員距離時間序列，
要判斷防守方怎麼應對這次掩護。

角色：H = 持球者，S = 掩護者，Dh = 原本守持球者的防守者，Ds = 原本守掩護者的防守者。
時間 0.0s = 掩護者撞上防守者的那一刻。距離單位英尺。

防守方式（看 Ds 怎麼應對，Dh 怎麼繞過掩護不影響分類）：
- drop（退防）：Ds 沒有上前堵持球者，留在掩護者附近或往籃框退；Dh 自己繞過掩護去追持球者。
- hedge（延誤）：Ds 上前堵持球者一下，然後退回去守掩護者；Dh 繞過掩護回到持球者身上。
- blitz（包夾）：Ds 上前堵持球者並且留著不走，Ds 和 Dh 兩人一起夾持球者
  （持球者常因此把球傳出去，之後防守會輪轉，兩人不一定一直貼著）。
- switch（換防）：Ds 接手去守持球者，Dh 離開持球者改去守掩護者。

判讀提示：
- 撞上那一刻 Ds、Dh 本來就都在持球者附近，不代表包夾；要看撞後 0.6~2.4 秒的變化。
- 換防的關鍵：撞後一段時間「離持球者最近的人」變成 Ds，且 Dh 跑到掩護者附近。
- 延誤 vs 退防：延誤的 Ds 會明顯靠近持球者再拉開；退防的 Ds 和持球者距離一直不太近。
- 座標有量測雜訊（{noise}），單一時間點的跳動不要過度解讀。
看不出來就回 unclear，不要硬猜。

只輸出 JSON：{{"coverage": "switch|hedge|blitz|drop|unclear", "confidence": 0~1, "reason": "一兩句理由，引用關鍵時間點的數字"}}"""

SYSTEM_V2 = """你是籃球防守戰術分析師。你會拿到一次擋拆（pick-and-roll）前後的球員距離時間序列，
要判斷防守方怎麼應對這次掩護。

角色：H = 持球者，S = 掩護者，Dh = 原本守持球者的防守者，Ds = 原本守掩護者的防守者。
時間 0.0s = 掩護者撞上防守者的那一刻。距離單位英尺。

防守方式（看 Ds 怎麼應對，Dh 怎麼繞過掩護不影響分類）：
- drop（退防）：Ds 沒有上前堵持球者，留在掩護者附近或往籃框退；Dh 自己繞過掩護去追持球者。
  這是最常見的防守（約六成），沒有明確證據顯示其他三種時，選 drop。
- hedge（延誤）：撞上之後 Ds 主動上前到持球者面前（Ds→H 明顯比撞上那刻更小，通常 ≤5ft），
  然後明顯退回掩護者（Ds→S 變小、Ds→H 拉開），同時 Dh 回到持球者身上（離 H 最近的人變回 Dh）。
  注意：撞上那一刻持球者正經過掩護者，Ds 本來就會離持球者很近，這不算延誤；要看撞上後 Ds 有沒有「再靠近、再退回」。
- blitz（包夾）：撞上後約 0.2~1.0 秒內，Ds 和 Dh 同時在持球者身邊（H 身邊 6ft 內有 2 個防守者，Ds→H、Dh→H 都小）。
  包夾常逼持球者把球傳出去，之後防守輪轉，Dh 可能離開持球者、看起來像換防——只要前面有一段兩人同時夾住，就算 blitz。
- switch（換防）：Ds 接手去守持球者，而 Dh「一開始就」離開持球者改去守掩護者（Dh→S 小、Dh→H 大），
  中間沒有兩人同時貼住持球者的階段。

判讀提示：
- 先看撞後 0.2~1.0 秒：H 身邊 6ft 內是不是有 2 人（→ 包夾候選）、離 H 最近的人是誰。
- 再看撞後 1.0~2.4 秒：誰留在 H 身邊、Ds 有沒有退回 S。
- 座標有量測雜訊（{noise}），單一時間點的跳動不要過度解讀，要看連續幾個時間點的趨勢。
看不出來就回 unclear，不要硬猜。

只輸出 JSON：{{"coverage": "switch|hedge|blitz|drop|unclear", "confidence": 0~1, "reason": "一兩句理由，引用關鍵時間點的數字"}}"""

PROMPTS = {'v1': SYSTEM, 'v2': SYSTEM_V2}

NOISE = {'broadcast': '轉播畫面換算，約 ±3 英尺', 'nba': '專業追蹤系統，誤差很小'}


# ── 樣本 → 文字 ──────────────────────────────────────────────────────────────

def describe(c, ev, ds, dh):
    ct, team, H, S, cfr = c['court_tracks'], c['track_team'], c['handler_tid'], ev['tid'], ev['idx']
    n = len(ct)
    frames = [cfr + o for o in OFFS]
    P = {k: series(ct, p, frames) for k, p in (('Ds', ds), ('Dh', dh), ('H', H), ('S', S))}
    B = np.array(cm.basket_of(c), float)
    defs = [p for p, t in team.items() if t == 'defense']
    lines = ['時間   Ds→H  Dh→H  Ds→S  Dh→S  Ds→籃框 Dh→籃框  離H最近  H身邊6ft內防守人數']
    for i, (o, f) in enumerate(zip(OFFS, frames)):
        d = lambda a, b: np.linalg.norm(P[a][i] - (P[b][i] if isinstance(b, str) else b))
        near, cnt = '?', 0
        if 0 <= f < n:
            here = {p: np.array(ct[f][p]) for p in defs if p in ct[f]}
            here[ds], here[dh] = P['Ds'][i], P['Dh'][i]
            dH = {p: np.linalg.norm(q - P['H'][i]) for p, q in here.items()}
            m = min(dH, key=dH.get)
            near = 'Ds' if m == ds else ('Dh' if m == dh else '其他防守者')
            cnt = sum(v <= 6 for v in dH.values())
        lines.append(f"{o / 5:+.1f}s  {d('Ds', 'H'):5.1f} {d('Dh', 'H'):5.1f} {d('Ds', 'S'):5.1f} {d('Dh', 'S'):5.1f}"
                     f"  {d('Ds', B):6.1f}  {d('Dh', B):6.1f}   {near:6s}  {cnt}")
    return '\n'.join(lines)


def broadcast_samples():
    out = []
    for s in cv2_.collect():
        c = pickle.load(open(ce.CACHE / f"{s['clip']}.pkl", 'rb'))
        out.append({'key': f"bc:{s['clip']}:{s['frame']}", 'group': s['clip'], 'y': s['y'],
                    'text': describe(c, s['ev'], s['ds'], s['dh']), 'X': s['old'].flatten()})
    return out


def nba_samples():
    from coverage_v2 import pick_ds_dh
    out = []
    for d in NBA_BATCHES:
        L = list(csv.DictReader(open(d / 'label.csv', encoding='utf-8-sig')))
        S = pickle.load(open(d / 'probe_samples.pkl', 'rb'))
        col = [k for k in L[0] if k.startswith('防守方式')][0]
        for r, s in zip(L, S):
            y = CN.get(r[col].strip())
            if y is None or r['有把握(t/f)'].strip() != 't':
                continue
            c = {'court_tracks': s['court_tracks'], 'track_team': s['track_team'], 'handler_tid': s['handler_tid']}
            ev = {'tid': s['screener'], 'idx': s['contact_idx']}
            ds, dh = pick_ds_dh(c, ev)
            f = cm.features(c, ev)
            out.append({'key': f"nba:{d.name}:{s['file']}", 'group': d.name, 'y': y,
                        'text': describe(c, ev, ds, dh), 'X': f})
    return out


# ── LLM ──────────────────────────────────────────────────────────────────────

def load_cache():
    out = {}
    if CACHE.exists():
        for line in open(CACHE, encoding='utf-8'):
            r = json.loads(line)
            out[r['h']] = r['resp']
    return out


def ask(client, model, effort, system, user):
    kw = {'model': model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
          'response_format': {'type': 'json_object'}}
    if effort and not model.startswith('gpt-4'):
        kw['reasoning_effort'] = effort
    r = client.chat.completions.create(**kw)
    try:
        j = json.loads(r.choices[0].message.content)
    except Exception:
        j = {'coverage': 'unclear', 'confidence': 0, 'reason': 'JSON 解析失敗'}
    if j.get('coverage') not in ('switch', 'hedge', 'blitz', 'drop', 'unclear'):
        j['coverage'] = 'unclear'
    return j


def build_prompt(sample, shots, noise):
    ex = ''
    for e in shots:
        ex += f"\n\n【範例：答案 {e['y']}（{ZH[e['y']]}）】\n{e['text']}"
    head = f"以下是幾個已標好答案的例子：{ex}\n\n" if shots else ''
    return f"{head}【請判斷這一次】\n{sample['text']}"


def pick_shots(pool, k, rng):
    out = []
    for cl in ('switch', 'hedge', 'blitz', 'drop'):
        c = [p for p in pool if p['y'] == cl]
        if c:
            out += list(rng.choice(c, min(k, len(c)), replace=False))
    rng.shuffle(out)
    return out


class Similarity:
    """
    KATE 式找相似範例：108 個距離特徵（coverage_ml）標準化後，用隨機森林特徵重要度加權的歐氏距離。
    標準化用範例池自己的平均 / 標準差；重要度由呼叫端給（轉播 = 只用訓練份訓練的 RF，不偷看考題）。
    """

    def __init__(self, pool, weights, nan_fill):
        self.pool = pool
        X = cmod._fill_nan(np.array([p['X'] for p in pool]), nan_fill)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.Z = (X - self.mu) / self.sd
        self.w = weights / weights.sum()
        self.nan_fill = nan_fill

    def order(self, sample):
        """池裡的範例由近到遠的 index。"""
        x = cmod._fill_nan(np.asarray(sample['X'], float)[None], self.nan_fill)[0]
        z = (x - self.mu) / self.sd
        d = np.sqrt((((self.Z - z) ** 2) * self.w).sum(1))
        return np.argsort(d)


def pick_knn(sample, sim, mode, k):
    """knn_class：每類最相似的 k 個；knn：不分類最相似的 4k 個。回傳時最相似的放最後（最靠近題目）。"""
    order = sim.order(sample)
    if mode == 'knn':
        chosen = list(order[:4 * k])
    else:
        chosen = []
        for cl in ('switch', 'hedge', 'blitz', 'drop'):
            chosen += [i for i in order if sim.pool[i]['y'] == cl][:k]
        rank = {i: r for r, i in enumerate(order)}
        chosen.sort(key=lambda i: rank[i])
    return [sim.pool[i] for i in reversed(chosen)]


def knn_vote(sample, sim, k):
    """對照組：直接抄最相似 k 個的多數答案（平手取最近那個的答案）。"""
    order = sim.order(sample)[:k]
    labs = [sim.pool[i]['y'] for i in order]
    cnt = collections.Counter(labs)
    top = max(cnt.values())
    return next(l for l in labs if cnt[l] == top)


def rf_weights(pool, nan_fill, seed=0):
    X = cmod._fill_nan(np.array([p['X'] for p in pool]), nan_fill)
    y = np.array([cmod.to3(p['y']) for p in pool], dtype=object)
    return cmod._rf(seed).fit(X, y).feature_importances_


def run_llm(items, model, effort, workers, prompt='v1'):
    """items: [(sample, shots, noise)] → 預測 list（含快取）。"""
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    from openai import OpenAI
    client = OpenAI()
    cache = load_cache()
    jobs, res = [], [None] * len(items)
    for i, (s, shots, noise) in enumerate(items):
        system = PROMPTS[prompt].format(noise=NOISE[noise])
        user = build_prompt(s, shots, noise)
        h = hashlib.sha1(f"{model}|{effort}|{system}|{user}".encode()).hexdigest()
        if h in cache:
            res[i] = cache[h]
        else:
            jobs.append((i, h, system, user))
    print(f"  {len(items)} 題，快取 {len(items) - len(jobs)}、要問 {len(jobs)}")
    with cf.ThreadPoolExecutor(workers) as ex, open(CACHE, 'a', encoding='utf-8') as fo:
        futs = {ex.submit(ask, client, model, effort, sy, us): (i, h) for i, h, sy, us in jobs}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            i, h = futs[fu]
            try:
                res[i] = fu.result()
            except Exception as e:
                print('  失敗：', str(e)[:120])
                res[i] = {'coverage': 'unclear', 'confidence': 0, 'reason': f'API 錯誤'}
                continue
            fo.write(json.dumps({'h': h, 'resp': res[i]}, ensure_ascii=False) + '\n')
            fo.flush()
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)}")
    return res


# ── 評估 ─────────────────────────────────────────────────────────────────────

def report(name, y, pred):
    y, pred = np.array(y, dtype=object), np.array(pred, dtype=object)
    ans = pred != 'unclear'
    four = ['switch', 'hedge', 'blitz', 'drop']
    per4 = {c: ((pred == c) & (y == c)).sum() / max(1, (y == c).sum()) for c in four}
    to3 = lambda v: np.array([cmod.to3(x) if x != 'unclear' else 'unclear' for x in v], dtype=object)
    y3, p3 = to3(y), to3(pred)
    per3 = [((p3 == c) & (y3 == c)).sum() / max(1, (y3 == c).sum()) for c in cmod.CLASSES]
    print(f"  {name:28s} 回答 {ans.mean():4.0%}  四類整體 {np.mean(pred == y):4.0%}  "
          f"三類整體 {np.mean(p3 == y3):4.0%}  三類平均 {np.mean(per3):4.0%}   "
          + "  ".join(f"{ZH[c]} {per4[c]:4.0%}" for c in four))
    if CONFUSION:
        cols = four + ['unclear']
        print("      答案＼判斷 " + " ".join(f"{ZH[c]:>4s}" for c in cols))
        for a in four:
            print(f"      {ZH[a]:6s}    " + " ".join(f"{((y == a) & (pred == c)).sum():5d}" for c in cols))


CONFUSION = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-5.4-mini')
    ap.add_argument('--effort', default='medium', help='reasoning_effort（gpt-5 / o 系列）')
    ap.add_argument('--shots', type=int, default=2, help='每類幾個範例（0 = 零樣本）')
    ap.add_argument('--set', choices=['broadcast', 'nba', 'both'], default='both')
    ap.add_argument('--limit', type=int, default=0, help='每份考題只考前 N 題（除錯用）')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--show', type=int, default=0, help='印幾筆 LLM 的理由')
    ap.add_argument('--confusion', action='store_true', help='印混淆矩陣')
    ap.add_argument('--prompt', choices=['v1', 'v2'], default='v1', help='v2：包夾/延誤定義更嚴格、退防為預設')
    ap.add_argument('--shot_mode', choices=['random', 'knn_class', 'knn'], default='random',
                    help='範例怎麼挑：random 每類隨機；knn_class 每類最相似；knn 不分類最相似（KATE）')
    ap.add_argument('--nba_pool', choices=['broadcast', 'other_batch'], default='broadcast',
                    help='NBA 題目的範例來源：轉播資料，或另一批 NBA 標記')
    args = ap.parse_args()
    global CONFUSION
    CONFUSION = args.confusion
    rng = np.random.RandomState(0)
    print(f"模型 {args.model}（effort {args.effort}），prompt {args.prompt}，每類 {args.shots} 個範例"
          f"（{args.shot_mode}，NBA 範例來源 {args.nba_pool}）")
    M = cmod.load_model()
    knn = args.shot_mode != 'random' and args.shots > 0
    k_vote = 4 * args.shots

    BC = broadcast_samples()
    # 同 coverage_model 的切法（seed 0）：每題的範例只從它不在的那幾份挑
    y_bc = np.array([cmod.to3(s['y']) for s in BC], dtype=object)
    g_bc = np.array([s['group'] for s in BC])
    fold_of = np.zeros(len(BC), int)
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(BC)), y_bc, g_bc))
    for k, (_, te) in enumerate(splits):
        fold_of[te] = k

    sets = []
    if args.set in ('broadcast', 'both'):
        idx = list(range(len(BC)))[:args.limit or None]
        items, votes, sims = [], [], {}
        for i in idx:
            pool = [BC[j] for j in range(len(BC)) if fold_of[j] != fold_of[i]]
            if knn:
                if fold_of[i] not in sims:             # 每份一個：重要度只用訓練份的 RF
                    sims[fold_of[i]] = Similarity(pool, rf_weights(pool, M['nan_fill']), M['nan_fill'])
                sim = sims[fold_of[i]]
                shots = pick_knn(BC[i], sim, args.shot_mode, args.shots)
                votes.append(knn_vote(BC[i], sim, k_vote))
            else:
                shots = pick_shots(pool, args.shots, rng) if args.shots else []
            items.append((BC[i], shots, 'broadcast'))
        sets.append(('轉播', [BC[i] for i in idx], items, votes))
    if args.set in ('nba', 'both'):
        NB_all = nba_samples()
        NB = NB_all[:args.limit or None]
        items, votes, sims = [], [], {}
        for s in NB:
            if knn:
                if args.nba_pool == 'broadcast':
                    key, pool = 'bc', BC
                else:
                    key, pool = s['group'], [p for p in NB_all if p['group'] != s['group']]
                if key not in sims:
                    # 重要度用轉播 RF（沒看過 NBA）；池子是 NBA 時也一樣，避免偷看考題
                    sims[key] = Similarity(pool, M['model'].feature_importances_, M['nan_fill'])
                shots = pick_knn(s, sims[key], args.shot_mode, args.shots)
                votes.append(knn_vote(s, sims[key], k_vote))
            else:
                shots = pick_shots(BC, args.shots, rng) if args.shots else []
            items.append((s, shots, 'nba'))
        sets.append(('NBA', NB, items, votes))

    for name, S, items, votes in sets:
        print(f"\n── {name}（{len(S)} 題）：", dict(collections.Counter(ZH[s['y']] for s in S)))
        pred = run_llm(items, args.model, args.effort, args.workers, args.prompt)
        p = [r['coverage'] for r in pred]
        y = [s['y'] for s in S]
        report(f'LLM {args.model}', y, p)
        if votes:
            report(f'kNN 抄最相似 {k_vote} 個的答案', y, votes)
        # 隨機森林對照：轉播 = 同切法交叉驗證；NBA = 全部轉播資料訓練的 coverage_rf
        if name == '轉播':
            X = cmod._fill_nan(np.array([s['X'] for s in S]), M['nan_fill'])
            rf = np.empty(len(S), dtype=object)
            yy = np.array([cmod.to3(s['y']) for s in S], dtype=object)
            gg = np.array([s['group'] for s in S])
            fo = fold_of[:len(S)]
            for k in range(5):
                tr, te = np.flatnonzero(fo != k), np.flatnonzero(fo == k)
                if len(te):
                    rf[te] = cmod._rf(0).fit(X[tr], yy[tr]).predict(X[te])
        else:
            X = cmod._fill_nan(np.array([s['X'] for s in S]), M['nan_fill'])
            rf = M['model'].predict(X)
        report('隨機森林（三類，延誤併退防）', y, list(rf))
        for s, r in list(zip(S, pred))[:args.show]:
            print(f"    [{ZH[s['y']]}] → {ZH[r['coverage']]} ({r.get('confidence')})：{r.get('reason')}")


if __name__ == '__main__':
    main()
