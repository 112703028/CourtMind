"""
sportvu_coverage.py — SportVU 擋拆的防守方式：LLM 弱標記 + 主動學習候選（補包夾資料）

背景（2026-10-24）：包夾修正後只剩 38 筆。在使用者標過的 59 筆 SportVU 探針上量：
  規則（coverage_eval.classify）判包夾的精準度 29%、判退防 100%；LLM（gpt-5.5）判包夾 62%、退防 94%、換防 75%。
  → 規則判退防直接當弱標記（免費），其他交給 LLM。

步驟：
  collect：沒用過的比賽（跳過既有探針用過的 game）→ find_screens（外線）→ 規則 → stubs/sportvu_cov_cands.pkl
  llm    ：規則不是退防的候選問 LLM（快取 stubs/sportvu_cov_llm.jsonl）
  render ：LLM 判包夾的挑 N 個做動畫 → output_videos/nba_probe_blitz/（label.csv 給使用者標）

用法：
  python nba_data_pipeline/sportvu_coverage.py collect --skip 150 --games 20
  python nba_data_pipeline/sportvu_coverage.py llm
  python nba_data_pipeline/sportvu_coverage.py render --n 60
"""
import os
import csv
import glob
import json
import pickle
import hashlib
import argparse
import tempfile
import shutil
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
CANDS = ROOT / 'stubs' / 'sportvu_cov_cands.pkl'
LLM_CACHE = ROOT / 'stubs' / 'sportvu_cov_llm.jsonl'
OUT_DIR = ROOT / 'output_videos' / 'nba_probe_blitz'


def used_games():
    g = set()
    for d in (ROOT / 'output_videos').glob('nba_probe*'):
        p = d / 'probe_samples.pkl'
        if p.exists():
            g |= {s.get('game') for s in pickle.load(open(p, 'rb'))}
    return g


def _collect_one(path7z):
    import py7zr
    from extract_sequences import parse_moment, find_ball_handler, dist
    from nba_coverage_probe import (load_timeline, segments, find_screens, clip_tracks, nearest_def_mean,
                                    FPS, PRE_S, POST_S)
    import coverage_eval as ce
    import coverage_v2 as cv2_
    import llm_coverage as lc
    tmp = tempfile.mkdtemp(prefix='svc_')
    out = []
    try:
        with py7zr.SevenZipFile(path7z, 'r') as z:
            z.extractall(tmp)
        js = glob.glob(os.path.join(tmp, '**', '*.json'), recursive=True)
        if not js:
            return Path(path7z).stem, []
        gameid, frames, roster = load_timeline(js[0])
        for seg in segments(frames):
            if len(seg) < (PRE_S + POST_S) * FPS + 10:
                continue
            parsed = [parse_moment(m) for m in seg]
            handlers = find_ball_handler(parsed, None, None)
            for s in find_screens(parsed, handlers):
                f0, f1 = s['cf'] - int(PRE_S * FPS), s['cf'] + int(POST_S * FPS)
                if f0 < 0 or f1 > len(parsed):
                    continue
                pl = parsed[s['cf']]['players']
                hx, hy = pl[s['H']]['pos']
                rim = (88.75, 25.0) if hx >= 47 else (5.25, 25.0)
                if dist((hx, hy), rim) < 16:
                    continue
                defs = [p for p, i in pl.items() if i['team'] != s['team']]
                pre = list(range(s['cf'] - FPS, s['cf']))
                s['Dh'] = nearest_def_mean(parsed, pre, s['H'], defs)
                s['Ds'] = nearest_def_mean(parsed, pre, s['S'], [d for d in defs if d != s['Dh']])
                s['period'], s['clock'] = seg[s['cf']][0], seg[s['cf']][2]
                win = parsed[f0:f1]
                s['cf'] -= f0
                c = clip_tracks(win, s)
                ev = {'tid': s['S'], 'idx': c['contact_idx']}
                rule, _, _ = ce.classify(c, ev, 1.0, 1.5, 6.5, 8.5)
                ds, dh = cv2_.pick_ds_dh(c, ev)
                if ds is None:
                    continue
                out.append({'game': gameid, 's': s, 'c': c, 'rule': rule, 'text': lc.describe(c, ev, ds, dh),
                            'win': win if rule != 'drop' else None,          # 動畫只會用到非退防的
                            'roster': {p: roster.get(p, {}) for p in pl}})
        return gameid, out
    except Exception as e:
        return f'{Path(path7z).stem} 失敗：{str(e)[:80]}', []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def collect(args):
    from multiprocessing import Pool
    skip = used_games()
    zs = sorted(glob.glob(str(ROOT / 'nba-movement-data' / 'data' / '*.7z')))[args.skip:args.skip + args.games]
    allc = []
    with Pool(args.workers) as pool:
        for gid, out in pool.imap_unordered(_collect_one, zs):
            if gid in skip:
                print(f'  {gid}：探針用過，跳過'); continue
            allc += out
            print(f'  {gid}：外線擋拆 {len(out)} 個  規則 {dict(collections.Counter(x["rule"] for x in out))}', flush=True)
    pickle.dump(allc, open(CANDS, 'wb'))
    print(f'共 {len(allc)} 個 → {CANDS}；規則 {dict(collections.Counter(x["rule"] for x in allc))}')


def llm(args):
    import llm_coverage as lc
    from coverage_model import collect_training, _llm_client
    import concurrent.futures as cf
    cl = _llm_client()
    allc = pickle.load(open(CANDS, 'rb'))
    pool = [{'y': s['y'], 'text': s['text']} for s in collect_training(verbose=False) if not s['exclude'] and not s['either']]
    system = lc.PROMPTS['v1'].format(noise=lc.NOISE['nba'])
    cache = {}
    if LLM_CACHE.exists():
        for line in open(LLM_CACHE, encoding='utf-8'):
            r = json.loads(line); cache[r['h']] = r['resp']
    rng = np.random.RandomState(0)
    # 控制費用：規則判包夾的只抽 --max_blitz 個（規則包夾精準度 29%，大多是退防），其他非退防全問
    blitz_idx = [i for i, x in enumerate(allc) if x['rule'] == 'blitz']
    keep_blitz = set(np.random.RandomState(1).choice(blitz_idx, min(args.max_blitz, len(blitz_idx)), replace=False).tolist())
    for i, x in enumerate(allc):
        x['asked'] = x['rule'] != 'drop' and (x['rule'] != 'blitz' or i in keep_blitz)
    jobs = []
    for i, x in enumerate(allc):
        if not x['asked']:
            continue
        user = lc.build_prompt({'text': x['text']}, lc.pick_shots(pool, 2, rng), 'nba')
        h = hashlib.sha1(f'gpt-5.5|{system}|{user}'.encode()).hexdigest()
        x['llm_h'] = h
        if h not in cache:
            jobs.append((i, h, user))
    n_ask = sum(x['asked'] for x in allc)
    print(f'問 LLM：{n_ask} 個候選，快取 {n_ask - len(jobs)}、要問 {len(jobs)}', flush=True)
    with cf.ThreadPoolExecutor(8) as ex, open(LLM_CACHE, 'a', encoding='utf-8') as fo:
        futs = {ex.submit(lc.ask, cl, 'gpt-5.5', 'medium', system, u): (i, h) for i, h, u in jobs}
        for n, fu in enumerate(cf.as_completed(futs), 1):
            i, h = futs[fu]
            try:
                cache[h] = fu.result()
            except Exception as e:
                print('  失敗：', str(e)[:150]); continue
            fo.write(json.dumps({'h': h, 'resp': cache[h]}, ensure_ascii=False) + '\n'); fo.flush()
            if n % 50 == 0:
                print(f'  {n}/{len(jobs)}', flush=True)
    for x in allc:
        x['llm'] = cache.get(x.get('llm_h'), {}).get('coverage') if x['asked'] else None
        x['weak'] = 'drop' if x['rule'] == 'drop' else x['llm']      # 沒問的規則包夾 → None（不用）
    pickle.dump(allc, open(CANDS, 'wb'))
    print('弱標記：', dict(collections.Counter(x['weak'] for x in allc)))


def render(args):
    from nba_coverage_probe import render as draw, clip_tracks
    allc = pickle.load(open(CANDS, 'rb'))
    rng = np.random.RandomState(args.seed)
    cand = [i for i, x in enumerate(allc) if x.get('llm') == 'blitz' and x['win'] is not None]
    pick = list(rng.choice(cand, min(args.n, len(cand)), replace=False))
    print(f'LLM 判包夾 {len(cand)} 個，挑 {len(pick)} 個做動畫')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, samples = [], []
    for k, i in enumerate(pick, 1):
        x = allc[i]; s, win, roster = x['s'], x['win'], x['roster']
        J = lambda p: roster.get(p, {}).get('jersey', '?')
        name = f"{k:02d}_{x['game']}_Q{s['period']}_{int(s['clock'] // 60)}m{int(s['clock'] % 60):02d}s.mp4"
        draw(OUT_DIR / name, win, s, roster, 0, len(win))
        rows.append(f"{k:02d},{name},{s['period']},{s['clock']:.1f},{roster.get(s['H'], {}).get('team')},"
                    f"#{J(s['H'])},#{J(s['S'])},#{J(s['Ds'])},#{J(s['Dh'])},,,")
        samples.append({'id': k, 'file': name, 'game': x['game'], **x['c'], 'Ds': s['Ds'], 'Dh': s['Dh'],
                        'rule_kind': x['rule'], 'llm': x['llm'], 'cand_index': int(i)})
    with open(OUT_DIR / 'label.csv', 'w', encoding='utf-8-sig') as fo:
        fo.write('編號,檔名,節,時間(秒),進攻隊,持球者H,掩護者S,Ds守掩護者,Dh守持球者,'
                 '防守方式(換防/延誤/包夾/退防/其他/沒掩護),有把握(t/f),備註\n')
        fo.write('\n'.join(rows) + '\n')
    pickle.dump(samples, open(OUT_DIR / 'probe_samples.pkl', 'wb'))
    print(f'已輸出 {len(rows)} 支動畫 → {OUT_DIR}（LLM 的判斷不寫進表格，避免影響標記）')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('collect'); a.add_argument('--skip', type=int, default=150); a.add_argument('--games', type=int, default=20)
    a.add_argument('--workers', type=int, default=8)
    l = sub.add_parser('llm'); l.add_argument('--max_blitz', type=int, default=300, help='規則判包夾的最多問幾個')
    r = sub.add_parser('render'); r.add_argument('--n', type=int, default=60); r.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    {'collect': collect, 'llm': llm, 'render': render}[args.cmd](args)
