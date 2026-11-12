"""
e2e_eval.py — 端到端評估集（2026-11-07）：固定一組片段，跑完整條分析（規則事實 + 主播解說），人工逐項標對錯，之後每次改進用同一組重跑比較。

步驟：
  python nba_data_pipeline/e2e_eval.py select          # 抽片段（固定 seed）→ output_videos/e2e_eval/clips.json
  python nba_data_pipeline/e2e_eval.py run [--tag v1]  # 每支：規則事實（list_screens）+ 主播解說 → facts_<tag>.json、narration_<tag>/
  python nba_data_pipeline/e2e_eval.py sheet [--tag v1] # 一行一個事實的檢查表 check_<tag>.csv + 解說全文 narration_<tag>.md
  python nba_data_pipeline/e2e_eval.py video           # 原片 + 框 / 編號 / 影格號碼（跟事實裡的編號、影格一致）
  python nba_data_pipeline/e2e_eval.py score [--tag v1] # 讀標好的檢查表 → 各項準確率
注意：這批片段都在 ScreenNet / 防守應對的訓練資料裡 → 擋拆偵測會比新影片好看；要測泛化得換沒訓練過的影片。
"""
import sys
import csv
import json
import random
import pickle
import argparse
import collections
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
OUT = ROOT / 'output_videos' / 'e2e_eval'
BATCH = ROOT / 'output_videos' / 'batch'
QUESTION = ("請像轉播主播一樣，詳細解說這球的進攻戰術。先用 list_screens 掌握全局，再對關鍵擋拆用精確工具查證 roll/pop、"
            "防守方式、擋誰，並看流程脈絡，最後給一段生動的解說。")


def _clip_feats(pk):
    d = pickle.load(open(pk, 'rb'))
    fps, ev = d['fps'], d['events']
    peaks = sorted(e['peak_frame'] for e in ev)
    return dict(n_ev=len(ev),
                pass_after=any(0 <= (p['release'] - e) / fps <= 6 for p in d.get('passes') or [] if p['kind'] == 'pass' for e in peaks),
                shot_after=any(-0.5 <= (s['release_frame'] - e) / fps <= 8 for s in d.get('shots') or [] for e in peaks),
                multi=any(0 < (b - a) / fps <= 2.0 for a, b in zip(peaks, peaks[1:])))


def select(n_pass=5, n_shot=5, n_multi=3, n_rand=5, seed=7):
    """分層抽：擋拆後有傳球 / 有出手 / 雙擋或連續擋 / 隨機，每支至少一個擋拆事件。"""
    feats = {}
    for pk in sorted(BATCH.glob('screen_*_events.pkl')):
        f = _clip_feats(pk)
        if f['n_ev']:
            feats[pk.stem.replace('_events', '')] = f
    rng = random.Random(seed)
    pool, chosen = sorted(feats), []
    for key, n in (('pass_after', n_pass), ('multi', n_multi), ('shot_after', n_shot)):
        cand = [c for c in pool if feats[c][key] and c not in chosen]
        chosen += rng.sample(cand, min(n, len(cand)))
    rest = [c for c in pool if c not in chosen]
    chosen += rng.sample(rest, n_rand)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({'seed': seed, 'clips': chosen, 'feats': {c: feats[c] for c in chosen}}, open(OUT / 'clips.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'有擋拆事件的片段 {len(feats)} 支 → 抽 {len(chosen)} 支：{chosen}')


def _clips():
    return json.load(open(OUT / 'clips.json', encoding='utf-8'))['clips']


def run(tag='v1', model='gpt-5.5'):
    from shot_detect import apply_vlm_made
    from tactics_agent import build_ctx, run_agent, t_list_screens
    fp = OUT / f'facts_{tag}.json'
    facts = json.load(open(fp, encoding='utf-8')) if fp.exists() else {}
    nd = OUT / f'narration_{tag}'
    nd.mkdir(parents=True, exist_ok=True)
    for c in _clips():
        if c in facts and (nd / f'{c}.txt').exists():
            continue
        pk = BATCH / f'{c}_events.pkl'
        data = pickle.load(open(pk, 'rb'))
        apply_vlm_made(data, pk)
        ctx = build_ctx(data)
        facts[c] = {'fps': ctx['fps'], 'screens': t_list_screens(ctx)}
        json.dump(facts, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
        ans = run_agent(QUESTION, ctx, model=model, verbose=False)
        (nd / f'{c}.txt').write_text(ans, encoding='utf-8')
        print(f'{c}：{len(facts[c]["screens"])} 個擋拆事件、解說 {len(ans)} 字', flush=True)


ACTION = {'roll': '順下', 'pop': '外拉', 'unclear': '不明顯'}
COVER = {'switch': '換防', 'blitz': '包夾', 'drop': '退防 / 延誤', 'hedge': '延誤', 'unclear': '不明確'}


def _fmt_passes(ps):
    if ps is None:
        return '（沒有傳球資料）'
    if not ps:
        return '沒偵測到傳球'
    out = []
    for p in ps:
        if p['kind'] == 'pass':
            out.append(f"{p['seconds_after_screen']} 秒後 #{p['from']}→#{p['to']}（{p['to_role']}，{p['to_zone']}）")
        else:
            out.append(f"{p['seconds_after_screen']} 秒後 #{p['from']} 的球被 #{p['to']} 拿走")
    return '；'.join(out)


def _fmt_outcome(oc):
    if oc is None:
        return '（沒有投籃資料）'
    if oc.get('shot'):
        made = {True: '進', False: '沒進', None: '看不出進沒進'}[oc['made']]
        return f"{oc['seconds_after_screen']} 秒後 #{oc['shooter_id']}（{oc['shooter_role']}）{oc['zone']}{oc['kind']}，{made}"
    return oc.get('note', '沒有出手')


def sheet(tag='v1'):
    facts = json.load(open(OUT / f'facts_{tag}.json', encoding='utf-8'))
    rows, k = [], 0

    def add(c, ev, t, item, said):
        nonlocal k
        k += 1
        rows.append([f'E{k:03d}', c, ev, t, item, said, '', ''])

    md = [f'# 端到端評估：主播解說（{tag}）\n', '每支片段的解說全文。檢查表最後一列「解說」請看這裡：有沒有講錯、有沒有編造資料裡沒有的東西。\n']
    for c in _clips():
        fps = facts[c]['fps']
        for e in facts[c]['screens']:
            f0 = min(s['frame'] for s in e['screens'])
            t = f'{f0 / fps:.1f}s（影格 {f0}）'
            ev = f"#{e['event']}"
            kind = {'single': '單擋', 'double': '雙擋（同時）', 'staggered': '連續擋（先後）'}.get(e['play_type'].split(' ')[0], e['play_type'])
            add(c, ev, t, '有這次擋拆嗎', f"{kind}，{e['n_screens']} 次擋")
            add(c, ev, t, '持球者', f"#{e['handler']}")
            for j, s in enumerate(e['screens'], 1):
                tag_s = f'（第 {j} 擋，影格 {s["frame"]}）' if e['n_screens'] > 1 else ''
                add(c, ev, t, f'掩護者{tag_s}', f"#{s['screener']}")
                add(c, ev, t, f'被擋的防守者{tag_s}', f"#{s['defender']}")
                add(c, ev, t, f'順下 / 外拉{tag_s}', ACTION.get(s['action'], s['action']))
                add(c, ev, t, f'防守應對{tag_s}', f"{COVER.get(s['coverage'], s['coverage'])}（{s['coverage_confidence']}）")
                if 'handler_action' in s:                    # v3 起：戰術細節
                    ha, hp = s['handler_action'], s['help']
                    add(c, ev, t, f'持球者怎麼用掩護{tag_s}', ha['use'] if 'direction' not in ha else f"{ha['use']}，{ha['direction']}")
                    if 'next' in ha:
                        add(c, ev, t, f'持球者接下來{tag_s}', ha['next'] + (f"（{ha['next_after_s']} 秒後）" if ha.get('next_after_s') is not None else ''))
                    for key, who in (('roller', '補防順下'), ('handler', '協防持球者')):
                        if key in hp:
                            v = hp[key]
                            if not isinstance(v, dict) and key == 'handler' and v == '沒人補防':
                                v = '沒人上前協防'
                            add(c, ev, t, f'{who}{tag_s}', f"#{v['helper']}（{v['after_s']} 秒後）" if isinstance(v, dict) else v)
                    if 'offball' in s:                       # v4 起：無球跑位
                        moves = [o for o in s['offball'] if o['move'] != '原地附近' or 'screen' in o]
                        add(c, ev, t, f'無球跑位{tag_s}', '；'.join(
                            f"#{o['player']} {o['move']}（{o['from']}→" + (f"{o['via']}→" if o.get('via') else '') + f"{o['to']}）" + (
                                f"，{o['screen']['after_s']} 秒後{o['screen']['kind']}（擋 #{o['screen']['defender']}）" if 'screen' in o else '')
                            for o in moves) or '其他人都原地附近')
                    add(c, ev, t, f'擋拆後空檔{tag_s}', '；'.join(f"#{o['player']} {o['zone']}（{o['after_s']} 秒後，最近防守者 {o['defender_ft']} 呎）"
                                                            for o in s['open_looks']) or '沒有大空檔')
            add(c, ev, t, '擋拆後傳球', _fmt_passes(e.get('passes_after')))
            add(c, ev, t, '出手結果', _fmt_outcome(e.get('outcome')))
            oc = e.get('outcome') or {}
            if oc.get('shot') and 'defense_level' in oc:
                add(c, ev, t, '出手時的防守', f"#{oc['shooter_id']} 離最近防守者 {oc['closest_defender_ft']} 呎（{oc['defense_level']}）"
                    if oc['closest_defender_ft'] is not None else '不明（有防守者在畫面外）')
            for p_ in [q for q in (e.get('passes_after') or []) if q['kind'] == 'pass' and 'catch_defense_level' in q]:
                add(c, ev, t, '接球時的防守', f"#{p_['to']} 接球時離最近防守者 {p_['catch_defender_ft']} 呎（{p_['catch_defense_level']}）"
                    if p_['catch_defender_ft'] is not None else f"#{p_['to']}：不明")
        add(c, '整支', '', '漏掉的擋拆', '（有漏掉的請在「正確答案」寫時間 / 影格和誰幫誰擋；沒漏填 t）')
        add(c, '整支', '', '解說', f'見 narration_{tag}.md 的 {c}（t = 沒講錯；f = 備註寫哪句錯 / 編造）')
        md += [f'\n## {c}\n', (OUT / f'narration_{tag}' / f'{c}.txt').read_text(encoding='utf-8'), '\n']
    with open(OUT / f'check_{tag}.csv', 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['編號', 'clip', '事件', '時間', '項目', '系統說', '對嗎(t/f/?)', '正確答案 / 備註'])
        w.writerows(rows)
    (OUT / f'narration_{tag}.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'{len(rows)} 行 → {OUT / f"check_{tag}.csv"}；解說 → {OUT / f"narration_{tag}.md"}')


def video():
    from concurrent.futures import ProcessPoolExecutor
    todo = [c for c in _clips() if not (OUT / f'{c}.mp4').exists()]
    with ProcessPoolExecutor(6) as ex:
        list(ex.map(_render, todo))
    print(f'影片 {len(list(OUT.glob("*.mp4")))} 支 → {OUT}')


def _render(c):
    import ids_framenum_video as IV
    IV.render(BATCH / f'{c}_events.pkl', ROOT / 'input_videos' / 'screen' / f'{c}.mp4', OUT / f'{c}.mp4')


def score(tag='v1'):
    rows = list(csv.reader(open(OUT / f'check_{tag}.csv', encoding='utf-8-sig')))[1:]
    cnt = collections.defaultdict(collections.Counter)
    for r in rows:
        item = r[4].split('（')[0]
        lab = r[6].strip().lower()[:1]
        if lab in ('t', 'f', '?'):
            cnt[item][lab] += 1
    print(f'{"項目":14s} 對 / 判過   準確率（? 不算）')
    for item, c in cnt.items():
        n = c['t'] + c['f']
        print(f'{item:14s} {c["t"]:3d} / {n:3d}   {c["t"] / n:.0%}' + (f'   ?{c["?"]}' if c['?'] else '') if n else f'{item:14s} —')


def carry(old_tag, new_tag):
    """新版檢查表沿用舊版標記：同片段 / 事件 / 時間 / 項目、系統說法一樣 → 抄舊的對錯和備註；不一樣的留空給人看（備註寫 v1 說什麼）。
    整支的「漏掉的擋拆」「解說」：漏掉的擋拆看事件數有沒有變；解說一律重看（文字每次都不同）。"""
    old = {tuple(r[1:6]): r for r in list(csv.reader(open(OUT / f'check_{old_tag}.csv', encoding='utf-8-sig')))[1:]}
    old_by_item = collections.defaultdict(list)
    for k, r in old.items():
        old_by_item[(k[0], k[1], k[3])].append(r)
    rows = list(csv.reader(open(OUT / f'check_{new_tag}.csv', encoding='utf-8-sig')))
    todo = []
    for r in rows[1:]:
        k = tuple(r[1:6])
        if r[4] == '解說':
            todo.append(r); continue
        if k in old:
            r[6], r[7] = old[k][6], old[k][7]
        elif r[4] == '漏掉的擋拆' and old_by_item.get((r[1], r[2], r[4])):
            r[6], r[7] = old_by_item[(r[1], r[2], r[4])][0][6:8]
        else:
            prev = old_by_item.get((r[1], r[2], r[4]))
            r[7] = f'（{old_tag} 說：{prev[0][5]}，標 {prev[0][6]}；{prev[0][7]}）' if prev else f'（{old_tag} 沒有這一行）'
            todo.append(r)
    with open(OUT / f'check_{new_tag}.csv', 'w', newline='', encoding='utf-8-sig') as fo:
        csv.writer(fo).writerows(rows)
    print(f'{len(rows) - 1} 行，沿用 {old_tag} 的 {len(rows) - 1 - len(todo)} 行，要重看 {len(todo)} 行：')
    for r in todo:
        print('  ', r[0], r[1], r[2], r[4], '→', r[5], r[7])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['select', 'run', 'sheet', 'video', 'score', 'carry'])
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--old', default='v1', help='carry：沿用哪一版的標記')
    a = ap.parse_args()
    if a.cmd == 'carry':
        carry(a.old, a.tag)
    {'select': select, 'video': video}.get(a.cmd, lambda: None)()
    if a.cmd in ('run', 'sheet', 'score'):
        {'run': run, 'sheet': sheet, 'score': score}[a.cmd](a.tag)
