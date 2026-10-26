"""
apply_cache_fixes.py — 把 cache_label_fixes.json 的人工修正寫進訓練快取（stubs/train_contact_cache/<clip>.pkl）。
重建快取（convert_contact / ingest_new_labels）之後要再跑一次。可重複執行（每種修正各記在 _fixes_applied，已套用的跳過）。

  swap_handler_screener：持球者 ↔ 掩護者互換（handler_tid、events 的 tid、screener_tids）
  add_events：補沒標到的真掩護 [{idx(5fps 影格), tid(掩護者), src}]（contact=True；掩護者加進 screener_tids、分隊設進攻）
  set_team：改分隊 {pid: 'offense' | 'defense'}

用法：python nba_data_pipeline/apply_cache_fixes.py
"""
import json
import pickle
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE.parent / 'stubs' / 'train_contact_cache'
FIX = HERE / 'cache_label_fixes.json'


def main():
    fixes = {k: v for k, v in json.load(open(FIX, encoding='utf-8')).items() if not k.startswith('_')}
    for clip, fx in fixes.items():
        p = CACHE / f'{clip}.pkl'
        c = pickle.load(open(p, 'rb'))
        done = c.setdefault('_fixes_applied', {})
        msgs = []
        if fx.get('swap_handler_screener') and not done.get('swap_handler_screener'):
            H = c['handler_tid']
            S = next(e['tid'] for e in c['events'] if e['tid'] is not None)
            c['handler_tid'] = S
            for e in c['events']:
                if e['tid'] == S:
                    e['tid'] = H
            c['screener_tids'] = sorted({H if t == S else t for t in c['screener_tids']})
            for p_ in (H, S):
                c['track_team'][p_] = 'offense'
            done['swap_handler_screener'] = True
            msgs.append(f'持球者 #{H} ↔ 掩護者 #{S}')
        if fx.get('set_team') and not done.get('set_team'):
            for pid, t in fx['set_team'].items():
                c['track_team'][int(pid)] = t
            done['set_team'] = True
            msgs.append(f"分隊 {fx['set_team']}")
        if fx.get('add_events') and not done.get('add_events'):
            step = c['fps'] / 5
            for a in fx['add_events']:
                if any(e['tid'] == a['tid'] and abs(e['idx'] - a['idx']) <= 2 for e in c['events']):
                    continue
                c['events'].append({'frame': int(round(a['idx'] * step)), 'idx': a['idx'], 'contact': True,
                                    'tid': a['tid'], 'tid_method': 'user_fix'})
                if a['tid'] not in c['screener_tids']:
                    c['screener_tids'] = list(c['screener_tids']) + [a['tid']]
                c['track_team'][a['tid']] = 'offense'
            c['events'].sort(key=lambda e: e['idx'])
            done['add_events'] = True
            msgs.append(f"補掩護 {[(a['tid'], a['idx']) for a in fx['add_events']]}")
        pickle.dump(c, open(p, 'wb'))
        print(f"{clip}：{'；'.join(msgs) if msgs else '已套用過，跳過'}")


if __name__ == '__main__':
    main()
