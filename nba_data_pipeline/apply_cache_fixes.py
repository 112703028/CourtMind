"""
apply_cache_fixes.py — 把 cache_label_fixes.json 的人工修正寫進訓練快取（stubs/train_contact_cache/<clip>.pkl）。
重建快取（convert_contact / ingest_new_labels）之後要再跑一次。可重複執行（每種修正各記在 _fixes_applied，已套用的跳過）。

  swap_handler_screener：持球者 ↔ 掩護者互換（handler_tid、events 的 tid、screener_tids）
  add_events：補沒標到的真掩護 [{idx(5fps 影格), tid(掩護者), src}]（contact=True；掩護者加進 screener_tids、分隊設進攻）
  set_team：改分隊 {pid: 'offense' | 'defense'}
  set_handler：改持球者（pid；分隊設進攻）
  exclude：整支片段不用（pending=True；build_court2 / finetune_seq / coverage_model 都會跳過），例：真正的持球者沒被追蹤到
  remove_events：拿掉標錯的掩護 [{idx, tid, src}]（例：其實是無球掩護）；該掩護者沒有其他掩護就從 screener_tids 拿掉

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
        if fx.get('add_events'):
            # 逐筆檢查（同一掩護者 ±2 窗內已有就跳過）—— 不用「整支只補一次」，之後追加的才會套上（2026-11-13 156 / 157 第二次補）
            step = c['fps'] / 5
            added = []
            for a in fx['add_events']:
                if any(e['tid'] == a['tid'] and abs(e['idx'] - a['idx']) <= 2 for e in c['events']):
                    continue
                c['events'].append({'frame': int(round(a['idx'] * step)), 'idx': a['idx'], 'contact': True,
                                    'tid': a['tid'], 'tid_method': 'user_fix'})
                if a['tid'] not in c['screener_tids']:
                    c['screener_tids'] = list(c['screener_tids']) + [a['tid']]
                c['track_team'][a['tid']] = 'offense'
                added.append((a['tid'], a['idx']))
            c['events'].sort(key=lambda e: e['idx'])
            done['add_events'] = True
            if added:
                msgs.append(f"補掩護 {added}")
        if fx.get('set_handler') is not None and not done.get('set_handler'):
            old = c['handler_tid']
            c['handler_tid'] = int(fx['set_handler'])
            c['track_team'][c['handler_tid']] = 'offense'
            done['set_handler'] = True
            msgs.append(f"持球者 #{old} → #{c['handler_tid']}")
        if fx.get('remove_events') and not done.get('remove_events'):
            rm = {(a['tid'], a['idx']) for a in fx['remove_events']}
            n0 = len(c['events'])
            c['events'] = [e for e in c['events'] if (e['tid'], e['idx']) not in rm]
            left = {e['tid'] for e in c['events'] if e['tid'] is not None}
            c['screener_tids'] = [t for t in c['screener_tids'] if t in left]
            done['remove_events'] = True
            msgs.append(f"拿掉掩護 {sorted(rm)}（{n0 - len(c['events'])} 個）")
        if fx.get('exclude') and not done.get('exclude'):
            c['pending'], c['excluded_reason'] = True, fx['exclude']
            done['exclude'] = True
            msgs.append(f"整支不用：{fx['exclude']}")
        pickle.dump(c, open(p, 'wb'))
        print(f"{clip}：{'；'.join(msgs) if msgs else '已套用過，跳過'}")


if __name__ == '__main__':
    main()
