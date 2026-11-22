"""
new_clip_sheet.py — 使用者從 YouTube 挑的新片段（手遞手 dho_* / 無球掩護 offball_*）→ 帶編號 + 影格的影片 + 標記表（2026-11-22）。

pod 跑完 run_new_clips_batch.sh → 下載 output_videos/pipeline/<片段>_events.pkl → 這裡畫框（ids_framenum_video）+ 列系統偵測結果當參考。
使用者標「所有掩護」（持球 + 無球 + 手遞手）：影格:掩護者>被掩護的人，手遞手後面加 dho；多個用 ; 隔開；沒有填「無」。
持球 / 無球之後用持球者自動分（被掩護的人那刻有沒有拿球），不用標。

用法：python nba_data_pipeline/new_clip_sheet.py dho        # → output_videos/new_label/dho/（影片 + label.csv）
     python nba_data_pipeline/new_clip_sheet.py offball
"""
import sys
import csv
import pickle
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

PIPE = ROOT / 'output_videos' / 'pipeline'


def main(kind):
    import ids_framenum_video as IV
    import dho
    out = ROOT / 'output_videos' / 'new_label' / kind
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pk in sorted(PIPE.glob(f'{kind}_*_events.pkl'), key=lambda p: int(p.stem.split('_')[1])):
        c = pk.stem.replace('_events', '')
        d = pickle.load(open(pk, 'rb'))
        scr = '；'.join(f"{e['peak_frame']}:#{e['screener_id']}>#{e['handler_id']}" for e in d.get('events') or [])
        try:
            dh = '；'.join(f"{p['frame']}:#{p['giver']}>#{p['receiver']}" for p in dho.detect(d))
        except Exception:
            dh = ''
        rows.append([c, len(d['court_coords']), scr, dh, '', ''])
        if not (out / f'{c}.mp4').exists():
            IV.render(pk, ROOT / 'input_videos' / kind / f'{c}.mp4', out / f'{c}.mp4')
        print(c, flush=True)
    with open(out / 'label.csv', 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.writer(fo)
        w.writerow(['clip', '影格數', '系統偵測的擋拆（參考）', '系統判的手遞手（參考）',
                    '所有掩護（影格:掩護者>被掩護的人，手遞手加 dho；多個用 ; 隔開；沒有填「無」）', '備註'])
        w.writerows(rows)
    print(f'{len(rows)} 支 → {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'dho')
