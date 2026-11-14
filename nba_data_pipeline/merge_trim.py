"""
merge_trim.py — 片頭剪掉重跑（run_trim_batch.sh）的結果合併回本地批次（2026-11-14）。

從 pod 拉回 stubs/trim/<片段>_{sam3,touch,pose}.pkl、output_videos/trim/<片段>_events.pkl（+ _shot*.jpg）後：
  events.pkl → output_videos/batch（舊的備份到 stubs/backup_events_pretrim/）；籃框放大圖一起搬
  追球 → stubs/ball_batch/<片段>_ball.pkl（{'shot_sam3': ...}）、碰球 → _touch.pkl；骨架 → stubs/pose_hands/<片段>.pkl
之後跑 rescreen_batch.py（擋拆用同一套正式設定重算）、pose_pass.py add_batch（傳球）。

用法：python nba_data_pipeline/merge_trim.py
"""
import shutil
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRIM_S, TRIM_O = ROOT / 'stubs' / 'trim', ROOT / 'output_videos' / 'trim'
BATCH, BALL, POSE = ROOT / 'output_videos' / 'batch', ROOT / 'stubs' / 'ball_batch', ROOT / 'stubs' / 'pose_hands'
BACKUP = ROOT / 'stubs' / 'backup_events_pretrim'


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    n = 0
    for ev in sorted(TRIM_O.glob('screen_*_events.pkl')):
        c = ev.stem.replace('_events', '')
        need = [TRIM_S / f'{c}_sam3.pkl', TRIM_S / f'{c}_touch.pkl', TRIM_S / f'{c}_pose.pkl']
        if not all(p.exists() for p in need):
            print(f'{c}：缺檔，跳過 {[p.name for p in need if not p.exists()]}')
            continue
        old = BATCH / f'{c}_events.pkl'
        if old.exists() and not (BACKUP / old.name).exists():
            shutil.copy(old, BACKUP / old.name)
        shutil.copy(ev, old)
        for g in TRIM_O.glob(f'{c}_shot*.jpg'):
            shutil.copy(g, BATCH / g.name)
        s = pickle.load(open(need[0], 'rb'))
        pickle.dump({'n_frames': s['n_frames'], 'fps': s['fps'], 'shot_sam3': s['shot_sam3'], 'start_frame': s.get('start_frame')},
                    open(BALL / f'{c}_ball.pkl', 'wb'))
        shutil.copy(need[1], BALL / f'{c}_touch.pkl')
        shutil.copy(need[2], POSE / f'{c}.pkl')
        n += 1
    print(f'{n} 支合併回批次（舊 events 備份在 {BACKUP}）')


if __name__ == '__main__':
    main()
