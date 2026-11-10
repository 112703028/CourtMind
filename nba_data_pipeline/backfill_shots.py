"""
backfill_shots.py — 批次片段補投籃偵測 + 鏡頭切點（2026-11-07）。

output_videos/batch 的 476 支是沒開投籃偵測跑的 → events.pkl 的 shots / cuts 全空：投籃沒被傳球偵測排除、
跨鏡頭的換手也擋不掉、戰術分析的出手結果一直是「沒有偵測到出手」。現在 stubs/ball_batch 都有 10fps 追球了，
用跟 inference.py 一樣的 detect_shots 補算。

pod（GPU）：python nba_data_pipeline/backfill_shots.py run      → stubs/shots_batch/<片段>.pkl =（shots, cuts）+ 籃框放大圖
（本地 CPU 也能跑，一支 ~45 秒）
本地：     python nba_data_pipeline/backfill_shots.py merge    → 寫進 output_videos/batch/<片段>_events.pkl 的 shots / cuts
"""
import sys
import pickle
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
OUT = ROOT / 'stubs' / 'shots_batch_v2'          # 2026-11-10 第 2 版投籃規則（shot_v2）；第 1 版在 stubs/shots_batch
RF_CACHE = ROOT / 'stubs' / 'rf_cache'           # Roboflow 每張偵測存檔（shot_eval.py 同一份），改規則可以本地重算


def run(first=()):
    """first：先做這些片段（例：檢查表上的），其餘照編號。"""
    from utils import read_video
    from shot_detect import run_rf, detect_shots
    OUT.mkdir(parents=True, exist_ok=True)
    bps = sorted((ROOT / 'stubs' / 'ball_batch').glob('screen_*_ball.pkl'), key=lambda x: int(x.stem.split('_')[1]))
    bps = [b for b in bps if int(b.stem.split('_')[1]) in first] + [b for b in bps if int(b.stem.split('_')[1]) not in first]
    for bp in bps:
        name = bp.stem.replace('_ball', '')
        o, ev = OUT / f'{name}.pkl', ROOT / 'output_videos' / 'batch' / f'{name}_events.pkl'
        if o.exists() or not ev.exists():
            continue
        s = pickle.load(open(bp, 'rb'))['shot_sam3']
        d = pickle.load(open(ev, 'rb'))
        frames = read_video(str(ROOT / 'input_videos' / 'screen' / f'{name}.mp4'))
        rp = RF_CACHE / f'batch_{name}.pkl'
        if rp.exists():
            rf = pickle.load(open(rp, 'rb'))
        else:
            rf = run_rf(frames, s['orig_fps'])
            RF_CACHE.mkdir(parents=True, exist_ok=True)
            pickle.dump(rf, open(rp, 'wb'))
        shots, cuts = detect_shots(s, frames, rf, d['player_tracks'], grid_prefix=str(ROOT / 'output_videos' / 'batch' / name))
        pickle.dump((shots, cuts), open(o, 'wb'))
        print(f'{name}：出手 {len(shots)}、鏡頭切點 {cuts}', flush=True)
    print('SHOTS_DONE')


def merge():
    n = 0
    for o in sorted(OUT.glob('screen_*.pkl')):
        ev = ROOT / 'output_videos' / 'batch' / f'{o.stem}_events.pkl'
        if not ev.exists():
            continue
        shots, cuts = pickle.load(open(o, 'rb'))
        d = pickle.load(open(ev, 'rb'))
        d['shots'], d['cuts'] = shots, cuts
        pickle.dump(d, open(ev, 'wb'))
        n += 1
    print(f'{n} 支片段寫入 shots / cuts')


if __name__ == '__main__':
    if sys.argv[1] == 'run':      # python backfill_shots.py run [13,17,21 先做這些]
        run([int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else ())
    else:
        merge()
