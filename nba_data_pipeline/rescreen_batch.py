"""
rescreen_batch.py — 批次片段的擋拆偵測換成正式模型（2026-11-12）。

output_videos/batch 的 events.pkl 是 run_board_batch.sh 用整段 GRU（finetune_seq_final4gru_s{1,2,3}）+ 雙門檻算的，
還沒有接觸救回（inference.detect_screen_peaks rescue_lo=0.55）→ 重算一次（模型可換，見 CKPTS）。
只需要 events.pkl 裡的球場座標 / 分隊 / 持球者 → 本地 CPU 就能重算。舊的事件留在 'events_peak_v1'。
（screen_44 第二次掩護：GRU 0.64 < 0.7 漏掉；救回後抓到 f275。）

用法：python nba_data_pipeline/rescreen_batch.py [片段編號...]   # 不給 = 全部；做過的（screen_mode = MODE）跳過
"""
import sys
import glob
import pickle
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
MODE = 'seq_rescue_final5b_vit3'   # 做過的跳過（換模型 / 持球者來源就換名字）；2026-11-13 持球者改 SAM3 + Viterbi
CKPTS = [ROOT / 'nba_data_pipeline' / 'checkpoints' / f'finetune_seq_final5bgru_s{i}_best.pt' for i in (1, 2, 3)]   # 2026-11-13 換 final5b


def load_model():
    import inference as I
    from finetune_seq import load_seq
    ms = []
    for p in CKPTS:
        m, _ = load_seq(str(p), 'cpu')
        m.is_seq, m.rel_coords = True, False
        ms.append(m)
    model = I.SeqProbAverage(ms)
    model.is_seq, model.rel_coords = True, False
    return model


def rescreen(pk, model, rescue_lo=0.55, threshold=0.7, low=0.5):
    import inference as I
    d = pickle.load(open(pk, 'rb'))
    if d.get('screen_mode') == MODE:
        return None
    peaks, curves = I.detect_screen_peaks(model, 'cpu', d['court_coords'], d['player_assignment'], d['ball_acquisition'], d['fps'],
                                          anchor=3, threshold=threshold, low=low, yolo_defender_pids=d.get('yolo_defender_pids'),
                                          rescue_lo=rescue_lo)
    half, n = int(round(0.3 * d['fps'])), len(d['court_coords'])
    events = []
    for p in peaks:
        ev = {'start': max(0, p['frame'] - half), 'end': min(n - 1, p['frame'] + half), 'peak_frame': p['frame'],
              'handler_id': p['handler_id'], 'screener_id': p['screener_id'], 'score': p['score']}
        if p.get('rescued'):
            ev['rescued'] = True
        events.append(ev)
    d.setdefault('events_peak_v1', d['events'])
    d['events'] = events
    d['peak_curves'] = {f"{h}-{c}": sorted(cv.items()) for (h, c), cv in curves.items()}
    d['screen_mode'] = MODE
    pickle.dump(d, open(pk, 'wb'))
    return len(d['events_peak_v1']), len(events), sum(e.get('rescued', False) for e in events)


def main():
    import io, contextlib
    model = load_model()
    want = set(sys.argv[1:])
    pks = sorted(glob.glob(str(ROOT / 'output_videos' / 'batch' / 'screen_*_events.pkl')), key=lambda x: int(Path(x).stem.split('_')[1]))
    tot = [0, 0, 0, 0]
    for pk in pks:
        c = Path(pk).stem.split('_')[1]
        if want and c not in want:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            r = rescreen(pk, model)
        if r is None:
            continue
        tot[0] += 1; tot[1] += r[0]; tot[2] += r[1]; tot[3] += r[2]
        print(f'screen_{c}：舊 {r[0]} → 新 {r[1]}（救回 {r[2]}）', flush=True)
    print(f'RESCREEN_DONE {tot[0]} 支：擋拆事件 舊 {tot[1]} → 新 {tot[2]}（其中救回 {tot[3]}）')


if __name__ == '__main__':
    main()
