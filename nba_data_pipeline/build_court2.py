"""output_contact_court2_503：跟 output_contact_clean_503 同片段、同標記、同座標清理，
只把球場座標換成第 2 版（court_v2.py）。npz 記 court_v2=1。
要先有 stubs/court_kp（court_kp_dump.py）。"""
import sys, glob, pickle, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parent.parent
sys.path[:0] = [str(ROOT / 'nba_data_pipeline'), str(ROOT)]
from convert_contact import CACHE_DIR, build_windows, N_OFF, N_DEF
import court_v2

roles = np.array([0] + [1] * (N_OFF - 1) + [2] * N_DEF, dtype=np.int64)
T, P, C, CTR = [], [], [], []
no_poss, changed = [], 0
for p in sorted(glob.glob(str(CACHE_DIR / 'screen_*.pkl')), key=lambda q: int(Path(q).stem.split('_')[1])):
    cid = int(Path(p).stem.split('_')[1])
    if cid > 503:
        continue
    c = pickle.load(open(p, 'rb'))
    if c.get('pending'):
        continue
    if cid < 185:
        app = collections.Counter(q for fd in c['raw_tracks'] for q in fd)
        main = [q for q in c['track_team'] if app[q] >= c['n'] / 3]
        off = sum(1 for q in main if c['track_team'][q] == 'offense')
        if off < 4 or len(main) - off < 4:
            continue
    ct2 = court_v2.court_tracks(c)
    if ct2 is None:
        no_poss.append(cid)
    else:
        c['court_tracks'] = ct2
    ws = build_windows(c, 1.0, 1.5, 0.01, 3, clean=True)
    for traj, pw, center in ws:
        T.append(traj); P.append(pw); C.append(cid); CTR.append(center)
T, P = np.stack(T), np.stack(P)
C, CTR = np.array(C, np.int32), np.array(CTR, np.int32)
lab = (P.max(axis=1) >= 0.5).astype(np.float32)
out = ROOT / 'nba_data_pipeline' / (sys.argv[1] if len(sys.argv) > 1 else 'output_contact_court2_503')   # 可指定輸出名
out.mkdir(parents=True, exist_ok=True)
for name, m in (('positive_sequences.npz', lab == 1), ('negative_sequences.npz', lab == 0)):
    np.savez_compressed(str(out / name), trajectories=T[m], roles=np.repeat(roles[None], m.sum(), 0),
                        labels=lab[m], pairwise_labels=P[m], clip_ids=C[m], centers=CTR[m],
                        anchor=np.int32(3), sigma=np.float32(1.0), clean_coords=np.float32(1.0),
                        court_v2=np.float32(1.0))
old = sum(len(np.load(ROOT / 'nba_data_pipeline/output_contact_clean_503' / n)['clip_ids']) for n in ('positive_sequences.npz', 'negative_sequences.npz'))
print(f'{out.name}：{len(set(C.tolist()))} 支、{len(T)} 窗（clean_503 是 {old} 窗）、正 {int(lab.sum())}；'
      f'沒有關鍵點檔 {len(no_poss)} 支 {no_poss[:20]}')
