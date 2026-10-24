"""
scan_coverage.py — 自動掃所有影片的所有擋拆事件，判 coverage（switch/hedge/drop）。

對每支影片：find_screen_instances 找擋拆 → defensive_matchup_hmm 配對
→ classify_coverage_hmm 判每次擋拆是 switch/hedge/drop/unclear。

用法：
  python scan_coverage.py                    # 全部有 events pkl 的影片
  python scan_coverage.py --vid 130          # 單支
"""
import os
import sys
import glob
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tactics_tools import (find_screen_instances, defensive_matchup_hmm,
                           classify_coverage_hmm, infer_target_basket)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='output_videos')
    ap.add_argument('--vid', default=None)
    args = ap.parse_args()

    if args.vid:
        pkls = [os.path.join(args.outdir, f'screens_v{args.vid}_sam3_events.pkl')]
    else:
        pkls = sorted(glob.glob(os.path.join(args.outdir, 'screens_v*_sam3_events.pkl')))

    tally = {}
    for pkl in pkls:
        if not os.path.exists(pkl):
            print(f"{pkl} 不存在"); continue
        name = os.path.basename(pkl).replace('_sam3_events.pkl', '').replace('screens_', '')
        d = pickle.load(open(pkl, 'rb'))
        cc = d['court_coords']; pa = d['player_assignment']; ev = d.get('events', [])
        fps = d.get('fps', 30.0); n = len(cc)
        if not ev:
            continue
        basket = infer_target_basket(cc, 0, n - 1)
        matchup = defensive_matchup_hmm(cc, pa, ev[0]['handler_id'], basket)
        inst = find_screen_instances(cc, pa, ev)
        if not inst:
            print(f"═══ {name} ═══  (無擋拆事件)")
            continue
        print(f"═══ {name} ═══  handler=#{ev[0]['handler_id']}  {len(inst)} 次擋拆")
        for it in inst:
            cov = classify_coverage_hmm(
                cc, matchup, it['handler_id'], it['screener_id'],
                it['defender_id'], it['frame'], basket, fps=fps)
            tally[cov['coverage']] = tally.get(cov['coverage'], 0) + 1
            print(f"  f{it['frame']:3d} screener#{it['screener_id']} 擋 "
                  f"守screener的#{it['defender_id']}  "
                  f"→ [{cov['coverage']:8s}] ({cov['confidence']}) {cov['detail']}")
        print()

    print("══════════════════════════════")
    print(f"coverage 統計: {tally}")


if __name__ == '__main__':
    main()
