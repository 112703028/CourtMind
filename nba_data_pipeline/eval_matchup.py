"""
eval_matchup.py — 拿逐幀 GT（eval_matchup_gt.json）評 HMM 配對準確率。

對每筆 GT [frame, 防守者, 該守進攻者]，看 HMM 在該幀判的對不對。
用來比較不同 blend / sigma / switch_pen 參數對配對命中率的影響。

用法：
  python eval_matchup.py                       # 預設參數
  python eval_matchup.py --blend 0 --blend 0.5 # 比較多組 blend
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tactics_tools import (defensive_matchup_hmm, matchup_at_frame,
                           infer_target_basket)

HERE = Path(__file__).parent


def run(gt, outdir, blend, sigma, switch_pen, min_seg, verbose):
    hit = miss = 0
    miss_detail = []
    for v, g in gt.items():
        pkl = os.path.join(outdir, f'screens_v{v}_sam3_events.pkl')
        if not os.path.exists(pkl):
            continue
        d = pickle.load(open(pkl, 'rb'))
        cc = d['court_coords']; pa = d['player_assignment']; ev = d.get('events', [])
        if not ev:
            continue
        h = ev[0]['handler_id']; n = len(cc)
        basket = infer_target_basket(cc, 0, n - 1)
        m = defensive_matchup_hmm(cc, pa, h, basket, blend=blend,
                                  sigma=sigma, switch_pen=switch_pen, min_seg=min_seg)
        for f, D, O_gt, note in g['frames']:
            if D not in cc[f]:
                continue   # SAM3 漏追該防守者，不算配對錯（是感知層問題）
            O_hmm = matchup_at_frame(m, f).get(D)
            if O_hmm == O_gt:
                hit += 1
            else:
                miss += 1
                miss_detail.append((v, f, D, O_gt, O_hmm, note))
    return hit, miss, miss_detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=str(HERE / 'eval_matchup_gt.json'))
    ap.add_argument('--outdir', default='output_videos')
    ap.add_argument('--blend', type=float, action='append',
                    help='可給多次比較，如 --blend 0 --blend 0.5')
    ap.add_argument('--sigma', type=float, default=5.0)
    ap.add_argument('--switch_pen', type=float, default=8.0)
    ap.add_argument('--min_seg', type=int, default=30)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding='utf-8'))['videos']
    blends = args.blend if args.blend else [0.5]

    for blend in blends:
        hit, miss, md = run(gt, args.outdir, blend, args.sigma,
                            args.switch_pen, args.min_seg, args.verbose)
        tot = hit + miss
        print(f"══ blend={blend}  sigma={args.sigma} switch_pen={args.switch_pen} ══")
        print(f"  配對命中：{hit}/{tot}  ({hit/tot*100:.0f}%)" if tot else "  無資料")
        if md:
            print(f"  未命中：")
            for v, f, D, Og, Oh, note in md:
                print(f"    v{v} f{f} #{D}: 應守#{Og} HMM判#{Oh}  {note}")
        print()


if __name__ == '__main__':
    main()
