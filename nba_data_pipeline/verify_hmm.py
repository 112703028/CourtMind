"""
verify_hmm.py — 驗證逐時刻防守配對（HMM）。

吃 inference.py dump 的 events pkl（court_coords / player_assignment / events / fps），
用 tactics_tools.defensive_matchup_hmm 印出每個防守者的逐段配對，並做健檢：
  - 初始配對是否乾淨一對一（每個防守者各守一個進攻者）
  - handler 有沒有被穩定的人守（擋拆影片一定有）
  - 找出換防（同防守者多段）與兩人守一（包夾/hedge）的時間點

不碰 GPU，讀 pkl 即可。用法：
  python verify_hmm.py --pkl output_videos/screens_v129_sam3_clean_events.pkl
  python verify_hmm.py --pkl ...pkl --at 2.0   # 印 2.0 秒的配對快照
"""
import os
import sys
import pickle
import argparse
import collections as _c
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tactics_tools import (defensive_matchup_hmm, matchup_at_frame,
                           double_teams, infer_target_basket)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--at', type=float, default=None,
                    help='印某一秒的配對快照（對人工 GT 用）')
    ap.add_argument('--sigma', type=float, default=5.0)
    ap.add_argument('--switch_pen', type=float, default=8.0)
    ap.add_argument('--min_seg', type=int, default=30)
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, 'rb'))
    cc = d['court_coords']
    pa = d['player_assignment']
    events = d.get('events', [])
    fps = d.get('fps', 30.0)
    n = len(cc)

    if not events:
        print(f"{os.path.basename(args.pkl)}: 無 events，無法定 handler，跳過")
        return
    handler = events[0]['handler_id']
    basket = infer_target_basket(cc, 0, n - 1)

    matchup = defensive_matchup_hmm(
        cc, pa, handler, basket,
        sigma=args.sigma, switch_pen=args.switch_pen, min_seg=args.min_seg)

    print(f"=== {os.path.basename(args.pkl)} ===")
    print(f"handler=#{handler}  n={n} fps={fps:.0f}  "
          f"防守者{sorted(matchup)}")
    print(f"參數 sigma={args.sigma} switch_pen={args.switch_pen} "
          f"min_seg={args.min_seg}幀(~{args.min_seg/fps:.1f}s)\n")

    # 逐段配對
    for D in sorted(matchup):
        segs = matchup[D]
        line = ' -> '.join(f"#{o}({a/fps:.1f}-{b/fps:.1f}s)" for o, a, b in segs)
        print(f"  #{D}: {line}")

    # 健檢 1：初始配對一對一
    init = {}
    for D in sorted(matchup):
        segs = matchup[D]
        if segs:
            init[D] = segs[0][0]   # 第一段守的對象
    dup = [o for o, c in _c.Counter(init.values()).items() if c > 1]
    print(f"\n初始配對: {[f'#{D}守#{o}' for D, o in init.items()]}")
    print(f"  一對一: {'[X] 有重複守 ' + str(dup) if dup else '[OK]'}")

    # 健檢 2：handler 有沒有被守
    hd = [D for D, o in init.items() if o == handler]
    print(f"  守handler#{handler}的: {hd if hd else '[X] 沒人守 handler'}")

    # 健檢 3：換防（同防守者 >=2 段）
    switches = {D: segs for D, segs in matchup.items() if len(segs) >= 2}
    if switches:
        print(f"\n換防（防守者換守備對象）:")
        for D in sorted(switches):
            segs = switches[D]
            for i in range(1, len(segs)):
                po, _, pb = segs[i - 1]
                no, na, _ = segs[i]
                print(f"  #{D}: {pb/fps:.1f}s 從守#{po} -> {na/fps:.1f}s 換守#{no}")

    # 健檢 4：兩人守一（包夾/hedge）
    print(f"\n兩人以上守一個（包夾/hedge 訊號）:")
    prev = None
    any_dbl = False
    for f in range(0, n, 10):
        dt = double_teams(matchup, f)
        key = tuple(sorted(dt.items()))
        if dt and key != prev:
            any_dbl = True
            desc = ', '.join(f"守#{o}={['#'+str(x) for x in ds]}"
                             for o, ds in sorted(dt.items()))
            print(f"  {f/fps:.1f}s: {desc}")
        prev = key if dt else prev
    if not any_dbl:
        print("  (取樣沒抓到明顯包夾)")

    # 快照（對 GT）
    if args.at is not None:
        f = int(args.at * fps)
        snap = matchup_at_frame(matchup, f)
        print(f"\n{args.at:.1f}秒(f{f}) 配對快照:")
        for D in sorted(snap):
            print(f"  #{D} 守 #{snap[D]}")


if __name__ == '__main__':
    main()
