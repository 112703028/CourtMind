"""
eval_perception.py — 拿人工 GT（eval_gt.json）評 SAM3 感知層輸出。

量化「HMM 拿到的分組」跟真值差多少，分四種錯：
  - SAM3 全漏追：GT 真球員 SAM3 完全沒追到（常見於深膚色球員）
  - 門檻誤濾：真球員有追到但出現幀數 < 門檻被濾掉
  - 分隊錯：真球員進了分組但分錯邊
  - 誤抓混入：HMM 分組裡有 GT 沒有的 id（替補/裁判/跑馬燈沒濾乾淨）

改感知層（門檻/分隊/referee concept/追蹤）後跑這個看命中率有沒有升。

用法：
  python eval_perception.py                       # 全部
  python eval_perception.py --min_frac 0.25       # 調出現門檻試效果
  python eval_perception.py --vid 117             # 只看一支
"""
import os
import sys
import json
import pickle
import argparse
import collections as _c
from pathlib import Path

HERE = Path(__file__).parent


def hmm_groups(pkl, min_frac):
    """重現 defensive_matchup_hmm 的分組邏輯：多數決隊伍 + 出現>=min_frac幀。
    回 (handler, offense_used, defense_used, tracked_all, appear, n)。"""
    d = pickle.load(open(pkl, 'rb'))
    cc = d['court_coords']
    pa = d['player_assignment']
    ev = d.get('events', [])
    n = len(cc)
    h = ev[0]['handler_id'] if ev else None
    votes = {}
    for fr in pa:
        for pid, t in fr.items():
            votes.setdefault(pid, _c.Counter())[t] += 1
    maj = {pid: c.most_common(1)[0][0] for pid, c in votes.items()}
    offt = maj.get(h)
    appear = {pid: sum(1 for f in cc if pid in f) for pid in maj}
    thr = int(n * min_frac)
    off = set(p for p in maj if maj[p] == offt and appear[p] >= thr)
    dfn = set(p for p in maj if maj[p] != offt and appear[p] >= thr)
    return h, off, dfn, set(maj), appear, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=str(HERE / 'eval_gt.json'))
    ap.add_argument('--outdir', default='output_videos')
    ap.add_argument('--min_frac', type=float, default=1/3,
                    help='出現幀數門檻（佔全片比例）；改這個看門檻對命中率的影響')
    ap.add_argument('--vid', default=None, help='只看某一支')
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding='utf-8'))['videos']
    vids = [args.vid] if args.vid else sorted(gt)

    clean = 0
    total = 0
    err_counter = _c.Counter()
    for v in vids:
        if v not in gt:
            print(f"v{v}: 無 GT，跳過"); continue
        pkl = os.path.join(args.outdir, f'screens_v{v}_sam3_events.pkl')
        if not os.path.exists(pkl):
            print(f"v{v}: {pkl} 不存在，跳過"); continue
        g = gt[v]
        off_gt = set(x for x in g['offense'] if isinstance(x, int))
        def_gt = set(x for x in g['defense'] if isinstance(x, int))
        # 非整數佔位符（如 "missed_a"）= SAM3 連 id 都沒給的真球員（通常深膚色漏追）
        off_placeholder = [x for x in g['offense'] if not isinstance(x, int)]
        def_placeholder = [x for x in g['defense'] if not isinstance(x, int)]
        h, off_u, def_u, tracked, appear, n = hmm_groups(pkl, args.min_frac)
        total += 1

        missed_off = [p for p in off_gt if p not in tracked]
        missed_def = [p for p in def_gt if p not in tracked]
        filtered_off = [p for p in off_gt if p in tracked and p not in off_u and p not in def_u]
        filtered_def = [p for p in def_gt if p in tracked and p not in off_u and p not in def_u]
        wrong_off = [p for p in off_gt if p in def_u]
        wrong_def = [p for p in def_gt if p in off_u]
        false_off = [p for p in off_u if p not in off_gt and p not in def_gt]
        false_def = [p for p in def_u if p not in off_gt and p not in def_gt]

        if missed_off or missed_def or off_placeholder or def_placeholder:
            err_counter['SAM3全漏追'] += 1
        if filtered_off or filtered_def:
            err_counter['門檻誤濾'] += 1
        if wrong_off or wrong_def:
            err_counter['分隊錯'] += 1
        if false_off or false_def:
            err_counter['誤抓混入'] += 1

        # 乾淨 = 分組完全對 且 沒有 SAM3 連 id 都沒給的真球員（佔位符）
        is_clean = (off_u == off_gt and def_u == def_gt
                    and not off_placeholder and not def_placeholder)
        if is_clean:
            clean += 1

        n_off_true = len(off_gt) + len(off_placeholder)
        n_def_true = len(def_gt) + len(def_placeholder)
        print(f"═══ v{v}  handler=#{h} ═══  {'[OK] 乾淨5v5' if is_clean else '[X]'}")
        print(f"  GT   進攻{sorted(off_gt)}+{off_placeholder}={n_off_true}人 "
              f"防守{sorted(def_gt)}+{def_placeholder}={n_def_true}人")
        print(f"  HMM  進攻{sorted(off_u)} 防守{sorted(def_u)}")
        if off_placeholder or def_placeholder:
            print(f"  [SAM3全漏追-無id] 進攻{off_placeholder} 防守{def_placeholder}")
        if missed_off or missed_def:
            print(f"  [SAM3全漏追] 進攻{missed_off} 防守{missed_def}")
        if filtered_off or filtered_def:
            fo = [(p, appear.get(p)) for p in filtered_off]
            fd = [(p, appear.get(p)) for p in filtered_def]
            print(f"  [門檻誤濾] 進攻{fo} 防守{fd}  (幀數/{n})")
        if wrong_off or wrong_def:
            print(f"  [分隊錯] 進攻→防守{wrong_off} 防守→進攻{wrong_def}")
        if false_off or false_def:
            print(f"  [誤抓混入] 進攻{false_off} 防守{false_def}")
        print()

    print("══════════════════════════════")
    print(f"乾淨 5v5 命中：{clean}/{total}  (min_frac={args.min_frac:.2f})")
    if err_counter:
        print("錯誤分佈（幾支影片犯此錯）：")
        for k, c in err_counter.most_common():
            print(f"  {k}: {c} 支")


if __name__ == '__main__':
    main()
