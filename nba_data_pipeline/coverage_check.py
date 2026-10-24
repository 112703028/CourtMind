"""
coverage_check.py — 用擋拆「撞上那一刻」當基準，判斷防守方式（switch / hedge / drop），先在少數有人工答案的影片上試。

規則（擋拆接觸幀 cf，持球者 H、掩護者 S）：
  1. 撞上前 pre 秒：守 H 的防守者 Dh、守 S 的防守者 Ds = 這段時間平均距離最近的防守方球員
     （擋拆前兩人還分開，最近的人就是在守他的人；擋拆當下會擠在一起，不用那段）
  2. 撞上後 post 秒，看 Dh、Ds 的位置：
     switch：最後 0.5 秒 Ds 離 H 比 Dh 近、且 Dh 離 S 比 Ds 近（兩人交換）
     hedge ：沒有交換，但有一段時間 Dh、Ds 同時都在 H 的 near_ft 英尺內（兩人一起堵持球者）
     drop  ：Ds 一直沒有進到 H 的 near_ft 英尺內
     其他  ：unclear
  （之前發現 HMM 配對在擋拆當下分不清換防和包夾，所以撞上之後直接看距離）

用法：
  python nba_data_pipeline/coverage_check.py
"""
import sys
import pickle
import collections
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# 人工答案（使用者 2026-10 口述，球員編號 = 完整影片 SAM3 的編號）
CASES = [
    # (影片, 接觸幀(原片), 持球者, 掩護者, 答案)
    (124, 171, 5, 6, 'hedge'),
    (127, 72, 0, 2, 'hedge'),
    (130, 246, 6, 1, 'hedge'),
    (129, 150, 1, 2, 'switch'),
]


def team_of(player_assignment):
    """整支影片多數決隊伍。"""
    v = collections.defaultdict(collections.Counter)
    for fa in player_assignment:
        for p, t in fa.items():
            v[p][t] += 1
    return {p: c.most_common(1)[0][0] for p, c in v.items()}


def pos(cc, pid, f):
    return np.array(cc[f][pid]) if pid in cc[f] else None


def mean_dist(cc, a, b, frames):
    ds = [np.linalg.norm(pos(cc, a, f) - pos(cc, b, f)) for f in frames
          if pid_in(cc, a, f) and pid_in(cc, b, f)]
    return float(np.mean(ds)) if ds else float('inf')


def pid_in(cc, p, f):
    return 0 <= f < len(cc) and p in cc[f]


def classify(cc, team, H, S, cf, fps, pre=1.0, post=1.5, near_ft=6.0, verbose=True):
    n = len(cc)
    pre_f = list(range(max(0, cf - int(pre * fps)), max(0, cf - int(0.2 * fps))))
    post_f = list(range(cf, min(n, cf + int(post * fps))))
    end_f = post_f[-int(0.5 * fps):]
    off_team = team.get(H)
    defenders = [p for p, t in team.items() if t != off_team]

    dh = min(defenders, key=lambda d: mean_dist(cc, d, H, pre_f))
    ds = min((d for d in defenders if d != dh), key=lambda d: mean_dist(cc, d, S, pre_f))

    def d(a, b, f):
        return (np.linalg.norm(pos(cc, a, f) - pos(cc, b, f))
                if pid_in(cc, a, f) and pid_in(cc, b, f) else np.nan)

    swapped = (np.nanmean([d(ds, H, f) for f in end_f]) < np.nanmean([d(dh, H, f) for f in end_f])
               and np.nanmean([d(dh, S, f) for f in end_f]) < np.nanmean([d(ds, S, f) for f in end_f]))
    both_near = [f for f in post_f if d(dh, H, f) <= near_ft and d(ds, H, f) <= near_ft]
    ds_near = [f for f in post_f if d(ds, H, f) <= near_ft]

    if swapped:
        cov, why = 'switch', f'最後 0.5 秒 #{ds} 改守持球者、#{dh} 改守掩護者'
    elif both_near:
        cov, why = 'hedge', f'#{dh}、#{ds} 同時貼近持球者 {len(both_near) / fps:.1f} 秒'
    elif not ds_near:
        cov, why = 'drop', f'#{ds} 一直沒有靠近持球者'
    else:
        cov, why = 'unclear', f'#{ds} 靠近過持球者，但 #{dh} 不在旁邊、也沒交換'

    if verbose:
        print(f"  撞上前守持球者 #{H} 的是 #{dh}（平均 {mean_dist(cc, dh, H, pre_f):.1f} 英尺），"
              f"守掩護者 #{S} 的是 #{ds}（{mean_dist(cc, ds, S, pre_f):.1f} 英尺）")
        print("  撞上後每 0.25 秒的距離（英尺）：")
        print("     時間  Dh→持球者  Ds→持球者  Dh→掩護者  Ds→掩護者")
        for f in post_f[::max(1, int(0.25 * fps))]:
            print(f"    +{(f - cf) / fps:4.2f}s   {d(dh, H, f):6.1f}     {d(ds, H, f):6.1f}"
                  f"     {d(dh, S, f):6.1f}     {d(ds, S, f):6.1f}")
    return cov, why, dh, ds


def main():
    names = {'switch': '換防', 'hedge': '包夾', 'drop': '退防', 'unclear': '判斷不出'}
    ok = 0
    for v, cf, H, S, ans in CASES:
        dta = pickle.load(open(HERE.parent / 'output_videos' / f'screens_v{v}_sam3_events.pkl', 'rb'))
        cc, fps = dta['court_coords'], dta['fps']
        team = team_of(dta['player_assignment'])
        print(f"═══ v{v}  接觸幀 {cf}（{cf / fps:.1f} 秒）  持球者 #{H}  掩護者 #{S}  你的答案：{names[ans]} ═══")
        cov, why, dh, ds = classify(cc, team, H, S, cf, fps)
        hit = cov == ans
        ok += hit
        print(f"  → 判斷：{names[cov]}（{why}）  {'✓ 對' if hit else '✗ 錯'}\n")
    print(f"4 支裡判對 {ok} 支")


if __name__ == '__main__':
    main()
