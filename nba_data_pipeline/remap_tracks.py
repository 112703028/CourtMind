"""
remap_tracks.py — 重新追蹤後，把新的球員編號換回舊編號（2026-11-14，片頭剪掉重跑 SAM3 用）。

重跑 SAM3 編號會整個重排 → 使用者用舊編號標過的東西（傳球檢查表、端到端評估…）會對不上。
新舊兩份追蹤在同一張影格的框 IoU：共同出現的影格裡平均 IoU ≥ 0.5、而且共同出現 ≥ 新追蹤的 30% → 沿用舊編號（一對一、分數高的先配）；
配不到的給新編號（舊的最大編號 + 1 起）。舊追蹤裡配不到新追蹤的（例 screen_366 的 #3：前一段畫面帶過來的假球員）自然消失。

用法：python nba_data_pipeline/remap_tracks.py <新的 sam3.pkl> <舊的 events.pkl 或 sam3.pkl>   # 直接改寫新的 pkl（player_tracks）
"""
import sys
import pickle


def _box(v):
    return v.get('bbox') if isinstance(v, dict) else v


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def remap(new_tracks, old_tracks, min_iou=0.5, min_frac=0.3):
    """→ (換好編號的 new_tracks, {新編號: 舊編號 or 新給的編號})"""
    import collections
    tot, cnt, life = collections.Counter(), collections.Counter(), collections.Counter()
    for nf, of in zip(new_tracks, old_tracks):
        for q, v in nf.items():
            life[q] += 1
            bq = _box(v)
            if bq is None:
                continue
            for o, w in of.items():
                bo = _box(w)
                if bo is not None:
                    s = _iou(bq, bo)
                    if s > 0.1:
                        tot[(q, o)] += s; cnt[(q, o)] += 1
    cands = sorted(((tot[k] / cnt[k], cnt[k], k) for k in tot
                    if tot[k] / cnt[k] >= min_iou and cnt[k] >= min_frac * life[k[0]]), reverse=True)
    mp, used = {}, set()
    for _, _, (q, o) in cands:
        if q not in mp and o not in used:
            mp[q] = o; used.add(o)
    nxt = max([o for of in old_tracks for o in of] + [-1]) + 1
    for q in sorted(life):
        if q not in mp:
            mp[q] = nxt; nxt += 1
    return [{mp[q]: v for q, v in fd.items()} for fd in new_tracks], mp


def main():
    new_p, old_p = sys.argv[1], sys.argv[2]
    new = pickle.load(open(new_p, 'rb'))
    old = pickle.load(open(old_p, 'rb'))
    tracks, mp = remap(new['player_tracks'], old['player_tracks'])
    kept = {q: o for q, o in mp.items() if any(o in fd for fd in old['player_tracks'])}
    gone = sorted({o for fd in old['player_tracks'] for o in fd} - set(mp.values()))
    new['player_tracks'], new['id_remap'] = tracks, mp
    pickle.dump(new, open(new_p, 'wb'))
    print(f'沿用舊編號 {len(kept)} 個、新給 {len(mp) - len(kept)} 個；舊的配不到（消失）：{gone}')


if __name__ == '__main__':
    main()
