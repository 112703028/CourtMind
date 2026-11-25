"""
offball_new_labels.py — 使用者標的新來源 105 段（output_videos/new_label/offball/label.csv）：整理標記 + 持球 / 無球分類 + 現在的 ScreenNet 抓到多少（2026-11-25）。

標記格式：影格:#掩護者>#被掩護的人，後面可加 dho（手遞手）/ 假擋；多個用逗號或分號隔開。
持球 / 無球：被掩護的人在**撞上前** 0.5 秒內有沒有被判成持球者（events.pkl 的 ball_acquisition）。
  原本看前後 0.5 秒 → 掩護完馬上接球的無球掩護（Pin-down 跑出來接球）被算成持球：持球 66 / 無球 56 → 改成撞上前：28 / 94。
  持球者用的是推論時的版本（新片段還沒做手腕修正），仍不完全準。
備註「#4 和 #10 後來交換標籤」（offball_002）等只記下，不自動處理。

用法：python nba_data_pipeline/offball_new_labels.py
"""
import re
import csv
import pickle
import collections
from pathlib import Path

ROOT = Path(__file__).parent.parent
SHEET = ROOT / 'output_videos' / 'new_label' / 'offball' / 'label.csv'
PIPE = ROOT / 'output_videos' / 'pipeline'
PAT = re.compile(r'(\d+)\s*:?\s*[#$](\d+)\s*>\s*#(\d+)\s*(\(?dho\)?)?\s*(假擋|甲黨)?', re.I)


def parse(cell):
    """→ [(影格, 掩護者, 被掩護的人, 是不是手遞手, 是不是假擋)]；被掩護的人寫 ? 的略過。"""
    return [(int(f), int(s), int(h), bool(d), bool(fake)) for f, s, h, d, fake in PAT.findall(cell)]


def labels():
    return {r[0]: parse(r[4]) for r in list(csv.reader(open(SHEET, encoding='utf-8-sig')))[1:]}


def main():
    lab = labels()
    stat = collections.Counter()
    found = collections.Counter()
    for c, items in lab.items():
        pk = PIPE / f'{c}_events.pkl'
        if not pk.exists():
            stat['沒有 events.pkl'] += 1
            continue
        d = pickle.load(open(pk, 'rb'))
        fps, ba = d['fps'], d['ball_acquisition']
        ev = [(e['peak_frame'], e['screener_id'], e['handler_id']) for e in d.get('events') or []]
        for f, S, H, dho, fake in items:
            if fake:
                stat['假擋'] += 1
                continue
            win = range(max(0, f - int(0.5 * fps)), min(len(ba), f + 1))
            kind = '手遞手' if dho else ('持球' if any(ba[g] == H for g in win) else '無球')
            stat[kind] += 1
            hit_s = any(es == S and abs(ef - f) <= fps for ef, es, eh in ev)
            hit_pair = any(es == S and eh == H and abs(ef - f) <= fps for ef, es, eh in ev)
            found[kind, '掩護者對'] += hit_s
            found[kind, '掩護者 + 被掩護的人都對'] += hit_pair
    n = sum(stat[k] for k in ('持球', '無球', '手遞手'))
    print(f'{len(lab)} 段、掩護 {n} 個：持球 {stat["持球"]}、無球 {stat["無球"]}、手遞手 {stat["手遞手"]}；假擋 {stat["假擋"]}；沒有 events.pkl {stat["沒有 events.pkl"]}')
    print('現在的 ScreenNet（±1 秒）：')
    for k in ('持球', '無球', '手遞手'):
        print(f'  {k}：掩護者對 {found[k, "掩護者對"]}/{stat[k]}、兩人都對 {found[k, "掩護者 + 被掩護的人都對"]}/{stat[k]}')


if __name__ == '__main__':
    main()
