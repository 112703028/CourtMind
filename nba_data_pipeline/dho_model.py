"""
dho_model.py — 用 ScreenNet 判斷手遞手（2026-11-24），不動正式擋拆模型。

上次手遞手頭跟擋拆共用主幹一起訓練 → 擋拆 ±0.4 秒掉 0.019 ~ 0.059（實驗 37）。這裡兩種做法都不改正式模型：
  A（probe）：正式 ScreenNet（finetune_contact_final5b_s1）主幹凍結 → 掩護者 / 持球者 / 防守整體特徵
             + 兩人 10 格的「有沒有拿球」→ 小 MLP 判斷「這組掩護是不是手遞手」。訓練只用 NBA 弱標籤（output_contact_nbadho）。
  B（獨立模型）：pod 實驗的 holdnba（x, y, 持球通道 + 手遞手頭 + NBA 弱標籤，自己的一份權重）只用它的手遞手頭；
             舊片段用「這支片段在驗證份」的那個模型（沒訓練過），新來源用 15 個平均。
擋拆事件照樣來自正式模型（events.pkl 的 events），對每個擋拆組窗口（同推論：5fps、10 格、撞上在第 3 格、持球者 slot 0、掩護者 slot 1）。

評估（以事件為單位）：舊 476 支（已知是 / 不是 = dho_check + dho_check3）、新來源 29 段（每段那一個手遞手；其他擋拆當不是）。
用法：python nba_data_pipeline/dho_model.py train_a          # 訓練 A → models/dho_probe.pt
     python nba_data_pipeline/dho_model.py eval             # A / B 在兩批上的結果（掃門檻）
"""
import sys
import csv
import re
import glob
import pickle
import collections
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from model import N_OFFENSE, N_DEFENSE, N_PLAYERS
from convert_contact import pair_index

CK = HERE / 'checkpoints'
PROD = CK / 'finetune_contact_final5b_s1_best.pt'
PROBE = ROOT / 'models' / 'dho_probe.pt'
SEQ_LEN, ANCHOR = 10, 3
ROLES = torch.tensor([[0] + [1] * (N_OFFENSE - 1) + [2] * N_DEFENSE])


def load_base(path=PROD):
    from eval_contact import load_model
    m, ck = load_model(path)
    for p in m.parameters():
        p.requires_grad = False
    return m


class Probe(nn.Module):
    """[掩護者特徵, 持球者特徵, 防守總和, 持球者 10 格拿球, 掩護者 10 格拿球] → 手遞手 logit。"""

    def __init__(self, d):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(3 * d + 2 * SEQ_LEN, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.ff(x).squeeze(-1)


def probe_features(base, traj):
    """traj (B, 10, 10, 3)：x, y, 持球 → (B, 3d + 20)。主幹只吃 x, y。"""
    with torch.no_grad():
        z = base.base(traj[..., :2], ROLES.expand(len(traj), -1))
    return torch.cat([z[:, 0], z[:, 1], z[:, N_OFFENSE:].sum(1), traj[:, 0, :, 2], traj[:, 1, :, 2]], dim=1)


def train_a(epochs=6, seed=0):
    torch.manual_seed(seed)
    base = load_base()
    X, Y = [], []
    for name in ('positive_sequences.npz',):
        d = np.load(HERE / 'output_contact_nbadho' / name)
        pw, pd = d['pairwise_labels'][:, pair_index(0, 1)], d['pairwise_dho'][:, pair_index(0, 1)]
        keep = pw >= 0.5                                        # 只看「slot 1 真的在掩護」的窗：學 P(手遞手 | 掩護)
        T = torch.from_numpy(d['trajectories'][keep])
        for i in range(0, len(T), 2048):
            X.append(probe_features(base, T[i:i + 2048]))
        Y.append(torch.from_numpy((pd[keep] >= 0.5).astype(np.float32)))
    X, Y = torch.cat(X), torch.cat(Y)
    print(f'NBA 掩護窗 {len(Y)}（手遞手 {int(Y.sum())}）')
    probe = Probe(base.head.ff[0].in_features // 3)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    perm = torch.randperm(len(Y))
    nv = len(Y) // 10
    va, tr = perm[:nv], perm[nv:]
    for ep in range(epochs):
        probe.train()
        for i in range(0, len(tr), 512):
            idx = tr[torch.randperm(len(tr))[:512]] if False else tr[i:i + 512]
            opt.zero_grad(); loss = lossf(probe(X[idx]), Y[idx]); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            p = torch.sigmoid(probe(X[va])) >= 0.5
        y = Y[va] >= 0.5
        tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
        print(f'epoch {ep + 1}：NBA 驗證 F1 {2 * tp / max(1, 2 * tp + fp + fn):.3f}（準 {tp / max(1, tp + fp):.0%}、抓 {tp / max(1, tp + fn):.0%}）')
    PROBE.parent.mkdir(exist_ok=True)
    torch.save({'probe': probe.state_dict(), 'd': base.head.ff[0].in_features // 3}, PROBE)


# ── 擋拆事件 → 窗口 ──────────────────────────────────────────────────────────

def event_window(d, cc, team, S, H, f, holder):
    """同推論：5fps 10 格、撞上在第 ANCHOR 格；slot 0 持球者、slot 1 掩護者、其他進攻按離持球者排、防守最近 5 個。→ (10, 10, 3) 或 None。"""
    from inference import positions_for
    step = d['fps'] / 5.0
    n = len(cc)
    window = [int(round(f + (k - ANCHOR) * step)) for k in range(SEQ_LEN)]
    if window[0] < 0 or window[-1] >= n or H not in team or S not in team:
        return None
    cnt = collections.Counter(p for g in window for p in cc[g])
    present = {p for p, c in cnt.items() if c >= SEQ_LEN // 2} | {H, S}
    hpos = next((cc[g][H] for g in reversed(window) if H in cc[g]), None)

    def dist_to_h(p):
        pos = next((cc[g][p] for g in reversed(window) if p in cc[g]), None)
        if hpos is None:
            return 0.0
        return float('inf') if pos is None else float(np.hypot(pos[0] - hpos[0], pos[1] - hpos[1]))
    off = [p for p in present if team.get(p) == team[H]]
    dfn = [p for p in present if p in team and team[p] != team[H]]
    order = ([H, S] + sorted((p for p in off if p not in (H, S)), key=dist_to_h) + [None] * N_OFFENSE)[:N_OFFENSE]
    order += (sorted(dfn, key=dist_to_h) + [None] * N_DEFENSE)[:N_DEFENSE]
    tr = np.zeros((N_PLAYERS, SEQ_LEN, 3), np.float32)
    for i, p in enumerate(order):
        if p is not None:
            tr[i, :, :2] = positions_for(p, cc, window)
            tr[i, :, 2] = [1.0 if holder[g] == p else 0.0 for g in window]
    return tr


def clip_events(pk):
    """events.pkl → [(S, H, peak, 窗口)]"""
    from tactics_tools import clean_court_tracks, clip_teams
    d = pickle.load(open(pk, 'rb'))
    cc = clean_court_tracks(d['court_coords'])[0]
    team = clip_teams(d['player_assignment'])
    out = []
    for e in d.get('events') or []:
        w = event_window(d, cc, team, e['screener_id'], e['handler_id'], e['peak_frame'], d['ball_acquisition'])
        if w is not None:
            out.append((e['screener_id'], e['handler_id'], e['peak_frame'], w, d['fps']))
    return out


_CACHE = {}


def probe_scores(d):
    """events.pkl dict → {(掩護者, 持球者, 撞上): A probe 的手遞手分數}（dho.add 用來複查「從擋拆出發」）。"""
    from tactics_tools import clean_court_tracks, clip_teams
    if 'probe' not in _CACHE:
        base = load_base()
        ck = torch.load(PROBE, map_location='cpu')
        probe = Probe(ck['d']); probe.load_state_dict(ck['probe']); probe.eval()
        _CACHE['probe'] = (base, probe)
    base, probe = _CACHE['probe']
    cc = clean_court_tracks(d['court_coords'])[0]
    team = clip_teams(d['player_assignment'])
    keys, W = [], []
    for e in d.get('events') or []:
        w = event_window(d, cc, team, e['screener_id'], e['handler_id'], e['peak_frame'], d['ball_acquisition'])
        if w is not None:
            keys.append((e['screener_id'], e['handler_id'], e['peak_frame'])); W.append(w)
    if not W:
        return {}
    with torch.no_grad():
        p = torch.sigmoid(probe(probe_features(base, torch.from_numpy(np.stack(W))))).numpy()
    return {k: float(v) for k, v in zip(keys, p)}


def labels_old():
    lab = {}
    for fn in ('dho_check/dho_check.csv', 'dho_check3/dho_check3.csv'):
        for r in csv.DictReader(open(ROOT / 'output_videos' / fn, encoding='utf-8-sig')):
            t = r['手遞手嗎(t/f/?)'].strip()
            if t in ('t', 'f'):
                lab.setdefault(r['clip'], []).append((int(r['交接影格']), int(r['給球者'].lstrip('#')), int(r['接球者'].lstrip('#')), t == 't'))
    return lab


def labels_new():
    pat = re.compile(r'(\d+)\s*:\s*#(\d+)\s*>\s*#(\d+)')
    return {r[0]: [(int(a), int(b), int(c), True) for a, b, c in pat.findall(r[4])]
            for r in list(csv.reader(open(ROOT / 'output_videos' / 'new_label' / 'dho' / 'label.csv', encoding='utf-8-sig')))[1:]}


def evaluate():
    base = load_base()
    ck = torch.load(PROBE, map_location='cpu')
    probe = Probe(ck['d']); probe.load_state_dict(ck['probe']); probe.eval()
    from eval_contact import load_model
    Bm = {}
    for s in (1, 2, 3):
        for k in range(5):
            m, c = load_model(CK / f'finetune_contact_holdnba_s{s}_cv{k}_best.pt')
            Bm[(s, k)] = (m, set(c['val_clips']))
    sets = {'舊 476 支': (sorted(glob.glob(str(ROOT / 'output_videos' / 'batch' / 'screen_*_events.pkl'))), labels_old(), False),
            '新來源 29 段': (sorted(glob.glob(str(ROOT / 'output_videos' / 'pipeline' / 'dho_*_events.pkl'))), labels_new(), True)}
    for name, (pks, lab, assume_neg) in sets.items():
        rows = []                                  # (A 分數, B 分數, 答案 True/False/None)
        for pk in pks:
            c = Path(pk).stem.replace('_events', '')
            ev = clip_events(pk)
            if not ev:
                continue
            W = torch.from_numpy(np.stack([w for *_, w, _ in ev]))
            with torch.no_grad():
                pa = torch.sigmoid(probe(probe_features(base, W))).numpy()
                cid = int(c.split('_')[1])
                ms = [m for (s, k), (m, vc) in Bm.items() if (assume_neg or cid in vc)]
                pb = (np.mean([torch.sigmoid(m(W, ROLES.expand(len(W), -1))[:, 20 + pair_index(0, 1)]).numpy() for m in ms], axis=0)
                      if ms else np.full(len(W), -1.0))            # 不在任何驗證份的片段（沒標記）→ B 不判
            for (S, H, f, _, fps), a, b in zip(ev, pa, pb):
                t = None
                for hf, A, B, tt in lab.get(c, []):
                    if A == S and B == H and -0.6 * fps <= f - hf <= 2.1 * fps:
                        t = tt
                if t is None and assume_neg:
                    t = False
                rows.append((float(a), float(b), t))
        npos = sum(1 for r in rows if r[2] is True)
        print(f'\n== {name}：擋拆事件 {len(rows)}、其中已知手遞手 {npos}、已知不是 {sum(1 for r in rows if r[2] is False)}')
        for k, nm in ((0, 'A probe'), (1, 'B 獨立模型')):
            for thr in (0.3, 0.5, 0.7, 0.9):
                sel = [r for r in rows if r[k] >= thr]
                tp = sum(1 for r in sel if r[2] is True); fp = sum(1 for r in sel if r[2] is False)
                print(f'  {nm} 門檻 {thr}：判 {len(sel)}、對 {tp}、已知錯 {fp}、沒看過 {len(sel) - tp - fp}；抓到 {tp}/{npos}')


if __name__ == '__main__':
    {'train_a': train_a, 'eval': evaluate}[sys.argv[1]]()
