"""
coverage_net.py — 防守應對：拿 ScreenNet 的主幹（看 10 人軌跡）當底子 + 新的分類頭（遷移學習）

隨機森林只看 H、S、Dh、Ds 與籃框之間 6 種距離；這裡讓模型看 10 個人完整的軌跡（含協防者輪轉），
「怎麼看軌跡」繼承自 ScreenNet（459 支片段、3 萬多個掩護窗學的；它只學過找掩護，沒看過防守方式標記 → 不洩漏）。

輸入：撞上前 1 秒 ~ 撞上後 2.4 秒（5fps 18 張）、座標已清理。slot = [H, S, 其他進攻（離 H 近到遠）, Dh, Ds, 其他防守]
     （ScreenNet 主幹只分持球者 / 進攻 / 防守，不知道誰是 Dh / Ds → 固定放 slot 5、6，分類頭直接取這幾格的摘要）
分類頭：H、S、Dh、Ds 的 64 維摘要 + 其他進攻平均 + 其他防守平均 → MLP → 換防 / 包夾 / 退防（含延誤）
訓練：凍結主幹只訓練頭（frozen），或之後再用小學習率一起微調（finetune）；左右 / 上下翻轉 + 位置雜訊；
     SportVU 人工包夾（coverage_model.sportvu_blitz 同一批）加雜訊只進訓練；類別權重平衡
評估：同 coverage_model 的按片段 5 份、5 種切法；換防 / 包夾都算對的 3 筆照規則；另算接進混合法（取代隨機森林當把關）

用法：python nba_data_pipeline/coverage_net.py [--modes frozen,finetune] [--seeds 5]
"""
import sys
import csv
import json
import pickle
import argparse
import collections
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from model import ScreenNetPairwise, N_PLAYERS
from coverage_model import collect_training, to3, CLASSES, CN, FIX_PATH
import coverage_eval as ce
import coverage_v2 as cv2_
from tactics_tools import clean_court_tracks

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OFFS = list(range(-5, 13))
W, HGT = 91.8, 49.2
BASE_CKPT = HERE / 'checkpoints' / 'finetune_contact_final_s1_best.pt'
ROLES = torch.tensor([[0] + [1] * 4 + [2] * 5])


def traj_of(c, ev, ds, dh):
    """→ (10, 18, 2) 或 None。缺的張用最近有的補；整段沒有的人 = 0。"""
    ct, team, H, S, cf = c['court_tracks'], c['track_team'], c['handler_tid'], ev['tid'], ev['idx']
    n = len(ct)
    frames = [cf + o for o in OFFS]
    pos_at = lambda p, f: ct[f].get(p) if 0 <= f < n else None
    hpos = next((pos_at(H, f) for f in (cf, cf - 1, cf + 1, cf - 2, cf + 2) if pos_at(H, f)), None)
    if hpos is None:
        return None

    def dist_h(p):
        q = next((pos_at(p, f) for f in (cf, cf - 1, cf + 1, cf - 2, cf + 2) if pos_at(p, f)), None)
        return np.inf if q is None else np.hypot(q[0] - hpos[0], q[1] - hpos[1])
    off = sorted((p for p, t in team.items() if t == 'offense' and p not in (H, S)), key=dist_h)
    de = sorted((p for p, t in team.items() if t == 'defense' and p not in (dh, ds)), key=dist_h)
    order = ([H, S] + off + [None] * 5)[:5] + ([dh, ds] + de + [None] * 5)[:5]
    T = np.zeros((N_PLAYERS, len(frames), 2), np.float32)
    for i, p in enumerate(order):
        if p is None:
            continue
        have = [k for k, f in enumerate(frames) if pos_at(p, f)]
        if not have:
            continue
        for k, f in enumerate(frames):
            kk = min(have, key=lambda h: abs(h - k))
            T[i, k] = pos_at(p, frames[kk])
    return T


def collect():
    """轉播（修正後標記）+ SportVU 人工包夾 → (T, y, group, either, key)；SportVU 的 group = None（只進訓練）"""
    fixes = json.load(open(FIX_PATH, encoding='utf-8')) if FIX_PATH.exists() else {}
    B = []
    for s in collect_training(verbose=False):
        if s['exclude']:
            continue
        tid, idx = map(int, s['key'].split(':')[1].split('@'))
        c = pickle.load(open(ce.CACHE / f"{s['clip']}.pkl", 'rb'))
        c['court_tracks'] = clean_court_tracks(c['court_tracks'])[0]
        ev = {'tid': tid, 'idx': idx}
        ds, dh = cv2_.pick_ds_dh(c, ev)
        fx = fixes.get(s['key'], {})
        dh, ds = fx.get('dh', dh), fx.get('ds', ds)
        T = traj_of(c, ev, ds, dh)
        if T is not None:
            B.append((T, to3(s['y']), s['clip'], bool(s['either']), s['key']))
    SVt = []
    for d in sorted((ROOT / 'output_videos').glob('nba_probe*')):
        if not (d / 'label.csv').exists() or not (d / 'probe_samples.pkl').exists():
            continue
        L = list(csv.DictReader(open(d / 'label.csv', encoding='utf-8-sig')))
        P = pickle.load(open(d / 'probe_samples.pkl', 'rb'))
        col = [k for k in L[0] if k.startswith('防守方式')][0]
        for r, s in zip(L, P):
            if CN.get(r[col].strip()) != 'blitz' or r.get('有把握(t/f)', '').strip() != 't':
                continue
            c = {'court_tracks': s['court_tracks'], 'track_team': s['track_team'], 'handler_tid': s['handler_tid']}
            ev = {'tid': s['screener'], 'idx': s['contact_idx']}
            ds, dh = cv2_.pick_ds_dh(c, ev)
            T = traj_of(c, ev, ds, dh)
            if T is not None:
                SVt.append(T)
    return B, np.array(SVt)


class CoverageNet(nn.Module):
    def __init__(self, base_ckpt=BASE_CKPT, d=64):
        super().__init__()
        ck = torch.load(base_ckpt, map_location='cpu', weights_only=False)
        sn = ScreenNetPairwise(ck.get('d_model', 64), ck.get('n_heads', 4), ck.get('n_layers', 2))
        sn.load_state_dict(ck['model'])
        self.base = sn.base
        self.head = nn.Sequential(nn.Linear(6 * d, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        e = self.base(x, ROLES.expand(len(x), -1).to(x.device))          # (B, 10, d)
        z = torch.cat([e[:, 0], e[:, 1], e[:, 5], e[:, 6], e[:, 2:5].mean(1), e[:, 7:].mean(1)], dim=-1)
        return self.head(z)


def augment(x, rng, noise=1.0):
    """左右 / 上下翻轉（球場對稱；補位的 0 不翻）+ 每人整段偏移 + 逐張抖動"""
    x = x.copy()
    real = np.abs(x).sum(-1, keepdims=True) > 0
    for b in range(len(x)):
        if rng.rand() < 0.5:
            x[b, ..., 0] = np.where(real[b, ..., 0], W - x[b, ..., 0], 0)
        if rng.rand() < 0.5:
            x[b, ..., 1] = np.where(real[b, ..., 0], HGT - x[b, ..., 1], 0)
    jit = rng.normal(0, noise, (len(x), N_PLAYERS, 1, 2)) + rng.normal(0, noise * 0.5, x.shape)
    return np.where(real, x + jit, 0).astype(np.float32)


def train_one(Xtr, ytr, mode, seed, epochs=60, ft_epochs=20):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = CoverageNet().to(DEV)
    cnt = collections.Counter(ytr.tolist())
    wcls = torch.tensor([len(ytr) / (3 * cnt.get(c, 1)) for c in range(3)], dtype=torch.float32).to(DEV)
    lossf = nn.CrossEntropyLoss(weight=wcls)

    def run(params, lr, n_ep, train_base):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
        for _ in range(n_ep):
            net.train()
            if not train_base:
                net.base.eval()
            perm = rng.permutation(len(Xtr))
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                xb = torch.from_numpy(augment(Xtr[b], rng)).to(DEV)
                loss = lossf(net(xb), torch.from_numpy(ytr[b]).to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()

    for p in net.base.parameters():
        p.requires_grad = False
    run(net.head.parameters(), 1e-3, epochs, False)
    if mode == 'finetune':
        for p in net.base.parameters():
            p.requires_grad = True
        run([{'params': net.base.parameters(), 'lr': 5e-5}, {'params': net.head.parameters(), 'lr': 3e-4}],
            3e-4, ft_epochs, True)
    net.eval()
    return net


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='frozen,finetune')
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(8)
    B, SV = collect()
    T = np.stack([b[0] for b in B]); y = np.array([CLASSES.index(b[1]) for b in B])
    g = np.array([b[2] for b in B]); either = np.array([b[3] for b in B]); keys = [b[4] for b in B]
    print(f'轉播 {len(y)} 筆（{dict(collections.Counter(CLASSES[i] for i in y))}）、SportVU 人工包夾 {len(SV)} 筆，'
          f'輸入 {T.shape[1:]}，主幹 = {BASE_CKPT.name}，{DEV}', flush=True)
    # LLM 回答（混合法用），對 key
    FE = pickle.load(open(ROOT / 'stubs' / 'coverage_fixed_eval.pkl', 'rb'))
    llm = {k.split(':', 1)[1]: (to3(a) if a != 'unclear' else 'unclear') for k, a in zip(FE['keys'], FE['A'])}
    A = np.array([llm.get(k, 'unclear') for k in keys], dtype=object)
    ok_fn = lambda p, i: (CLASSES[p] in ('switch', 'blitz')) if either[i] else p == y[i]
    for mode in args.modes.split(','):
        rows, hyb_rows = [], []
        for sd in range(args.seeds):
            P = np.zeros((len(y), 3))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(T, y, g):
                tr = tr[~either[tr]]
                Xtr = np.concatenate([T[tr], SV]); ytr = np.concatenate([y[tr], np.full(len(SV), CLASSES.index('blitz'))])
                net = train_one(Xtr, ytr, mode, sd)
                with torch.no_grad():
                    P[te] = torch.softmax(net(torch.from_numpy(T[te]).to(DEV)), -1).cpu().numpy()
            pred = P.argmax(1)
            ok = np.array([ok_fn(p, i) for i, p in enumerate(pred)])
            rows.append([ok[y == c].mean() for c in range(3)] + [ok.mean()])
            if sd == 0:                                     # 混合法：LLM 回答是用 random_state=0 的切法問的
                hp = np.array([CLASSES.index('drop') if P[i, 2] >= 0.7 else
                               (pred[i] if A[i] == 'unclear' else CLASSES.index(A[i])) for i in range(len(y))])
                hok = np.array([ok_fn(p, i) for i, p in enumerate(hp)])
                hyb_rows = [hok[y == c].mean() for c in range(3)] + [hok.mean()]
            print(f'  {mode} 切法 {sd + 1}/{args.seeds}：三類平均 {np.mean(rows[-1][:3]):.0%}', flush=True)
        r = np.mean(rows, 0)
        print(f'\n== {mode}（{args.seeds} 種切法平均）：換防 {r[0]:.0%}  包夾 {r[1]:.0%}  退防/延誤 {r[2]:.0%}  '
              f'三類平均 {np.mean(r[:3]):.0%}  整體 {r[3]:.0%}')
        print(f'   接進混合法（切法 1，退防 ≥0.7 不問 LLM）：換防 {hyb_rows[0]:.0%}  包夾 {hyb_rows[1]:.0%}  '
              f'退防/延誤 {hyb_rows[2]:.0%}  三類平均 {np.mean(hyb_rows[:3]):.0%}', flush=True)


if __name__ == '__main__':
    main()
