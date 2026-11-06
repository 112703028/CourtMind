"""
finetune_seq.py — C1：整段時序定位（E2E-Spot / T-DEED 的想法改成軌跡版）

現在的 ScreenNet：每個 10 張窗（2 秒）各自打分，相鄰窗互不知道、只看撞上前 0.6 秒 / 後 1.2 秒。
這裡在它後面加一個沿時間讀完整支片段的雙向 GRU：

  每個 (候選掩護者, 窗) → 沿用 ScreenNet 的 LSTM + transformer + 配對頭第一層 → 向量 h[t]、原本分數 w[t]
  同一個候選人整支片段的 h[1..T] → 雙向 GRU → 修正量 δ[t]
  最終 logit[t] = w[t] + δ[t]        （δ 的輸出層初始化成 0 → 一開始跟原模型一模一樣，只學修正）

初始化：第 k 份用「同一種子、沒看過第 k 份」的 court2fix2 交叉驗證模型（finetune_contact_court2fix2_s{s}_cv{k}），
不會偷看考題。考題、標籤（撞上張高斯 σ=1、沒撞到 1.5）、座標（第 2 版 + 清理）、選最佳 epoch 的方式（考題份的逐窗 F1）
都跟 court2fix2 一樣 → five_seed_eval 可以直接比。

輸出：每個種子 5 份考完 → stubs/seq_curves/<tag>_s<種子>.pkl（{clip: {候選人: {影格: 分數}}}，同 five_seed_eval 曲線格式）

正式模型（全部資料）：--final --init final4 --epochs N --threshold T → checkpoints/finetune_seq_<tag>_s<種子>_best.pt，
inference.py --ckpt 給這種 ckpt 會自動走整段 GRU + 連續一段算一次。

用法（pod）：
  python nba_data_pipeline/finetune_seq.py --build                       # 先建序列快取 stubs/seq_data.pkl（CPU，一次）
  python nba_data_pipeline/finetune_seq.py --seed 1 --tag seq            # 5 份依序訓練 + 存曲線
"""
import sys
import math
import glob
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
from convert_contact import CACHE_DIR, SEQ_LEN, N_OFF, N_DEF
from eval_contact import load_model, window_traj, ROLES

ANCHOR, SIGMA, SIGMA_NC, MIN_TARGET = 3, 1.0, 1.5, 0.01
DATA = ROOT / 'stubs' / 'seq_data.pkl'
OUTD = ROOT / 'stubs' / 'seq_curves'
CK = HERE / 'checkpoints'


# ── 資料：每支片段 × 每個候選人 → 連續的窗序列 ─────────────────────────────────

def clip_coords(c):
    import court_v2
    from tactics_tools import clean_court_tracks
    ct = court_v2.court_tracks(c)
    return clean_court_tracks(ct if ct is not None else c['court_tracks'])[0]


N_FEAT = 10


def geo_feats(ct, center, start, H, cand, dfn):
    """每個窗在撞上格（center）的幾何特徵（直接從清理後座標算，不經 ScreenNet 壓縮）：
    持球者↔候選人、候選人↔最近防守者、持球者↔最近防守者距離（/10 ft，上限 3）、兩人速度（ft/張 /10）、
    候選人往持球者靠近的速度、撞上格持球者 / 候選人有沒有座標（0/1）、整個窗兩人缺座標的比例。
    算不出來的填 0（缺座標旗標會告訴 GRU）。"""
    def pos(p, f):
        return ct[f].get(p) if 0 <= f < len(ct) else None

    def d(a, b):
        return math.dist(a, b) if a is not None and b is not None else None

    def cap(x, k=10.0, m=3.0):
        return 0.0 if x is None else min(x / k, m)

    ph, pc = pos(H, center), pos(cand, center)
    defs = [ct[center][p] for p in dfn if p in ct[center]] if 0 <= center < len(ct) else []
    d_hc = d(ph, pc)
    d_cd = min((math.dist(pc, q) for q in defs), default=None) if pc is not None else None
    d_hd = min((math.dist(ph, q) for q in defs), default=None) if ph is not None else None
    v_h = d(pos(H, center - 1), pos(H, center + 1))
    v_c = d(pos(cand, center - 1), pos(cand, center + 1))
    a0, a1 = d(pos(H, center - 1), pos(cand, center - 1)), d(pos(H, center + 1), pos(cand, center + 1))
    appr = (a0 - a1) if a0 is not None and a1 is not None else None
    win = range(start, min(len(ct), start + SEQ_LEN))
    return np.array([cap(d_hc), cap(d_cd), cap(d_hd), cap(v_h), cap(v_c),
                     0.0 if appr is None else float(np.clip(appr / 5.0, -3, 3)),
                     float(ph is None), float(pc is None),
                     float(np.mean([H not in ct[f] for f in win])), float(np.mean([cand not in ct[f] for f in win]))],
                    np.float32)


def clip_sequences(c, ct, train):
    """train=True 照 build_windows（標記的人一定算在場、防守者 < 2 人有座標的窗不要）；
    False 照 score_curves（評估 / 推論）。回 [dict(cand, frames, traj (L,10,10,2), y (L,))]，序列在缺窗處斷開。"""
    n, raw, team, H = c['n'], c['raw_tracks'], c['track_team'], c['handler_tid']
    ev = [e for e in c['events'] if e['tid'] is not None]
    per = collections.defaultdict(dict)
    for start in range(c['valid_start_idx'], n - SEQ_LEN + 1):
        wr, wc = raw[start:start + SEQ_LEN], ct[start:start + SEQ_LEN]
        cnt = collections.Counter(p for fd in wr for p in fd)
        present = {p for p, k in cnt.items() if k >= SEQ_LEN // 2}
        forced = (list(c['screener_tids']) + [e['tid'] for e in ev] + [H, c.get('defender_tid')]) if train else [H]
        for p in forced:
            if p is not None and cnt.get(p, 0) >= 1:
                present.add(p)
        off = [p for p in present if team.get(p) == 'offense']
        dfn = [p for p in present if team.get(p) == 'defense']
        if H is None or H not in off or len(off) < 2:
            continue
        hpos = next((wc[k][H] for k in range(SEQ_LEN - 1, -1, -1) if H in wc[k]), None)

        def dist_to_h(p):
            if hpos is None:
                return 0.0
            pos = next((wc[k][p] for k in range(SEQ_LEN - 1, -1, -1) if p in wc[k]), None)
            return float('inf') if pos is None else math.dist(pos, hpos)

        order_def = (sorted(dfn, key=dist_to_h) + [None] * N_DEF)[:N_DEF]
        center = start + ANCHOR
        for cand in (p for p in off if p != H):
            rest = sorted((p for p in off if p not in (H, cand)), key=dist_to_h)
            tr = window_traj(wc, ([H, cand] + rest + [None] * N_OFF)[:N_OFF] + order_def)
            if train and int((np.abs(tr[N_OFF:]).sum(axis=(1, 2)) > 0).sum()) < 2:
                continue
            y = max([math.exp(-((center - e['idx']) ** 2) / (2 * (SIGMA if e['contact'] else SIGMA_NC) ** 2))
                     for e in ev if e['tid'] == cand] + [0.0])
            per[cand][center] = (tr, y if y >= MIN_TARGET else 0.0, geo_feats(ct, center, start, H, cand, dfn))
    out = []
    for cand, d in per.items():
        fs = sorted(d)
        run = [fs[0]]
        for f in fs[1:] + [None]:
            if f is not None and f == run[-1] + 1:
                run.append(f); continue
            out.append(dict(cand=cand, frames=np.array(run), traj=np.stack([d[g][0] for g in run]).astype(np.float32),
                            y=np.array([d[g][1] for g in run], np.float32),
                            feat=np.stack([d[g][2] for g in run])))
            if f is not None:
                run = [f]
    return out


def build(path=DATA):
    """跟 build_court2.py 同一批片段（≤503、185 以前要至少 4 對 4）。"""
    data = {}
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
        ct = clip_coords(c)
        data[cid] = dict(train=clip_sequences(c, ct, True), eval=clip_sequences(c, ct, False))
        if len(data) % 50 == 0:
            print(f'{len(data)} 支', flush=True)
    pickle.dump(data, open(path, 'wb'))
    n_tr = sum(len(s['y']) for d in data.values() for s in d['train'])
    print(f'{len(data)} 支片段、訓練 {sum(len(d["train"]) for d in data.values())} 條序列 / {n_tr} 個窗 → {path}')


# ── 模型 ─────────────────────────────────────────────────────────────────────

class ScreenSeq(nn.Module):
    def __init__(self, pairwise, hidden=64, n_feat=0):
        super().__init__()
        self.base, self.ff = pairwise.base, pairwise.head.ff      # ff = Linear(3d,d) ReLU Dropout Linear(d,1)
        d = self.base.d_model
        self.n_feat = n_feat                                      # > 0：GRU 另外吃每窗的幾何特徵（geo_feats）
        self.gru = nn.GRU(d + 1 + n_feat, hidden, batch_first=True, bidirectional=True)
        self.delta = nn.Linear(2 * hidden, 1)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def window(self, traj):
        """traj (N,10,10,2) → h (N,d)、原本的分數 logit w (N,)。"""
        emb = self.base(traj, ROLES.to(traj.device).expand(len(traj), -1))
        pair = torch.cat([emb[:, 0], emb[:, 1], emb[:, N_OFF:].sum(1)], dim=-1)
        h = self.ff[2](self.ff[1](self.ff[0](pair)))
        return h, self.ff[3](h).squeeze(-1)

    def forward(self, seqs, feats=None):
        """seqs：list of (L_i,10,10,2) tensor；feats（n_feat > 0 時）：list of (L_i, n_feat) → list of (L_i,) logit。"""
        lens = [len(s) for s in seqs]
        h, w = self.window(torch.cat(seqs))
        x = [h, w[:, None]] + ([torch.cat(feats)] if self.n_feat else [])
        hs = torch.split(torch.cat(x, -1), lens)
        ws = torch.split(w, lens)
        pad = nn.utils.rnn.pad_sequence(hs, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(pad, torch.tensor(lens), batch_first=True, enforce_sorted=False)
        g, _ = self.gru(packed)
        g, _ = nn.utils.rnn.pad_packed_sequence(g, batch_first=True)
        dl = self.delta(g).squeeze(-1)
        return [ws[i] + dl[i, :lens[i]] for i in range(len(seqs))]


# ── 訓練 / 評估 ──────────────────────────────────────────────────────────────

def run_fold(data, seed, fold, args, device):
    base_ck = CK / f'finetune_contact_{args.init}_s{seed}_cv{fold}_best.pt'
    pw_model, ck = load_model(base_ck)
    val = set(ck['val_clips'])
    torch.manual_seed(seed); np.random.seed(seed)
    model = ScreenSeq(pw_model, args.hidden, N_FEAT if args.feat else 0).to(device)
    tr = [s for cid, d in data.items() if cid not in val for s in d['train']]
    va = [s for cid, d in data.items() if cid in val for s in d['train']]
    opt = torch.optim.Adam([{'params': list(model.base.parameters()) + list(model.ff.parameters()), 'lr': args.lr_base},
                            {'params': list(model.gru.parameters()) + list(model.delta.parameters()), 'lr': args.lr_head}])
    pos_w = torch.tensor(args.pos_weight, device=device)

    def loss_of(batch, train):
        model.train(train)
        out = model([torch.from_numpy(s['traj']).to(device) for s in batch], _feats(batch, model, device))
        y = torch.cat([torch.from_numpy(s['y']) for s in batch]).to(device)
        z = torch.cat(out)
        return nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=pos_w), z, y

    def val_f1():            # 跟 court2fix2 同樣的選法：考題份逐窗 F1（目標 ≥0.5 vs 機率 ≥0.5）
        tp = fp = fn = 0
        with torch.no_grad():
            for i in range(0, len(va), args.batch):
                _, z, y = loss_of(va[i:i + args.batch], False)
                p, t = torch.sigmoid(z) >= 0.5, y >= 0.5
                tp += int((p & t).sum()); fp += int((p & ~t).sum()); fn += int((~p & t).sum())
        return 2 * tp / max(1, 2 * tp + fp + fn)

    best, best_state, bad = val_f1(), {k: v.clone() for k, v in model.state_dict().items()}, 0
    print(f'  第 {fold} 份：訓練 {len(tr)} 條、考題 {len(val)} 支；起點（= 原模型）逐窗 F1 {best:.3f}', flush=True)
    rng = np.random.RandomState(seed * 100 + fold)
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(order), args.batch):
            batch = [tr[j] for j in order[i:i + args.batch]]
            if args.crop:                                   # 隨機切段（資料增強）：長序列只取一段
                batch = [_crop(s, args.crop, rng) for s in batch]
            if args.flip_aug:
                batch = [_flip(s, rng) for s in batch]
            loss, _, _ = loss_of(batch, True)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(batch)
        f1 = val_f1()
        tag = ''
        if f1 > best:
            best, best_state, bad, tag = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0, ' ← best'
        else:
            bad += 1
        print(f'    epoch {ep:2d} loss {tot / len(tr):.4f} 逐窗 F1 {f1:.3f}{tag}', flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    curves = {}
    with torch.no_grad():
        for cid in val:
            cv = collections.defaultdict(dict)
            for s in data[cid]['eval']:
                p = torch.sigmoid(model([torch.from_numpy(s['traj']).to(device)], _feats([s], model, device))[0]).cpu().numpy()
                for f, v in zip(s['frames'], p):
                    cv[s['cand']][int(f)] = float(v)
            curves[cid] = dict(cv)
    return curves, best


def train_final(data, args, device):
    """正式模型：起點 finetune_contact_<init>_s<seed>_best.pt（例 final4，全部資料訓練的），全部片段、固定 --epochs 輪。
    存 checkpoints/finetune_seq_<tag>_s<seed>_best.pt（inference.py 認 'seq' 欄位 → 整段 GRU + 連續一段算一次）。"""
    base_ck = CK / f'finetune_contact_{args.init}_s{args.seed}_best.pt'
    pw_model, ck = load_model(base_ck)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = ScreenSeq(pw_model, args.hidden, N_FEAT if args.feat else 0).to(device)
    tr = [s for d in data.values() for s in d['train']]
    opt = torch.optim.Adam([{'params': list(model.base.parameters()) + list(model.ff.parameters()), 'lr': args.lr_base},
                            {'params': list(model.gru.parameters()) + list(model.delta.parameters()), 'lr': args.lr_head}])
    pos_w = torch.tensor(args.pos_weight, device=device)
    rng = np.random.RandomState(args.seed)
    print(f'正式模型：起點 {base_ck.name}、{len(data)} 支 {len(tr)} 條序列、{args.epochs} 輪', flush=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(order), args.batch):
            batch = [tr[j] for j in order[i:i + args.batch]]
            if args.crop:
                batch = [_crop(s, args.crop, rng) for s in batch]
            if args.flip_aug:
                batch = [_flip(s, rng) for s in batch]
            z = torch.cat(model([torch.from_numpy(s['traj']).to(device) for s in batch], _feats(batch, model, device)))
            y = torch.cat([torch.from_numpy(s['y']) for s in batch]).to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(batch)
        print(f'  epoch {ep:2d} loss {tot / len(tr):.4f}', flush=True)
    out = CK / f'finetune_seq_{args.tag}_s{args.seed}_best.pt'
    meta = dict(ck.get('data_meta') or {}, decode='run', peak_threshold=args.threshold, peak_low=round(args.threshold - 0.2, 3))
    torch.save({'seq': True, 'model': model.state_dict(), 'hidden': args.hidden, 'n_feat': model.n_feat, 'base_ckpt': base_ck.name,
                'd_model': ck.get('d_model', 64), 'n_heads': ck.get('n_heads', 4), 'n_layers': ck.get('n_layers', 2),
                'epochs': args.epochs, 'pos_weight': args.pos_weight, 'data_meta': meta}, out)
    print(f'→ {out}（解碼：連續一段算一次、門檻 {args.threshold}、掉到 {args.threshold - 0.2:.2f} 以下才結束）')


def load_seq(path, device='cpu'):
    """讀 train_final 存的 ckpt → ScreenSeq（eval 模式）。"""
    from nba_data_pipeline.model import ScreenNetPairwise
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = ScreenSeq(ScreenNetPairwise(ck['d_model'], ck['n_heads'], ck['n_layers']), ck['hidden'], ck.get('n_feat', 0))
    m.load_state_dict(ck['model'])
    return m.to(device).eval(), ck


def decode_runs(curve, thr, gap, low=None):
    """連續超過門檻的一段算一次（取最高那張）。curve {影格: 分數}；相鄰兩點差 ≤ gap 算連續。回 [(影格, 分數)]。
    low（雙門檻）：到 thr 才開始一段、掉到 low 以下才結束 → 同一次掩護中間的小凹陷不會切成兩次
    （screen_300：同一人貼了 3.5 秒，中間掉到 0.68 被切成兩次）。交叉驗證 low = thr − 0.2：±0.4 秒持平、誤報 64 → 62。"""
    low = thr if low is None else low
    out, run, last = [], [], None
    for f, s in sorted(curve.items()):
        if run and s >= low and f - last <= gap:
            run.append((f, s))
        else:
            if run:
                out.append(max(run, key=lambda x: x[1]))
            run = [(f, s)] if s >= thr else []
        last = f
    if run:
        out.append(max(run, key=lambda x: x[1]))
    return out


def _feats(batch, model, device):
    return [torch.from_numpy(s['feat']).to(device) for s in batch] if model.n_feat else None


def _flip(s, rng, W=91.8, H=49.2):
    """整條序列同一個方向翻（左右 / 上下），缺的人（座標 0）不動；同 finetune.FlipAugment。"""
    fx, fy = rng.rand(2) < 0.5
    if not (fx or fy):
        return s
    t = s['traj'].copy()
    real = np.abs(t).sum(-1) > 0
    if fx:
        t[..., 0] = np.where(real, W - t[..., 0], t[..., 0])
    if fy:
        t[..., 1] = np.where(real, H - t[..., 1], t[..., 1])
    return dict(s, traj=t)


def _crop(s, L, rng):
    if len(s['y']) <= L:
        return s
    a = rng.randint(0, len(s['y']) - L + 1)
    return dict(s, traj=s['traj'][a:a + L], y=s['y'][a:a + L], frames=s['frames'][a:a + L],
                **({'feat': s['feat'][a:a + L]} if 'feat' in s else {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tag', default='seq')
    ap.add_argument('--init', default='court2fix2', help='起點：finetune_contact_<init>_s<seed>_cv<k>_best.pt')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr_base', type=float, default=3e-5)
    ap.add_argument('--lr_head', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--pos_weight', type=float, default=5.0)
    ap.add_argument('--crop', type=int, default=0, help='訓練時序列最長幾張（0 = 整段）')
    ap.add_argument('--flip_aug', action='store_true', help='訓練時整條序列隨機左右 / 上下翻（起點是 --flip_aug 訓練的模型時用）')
    ap.add_argument('--feat', action='store_true', help='GRU 另外吃每窗 10 個幾何特徵（序列快取要含 feat：--build --data 重建）')
    ap.add_argument('--data', default=str(DATA), help='序列快取路徑')
    ap.add_argument('--final', action='store_true', help='正式模型：起點 finetune_contact_<init>_s<seed>_best.pt、全部資料、固定 --epochs')
    ap.add_argument('--threshold', type=float, default=0.7, help='--final：存進 ckpt 給推論用的門檻（連續一段算一次）')
    args = ap.parse_args()
    if args.build:
        return build(Path(args.data))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = pickle.load(open(args.data, 'rb'))
    if args.final:
        return train_final(data, args, device)
    allc = {}
    for k in range(5):
        cv, f1 = run_fold(data, args.seed, k, args, device)
        allc.update(cv)
    OUTD.mkdir(parents=True, exist_ok=True)
    pickle.dump(allc, open(OUTD / f'{args.tag}_s{args.seed}.pkl', 'wb'))
    print(f'完成 → {OUTD / f"{args.tag}_s{args.seed}.pkl"}（{len(allc)} 支）')


if __name__ == '__main__':
    main()
