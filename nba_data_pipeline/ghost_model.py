"""
ghost_model.py — 5fps 的「正常跑法」軌跡預測模型（反事實掩護偵測的試驗）

為什麼：Step 1 預訓練是 SportVU 25fps 原始資料（10 張 = 0.4 秒），跟轉播的 5fps（10 張 = 2 秒）
時間尺度差 5 倍，不能直接拿來預測轉播軌跡。這裡用 SportVU 降到 5fps 重新訓練：
  輸入：10 人前 7 張位置（1.4 秒），slot = [持球者, 進攻×4, 防守×5]，球場座標（轉播尺寸 91.8×49.2 ft）/10
  輸出：10 人後 3 張的每張位移（0.6 秒，英尺）
  訓練時隨機拿掉 1~2 個非持球者（整條設 0 = 不在場上），模型才看得懂「這個人不在」→ 反事實才成立
  輸入加跟轉播相近的位置雜訊

用法：
  python nba_data_pipeline/ghost_model.py build               # → output_ghost/ghost_windows.npz
  python nba_data_pipeline/ghost_model.py train --epochs 8    # → checkpoints/ghost_5fps.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import ScreenNetPretrain, N_PLAYERS

HERE = Path(__file__).parent
OUT = HERE / 'output_ghost'
CKPT = HERE / 'checkpoints' / 'ghost_5fps.pt'
IN_LEN, PRED_LEN, WIN = 7, 3, 10
SCALE = 10.0


def build(args):
    from extract_sequences import parse_moment, find_ball_handler
    from nba_coverage_probe import load_timeline, segments, FPS, SX, SY
    from nba_contact_data import game_jsons
    step = FPS // 5
    W, G = [], []
    for gi, js in enumerate(game_jsons(args.games)):
        gameid, frames, _ = load_timeline(js)
        n0 = len(W)
        for seg in segments(frames):
            if len(seg) < WIN * step:
                continue
            parsed = [parse_moment(m) for m in seg]
            hand = find_ball_handler(parsed, None, None)
            idx = list(range(0, len(parsed), step))
            for s in range(0, len(idx) - WIN + 1, args.stride):
                fr = idx[s:s + WIN]
                h = hand[fr[IN_LEN - 1]]
                if h is None:
                    continue
                H, team = h
                pl = parsed[fr[0]]['players']
                if len(pl) != 10 or any(set(parsed[f]['players']) != set(pl) for f in fr):
                    continue
                off = sorted(p for p, i in pl.items() if i['team'] == team and p != H)
                de = sorted(p for p, i in pl.items() if i['team'] != team)
                if len(off) != 4 or len(de) != 5:
                    continue
                order = [H] + off + de
                W.append(np.array([[(parsed[f]['players'][p]['pos'][0] * SX, parsed[f]['players'][p]['pos'][1] * SY)
                                    for f in fr] for p in order], np.float32))
                G.append(gi)
        print(f'比賽 {gameid}：{len(W) - n0} 個窗', flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'ghost_windows.npz', windows=np.stack(W), game=np.array(G, np.int32))
    print(f'共 {len(W)} 個窗 → {OUT / "ghost_windows.npz"}')


def add_noise(w, rng, sigma):
    """每人整窗一個偏移（homography 偏差）+ 逐張抖動（腳點偵測）。w: (B, 10, T, 2)"""
    B = w.shape[0]
    return w + rng.normal(0, sigma, (B, N_PLAYERS, 1, 2)) + rng.normal(0, sigma * 0.5, w.shape)


def make_batch(w, rng, sigma, drop=True):
    """→ 輸入 (B,10,7,2)/SCALE、目標每張位移 (B,10,3,2)、loss 遮罩 (B,10)"""
    w = add_noise(w, rng, sigma).astype(np.float32)
    inp = w[:, :, :IN_LEN].copy()
    full = w[:, :, IN_LEN - 1:]                                  # 最後一張輸入 + 3 張未來
    tgt = np.diff(full, axis=2)                                  # (B,10,3,2)
    mask = np.ones(w.shape[:2], np.float32)
    if drop:
        B = len(w)
        k = rng.choice([0, 1, 2], size=B, p=[0.35, 0.45, 0.2])
        for b in range(B):
            for p in rng.choice(np.arange(1, N_PLAYERS), k[b], replace=False):
                inp[b, p] = 0.0
                mask[b, p] = 0.0
    inp = np.where(np.abs(inp).sum(-1, keepdims=True) > 0, inp / SCALE, 0.0)
    return torch.tensor(inp, dtype=torch.float32), torch.tensor(tgt), torch.tensor(mask)


def roles_for(B):
    return torch.tensor([[0] + [1] * 4 + [2] * 5] * B)


def ade(pred, tgt, mask):
    """平均位移誤差：3 張累積位置誤差的平均（英尺），只算 mask=1 的人。"""
    e = (pred.cumsum(2) - tgt.cumsum(2)).norm(dim=-1).mean(-1)   # (B,10)
    return (e * mask).sum() / mask.sum()


def train(args):
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    d = np.load(OUT / 'ghost_windows.npz')
    w, g = d['windows'], d['game']
    va = g == g.max()                                            # 最後一場當驗證
    wt, wv = w[~va], w[va]
    print(f'訓練 {len(wt)} 窗、驗證 {len(wv)} 窗')
    model = ScreenNetPretrain()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    vx, vy, vm = make_batch(wv, np.random.RandomState(1), args.noise, drop=True)
    # 基準：照最後一張的速度繼續走
    last_v = torch.tensor(np.diff(add_noise(wv, np.random.RandomState(1), args.noise)[:, :, IN_LEN - 2:IN_LEN], axis=2))
    print(f'基準（等速直走）驗證 ADE：{ade(last_v.repeat(1, 1, PRED_LEN, 1), vy, vm):.3f} ft')
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(wt))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            x, y, m = make_batch(wt[perm[i:i + args.batch]], rng, args.noise)
            pred = model(x, roles_for(len(x)))
            loss = (((pred - y) ** 2).sum(-1).mean(-1) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            v = ade(torch.cat([model(vx[i:i + 1024], roles_for(len(vx[i:i + 1024]))) for i in range(0, len(vx), 1024)]), vy, vm)
        mark = ''
        if v < best:
            best = v
            torch.save({'model': model.state_dict(), 'noise': args.noise, 'scale': SCALE}, CKPT)
            mark = ' ← best'
        print(f'epoch {ep + 1}/{args.epochs}  train MSE {tot / len(wt):.3f}  驗證 ADE {v:.3f} ft{mark}', flush=True)


def load(path=CKPT):
    ck = torch.load(path, map_location='cpu')
    m = ScreenNetPretrain()
    m.load_state_dict(ck['model'])
    m.eval()
    return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--games', type=int, default=0)
    b.add_argument('--stride', type=int, default=2, help='5fps 張數')
    t = sub.add_parser('train')
    t.add_argument('--epochs', type=int, default=8)
    t.add_argument('--batch', type=int, default=256)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--noise', type=float, default=1.0, help='位置雜訊（英尺）')
    a = ap.parse_args()
    build(a) if a.cmd == 'build' else train(a)
