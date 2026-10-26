"""
pretrain5.py — 5fps 自監督預訓練（取代 25fps 的 pretrain.py 當 ScreenNet 的起點）

為什麼：舊預訓練用 SportVU 25fps 原始資料（10 張 = 0.4 秒），ScreenNet 是轉播 5fps（10 張 = 2 秒），
時間尺度差 5 倍；即使這樣預訓練仍有 +0.09，代表方向對、只是尺度不對（HoopTransformer / TranSPORTmer /
NETS 都是大量無標記軌跡預訓練 + 少量標記微調）。

跟 ScreenNet 的輸入完全對齊：
  - 5fps、輸入 10 張（= ScreenNet 一個窗）→ 預測後 3 張每張位移
  - 絕對球場座標（英尺，轉播尺寸 91.8×49.2），不縮放、不以持球者為原點
  - slot = [持球者, 其他進攻依離持球者遠近, 防守依離持球者遠近]（同 convert_contact.build_windows）
  - 模型大小同 ScreenNet：d_model 64、n_heads 4、n_layers 2 → finetune.py --pretrain_ckpt 直接接
  - 訓練時：位置雜訊（每人整窗偏移 + 逐張抖動）、隨機拿掉 1~2 個非持球者（整條設 0 = 沒追到）

用法：
  python nba_data_pipeline/pretrain5.py build --games 100      # 一場一場解壓 → 窗 → 刪 json → output_pretrain5/*.npz
  python nba_data_pipeline/pretrain5.py train --epochs 30      # → checkpoints/pretrain5_best.pt（GPU 建議在 pod 跑）
"""
import argparse
import glob
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from model import ScreenNetPretrain, N_PLAYERS

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = HERE / 'output_pretrain5'
CKPT = HERE / 'checkpoints' / 'pretrain5_best.pt'
IN_LEN, PRED_LEN = 10, 3
WIN = IN_LEN + PRED_LEN


def game_windows(js_path, stride):
    from extract_sequences import parse_moment, find_ball_handler
    from nba_coverage_probe import load_timeline, segments, FPS, SX, SY
    step = FPS // 5
    gameid, frames, _ = load_timeline(js_path)
    W = []
    for seg in segments(frames):
        if len(seg) < WIN * step:
            continue
        parsed = [parse_moment(m) for m in seg]
        hand = find_ball_handler(parsed, None, None)
        idx = list(range(0, len(parsed), step))
        for s in range(0, len(idx) - WIN + 1, stride):
            fr = idx[s:s + WIN]
            h = hand[fr[IN_LEN - 1]]
            if h is None:
                continue
            H, team = h
            pl = parsed[fr[0]]['players']
            if len(pl) != 10 or any(set(parsed[f]['players']) != set(pl) for f in fr):
                continue
            last = parsed[fr[IN_LEN - 1]]['players']
            hx, hy = last[H]['pos']
            dist = lambda p: (last[p]['pos'][0] - hx) ** 2 + (last[p]['pos'][1] - hy) ** 2
            off = sorted((p for p, i in pl.items() if i['team'] == team and p != H), key=dist)
            de = sorted((p for p, i in pl.items() if i['team'] != team), key=dist)
            if len(off) != 4 or len(de) != 5:
                continue
            W.append(np.array([[(parsed[f]['players'][p]['pos'][0] * SX, parsed[f]['players'][p]['pos'][1] * SY)
                                for f in fr] for p in [H] + off + de], np.float32))
    return gameid, (np.stack(W) if W else np.zeros((0, N_PLAYERS, WIN, 2), np.float32))


def _build_one(args):
    path7z, stride = args
    import py7zr
    out = OUT / (Path(path7z).stem + '.npz')
    if out.exists():
        return Path(path7z).stem, -1
    tmp = tempfile.mkdtemp(prefix='p5_')
    try:
        with py7zr.SevenZipFile(path7z, 'r') as z:
            z.extractall(tmp)
        js = glob.glob(os.path.join(tmp, '**', '*.json'), recursive=True)
        if not js:
            return Path(path7z).stem, 0
        gid, W = game_windows(js[0], stride)
        np.savez_compressed(out, windows=W)
        return gid, len(W)
    except Exception as e:
        return f'{Path(path7z).stem} 失敗：{str(e)[:80]}', 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build(args):
    from multiprocessing import Pool
    OUT.mkdir(exist_ok=True)
    zs = sorted(glob.glob(str(ROOT / 'nba-movement-data' / 'data' / '*.7z')))[:args.games]
    t0, tot = time.time(), 0
    with Pool(args.workers) as pool:
        for k, (gid, n) in enumerate(pool.imap_unordered(_build_one, [(z, args.stride) for z in zs]), 1):
            tot += max(0, n)
            if k % 10 == 0 or k == len(zs):
                print(f'{k}/{len(zs)} 場  累計 {tot} 窗  {time.time() - t0:.0f} 秒', flush=True)
    print(f'→ {OUT}（{len(glob.glob(str(OUT / "*.npz")))} 場）')


def augment(w, rng, sigma, drop=True):
    """w: (B,10,13,2) → 輸入 (B,10,10,2)、目標每張位移 (B,10,3,2)、loss 遮罩 (B,10)"""
    B = len(w)
    w = w + rng.normal(0, sigma, (B, N_PLAYERS, 1, 2)) + rng.normal(0, sigma * 0.5, w.shape)
    w = w.astype(np.float32)
    inp = w[:, :, :IN_LEN].copy()
    tgt = np.diff(w[:, :, IN_LEN - 1:], axis=2)
    mask = np.ones((B, N_PLAYERS), np.float32)
    if drop:
        k = rng.choice([0, 1, 2], size=B, p=[0.35, 0.45, 0.2])
        for b in range(B):
            for p in rng.choice(np.arange(1, N_PLAYERS), k[b], replace=False):
                inp[b, p] = 0.0
                mask[b, p] = 0.0
    return torch.from_numpy(inp), torch.from_numpy(tgt), torch.from_numpy(mask)


def ade(pred, tgt, mask):
    e = (pred.cumsum(2) - tgt.cumsum(2)).norm(dim=-1).mean(-1)
    return (e * mask).sum() / mask.sum()


def train(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    files = sorted(glob.glob(str(OUT / '*.npz')))
    n_val = max(1, len(files) // 20)
    load = lambda fs: np.concatenate([np.load(f)['windows'] for f in fs if np.load(f)['windows'].size])
    wv, wt = load(files[:n_val]), load(files[n_val:])
    print(f'{len(files)} 場：訓練 {len(wt)} 窗、驗證 {len(wv)} 窗（{n_val} 場）；{dev}', flush=True)
    roles = torch.tensor([[0] + [1] * 4 + [2] * 5]).to(dev)
    model = ScreenNetPretrain(args.d_model, args.n_heads, args.n_layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    vx, vy, vm = augment(wv, np.random.RandomState(1), args.noise)
    vx, vy, vm = vx.to(dev), vy.to(dev), vm.to(dev)
    noisy = wv + 0                                                   # 基準：等速直走（同樣雜訊）
    base_v = vx[:, :, -1:] - vx[:, :, -2:-1]
    print(f'基準（等速直走）驗證 ADE：{ade(base_v.repeat(1, 1, PRED_LEN, 1), vy, vm):.3f} ft', flush=True)
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(wt))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            x, y, m = (t.to(dev) for t in augment(wt[perm[i:i + args.batch]], rng, args.noise))
            pred = model(x, roles.expand(len(x), -1))
            loss = (((pred - y) ** 2).sum(-1).mean(-1) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        model.eval()
        with torch.no_grad():
            v = ade(torch.cat([model(vx[i:i + 4096], roles.expand(len(vx[i:i + 4096]), -1))
                               for i in range(0, len(vx), 4096)]), vy, vm).item()
        mark = ''
        if v < best:
            best = v
            CKPT.parent.mkdir(exist_ok=True)
            torch.save({'model': model.state_dict(), 'd_model': args.d_model, 'n_heads': args.n_heads,
                        'n_layers': args.n_layers, 'epoch': ep + 1, 'val_ade': v,
                        'note': '5fps 自監督預訓練（10 張 → 3 張），絕對球場座標，雜訊 + 拿掉球員'}, CKPT)
            mark = ' ← best'
        print(f'epoch {ep + 1}/{args.epochs}  train MSE {tot / len(wt):.3f}  驗證 ADE {v:.3f} ft{mark}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--games', type=int, default=100)
    b.add_argument('--stride', type=int, default=2, help='5fps 張數')
    b.add_argument('--workers', type=int, default=8)
    t = sub.add_parser('train')
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--batch', type=int, default=512)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--noise', type=float, default=1.0, help='位置雜訊（英尺）')
    t.add_argument('--d_model', type=int, default=64)
    t.add_argument('--n_heads', type=int, default=4)
    t.add_argument('--n_layers', type=int, default=2)
    a = ap.parse_args()
    build(a) if a.cmd == 'build' else train(a)
