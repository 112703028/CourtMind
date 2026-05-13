"""
Step 1：自監督預訓練
  輸入：前 7 幀三人軌跡
  目標：預測後 3 幀的速度向量
  資料：NBA 負樣本（多樣化的籃球動作，不限於掩護）

使用方式：
  python pretrain.py --data_dir ./output_full --out_dir ./checkpoints
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import ScreenNetPretrain, INPUT_LEN, PRED_LEN

SEQ_LEN = INPUT_LEN + PRED_LEN   # = 10


class TrajectoryDataset(Dataset):
    """
    從 .npz 檔案載入軌跡序列。
    - 輸入：前 INPUT_LEN 幀
    - 目標：後 PRED_LEN 幀的速度（位移差）
    """

    def __init__(self, npz_paths):
        trajs_list, roles_list = [], []
        for path in npz_paths:
            data = np.load(path)
            trajs_list.append(data['trajectories'])   # (N, 10, 10, 2)
            roles_list.append(data['roles'])           # (N, 10)

        self.trajs = np.concatenate(trajs_list, axis=0).astype(np.float32)
        self.roles = np.concatenate(roles_list, axis=0).astype(np.int64)

        print(f"  Dataset loaded: {len(self.trajs):,} sequences")

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj = self.trajs[idx]    # (10, 10, 2)
        role = self.roles[idx]    # (10,)

        # 輸入：前 7 幀位置
        inp = torch.tensor(traj[:, :INPUT_LEN, :])    # (10, 7, 2)

        # 目標：後 3 幀的速度向量（位移差，論文選擇）
        future = traj[:, INPUT_LEN:, :]               # (10, 3, 2)
        # 速度 = 當前幀 - 前一幀（第一個速度用輸入最後一幀當基準）
        prev  = np.concatenate([traj[:, INPUT_LEN-1:INPUT_LEN, :], future[: , :-1, :]], axis=1)
        vel   = torch.tensor(future - prev)            # (10, 3, 2)

        return inp, torch.tensor(role), vel


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for inp, role, vel_target in loader:
        inp        = inp.to(device)        # (B, 10, 7, 2)
        role       = role.to(device)       # (B, 10)
        vel_target = vel_target.to(device) # (B, 10, 3, 2)

        optimizer.zero_grad()
        vel_pred = model(inp, role)        # (B, 3, 3, 2)

        loss = nn.functional.mse_loss(vel_pred, vel_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0

    for inp, role, vel_target in loader:
        inp        = inp.to(device)
        role       = role.to(device)
        vel_target = vel_target.to(device)

        vel_pred = model(inp, role)
        loss = nn.functional.mse_loss(vel_pred, vel_target)
        total_loss += loss.item()

    return total_loss / len(loader)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default=os.path.join(SCRIPT_DIR, 'output_10p'))
    parser.add_argument('--out_dir',   default=os.path.join(SCRIPT_DIR, 'checkpoints'))
    parser.add_argument('--d_model',   type=int,   default=64)
    parser.add_argument('--n_heads',   type=int,   default=4)
    parser.add_argument('--n_layers',  type=int,   default=2)
    parser.add_argument('--batch',     type=int,   default=512)
    parser.add_argument('--epochs',    type=int,   default=50)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--patience',  type=int,   default=10,
                        help='Early stopping patience')
    parser.add_argument('--neg_only',  action='store_true', default=True,
                        help='Use only negative sequences (more diverse movements)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 資料載入
    # 用負樣本做預訓練：負樣本包含各種籃球動作，比正樣本更多樣化
    paths = []
    if args.neg_only:
        neg_path = os.path.join(args.data_dir, 'negative_sequences.npz')
        if os.path.exists(neg_path):
            paths.append(neg_path)
    if not paths:
        for name in ('negative_sequences.npz', 'positive_sequences.npz'):
            p = os.path.join(args.data_dir, name)
            if os.path.exists(p):
                paths.append(p)

    print("Loading data...")
    dataset = TrajectoryDataset(paths)

    # 取最多 50 萬筆（避免記憶體不足），做 train/val 切分
    max_samples = min(len(dataset), 500_000)
    if max_samples < len(dataset):
        dataset, _ = random_split(dataset, [max_samples, len(dataset) - max_samples])

    val_size   = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    num_workers = 0 if os.name == 'nt' else 4   # Windows doesn't support fork
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=num_workers, pin_memory=False)

    print(f"Train: {train_size:,}  Val: {val_size:,}")

    # 模型
    model = ScreenNetPretrain(args.d_model, args.n_heads, args.n_layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
    )

    # 訓練
    best_val  = float('inf')
    no_improve = 0
    best_path  = os.path.join(args.out_dir, 'pretrain_best.pt')

    print(f"\nStarting pretraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)   # 根據驗證集的表現來調整學習率。如果 val_loss 很久沒下降，它會把學習率調小，讓模型搜尋更精細。

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save({
                'epoch':      epoch,
                'model':      model.state_dict(),
                'val_loss':   val_loss,
                'd_model':    args.d_model,
                'n_heads':    args.n_heads,
                'n_layers':   args.n_layers,
            }, best_path)
            tag = ' ← best'
        else:
            no_improve += 1
            tag = ''

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}"
              f"  lr={optimizer.param_groups[0]['lr']:.2e}{tag}")

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nPretraining done. Best val loss: {best_val:.4f}")
    print(f"Checkpoint saved → {best_path}")
    print(f"\nNext step: run finetune.py --pretrain_ckpt {best_path}")


if __name__ == '__main__':
    main()
