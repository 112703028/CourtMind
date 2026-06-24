"""
train.py — 訓練 TeamSiamese。

注意：
  - **不做色彩增強**（顏色是團隊判斷的關鍵訊號）
  - 只做位置增強（flip、crop、rotation）
  - 用 weighted sampler 解 positive (2327) : negative (583) 不平衡

執行:
  python nba_data_pipeline/team_classifier/train.py
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import TeamSiamese

# ── 設定 ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / 'team_classifier_data'
PAIRS_PATH   = DATA_DIR / 'pairs.json'
CROP_DIR     = DATA_DIR / 'crops'
OUT_DIR      = PROJECT_ROOT / 'models'
OUT_PATH     = OUT_DIR / 'team_siamese.pt'

SEED        = 42
VAL_RATIO   = 0.15
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 1e-4
WEIGHT_DECAY= 1e-5
IMG_SIZE    = 224

# ── Transforms ────────────────────────────────────────────────────────────────

# ImageNet 統計值（pretrained 必用）
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# Train: 只做位置增強，不做色彩增強
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # 多一點空間給 random crop
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# Val: 純 resize + normalize
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        crop1 = Image.open(CROP_DIR / p['crop1']).convert('RGB')
        crop2 = Image.open(CROP_DIR / p['crop2']).convert('RGB')
        return (
            self.transform(crop1),
            self.transform(crop2),
            torch.tensor(float(p['label']), dtype=torch.float32),
        )


# ── Train / Eval ───────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch_num):
    import time
    model.train()
    total_loss, n = 0.0, 0
    correct = 0
    n_batches = len(loader)
    t_start = time.time()
    t_batch_start = time.time()

    for batch_idx, (c1, c2, label) in enumerate(loader):
        # 看每 batch 載入 + 跑 GPU 的時間
        t_load = time.time() - t_batch_start

        t = time.time()
        c1, c2, label = c1.to(device), c2.to(device), label.to(device)
        t_to_gpu = time.time() - t

        t = time.time()
        optimizer.zero_grad()
        logit = model(c1, c2)
        loss = loss_fn(logit, label)
        loss.backward()
        optimizer.step()
        t_forward = time.time() - t

        total_loss += loss.item() * label.size(0)
        n += label.size(0)
        pred = (torch.sigmoid(logit) >= 0.5).float()
        correct += (pred == label).sum().item()

        # 每 5 batch 印一次
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            print(f"    [E{epoch_num} B{batch_idx+1}/{n_batches}] "
                  f"load={t_load:.2f}s gpu_send={t_to_gpu:.3f}s fwd_bwd={t_forward:.3f}s "
                  f"loss={loss.item():.4f}", flush=True)

        t_batch_start = time.time()

    print(f"  Epoch {epoch_num} took {time.time()-t_start:.1f}s total", flush=True)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, n = 0.0, 0
    correct, tp, fp, fn, tn = 0, 0, 0, 0, 0
    for c1, c2, label in loader:
        c1, c2, label = c1.to(device), c2.to(device), label.to(device)
        logit = model(c1, c2)
        loss = loss_fn(logit, label)
        '''
        為 loss_fn 回傳的是該 Batch 的「平均損失」，為了在最後計算整個驗證集的總平均，
        這裡必須把平均損失乘以當前 Batch 的實際樣本數（label.size(0)，通常是 64），
        還原成「該 Batch 的總損失」並累加到 total_loss。
        '''
        total_loss += loss.item() * label.size(0)
        n += label.size(0)

        pred = (torch.sigmoid(logit) >= 0.5).float()
        correct += (pred == label).sum().item()
        tp += ((pred == 1) & (label == 1)).sum().item()
        fp += ((pred == 1) & (label == 0)).sum().item()
        fn += ((pred == 0) & (label == 1)).sum().item()
        tn += ((pred == 0) & (label == 0)).sum().item()

    acc = correct / n
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return total_loss / n, acc, precision, recall, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load pairs
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)
    print(f"Total pairs: {len(pairs)}")

    # Train/Val split
    random.shuffle(pairs)
    n_val = int(len(pairs) * VAL_RATIO)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    # 統計 train labels
    train_labels = [p['label'] for p in train_pairs]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    print(f"Train labels: pos={n_pos}  neg={n_neg}")

    # WeightedRandomSampler — 不平衡資料的解法
    # 正樣本權重 1/n_pos，負樣本 1/n_neg → batch 內期望 50/50
    weights = [1 / n_pos if l == 1 else 1 / n_neg for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    import time

    # DataLoader
    print("[setup] Creating PairDataset...", flush=True)
    t = time.time()
    train_ds = PairDataset(train_pairs, train_transform)
    val_ds   = PairDataset(val_pairs, val_transform)
    print(f"[setup] Dataset ready in {time.time()-t:.2f}s", flush=True)

    print("[setup] Creating DataLoader (num_workers=0)...", flush=True)
    # num_workers=0 避免 docker shm 不夠用導致 worker crash
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)
    print(f"[setup] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    # Model
    print("[setup] Building TeamSiamese model...", flush=True)
    t = time.time()
    model = TeamSiamese().to(device)
    print(f"[setup] Model ready on {device} in {time.time()-t:.2f}s", flush=True)
    print(f"[setup] Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss()  # sampler 已平衡，不用 pos_weight

    # Train loop
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    print("\n[train] Starting training...\n", flush=True)
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_loss, val_acc, val_p, val_r, val_f1 = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        tag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, OUT_PATH)
            tag = " ← best"

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
              f"P={val_p:.3f} R={val_r:.3f} F1={val_f1:.3f}{tag}")

    print(f"\nBest F1: {best_f1:.4f}")
    print(f"Saved → {OUT_PATH}")


if __name__ == '__main__':
    main()
