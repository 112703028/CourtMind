"""
Step 2：弱標籤 Fine-tune（NBA movement data，幾何規則自動標註）
Step 3：人工標註 Fine-tune（你的 80 個片段）

兩個步驟用同一個腳本，差別只在 --data_dir 和 --pretrain_ckpt。

Step 2 使用方式：
  python finetune.py \
    --pretrain_ckpt ./checkpoints/pretrain_best.pt \
    --data_dir      ./output_10p \
    --out_dir       ./checkpoints \
    --tag           weak

Step 3 使用方式：
  python finetune.py \
    --pretrain_ckpt ./checkpoints/finetune_weak_best.pt \
    --data_dir      ./output_manual \
    --out_dir       ./checkpoints \
    --tag           manual \
    --epochs        30 \
    --lr            3e-4
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

from model import ScreenNetClassifier, ScreenNetPairwise, N_PAIRS, to_handler_relative

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class ScreenDataset(Dataset):
    """
    載入標註好的掩護序列。
    npz 格式（與 extract_sequences.py 輸出一致）：
      trajectories : (N, 10, 10, 2)
      roles        : (N, 10)
      labels       : (N,)  — 1=掩護, 0=非掩護
    """

    def __init__(self, npz_paths):
        trajs_list, roles_list, labels_list = [], [], []
        for path in npz_paths:
            data = np.load(path)
            trajs_list.append(data['trajectories'])
            roles_list.append(data['roles'])
            labels_list.append(data['labels'])
            print(f"  {os.path.basename(path)}: {len(data['labels']):,} sequences "
                  f"(pos={data['labels'].sum():,})")

        self.trajs  = np.concatenate(trajs_list,  axis=0).astype(np.float32)
        self.roles  = np.concatenate(roles_list,  axis=0).astype(np.int64)
        self.labels = np.concatenate(labels_list, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 全部 10 幀都餵入模型（分類任務使用完整序列）
        traj  = torch.tensor(self.trajs[idx])    # (10, 10, 2)
        role  = torch.tensor(self.roles[idx])    # (10,)
        label = torch.tensor(self.labels[idx])   # scalar
        return traj, role, label


class FlipAugment(Dataset):
    """
    訓練時隨機把球場左右翻 / 上下翻（球場對稱，掩護的幾何關係不變）→ 資料等於多 4 倍。
    座標是球場英尺（x 0~91.8、y 0~49.2）；補位用的 (0,0) 不翻，避免缺人變成角落的人。
    """
    W, H = 91.8, 49.2

    def __init__(self, base, rel=False):
        self.base = base
        self.rel = rel              # 相對座標（--rel_coords）：翻轉 = 取負號

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        traj, role, label = self.base[idx]
        flip_x, flip_y = np.random.rand(2) < 0.5
        if flip_x or flip_y:
            traj = traj.clone()
            real = (traj.abs().sum(-1, keepdim=True) > 0)      # (10, T, 1)
            for axis, on, size in ((0, flip_x, self.W), (1, flip_y, self.H)):
                if on:
                    v = -traj[..., axis] if self.rel else size - traj[..., axis]
                    traj[..., axis] = torch.where(real[..., 0], v, traj[..., axis])
        return traj, role, label


class PairwiseScreenDataset(Dataset):
    """
    載入 pairwise 標籤的掩護序列。
    npz 需包含 pairwise_labels : (N, 20)
    """

    def __init__(self, npz_paths):
        trajs_list, roles_list, pw_list, clip_list = [], [], [], []
        self.meta = {}
        for path in npz_paths:
            data = np.load(path)
            trajs_list.append(data['trajectories'])
            roles_list.append(data['roles'])
            pw_list.append(data['pairwise_labels'])
            # convert_contact.py 產的資料才有：每窗屬於哪支 clip（按 clip 切 train/val 用）
            if 'clip_ids' in data:
                clip_list.append(data['clip_ids'])
            for k in ('anchor', 'sigma', 'clean_coords'):
                if k in data:
                    self.meta[k] = float(data[k])
            n_pos = (data['pairwise_labels'].max(axis=1) >= 0.5).sum()
            print(f"  {os.path.basename(path)}: {len(data['trajectories']):,} sequences "
                  f"(pos={n_pos:,})")

        self.trajs    = np.concatenate(trajs_list, axis=0).astype(np.float32)
        self.roles    = np.concatenate(roles_list, axis=0).astype(np.int64)
        self.pw_labels= np.concatenate(pw_list,   axis=0).astype(np.float32)  # (N, 20)，可為軟標籤
        self.clip_ids = (np.concatenate(clip_list) if len(clip_list) == len(npz_paths) else None)

    def __len__(self):
        return len(self.pw_labels)

    def __getitem__(self, idx):
        traj  = torch.tensor(self.trajs[idx])      # (10, 10, 2)
        role  = torch.tensor(self.roles[idx])      # (10,)
        label = torch.tensor(self.pw_labels[idx])  # (20,)
        return traj, role, label

    @property
    def binary_labels(self):
        """任一對目標 >= 0.5 → 正樣本，供 WeightedRandomSampler 使用。
        （軟標籤的尾巴 0.01~0.4 不算正樣本；舊的 0/1 標籤結果不變）"""
        return (self.pw_labels.max(axis=1) >= 0.5).astype(np.float32)


def make_weighted_sampler(labels):
    """
    論文 Section 3.3：正負樣本不平衡，用 weighted sampling 補正。
    正樣本少 → 給每筆正樣本更高的抽樣權重。
    """
    n_pos = labels.sum()   # labels 是0/1陣列 
    n_neg = len(labels) - n_pos
    weight_pos = 1.0 / n_pos if n_pos > 0 else 1.0
    weight_neg = 1.0 / n_neg if n_neg > 0 else 1.0
    weights = np.where(labels == 1, weight_pos, weight_neg)  # 分配權重

    # WeightedRandomSampler 是一個 iterator，每次 DataLoader 要組一個 batch 時，它負責決定「從 dataset 的哪些 index 取資料」。
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


@torch.no_grad()
def eval_epoch_pairwise(model, loader, criterion, device, verbose=False):
    """Pairwise 模式的驗證：輸出 (B, 20)，計算多項指標。"""
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for traj, role, label in loader:
        traj  = traj.to(device)
        role  = role.to(device)
        label = label.to(device)

        logits = model(traj, role)          # (B, 20)
        loss   = criterion(logits, label)
        total_loss += loss.item()

        all_logits.append(logits.cpu())
        all_labels.append(label.cpu())

    all_logits = torch.cat(all_logits).numpy()   # (N, 20)
    all_labels = torch.cat(all_labels).numpy()   # (N, 20)
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()

    # 軟標籤（接觸幀為中心的高斯）以 0.5 為界轉成 0/1 才能算 AUC/F1；舊的 0/1 標籤不受影響
    all_labels = (all_labels >= 0.5).astype(np.float32)

    # 只對有正樣本的配對計算（與 AUC 一致）
    active_k = [k for k in range(N_PAIRS) if all_labels[:, k].sum() > 0]

    aucs, aps, f1s, precs, recs = [], [], [], [], []
    best_thresh_f1s = []

    for k in active_k:
        y_true = all_labels[:, k]
        y_prob = probs[:, k]

        try:
            aucs.append(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        aps.append(average_precision_score(y_true, y_prob))

        # F1/Precision/Recall at threshold=0.5
        y_pred = (y_prob >= 0.5).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))

        # 找讓 F1 最大的最佳 threshold
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        best_thresh_f1s.append((best_t, best_f1))

    macro_auc  = float(np.mean(aucs))  if aucs  else float('nan')
    macro_ap   = float(np.mean(aps))   if aps   else float('nan')
    macro_f1   = float(np.mean(f1s))   if f1s   else float('nan')
    macro_prec = float(np.mean(precs)) if precs else float('nan')
    macro_rec  = float(np.mean(recs))  if recs  else float('nan')
    avg_best_t = float(np.mean([t for t, _ in best_thresh_f1s])) if best_thresh_f1s else 0.5
    avg_best_f1= float(np.mean([f for _, f in best_thresh_f1s])) if best_thresh_f1s else float('nan')

    if verbose:
        print(f"  [Val] AUC={macro_auc:.4f}  AP={macro_ap:.4f}  "
              f"F1@0.5={macro_f1:.4f}  Prec={macro_prec:.4f}  Rec={macro_rec:.4f}")
        print(f"  [Val] Best-thresh F1={avg_best_f1:.4f} (avg threshold={avg_best_t:.2f})")
        print(f"  [Val] Active pairs={len(active_k)}/{N_PAIRS}  "
              f"Pos labels={int(all_labels.sum())}  Total={len(all_labels)}")

    return total_loss / len(loader), macro_auc, macro_f1


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for traj, role, label in loader:
        traj  = traj.to(device)
        role  = role.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(traj, role)          # (B,)
        loss   = criterion(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for traj, role, label in loader:
        traj  = traj.to(device)
        role  = role.to(device)
        label = label.to(device)

        logits = model(traj, role)
        loss   = criterion(logits, label)
        total_loss += loss.item()

        all_logits.append(logits.cpu())
        all_labels.append(label.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(all_labels, probs)
    except Exception:
        auc = float('nan')

    f1 = f1_score(all_labels, preds, zero_division=0)

    return total_loss / len(loader), auc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_ckpt', required=True,
                        help='Path to pretrain_best.pt or previous finetune ckpt')
    parser.add_argument('--data_dir',  default=os.path.join(SCRIPT_DIR, 'output_10p'))
    parser.add_argument('--out_dir',   default=os.path.join(SCRIPT_DIR, 'checkpoints'))
    parser.add_argument('--tag',       default='weak',
                        help='Checkpoint tag: "weak" or "manual"')
    parser.add_argument('--epochs',    type=int,   default=50)
    parser.add_argument('--batch',     type=int,   default=256)
    parser.add_argument('--lr',        type=float, default=5e-4)
    parser.add_argument('--patience',  type=int,   default=10)
    parser.add_argument('--freeze_base', action='store_true',
                        help='Freeze base weights, only train classification head')
    parser.add_argument('--pairwise', action='store_true',
                        help='Use PairwiseScreenHead (20 outputs) instead of single classifier')
    parser.add_argument('--val_frac', type=float, default=0.15,
                        help='有 clip_ids 時：拿多少比例的 clip 當驗證（按 clip 切，避免相鄰窗洩漏）')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_folds', type=int, default=0,
                        help='交叉驗證份數（>0 時忽略 --val_frac，用第 --fold 份 clip 當 val）')
    parser.add_argument('--fold', type=int, default=0, help='交叉驗證：第幾份當 val（0 起算）')
    parser.add_argument('--pos_weight_cap', type=float, default=20.0,
                        help='pairwise 正樣本 loss 權重上限；越大模型越偏向給高分（假尖峰變多）')
    parser.add_argument('--flip_aug', action='store_true',
                        help='訓練時隨機左右/上下翻轉球場座標（資料增強）')
    parser.add_argument('--train_clip_max', type=int, default=None,
                        help='只用編號小於這個值的 clip 訓練（驗證不受影響）；例 185 = 只用舊片段')
    parser.add_argument('--train_seed', type=int, default=None,
                        help='固定訓練的隨機性（權重初始化、抽樣順序）；--seed 只管 clip 怎麼切份，兩者分開')
    parser.add_argument('--extra_data_dir', default=None,
                        help='額外訓練資料目錄（clip_ids 為負數），一律放訓練集，例 output_contact_nba')
    parser.add_argument('--train_frac', type=float, default=1.0,
                        help='學習曲線：訓練 clip 只用這個比例（val 不變）')
    parser.add_argument('--no_pretrain', action='store_true',
                        help='對照組：不載入 --pretrain_ckpt 的權重（仍讀它的 d_model 等設定）')
    parser.add_argument('--rel_coords', action='store_true',
                        help='座標改成以持球者為原點（跟預訓練一致）；存進 ckpt data_meta，'
                             'eval_contact / inference 讀到會自動套用')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.train_seed is not None:
        # 2026-10-21：同設定在本地 CPU 跑 0.619、pod GPU 重跑 0.490 → 沒固定種子時訓練本身的變異很大
        import random
        random.seed(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)
        torch.cuda.manual_seed_all(args.train_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"固定訓練種子 {args.train_seed}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 資料載入 ────────────────────────────────────────────────────────────────
    paths = []
    for name in ('positive_sequences.npz', 'negative_sequences.npz'):
        p = os.path.join(args.data_dir, name)
        if os.path.exists(p):
            paths.append(p)

    if not paths:
        raise FileNotFoundError(f"No npz files found in {args.data_dir}")
    if args.extra_data_dir:
        # 額外訓練資料（例 nba_contact_data.py 的 NBA 窗）：clip_ids 是負數 → 下面一律放訓練集
        extra = [os.path.join(args.extra_data_dir, n) for n in ('positive_sequences.npz', 'negative_sequences.npz')]
        paths += [p for p in extra if os.path.exists(p)]
        print(f"額外訓練資料：{args.extra_data_dir}（只進訓練，不進驗證）")

    print("Loading data...")
    dataset = PairwiseScreenDataset(paths) if args.pairwise else ScreenDataset(paths)
    if args.rel_coords:
        # 預訓練（pretrain.py / weak finetune）吃的是 extract_sequences 的相對座標，
        # 接觸幀資料是絕對球場座標 → 對齊成相對座標，預訓練學到的特徵才用得上
        dataset.trajs = to_handler_relative(dataset.trajs)
        if hasattr(dataset, 'meta'):
            dataset.meta['rel_coords'] = 1.0
        print("座標改成以持球者為原點（--rel_coords）")

    val_clips = []
    if getattr(dataset, 'clip_ids', None) is not None:
        # 按 clip 切：同一支 clip 的窗只會在 train 或 val 其中一邊。
        # 逐窗隨機切的話，相鄰窗（只差一幀）會同時出現在兩邊，val 分數虛高。
        rng = np.random.RandomState(args.seed)
        clips = np.unique(dataset.clip_ids[dataset.clip_ids >= 0])   # 負數 = 額外資料，不參與切分
        extra_clips = np.unique(dataset.clip_ids[dataset.clip_ids < 0])
        rng.shuffle(clips)
        if args.n_folds > 0:
            # 交叉驗證：clip 分成 n_folds 份，第 fold 份當 val（同 seed 每份固定）
            val_clips = sorted(int(c) for c in np.array_split(clips, args.n_folds)[args.fold])
        elif args.val_frac > 0:
            n_val = max(1, int(round(len(clips) * args.val_frac)))
            val_clips = sorted(int(c) for c in clips[:n_val])
        # else：val_frac=0 → 全部拿來訓練（最終模型），沒有 val
        vset = set(val_clips)
        train_clips = [c for c in clips if c not in vset]
        if args.train_frac < 1.0:
            # 學習曲線：訓練 clip 只取一部分（val 不變，跟完整資料的交叉驗證同一份考題）
            sub = np.random.RandomState(args.seed + 1000).permutation(train_clips)
            train_clips = list(sub[:max(1, int(round(len(train_clips) * args.train_frac)))])
        if args.train_clip_max is not None:
            # 對照組：切份照常（考題含新片段），但訓練只用編號 < train_clip_max 的 clip（例 185 = 只用舊片段）
            train_clips = [c for c in train_clips if c < args.train_clip_max]
        tset = set(train_clips) | set(extra_clips.tolist())
        val_idx = [i for i, c in enumerate(dataset.clip_ids) if c in vset]
        train_idx = [i for i, c in enumerate(dataset.clip_ids) if c in tset]
        train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)
        train_size, val_size = len(train_idx), len(val_idx)
        fold_txt = f"（第 {args.fold + 1}/{args.n_folds} 份）" if args.n_folds > 0 else ""
        frac_txt = f"（只用 {args.train_frac:.0%}）" if args.train_frac < 1.0 else ""
        extra_txt = f" + 額外 {len(extra_clips)} 段" if len(extra_clips) else ""
        print(f"按 clip 切{fold_txt}：train {len(train_clips)} 支{frac_txt}{extra_txt}、"
              f"val {len(val_clips)} 支 {val_clips}")
    else:
        max_samples = min(len(dataset), 200_000)
        if max_samples < len(dataset):
            dataset, _ = random_split(dataset, [max_samples, len(dataset) - max_samples])

        val_size   = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # pairwise：任一配對目標 >= 0.5 算正樣本（軟標籤的尾巴不算）；一般模式：直接取 label
    if args.pairwise:
        train_labels = np.array([
            float(dataset[i][2].numpy().max() >= 0.5) for i in train_ds.indices
        ])
    else:
        train_labels = np.array([dataset[i][2].item() for i in train_ds.indices])

    pos_ratio = train_labels.mean()
    print(f"Train: {train_size:,}  Val: {val_size:,}  Positive ratio: {pos_ratio:.3f}")

    sampler = make_weighted_sampler(train_labels)

    num_workers = 0 if os.name == 'nt' else 4
    if args.flip_aug:
        print("訓練資料隨機左右/上下翻轉（--flip_aug）")
    train_loader = DataLoader(FlipAugment(train_ds, rel=args.rel_coords) if args.flip_aug else train_ds,
                              batch_size=args.batch,
                              sampler=sampler, num_workers=num_workers)
    val_loader   = (DataLoader(val_ds, batch_size=args.batch,
                               shuffle=False, num_workers=num_workers)
                    if val_size > 0 else None)   # 全部資料訓練最終模型時沒有 val

    # ── 模型：從預訓練載入 Base，分類頭隨機初始化 ─────────────────────────────
    ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')
    d_model  = ckpt.get('d_model',  64)
    n_heads  = ckpt.get('n_heads',  4)
    n_layers = ckpt.get('n_layers', 2)

    ModelClass = ScreenNetPairwise if args.pairwise else ScreenNetClassifier
    if args.no_pretrain:
        # 對照組：同樣架構、權重全部隨機初始化 → 看 NBA 預訓練到底有沒有幫助
        model = ModelClass(d_model, n_heads, n_layers).to(device)
        print("不載入預訓練權重（--no_pretrain），全部隨機初始化")
    else:
        model = ModelClass.from_pretrained(
            args.pretrain_ckpt, d_model, n_heads, n_layers
        ).to(device)

    if args.freeze_base:
        for p in model.base.parameters():
            p.requires_grad = False
        print("Base frozen — only training classification head")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}  (trainable: {trainable_params:,})")

    # pos_weight：pairwise 模式在配對層級計算，一般模式在序列層級計算
    n_pos = train_labels.sum()
    n_neg = train_size - n_pos
    if args.pairwise:
        # 每個正序列只有 1 個正配對，其餘 N_PAIRS-1 個都是負配對
        n_pos_pairs = n_pos * 1
        n_neg_pairs = n_pos * (N_PAIRS - 1) + n_neg * N_PAIRS
        pair_pw = min(n_neg_pairs / max(n_pos_pairs, 1), args.pos_weight_cap)  # cap 避免過度激進
        pos_weight = torch.full((N_PAIRS,), pair_pw, dtype=torch.float32).to(device)
        print(f"pos_weight（正樣本在 loss 的權重）= {pair_pw:.1f}（上限 {args.pos_weight_cap}）")
    else:
        scalar_pw = n_neg / max(n_pos, 1)
        pos_weight = torch.tensor([scalar_pw], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-6
    )

    # ── 訓練迴圈 ────────────────────────────────────────────────────────────────
    best_auc   = 0.0
    best_f1    = 0.0
    no_improve = 0
    best_path  = os.path.join(args.out_dir, f'finetune_{args.tag}_best.pt')

    eval_fn = eval_epoch_pairwise if args.pairwise else eval_epoch
    # pairwise 模式用 F1 做 early stopping（AUC 容易過早收斂，F1 反映實際預測能力）
    use_f1_stopping = args.pairwise

    print(f"\nStarting fine-tune ({args.tag}, {'pairwise' if args.pairwise else 'binary'}) "
          f"for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        if val_loader is None:
            # 沒有 val（最終模型）：固定訓練 args.epochs 輪（輪數用交叉驗證的最佳輪數決定），
            # 每輪都存，最後一輪就是成品
            scheduler.step(train_loss)
            val_loss, val_auc, val_f1 = train_loss, float('nan'), float('nan')
            improved = True
        else:
            val_loss, val_auc, val_f1 = eval_fn(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            monitor = val_f1 if use_f1_stopping else val_auc
            best_monitor = best_f1 if use_f1_stopping else best_auc
            improved = monitor > best_monitor
        if improved:
            best_auc   = val_auc
            best_f1    = val_f1
            no_improve = 0
            torch.save({
                'epoch':    epoch,
                'model':    model.state_dict(),
                'val_auc':  val_auc,
                'val_f1':   val_f1,
                'd_model':  d_model,
                'n_heads':  n_heads,
                'n_layers': n_layers,
                # 事件定位資料（convert_contact.py）才有：推論要用同一個 anchor 對回影格，
                # 事件級驗證只能用 val_clips（沒訓練過的 clip）
                'val_clips': val_clips,
                'data_meta': getattr(dataset, 'meta', {}),   # {anchor, sigma}
                'fold': args.fold if args.n_folds > 0 else None,
                'n_folds': args.n_folds,
            }, best_path)
            tag = ' ← best'
        else:
            no_improve += 1
            tag = ''

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={val_loss:.4f}  AUC={val_auc:.4f}  F1={val_f1:.4f}"
              f"  lr={optimizer.param_groups[0]['lr']:.2e}{tag}")

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if val_loader is None:
        print(f"\n全部資料訓練完成（{args.epochs} 輪）→ {best_path}")
        return

    # 用最佳 checkpoint 印詳細指標
    print(f"\n── Final evaluation on validation set ──")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if args.pairwise:
        eval_epoch_pairwise(model, val_loader, criterion, device, verbose=True)
    print(f"Best AUC: {best_auc:.4f}  Best F1: {best_f1:.4f}  (best epoch {ckpt['epoch']})")
    print(f"Checkpoint saved → {best_path}")

    if args.tag == 'weak':
        print(f"\nNext step: run finetune.py with --pretrain_ckpt {best_path} --tag manual")


if __name__ == '__main__':
    main()
