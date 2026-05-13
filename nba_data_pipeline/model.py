"""
MiniScreenNet：LSTM + Transformer + Team-wise Pooling
對應 NETS 論文架構，輸入為 10 名球員（5進攻 + 5防守）

角色編碼（roles）：
  0 = ball handler（進攻持球者，slot 0）
  1 = other offense（其他進攻球員，slot 1~4）
  2 = defense（防守球員，slot 5~9）

輸入 slot 順序：
  [handler, off1, off2, off3, off4, def1, def2, def3, def4, def5]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_OFFENSE  = 5
N_DEFENSE  = 5
N_PLAYERS  = N_OFFENSE + N_DEFENSE   # 10
N_ROLES    = 3    # handler / other_offense / defense
INPUT_LEN  = 7    # 預訓練輸入幀數
PRED_LEN   = 3    # 預訓練預測幀數


# ── Base Model ────────────────────────────────────────────────────────────────

class MiniScreenNet(nn.Module):
    """
    Base 模型，對應論文 Figure 3 的 Base Transformer。

    流程：
      1. 每位球員的軌跡各跑一個 LSTM → 初始 embedding z^0
         對應論文：LSTM layer to embed trajectories
      2. 加入角色編碼（handler / offense / defense）
         對應論文：role one-hot encoding concatenated to LSTM output
      3. Transformer Encoder（10人互相 Attention）→ context-aware embedding z^N
         對應論文：multi-head self-attention, N layers
    """

    def __init__(self, d_model=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 共享一個 LSTM（論文對同角色共享權重）
        # 所有10名球員用同一個 LSTM，靠 role_emb 區分身份
        self.lstm = nn.LSTM(
            input_size=2,           # (x, y) 座標
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        # 角色投影：論文做法 — 3維 one-hot concat LSTM輸出後，投影回 d_model
        # z^0_o = proj(concat(LSTM_output, role_one_hot))
        # LSTM 輸出 d_model 維 + one-hot 3 維 → Linear 投影回 d_model 維
        self.role_proj = nn.Linear(d_model + N_ROLES, d_model)

        # Transformer Encoder，10個球員互相做 Attention
        # 對應論文：multi-head self-attention with N layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,         # Pre-LN，訓練更穩定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_layers)

    def forward(self, trajectories, roles):
        """
        Args:
            trajectories : (B, 10, T, 2)  — 10人的位置序列
            roles        : (B, 10)         — 每人的角色 id

        Returns:
            embeddings   : (B, 10, d_model) — context-aware embedding z^N
        """
        B, N, T, _ = trajectories.shape   # N=10

        # Step 1：每人各跑 LSTM，取最後時步的隱藏狀態
        # 對應論文 Figure 3 最底層的 LSTM 行
        lstm_outs = []
        for i in range(N):
            _, (h, _) = self.lstm(trajectories[:, i])   # h: (1, B, d)
            lstm_outs.append(h.squeeze(0))               # (B, d)   d:dimension
        x = torch.stack(lstm_outs, dim=1)                # (B, 10, d)

        # Step 2：論文做法 — one-hot concat 後投影
        # 對應論文：z^0_o = Linear(LSTM_output ⊕ role_one_hot)
        role_onehot = F.one_hot(roles, num_classes=N_ROLES).float()  # (B, 10, 3)
        x = self.role_proj(torch.cat([x, role_onehot], dim=-1))      # (B, 10, d)

        # Step 3：Transformer，10人互相 Attention
        # 對應論文公式 (1)(2)(3)
        x = self.transformer(x)                          # (B, 10, d)

        return x


# ── Team-wise Pooling ─────────────────────────────────────────────────────────

def team_wise_pool(embeddings):
    """
    對應論文 Figure 5 / 公式 (4) 的 team-wise pooling。

    論文：y = softmax(FF(z^N_B ⊕ Σ z^N_Ai ⊕ Σ z^N_Di))
    這裡：把 handler embedding、offense 加總、defense 加總 串接起來

    Args:
        embeddings: (B, 10, d_model)
          slot 0    = handler
          slot 1~4  = other offense
          slot 5~9  = defense

    Returns:
        pooled: (B, 3 * d_model)
    """
    handler_emb = embeddings[:, 0, :]           # (B, d) — 單獨保留 handler
    offense_sum = embeddings[:, 1:5, :].sum(1)  # (B, d) — 其他進攻者加總
    defense_sum = embeddings[:, 5:,  :].sum(1)  # (B, d) — 防守者加總
    return torch.cat([handler_emb, offense_sum, defense_sum], dim=-1)  # (B, 3d)


# ── Heads ─────────────────────────────────────────────────────────────────────

class TrajectoryHead(nn.Module):
    """
    自監督預訓練用：預測未來 PRED_LEN 幀的速度向量。
    對應論文 Figure 4 / Section 3.3 "Trajectory Prediction Head"。

    論文設計：handler/offense/defense 各用不同的 FF。
    """

    def __init__(self, d_model=128, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len

        def ff():
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, pred_len * 2),   # pred_len × (dx, dy)
            )

        self.ff_handler  = ff()   # slot 0
        self.ff_offense  = ff()   # slot 1~4（共享同一個 FF）
        self.ff_defense  = ff()   # slot 5~9（共享同一個 FF）

    def forward(self, embeddings):
        """
        Args:
            embeddings: (B, 10, d_model)

        Returns:
            pred_velocities: (B, 10, PRED_LEN, 2)
        """
        B = embeddings.shape[0]
        preds = []

        # slot 0：handler
        preds.append(self.ff_handler(embeddings[:, 0, :]))

        # slot 1~4：other offense（用同一個 FF）
        for i in range(1, N_OFFENSE):
            preds.append(self.ff_offense(embeddings[:, i, :]))

        # slot 5~9：defense（用同一個 FF）
        for i in range(N_OFFENSE, N_PLAYERS):
            preds.append(self.ff_defense(embeddings[:, i, :]))

        out = torch.stack(preds, dim=1)              # (B, 10, pred_len*2)
        return out.view(B, N_PLAYERS, self.pred_len, 2)


class ClassificationHead(nn.Module):
    """
    掩護分類用。
    對應論文 Figure 5 / Section 3.3 "Classification Head"。

    使用 team_wise_pool 後接 FF → 二元分類。
    """

    def __init__(self, d_model=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, d_model),   # 3d → d（pool 後的維度）
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),             # → logit
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: (B, 10, d_model)

        Returns:
            logits: (B,)
        """
        pooled = team_wise_pool(embeddings)    # (B, 3d)
        return self.net(pooled).squeeze(-1)    # (B,)


N_PAIRS = N_OFFENSE * (N_OFFENSE - 1)   # 5×4 = 20 個進攻配對


class PairwiseScreenHead(nn.Module):
    """
    Pairwise 掩護分類頭。
    對每一對 (offense_i, offense_j) 判斷 j 是否在幫 i 做掩護。

    共 N_OFFENSE × (N_OFFENSE-1) = 20 對，輸出 20 個 logit。
    所有配對共享同一個 FF（掩護條件與球員身份無關）。
    """

    def __init__(self, d_model=128, dropout=0.3):
        super().__init__()
        # 輸入：offense_i ⊕ offense_j ⊕ defense_sum → (3d,)
        self.ff = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),  # 維度從d_model -> 1 代表輸出的是掩護的機率
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: (B, 10, d_model)

        Returns:
            logits: (B, 20)  — 20 對 (i,j) 的掩護 logit
                    順序：for i in range(5): for j in range(5): if i!=j
        """
        offense     = embeddings[:, :N_OFFENSE, :]   # (B, 5, d)
        defense     = embeddings[:, N_OFFENSE:, :]   # (B, 5, d)
        defense_sum = defense.sum(dim=1)              # (B, d)

        scores = []
        for i in range(N_OFFENSE):       # i = 被掩護者（持球者）
            for j in range(N_OFFENSE):   # j = 掩護者
                if i == j:
                    continue
                pair = torch.cat([
                    offense[:, i, :],    # (B, d) 被掩護者 embedding
                    offense[:, j, :],    # (B, d) 掩護者 embedding
                    defense_sum,         # (B, d) 防守整體
                ], dim=-1)               # (B, 3d)
                scores.append(self.ff(pair).squeeze(-1))   # (B,)

        return torch.stack(scores, dim=1)   # (B, 20)


# ── Combined Models ───────────────────────────────────────────────────────────

class ScreenNetPretrain(nn.Module):
    """Base + TrajectoryHead，用於自監督預訓練。"""

    def __init__(self, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.base = MiniScreenNet(d_model, n_heads, n_layers)
        self.head = TrajectoryHead(d_model)

    def forward(self, trajectories, roles):
        return self.head(self.base(trajectories, roles))


class ScreenNetClassifier(nn.Module):
    """Base + ClassificationHead，用於掩護分類（fine-tune）。"""

    def __init__(self, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.base = MiniScreenNet(d_model, n_heads, n_layers)
        self.head = ClassificationHead(d_model)

    def forward(self, trajectories, roles):
        return self.head(self.base(trajectories, roles))

    @classmethod
    def from_pretrained(cls, ckpt_path, d_model=128, n_heads=4, n_layers=4):
        """
        從預訓練 checkpoint 載入 base 權重，分類頭隨機初始化。
        對應論文的 fine-tuning 流程：transfer weights from pretrained base。
        """
        obj = cls(d_model, n_heads, n_layers)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # 只載入 base 的權重，跳過 head
        pretrain_state = {
            k.replace('base.', ''): v
            for k, v in ckpt['model'].items()
            if k.startswith('base.')
        }
        obj.base.load_state_dict(pretrain_state)
        return obj


class ScreenNetPairwise(nn.Module):
    """Base + PairwiseScreenHead，偵測所有進攻球員配對的掩護關係。"""

    def __init__(self, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.base = MiniScreenNet(d_model, n_heads, n_layers)
        self.head = PairwiseScreenHead(d_model)

    def forward(self, trajectories, roles):
        return self.head(self.base(trajectories, roles))   # (B, 20)

    @classmethod
    def from_pretrained(cls, ckpt_path, d_model=128, n_heads=4, n_layers=4):
        obj = cls(d_model, n_heads, n_layers)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        pretrain_state = {
            k.replace('base.', ''): v
            for k, v in ckpt['model'].items()
            if k.startswith('base.')
        }
        obj.base.load_state_dict(pretrain_state)
        return obj
