"""
TeamSiamese — 判斷兩個球員 crop 是否同隊的 Siamese 網路。

架構：
  ResNet18 backbone（ImageNet pretrained，共享權重）
       ↓
  兩個 crop 各自編碼 → 512-d embedding
       ↓
  Concat → MLP head → 1-d logit
       ↓
  Sigmoid → 同隊機率
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TeamSiamese(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()

        # ResNet18 backbone（ImageNet 預訓練，去掉最後 fc）
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.embed_dim = 512  # ResNet18 final feature dim

        # Pairwise head：concat 兩個 embedding 後判斷
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x):
        """單 crop encoding，用於 inference 時 cache。"""
        return self.encoder(x)

    def compare(self, emb1, emb2):
        """兩個 embedding 做同隊判斷，回 logit。"""
        return self.head(torch.cat([emb1, emb2], dim=1)).squeeze(-1)

    def forward(self, crop1, crop2):
        """(B, 3, 224, 224) × 2 → (B,) logit"""
        emb1 = self.encode(crop1)
        emb2 = self.encode(crop2)
        return self.compare(emb1, emb2)
