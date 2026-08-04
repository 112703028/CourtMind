"""
TeamAssignerSiamese — 用訓練好的 Siamese 模型做分隊。

Drop-in 替換現有 team_assigner.TeamAssigner，介面相容：
  - 同樣的 get_player_teams_across_frames(frames, player_tracks) → list[dict[pid: team_id]]
  - team_id = 1 (跟 handler 同隊 / offense), 2 (異隊 / defense)

不同之處：
  - 不用 fit (Siamese 已預訓練，直接 inference)
  - **需要每幀知道 handler_id**（pairwise 比較的 anchor）

Usage:
  team_assigner = TeamAssignerSiamese('models/team_siamese.pt')
  player_assignment = team_assigner.get_player_teams_across_frames(
      frames, player_tracks, ball_acquisition
  )
"""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model import TeamSiamese

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import read_stub, save_stub


# 同 train.py 的 val_transform
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
IMG_SIZE  = 224

_val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def _scale_bbox(bbox, factor=0.5):
    """同 extract_pairs.py 的 scale_bbox（取中央區域聚焦球衣）。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = w * factor, h * factor
    return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]


def _crop_pil(frame_bgr, bbox, min_size=5):
    """BGR numpy frame → PIL RGB crop（縮 0.5 取中央區域）。"""
    scaled = _scale_bbox(bbox)
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in scaled]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    # BGR → RGB
    return Image.fromarray(crop[:, :, ::-1])


class TeamAssignerSiamese:
    """
    Siamese 模型 + handler anchor 的分隊。

    每幀流程：
      1. 用 ball_acquisition[f_idx] 找出該幀的 handler_id
      2. 切 handler crop，encode 成 embedding（cache）
      3. 對每個其他 player_id：
         - 切 crop，encode embedding
         - model.compare(handler_emb, player_emb) → 同隊機率
         - prob > 0.5 → team 1（offense），otherwise team 2（defense）
    """

    def __init__(self, ckpt_path='models/team_siamese.pt', device=None, threshold=0.5,
                 debug=False):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.threshold = threshold
        self._debug = debug

        # Load model
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model = TeamSiamese().to(device).eval()
        self.model.load_state_dict(ckpt['model'])
        print(f"  Loaded TeamSiamese from {ckpt_path} (epoch={ckpt.get('epoch')}, "
              f"val_f1={ckpt.get('val_f1', 0):.3f})")

    @torch.no_grad()
    def _encode_crops(self, crops_pil):
        """list[PIL] → (N, 512) embeddings tensor on device."""
        if not crops_pil:
            return torch.empty(0, 512, device=self.device)
        tensors = torch.stack([_val_transform(c) for c in crops_pil]).to(self.device)
        return self.model.encode(tensors)

    @torch.no_grad()
    def predict_teams_one_frame(self, frame, frame_tracks, handler_id):
        """
        對單一幀的所有 player 預測 team。

        Returns:
            dict[pid: team_id], team_id ∈ {1, 2}
        """
        if handler_id is None or handler_id < 0:
            return {}
        if handler_id not in frame_tracks:
            return {}

        handler_bbox = frame_tracks[handler_id].get('bbox')
        if handler_bbox is None:
            return {}

        # Handler crop
        h_crop = _crop_pil(frame, handler_bbox)
        if h_crop is None:
            return {}

        # 蒐集其他 player crops
        other_pids = []
        other_crops = []
        for pid, info in frame_tracks.items():
            if pid == handler_id:
                continue
            bbox = info.get('bbox')
            if bbox is None:
                continue
            crop = _crop_pil(frame, bbox)
            if crop is None:
                continue
            other_pids.append(pid)
            other_crops.append(crop)

        if not other_crops:
            return {handler_id: 1}  # 只有 handler 自己

        # Encode 所有 crops 一次（batch）
        all_crops = [h_crop] + other_crops
        embeddings = self._encode_crops(all_crops)
        h_emb = embeddings[0:1]            # (1, 512)
        other_embs = embeddings[1:]        # (N, 512)

        # 對每個 other 跟 handler 配對
        # h_emb 廣播：(1, 512) → (N, 512) 跟 other_embs concat
        h_repeat = h_emb.expand_as(other_embs)
        logits = self.model.compare(h_repeat, other_embs)   # (N,)
        probs = torch.sigmoid(logits).cpu().numpy()

        # Decision
        result = {handler_id: 1}     # handler 自己永遠 team 1
        for pid, p in zip(other_pids, probs):
            result[pid] = 1 if p >= self.threshold else 2

        # Debug: 印每個 pair 的 prob（只在 stride 幀）
        if getattr(self, '_debug_this_frame', False):
            probs_str = ', '.join(f'{pid}={p:.2f}' for pid, p in zip(other_pids, probs))
            print(f"    handler={handler_id}  probs: {probs_str}")

        return result

    def get_player_teams_across_frames(self, video_frames, player_tracks,
                                       ball_acquisition,
                                       read_from_stub=False, stub_path=None):
        """
        對所有 frame 分隊。

        Args:
            video_frames:       list of BGR numpy frames
            player_tracks:      list[dict[pid: {'bbox': [...]}]]
            ball_acquisition:   list[int]，每幀的 handler player_id（-1 表示無）
            read_from_stub:     是否從 stub 讀
            stub_path:          stub 檔路徑
        """
        cached = read_stub(read_from_stub, stub_path)
        if cached is not None and len(cached) == len(video_frames):
            return cached

        print(f"  Predicting per-frame teams (Siamese)...")
        player_assignment = []
        debug_stride = 30   # 每 30 幀印一次 debug
        for f_idx, frame in enumerate(video_frames):
            handler_id = ball_acquisition[f_idx] if f_idx < len(ball_acquisition) else -1
            # 只在特定幀印 debug（省 log）
            self._debug_this_frame = self._debug and (f_idx % debug_stride == 0)
            if self._debug_this_frame:
                print(f"  [SIAM] frame {f_idx}:")
            frame_pred = self.predict_teams_one_frame(
                frame, player_tracks[f_idx], handler_id
            )
            player_assignment.append(frame_pred)

        if stub_path is not None:
            save_stub(stub_path, player_assignment)
        return player_assignment
