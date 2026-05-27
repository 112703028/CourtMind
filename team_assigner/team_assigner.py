import sys

sys.path.append('../')
from utils import read_stub, save_stub

try:
    import torch
except ImportError:
    torch = None

try:
    from sports import TeamClassifier
except ImportError:
    TeamClassifier = None


class TeamAssigner:
    """
    用 sports.TeamClassifier（SigLIP + UMAP + K-means）做無監督分隊。

    不需要 prompt（不用知道球衣顏色），自己從影片整體 visual pattern 學兩隊。

    Args:
        fit_stride (int): 每幾幀取一次 crops 來 fit classifier，預設 30
        crop_scale (float): bbox 縮小比例（0.4 = 取中央 40% 球衣區域），預設 0.4
        device (str): 'cuda' / 'cpu' / None（自動偵測）
    """
    def __init__(self, fit_stride=30, crop_scale=0.4, device=None):
        if TeamClassifier is None:
            raise ImportError(
                "sports.TeamClassifier 不可用。"
                "pip install git+https://github.com/roboflow/sports.git@feat/basketball"
            )

        self.fit_stride = fit_stride
        self.crop_scale = crop_scale

        if device is None:
            device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
        self.device = device

        self.classifier = None

    def _scale_bbox(self, bbox):
        """縮小 bbox 到中央 crop_scale 比例（聚焦球衣區域，避開頭/腳/背景）。"""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        nw, nh = w * self.crop_scale, h * self.crop_scale
        return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]

    def _crop_player(self, frame, bbox, min_size=5):
        """從 frame 切出縮小後 bbox 的區域，回 None 若無效。"""
        scaled = self._scale_bbox(bbox)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in scaled]
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < min_size or crop.shape[1] < min_size:
            return None
        return crop

    def load_model(self, crops):
        """用收集到的 crops fit TeamClassifier。"""
        print(f"  Fitting TeamClassifier on {self.device} with {len(crops)} crops...")
        self.classifier = TeamClassifier(device=self.device)
        self.classifier.fit(crops) # 抓出兩隊的特徵

    def get_player_color(self, frame, bbox):
        """單一球員的 team 預測（classifier 必須已 fit）。"""
        if self.classifier is None:
            raise RuntimeError("classifier 尚未 fit，先跑 get_player_teams_across_frames")
        crop = self._crop_player(frame, bbox)
        if crop is None:
            return 0
        return int(self.classifier.predict([crop])[0])

    def get_player_teams_across_frames(self, video_frames, player_tracks,
                                       read_from_stub=False, stub_path=None):
        """
        對所有 frame 分隊。回 list[dict[player_id: team_id]]。
        """
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        # 1. 收集 fit 用 crops
        print(f"  Collecting crops (stride={self.fit_stride}, scale={self.crop_scale})...")
        crops = []
        for f_idx in range(0, len(video_frames), self.fit_stride):
            if f_idx >= len(player_tracks):
                break
            for pid, track in player_tracks[f_idx].items():  # 遍歷一frame的所有 player
                bbox = track.get('bbox') if isinstance(track, dict) else None
                if bbox is None:
                    continue
                crop = self._crop_player(video_frames[f_idx], bbox)
                if crop is not None:
                    crops.append(crop)

        if len(crops) < 10:
            print(f"  ⚠ crops 太少（{len(crops)}），全部標 team=1")
            player_assignment = [{pid: 1 for pid in ft} for ft in player_tracks]
            save_stub(stub_path, player_assignment)
            return player_assignment

        # 2. fit
        self.load_model(crops)

        # 3. 對每幀每個 player 預測
        print(f"  Predicting per-frame teams...")
        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            frame_pred = {}
            batch_crops, batch_pids = [], [] # 每個frame 要更新一次
            for player_id, track in player_track.items():
                bbox = track.get('bbox') if isinstance(track, dict) else None
                if bbox is None:
                    continue
                crop = self._crop_player(video_frames[frame_num], bbox)
                if crop is None:
                    frame_pred[player_id] = 1
                    continue
                batch_crops.append(crop)
                batch_pids.append(player_id)

            if batch_crops:
                teams = self.classifier.predict(batch_crops)
                for pid, t in zip(batch_pids, teams):
                    frame_pred[pid] = int(t) + 1  # 0/1 → 1/2

            player_assignment.append(frame_pred)

        save_stub(stub_path, player_assignment)
        return player_assignment