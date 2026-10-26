"""
shot_detect.py — 推論用的投籃偵測（把 shot_owner.py 的規則包成一個函式）。

輸入：
  shot_sam3：run_sam3_tracks.py --ball 存的 10fps 抽樣影格 球員遮罩 + 球遮罩（pkl 的 'shot_sam3'）
  frames：原片影格（切鏡頭偵測用）
  rf：Roboflow NBA 偵測（籃框 / 跳投 / 上籃 / 球進框；run_rf() 產生，格式同 shot_detect_rf.py）
規則（2026-10-26，84 支延長片段人工核對，63 支有答案：出手時間 + 出手者全對 48）：
  一定要 10fps：5fps 追球，球飛起來那 0.2 秒會跟丟（screen_286）；球員遮罩借 5fps 的也掉到 45。
  球在空中（沒人碰）的時段 → 出手者 = 離手前 1.5 秒碰球最多的人；投籃動作幅度 ≥ 0.9 倍出手者身高；
  球離籃框 ≤ 1.0 倍身高或飛出畫面上方 = 強候選。類型 / 進球用 Roboflow（跳投 / 上籃、短暫的球進框）。
輸出：list[dict]（每次出手）：release_frame（原片影格）、t、shooter（已對回 player_tracks 的編號；
      shooter_raw = 投籃那次 SAM3 的原編號）、
      kind（jump-shot / layup / unknown）、made（True / False / None = 看不出來）、strong、amp、rim、segment
"""
from pathlib import Path

import numpy as np

import shot_owner as so
from shot_events import find_cuts, analyze, IN_BASKET

ROOT = Path(__file__).parent.parent
RF_WEIGHTS = ROOT / 'models' / 'rf_basketball_det_v4.onnx'


def run_rf(frames, fps, target_fps=10, weights=RF_WEIGHTS):
    """Roboflow NBA 偵測（ONNX，本機權重，不需金鑰）每 fps/target_fps 張跑一次 → shot_events 用的格式。"""
    from shot_detect_rf import OnnxDetector, KEEP
    det = OnnxDetector(weights)
    stride = max(1, int(round(fps / target_fps)))
    out = [(f, [d for d in det.infer(frames[f], conf=0.25, iou=0.9) if d[0] in KEEP])
           for f in range(0, len(frames), stride)]
    return dict(fps=fps, n=len(frames), stride=stride, size=frames[0].shape[1::-1], frames=out)


def map_ids(shot_sam3, player_tracks):
    """投籃偵測那次 SAM3（10fps）的球員編號 → 擋拆偵測用的 player_tracks（5fps 內插回原片）編號：
    同一張原片影格 bbox IoU ≥ 0.5，整段多數決。對不到的保留原編號（會在戰術分析對不到位置 / 隊伍）。"""
    from collections import Counter
    votes = {}
    for i, f in enumerate(shot_sam3['frame_idx']):
        if f >= len(player_tracks):
            break
        ref = {p: (v.get('bbox') if isinstance(v, dict) else v) for p, v in player_tracks[f].items()}
        for pid, b in shot_sam3['players'][i].items():
            best = max(((so.iou(b, rb), q) for q, rb in ref.items() if rb is not None), default=(0, None))
            if best[0] >= 0.5:
                votes.setdefault(pid, Counter())[best[1]] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


def detect_shots(shot_sam3, frames, rf, player_tracks=None):
    s = shot_sam3
    shape = frames[0].shape[:2]
    cuts = find_cuts(frames)
    bounds = [0] + cuts + [len(frames)]
    chosen, _ = so.pick_ball(s, rf)
    tch = so.smooth_touches(so.touches(s, shape, chosen))
    fps_o = s['orig_fps']
    idmap = map_ids(s, player_tracks) if player_tracks is not None else {}
    shots = []
    for seg, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if b - a < fps_o * 1.0:                         # 太短的鏡頭（轉場 / 重播片頭）不看
            continue
        rf_events, _, _ = analyze(None, rf, cuts, rng=(a, b))      # Roboflow：跳投 / 上籃候選、進球
        for c in so.shot_candidates(s, tch, rf, (a, b)):
            near = [e for e in rf_events if -0.6 <= e['t'] - c['release'] <= 1.2]
            e = min(near, key=lambda e: abs(e['t'] - c['release'])) if near else None
            made = any(x['made'] and -0.6 <= x['t'] - c['release'] <= 2.0 for x in rf_events)
            saw_rim_area = any(x[0] == IN_BASKET for f, dets in rf['frames']
                               if c['release'] * fps_o <= f <= (c['release'] + 2.0) * fps_o for x in dets) or c['rim'] is not None
            shots.append(dict(release_frame=int(round(c['release'] * fps_o)), t=c['release'],
                              shooter=idmap.get(c['shooter'], c['shooter']), shooter_raw=c['shooter'],
                              kind=e['kind'] if e else 'unknown',
                              made=True if made else (False if saw_rim_area else None),
                              strong=c['strong'], amp=c['amp'], rim=c['rim'], segment=seg))
    return shots, cuts
