"""
sam3_concept_check.py — 阶段 2 概念驗證：SAM3 的 concept prompt 能不能乾淨只框球員。

背景（見 docs/superpowers/specs/2026-09-05-sam3-nonplayer-filtering-design.md）：
現有 SAM2 pipeline 追蹤池混入非球員（裁判/籃框/球/計時器），驗證過所有幾何/embedding
信號都分不乾淨。SAM3 的 Promptable Concept Segmentation 用文字 "basketball player"
從語義層只追球員，是唯一對症的方案。本腳本驗證這個假設在**我方影片**上成不成立。

⚠ 這是**獨立驗證腳本**，跑在 SAM3 的 conda 環境（python=3.12），不 import 現有
basketball_analysis 的 utils/pipeline（那是另一個環境）。只依賴 sam3 + opencv + numpy。

前置：
  1. 依 github.com/facebookresearch/sam3 裝好 sam3 環境
  2. HF 申請 facebook/sam3 checkpoint access → hf auth login
  3. 準備影片（mp4）

跑法（SAM3 環境）：
  python sam3_concept_check.py \
      --video input_videos/screen/screen_130.mp4 \
      --prompt "basketball player" \
      --out output_videos/sam3_check_v130.mp4

驗收：人工看輸出影片，SAM3 是否乾淨只框 10 個真球員、排除裁判/籃框/球/計時器。
      對照 referee_labels 裡各支的非球員，確認 SAM3 有沒有把它們排除掉。
"""

import os
import argparse

import numpy as np
import cv2
import torch

from sam3.model_builder import build_sam3_video_predictor


def masks_to_bbox(mask):
    """binary mask (H,W bool/0-1) → [x1,y1,x2,y2] 或 None。"""
    m = np.asarray(mask)
    if m.ndim > 2:
        m = m.squeeze()
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def extract_obj_masks(frame_output):
    """
    從 SAM3 單幀 outputs 取出 {obj_id: mask}。

    SAM3 video predictor 的 per-frame outputs 結構（實跑確認）：
      out_obj_ids     : array of int，每個追蹤物體的 id
      out_probs       : array of float，每個物體是該 concept 的信心
      out_boxes_xywh  : 歸一化 xywh 框
      out_binary_masks: 每個物體的 binary mask
    """
    if not hasattr(frame_output, 'keys'):
        return None
    if 'out_obj_ids' not in frame_output or 'out_binary_masks' not in frame_output:
        return None
    ids = frame_output['out_obj_ids']
    masks = frame_output['out_binary_masks']
    out = {}
    for oid, m in zip(ids, masks):
        mm = m.cpu().numpy() if hasattr(m, 'cpu') else np.asarray(m)
        out[int(oid)] = mm
    return out


def draw_frame(frame_bgr, obj_masks):
    """把每個 obj 的 bbox + id 畫上去（半透明 mask 疊色）。"""
    img = frame_bgr.copy()
    rng = np.random.default_rng(0)
    for oid, mask in sorted(obj_masks.items()):
        bbox = masks_to_bbox(mask)
        if bbox is None:
            continue
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        # mask 疊色
        m = np.asarray(mask)
        if m.ndim > 2:
            m = m.squeeze()
        if m.shape[:2] == img.shape[:2]:
            img[m > 0] = (0.5 * img[m > 0] + 0.5 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"#{oid}", (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='mp4 影片路徑')
    ap.add_argument('--prompt', default='basketball player',
                    help='concept 文字提示（預設 basketball player）')
    ap.add_argument('--out', required=True, help='輸出標框影片路徑（mp4）')
    ap.add_argument('--single_gpu', action='store_true',
                    help='只用一張 GPU（預設用全部）')
    ap.add_argument('--max_frames', type=int, default=0,
                    help='只處理前 N 幀（0=整支）。GB10 受限時先用小 N 驗證 concept 能力，'
                         '不賭整支跑完')
    args = ap.parse_args()

    if args.single_gpu:
        gpus = [torch.cuda.current_device()]
    else:
        gpus = list(range(torch.cuda.device_count()))
    print(f"GPUs: {gpus}")

    # 用 ffmpeg 把影片抽成 jpg 幀資料夾（container cv2 沒 FFMPEG 讀不了影片，
    # 見 CLAUDE.md 坑10；ffmpeg 繞過）。SAM3 的 start_session 可直接吃 jpg 資料夾。
    import subprocess, tempfile, shutil
    frame_dir = tempfile.mkdtemp(prefix='sam3_frames_')
    vf = []
    if args.max_frames and args.max_frames > 0:
        vf = ['-frames:v', str(args.max_frames)]
        print(f"  截取前 {args.max_frames} 幀（省 GB10 資源）")
    cmd = ['ffmpeg', '-y', '-i', args.video, *vf,
           '-start_number', '0', os.path.join(frame_dir, '%05d.jpg')]
    print(f"  ffmpeg 抽幀 → {frame_dir}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    jpgs = sorted(f for f in os.listdir(frame_dir) if f.endswith('.jpg'))
    if not jpgs:
        print("  ffmpeg 抽幀失敗，stderr：")
        print(r.stderr[-1500:])
        shutil.rmtree(frame_dir, ignore_errors=True)
        return
    print(f"  抽出 {len(jpgs)} 幀")
    video_for_sam3 = frame_dir   # 餵 jpg 資料夾給 SAM3

    predictor = build_sam3_video_predictor(gpus_to_use=gpus)

    # 1. 開 session
    print(f"start_session: {video_for_sam3}")
    resp = predictor.handle_request(request=dict(
        type="start_session", resource_path=video_for_sam3))
    session_id = resp["session_id"]

    # 2. 加 concept prompt
    print(f'add_prompt: text="{args.prompt}"')
    resp = predictor.handle_request(request=dict(
        type="add_prompt", session_id=session_id,
        frame_index=0, text=args.prompt))
    # 印第一幀 output 結構，方便確認/修正 extract_obj_masks 的鍵名
    first_out = resp["outputs"]
    print("  第一幀 outputs 結構 keys:",
          list(first_out.keys()) if hasattr(first_out, 'keys') else type(first_out))

    # 3. propagate 全片，逐幀收 output
    print("propagate_in_video ...")
    outputs_per_frame = {}
    for r in predictor.handle_stream_request(request=dict(
            type="propagate_in_video", session_id=session_id)):
        outputs_per_frame[r["frame_index"]] = r["outputs"]
    n_frames = len(outputs_per_frame)
    print(f"  拿到 {n_frames} 幀 output")

    # 探測 output 結構（用中間一幀）
    probe = outputs_per_frame[sorted(outputs_per_frame)[n_frames // 2]]
    if extract_obj_masks(probe) is None:
        print("\n⚠ extract_obj_masks 取不到 mask/id，請看實際結構後修正鍵名：")
        print("  probe keys:", list(probe.keys()) if hasattr(probe, 'keys') else type(probe))
        print("  probe repr（前 500 字）:", str(probe)[:500])
        print("修正 extract_obj_masks 的 masks_key/ids_key 後重跑。")
        return

    # 4. 讀 jpg 幀（cv2 讀 jpg 不需要影片 codec）+ 畫框
    jpg_paths = [os.path.join(frame_dir, f) for f in jpgs]
    frames = [cv2.imread(p) for p in jpg_paths]
    print(f"  讀到 {len(frames)} 幀")

    # 統計每幀 obj 數（驗證污染有沒有被排除的量化指標）
    obj_counts = []
    out_frames = []
    for f_idx in range(len(frames)):
        fo = outputs_per_frame.get(f_idx)
        obj_masks = extract_obj_masks(fo) if fo is not None else {}
        obj_masks = obj_masks or {}
        obj_counts.append(len(obj_masks))
        out_frames.append(draw_frame(frames[f_idx], obj_masks))

    if obj_counts:
        import statistics
        print(f"\n每幀 SAM3 追到的 obj 數：median={statistics.median(obj_counts):.0f} "
              f"min={min(obj_counts)} max={max(obj_counts)}")
        print("（真球員應該是 10；若明顯 >10 表示仍有非球員被追進來）")

    # 存輸出：先存標框 jpg（一定成功），再用 ffmpeg 合成 mp4（container cv2 寫 mp4
    # 可能沒 codec，改用 ffmpeg 繞過）
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    draw_dir = tempfile.mkdtemp(prefix='sam3_drawn_')
    for i, fr in enumerate(out_frames):
        cv2.imwrite(os.path.join(draw_dir, f'{i:05d}.jpg'), fr)
    cmd_out = ['ffmpeg', '-y', '-framerate', '30',
               '-i', os.path.join(draw_dir, '%05d.jpg'),
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', args.out]
    r2 = subprocess.run(cmd_out, capture_output=True, text=True)
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        print(f"\n標框影片 → {args.out}")
    else:
        # ffmpeg 合成失敗 → 至少留 jpg 讓你看
        keep = os.path.splitext(args.out)[0] + '_frames'
        shutil.move(draw_dir, keep)
        print(f"\n⚠ mp4 合成失敗，標框 jpg 存在 → {keep}")
        print(r2.stderr[-800:])
    shutil.rmtree(draw_dir, ignore_errors=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    print("人工驗收：SAM3 是否乾淨只框球員、排除裁判/籃框/球/計時器。")


if __name__ == '__main__':
    main()
