# SAM3 追蹤整合實作 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 SAM3 concept segmentation 取代 SAM2 當球員追蹤器，從源頭只追球員（排除裁判/籃框/球等非球員污染），下游 inference 幾乎不改。

**Architecture:** 兩階段解耦。階段一：`run_sam3_tracks.py` 在 sam3_env container（cu13.1）跑 SAM3 concept `"basketball player"`，產出 `player_tracks`（格式同 SAM2）dump 成 pkl。階段二：`inference.py`（basketball_analysis container）加 `--sam3_tracks` 參數，給定就讀 pkl 當 player_tracks、跳過自己的 SAM2，下游 court/分隊/handler/擋拆全部不變。兩 container 各用各的 torch，透過 pkl 傳遞，不衝突。

**Tech Stack:** SAM3（sam3_env, torch 2.10 nv, cu13.1）、現有 inference pipeline（basketball_analysis, torch 2.7）、pickle 傳遞、ffmpeg 抽幀。

**Spec:** `docs/superpowers/specs/2026-09-05-sam3-nonplayer-filtering-design.md`

## Global Constraints

- SAM3 腳本跑在 sam3_env container，用 `HF_TOKEN` 環境變數認證（hf CLI 壞），`PYTHONIOENCODING=utf-8`。
- 影片讀取用 ffmpeg 抽 jpg 資料夾餵 SAM3（container cv2 沒 FFMPEG）。
- `player_tracks` 格式：`list[dict[int_pid: {'bbox': [x1,y1,x2,y2]}]]`，每個 frame 一個 dict，bbox 是像素座標（float 或 int）。**下游只讀 `info['bbox']`**，這是唯一契約。
- **驗證方式**：這個 pipeline 無法用 pytest（需 GPU/SAM3/影片/跨 container）。每個任務用「跑真實影片 + 對照輸出」驗證，不寫 pytest 測試。
- SAM3 的 obj_id 每次跑重新分配（同 SAM2），只保證單支影片內一致。

---

### Task 1: `run_sam3_tracks.py` — SAM3 產 player_tracks pkl（sam3_env）

**Files:**
- Create: `nba_data_pipeline/run_sam3_tracks.py`
- Reference（複用邏輯）：`nba_data_pipeline/sam3_concept_check.py`（ffmpeg 抽幀 + SAM3 concept + mask→bbox 已驗證）

**Interfaces:**
- Consumes: SAM3 video predictor API（`build_sam3_video_predictor`、`start_session`、`add_prompt` text、`propagate_in_video`）；每幀 outputs 鍵 `out_obj_ids`、`out_binary_masks`。
- Produces: pkl 檔含 `{'player_tracks': list[dict[pid:{'bbox':[x1,y1,x2,y2]}]], 'n_frames': int, 'fps': float}`。`player_tracks` 每 frame 一個 dict，格式同 SAM2 `run_sam2_tracks` 回傳。

- [ ] **Step 1: 寫 run_sam3_tracks.py 主體**

複用 sam3_concept_check.py 的 ffmpeg 抽幀 + run_concept + masks_to_bbox。核心：

```python
"""
run_sam3_tracks.py — SAM3 concept "basketball player" 追蹤 → player_tracks pkl。
取代 SAM2 的 run_sam2_tracks。跑在 sam3_env（cu13.1）。

用法：
  python run_sam3_tracks.py --video input_videos/screen/screen_130.mp4 \
      --out stubs/screen_130_sam3_tracks.pkl
"""
import os, argparse, subprocess, tempfile, shutil, pickle
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor


def masks_to_bbox(mask):
    m = np.asarray(mask)
    if m.ndim > 2:
        m = m.squeeze()
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def get_fps(video):
    r = subprocess.run(['ffprobe', '-v', '0', '-of', 'csv=p=0',
                        '-select_streams', 'v:0', '-show_entries',
                        'stream=r_frame_rate', video],
                       capture_output=True, text=True)
    try:
        num, den = r.stdout.strip().split('/')
        return float(num) / float(den)
    except Exception:
        return 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompt', default='basketball player')
    ap.add_argument('--single_gpu', action='store_true')
    args = ap.parse_args()

    gpus = [torch.cuda.current_device()] if args.single_gpu \
        else list(range(torch.cuda.device_count()))

    frame_dir = tempfile.mkdtemp(prefix='sam3t_')
    subprocess.run(['ffmpeg', '-y', '-i', args.video, '-start_number', '0',
                    os.path.join(frame_dir, '%05d.jpg')],
                   capture_output=True, text=True)
    jpgs = sorted(f for f in os.listdir(frame_dir) if f.endswith('.jpg'))
    if not jpgs:
        print("ffmpeg 抽幀失敗"); shutil.rmtree(frame_dir, ignore_errors=True); return
    n = len(jpgs)
    print(f"抽出 {n} 幀")

    predictor = build_sam3_video_predictor(gpus_to_use=gpus)
    resp = predictor.handle_request(request=dict(
        type="start_session", resource_path=frame_dir))
    sid = resp["session_id"]
    predictor.handle_request(request=dict(
        type="add_prompt", session_id=sid, frame_index=0, text=args.prompt))

    # 逐幀收 output → player_tracks（同 SAM2 格式）
    per_frame = {}
    for r in predictor.handle_stream_request(request=dict(
            type="propagate_in_video", session_id=sid)):
        per_frame[r["frame_index"]] = r["outputs"]

    player_tracks = []
    for f in range(n):
        fo = per_frame.get(f)
        frame_tracks = {}
        if fo is not None and 'out_obj_ids' in fo and 'out_binary_masks' in fo:
            for oid, m in zip(fo['out_obj_ids'], fo['out_binary_masks']):
                mm = m.cpu().numpy() if hasattr(m, 'cpu') else np.asarray(m)
                bbox = masks_to_bbox(mm)
                if bbox is not None:
                    frame_tracks[int(oid)] = {'bbox': bbox}
        player_tracks.append(frame_tracks)

    fps = get_fps(args.video)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as fp:
        pickle.dump({'player_tracks': player_tracks, 'n_frames': n, 'fps': fps}, fp)
    n_tracked = sum(1 for ft in player_tracks if ft)
    print(f"SAM3 tracked {n_tracked}/{n} frames → {args.out}")
    shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 本機語法檢查**

Run: `python -m py_compile nba_data_pipeline/run_sam3_tracks.py`
Expected: 無錯誤（本機沒 sam3 套件，只檢查語法）。

- [ ] **Step 3: 傳到 DGX sam3_env 跑一支**

```bash
scp nba_data_pipeline/run_sam3_tracks.py vipl-spark:/home/vipl/timwu/basketball_analysis/nba_data_pipeline/
```
在 DGX：
```bash
docker exec -i sam3_env bash <<'EOF'
export HF_TOKEN="<你的token>"
export PYTHONIOENCODING=utf-8
cd /workspace
python -X utf8 nba_data_pipeline/run_sam3_tracks.py \
  --video input_videos/screen/screen_130.mp4 \
  --out stubs/screen_130_sam3_tracks.pkl --single_gpu
EOF
```
Expected: 印 `抽出 N 幀` + `SAM3 tracked N/N frames`，產出 pkl。

- [ ] **Step 4: 驗證 pkl 格式正確**

```bash
docker exec -i sam3_env bash -c "cd /workspace && python -X utf8 -c \"
import pickle
d=pickle.load(open('stubs/screen_130_sam3_tracks.pkl','rb'))
print('keys:', list(d.keys()))
print('n_frames:', d['n_frames'], 'fps:', d['fps'])
pt=d['player_tracks']
print('frame100 pids:', sorted(pt[100].keys()))
print('frame100 一個 bbox:', list(pt[100].values())[0])
# 驗證每幀 obj 數（應接近 10 真球員，不像 SAM2 的 14）
import statistics
cnts=[len(ft) for ft in pt if ft]
print('每幀 obj 數 median:', statistics.median(cnts))
\""
```
Expected: keys 含 player_tracks/n_frames/fps；bbox 是 4 個數字的 list；每幀 obj median 接近 10（證明只追球員，非球員被排除）。

- [ ] **Step 5: Commit**

```bash
git add nba_data_pipeline/run_sam3_tracks.py
git commit -m "feat: run_sam3_tracks — SAM3 concept 追球員產 player_tracks pkl"
```

---

### Task 2: inference.py 加 `--sam3_tracks` 讀 pkl 跳過 SAM2

**Files:**
- Modify: `nba_data_pipeline/inference.py`（argparse 區約 985-987 加參數；SAM2 追蹤區約 988-994 加分支）

**Interfaces:**
- Consumes: Task 1 產的 pkl（`{'player_tracks':..., 'n_frames':..., 'fps':...}`）。
- Produces: 給定 `--sam3_tracks` 時，`player_tracks` 變數來自 pkl，跳過 `run_sam2_tracks` + `verify_sam2_with_yolo`；下游（handler 偵測、court、分隊、擋拆）不變。

- [ ] **Step 1: 加 argparse 參數**

在 inference.py 的 `--fps` 參數後（約 line 986）加：

```python
    parser.add_argument('--sam3_tracks', default=None,
                        help='SAM3 產的 player_tracks pkl 路徑。給定則讀它當追蹤結果，'
                             '跳過 SAM2（用 SAM3 乾淨追蹤取代）')
```

- [ ] **Step 2: 加讀 pkl 分支，跳過 SAM2**

找到 SAM2 追蹤區塊（約 line 988-994）：

```python
    print("Player tracking via SAM2...")
    player_tracks = run_sam2_tracks(frames, player_model, predictor, device=str(device))
    n_tracked = sum(1 for ft in player_tracks if ft)
    print(f"  SAM2 tracked {n_tracked}/{len(frames)} frames")

    # 用 YOLO 每幀重新偵測，清掉 SAM2 mask drift 的 bbox
    player_tracks = verify_sam2_with_yolo(frames, player_tracks, player_model)
```

改成：

```python
    if args.sam3_tracks:
        print(f"讀 SAM3 追蹤結果: {args.sam3_tracks}")
        import pickle as _pkl
        with open(args.sam3_tracks, 'rb') as _fp:
            _sam3 = _pkl.load(_fp)
        player_tracks = _sam3['player_tracks']
        if len(player_tracks) != len(frames):
            print(f"  ⚠ SAM3 幀數({len(player_tracks)}) != 影片幀數({len(frames)})，"
                  f"取較短對齊")
            m = min(len(player_tracks), len(frames))
            player_tracks = player_tracks[:m]
            frames = frames[:m]
        n_tracked = sum(1 for ft in player_tracks if ft)
        print(f"  SAM3 tracked {n_tracked}/{len(frames)} frames（跳過 SAM2 + verify）")
    else:
        print("Player tracking via SAM2...")
        player_tracks = run_sam2_tracks(frames, player_model, predictor, device=str(device))
        n_tracked = sum(1 for ft in player_tracks if ft)
        print(f"  SAM2 tracked {n_tracked}/{len(frames)} frames")
        # 用 YOLO 每幀重新偵測，清掉 SAM2 mask drift 的 bbox
        player_tracks = verify_sam2_with_yolo(frames, player_tracks, player_model)
```

- [ ] **Step 3: SAM2 predictor 只在需要時建（省 SAM3 模式的載入）**

找到 predictor 建立處（約 line 986）：

```python
    print(f"Loading SAM2 predictor on {device}...")
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_ckpt, device=str(device))
```

改成（只在非 SAM3 模式建）：

```python
    predictor = None
    if not args.sam3_tracks:
        print(f"Loading SAM2 predictor on {device}...")
        predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_ckpt, device=str(device))
```

- [ ] **Step 4: 本機語法檢查**

Run: `python -m py_compile nba_data_pipeline/inference.py`
Expected: 無錯誤。

- [ ] **Step 5: DGX 跑完整 inference（用 SAM3 tracks）**

先把 Task 1 產的 pkl 從 sam3_env 複製到 basketball_analysis 能讀的路徑（兩 container 都掛載 /workspace，同一主機目錄，直接共用）：
```bash
docker exec -i basketball_analysis bash <<'EOF'
cd /workspace
rm -f stubs/player_assignment_stub.pkl
python nba_data_pipeline/inference.py \
  --video input_videos/screen/screen_130.mp4 \
  --ckpt nba_data_pipeline/checkpoints/finetune_manual_best.pt \
  --handler_model models/handler_detector.pt \
  --sam3_tracks stubs/screen_130_sam3_tracks.pkl \
  --output_video output_videos/screens_v130_sam3.avi \
  --threshold 0.70 --debug 2>&1 | tee output_videos/v130_sam3.log
EOF
```
Expected: 印 `讀 SAM3 追蹤結果` + `SAM3 tracked N/N frames（跳過 SAM2 + verify）`，pipeline 跑完產出 events.pkl + 影片，**不報錯**（下游 court/分隊/handler/擋拆都吃得下 SAM3 的 player_tracks）。

- [ ] **Step 6: 對照驗證 SAM3 vs SAM2 的 obj 乾淨度**

```bash
docker exec -i basketball_analysis bash -c "cd /workspace && python -X utf8 -c \"
import pickle
d=pickle.load(open('output_videos/screens_v130_sam3_events.pkl','rb'))
cc=d['court_coords']
allids=set()
for fr in cc: allids.update(fr)
print('SAM3 版追到的 obj id:', sorted(allids), '共', len(allids), '個')
print('（對照：SAM2 版 v130 追 14 個含裁判/籃框；SAM3 應接近 10 真球員）')
\""
```
Expected: obj 數接近 10（非球員污染消失），證明 SAM3 追蹤乾淨。

- [ ] **Step 7: Commit**

```bash
git add nba_data_pipeline/inference.py
git commit -m "feat: inference 加 --sam3_tracks 讀 SAM3 pkl 跳過 SAM2"
```

---

### Task 3: 驗證 SAM3 tracks 品質 + 決定是否需要 verify（drift 清理）

**Files:**
- 無程式改動（純驗證任務）。若發現需要，才在後續 plan 加 verify。

**Interfaces:**
- Consumes: Task 2 產的 SAM3 版 events.pkl。
- Produces: 「SAM3 tracks 要不要 verify」的結論（記進 spec）。

- [ ] **Step 1: 跑 3 支影片對照 SAM3 vs SAM2 的擋拆偵測結果**

對 v128、v129、v130 各跑 SAM3 版（Task 1 產 tracks pkl → Task 2 inference），比對：
- 每幀 obj 數（SAM3 應 ~10，SAM2 ~14）
- 擋拆事件數量、handler 偵測率（`handler matched in X/N frames` 那行）
- court_coords 有沒有明顯飄移/漏幀（SAM2 的老問題）

```bash
# 對每支：先 sam3_env 產 tracks，再 basketball_analysis 跑 inference
for v in 128 129 130; do
  docker exec -i sam3_env bash -c "export HF_TOKEN='<token>'; export PYTHONIOENCODING=utf-8; cd /workspace; python -X utf8 nba_data_pipeline/run_sam3_tracks.py --video input_videos/screen/screen_\$v.mp4 --out stubs/screen_\${v}_sam3_tracks.pkl --single_gpu"
  docker exec -i basketball_analysis bash -c "cd /workspace; rm -f stubs/player_assignment_stub.pkl; python nba_data_pipeline/inference.py --video input_videos/screen/screen_\$v.mp4 --ckpt nba_data_pipeline/checkpoints/finetune_manual_best.pt --handler_model models/handler_detector.pt --sam3_tracks stubs/screen_\${v}_sam3_tracks.pkl --output_video output_videos/screens_v\${v}_sam3.avi --threshold 0.70 --debug 2>&1 | tee output_videos/v\${v}_sam3.log"
done
```
Expected: 三支都跑完不報錯；obj 數乾淨；handler 偵測率不低於 SAM2 版。

- [ ] **Step 2: 看標框影片確認追蹤穩定度**

抓回 3 支標框影片，人工看：球員框有沒有 SAM2 那種 mask drift（框飄到別人身上）、id 有沒有亂跳。
```bash
docker cp basketball_analysis:/workspace/output_videos/screens_v130_sam3.avi /home/vipl/timwu/basketball_analysis/output_videos/
scp vipl-spark:/home/vipl/timwu/basketball_analysis/output_videos/screens_v130_sam3.avi output_videos/
```
Expected: 追蹤穩定，明顯優於或不劣於 SAM2。

- [ ] **Step 3: 記錄結論到 spec，決定 verify 去留**

在 spec 的「阶段 3」加一段：SAM3 tracks 品質實測結果 + 「verify 是否需要」的結論。
- 若 SAM3 追蹤乾淨（幾乎不 drift）→ 確認省掉 verify_sam2_with_yolo，記錄。
- 若仍有 drift → 開後續 plan 加 SAM3 版 verify（用 handler_detector 或 YOLO 每幀對照清 drift）。

```bash
git add docs/superpowers/specs/2026-09-05-sam3-nonplayer-filtering-design.md
git commit -m "docs: SAM3 tracks 品質實測 + verify 去留結論"
```

---

## Self-Review

**Spec coverage:**
- 阶段 3「SAM3 單獨跑 dump player_tracks，下游讀 pkl 跳過 SAM2」→ Task 1（產 pkl）+ Task 2（讀 pkl）✅
- 「刪掉為 SAM2 污染寫的清非球員補救邏輯」→ **本 plan 暫不刪**（Task 2 只加分支跳過，不動補救邏輯）。理由：先讓 SAM3 路徑跑通、驗證乾淨，確認後再開獨立 plan 刪補救邏輯（降低單次改動風險）。已在此註明為後續工作。
- 「verify 去留」→ Task 3 明確驗證後決定 ✅
- handler 續用 handler_detector IOU 配對 → Task 2 下游不變，自動成立 ✅

**Placeholder scan:** `<你的token>`/`<token>` 是使用者執行時填的實際密鑰，非程式 placeholder（不該寫進檔案）。其餘無 TBD/TODO。

**Type consistency:** player_tracks 格式 `list[dict[int:{'bbox':[...]}]]` 在 Task 1 產出、Task 2 消費，一致。pkl 鍵 `player_tracks`/`n_frames`/`fps` 在 Task 1 dump、Task 2/驗證讀取，一致。

**後續工作（不在本 plan）：**
1. 刪 SAM2 污染的清非球員補救邏輯（inference.py ~1001-1056）— 確認 SAM3 乾淨後開獨立 plan。
2. 範圍 B：SAM3 追球 + v1 持球判定取代 handler_detector。
3. off-ball 掩護偵測（依賴乾淨追蹤，此 plan 完成後解鎖）。
