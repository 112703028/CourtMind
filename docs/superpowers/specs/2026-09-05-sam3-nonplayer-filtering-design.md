# 非球員污染 → SAM3 感知層評估與路線

日期：2026-09-05
狀態：決策記錄 + 分階段路線（阶段 2 概念驗證待硬體到位）

## 問題脈絡（為何走到 SAM3）

從一個具體 bug 一路查到感知層架構問題：

1. **表象**：v130 影片跑 `tactics_agent` 的 `list_screens`，#10 明明有明顯掩護卻沒顯示。
2. **第一層**：events.pkl 裡 #10 有 3 筆（含 score 0.99），但 `find_screen_instances`
   把它濾掉了 → 疑似 `contact_thr=3.5` 門檻太嚴（v125-130 這批座標偏移，接觸距離
   被量成 5-8ft）。
3. **第二層**：深查發現 #10 在關鍵擋拆幀大量漏偵測（48% 出現率），SAM2 追蹤飄掉。
4. **第三層（根因）**：SAM2 追了 **14 個 obj，真球員只有 10 個**。多出的是**非球員**：
   裁判、籃框、球、計時器，甚至裁判衣服的橘色局部。這些污染追蹤池、佔 obj 名額、
   干擾真球員的 mask。

**污染範圍（人工標注 11 支影片，重跑 SAM2 後的 ground truth）**：
- 裁判：每支 1-4 個（116/117/118/119/121/125/127/128/129/130 都有）
- 籃框：6 支有（118/119/121/127/128/130）
- 球：1 支（121）
- 計時器：1 支（127）
- **身體局部**：v116 有一個 obj 是「裁判衣服的橘色部分」→ SAM2 甚至會把單一人的
  一塊區域切成獨立 obj。這是阶段 2 驗證 SAM3 的重要測試點（SAM3 concept 應輸出
  完整實例，不該產生這種局部碎片）。

## 驗證：現有所有信號都分不乾淨非球員

依序驗證了每個「用現有工具/規則過濾非球員」的方法，全部失敗：

| 方法 | 能否乾淨分 | 證據 |
|------|-----------|------|
| bbox 形狀（`_filter_player_like_bboxes`） | ❌ | 裁判是站立人形，跟球員一樣 |
| 球場區域（離邊線） | ❌ | 底角球員也靠邊，離邊線<6ft 誤傷 15% 球員 |
| 分隊摇擺度 | ❌ 部分 | 裁判摇擺（median 0.113 vs 球員 0.051），但被遮擋球員也摇擺 |
| **TeamSiamese embedding 距離** | ❌ | 三類分布**單峰連續無分界線**（見下） |

**embedding 驗證數據**（11 支重跑，`verify_referee_embedding.py` 產出）：
- 球員 dist_nearest：median 6.3，max **18.6**
- 裁判：median 11.5，min **3.9**（跟球員嚴重重疊）
- 籃框/球/計時器：median 15.5，全部 ≥11.6（非人物體反而分得最開）
- 最佳門檻（thr=11）也只有「抓 68% 非球員、誤傷 17% 球員」→ 不可用

**結論**：裁判 vs 球員的區別是**語義層面**（「這是不是在打球的球員」），而現有 pipeline
沒有任何模組是為這個語義訓練的。TeamSiamese 只學「A、B 是否同隊」，從沒學過
「這是不是球員」。所有間接信號（形狀/位置/顏色距離）都不足以取代語義判斷。

## 決策：走 SAM3（Segment Anything with Concepts）

SAM3（Meta 2026-03，arXiv 2511.16719）的 **Promptable Concept Segmentation** 用文字
概念（`"basketball player"`）偵測+追蹤所有符合概念的實例。這是**唯一**能從語義層排除
非球員的工具——籃框/球/計時器/裁判都不是 "basketball player"，理論上不會被追進來。

### SAM3 關鍵事實（讀論文確認）

- ~850M 參數（vision 450M + text 300M + detector/tracker 100M）
- 保留 SAM2 的 point/box prompt（PVS），VOS 分數小勝 SAM2.1
- 支援 image exemplar（**正/負範例**）+ 交互式 refine 排除歧義
- **速度**：video latency 隨物體數線性增長。實時（30 FPS）需求：
  10 物體 → **2 張 H200**；28 物體 → 4 張；64 物體 → 8 張
- 支援 ≤30 秒短影片的 concept segmentation

### 硬體前提（已確認）

- SAM3 在被 vLLM 擠壓的單張 GB10 上**不可行**（更大、更慢、可能 OOM）——
  當前 GB10 連 SAM2 都因 vLLM 佔 84GB 而每幀 4.5 秒。
- **使用者已確認可申請/換更好的 GPU**（A100/H200 級）→ SAM3 成為可行的長期方向。
- 硬體到位是阶段 2/3 的前置條件。

## 分階段路線

### ~~阶段 1：修 `_filter_player_like_bboxes` 清非人~~（已否決）

原本想「先清籃框/球/計時器止血」，但**若最終換 SAM3，整個 YOLO-init + 形狀過濾邏輯
都會被刪除**，修它是投資在即將消失的架構上 → **不做**。若時間差期間需要乾淨結果，
用「下游按已標 id 手動排除」止血（SAM3 到位後直接丟掉），不改感知層。

### 阶段 2：SAM3 概念驗證 ✅ 已通過（2026-09-06）

**目的**：驗證「SAM3 的 `"basketball player"` concept 能在**我方影片**上乾淨只框球員」。

**結果（v130，污染最嚴重那支，60 幀）**：
- SAM3 只框出 **10 個 obj = 正好 10 個真球員**（SAM2 舊版追 14 個，混 4 個非球員）
- `out_probs` 全部 0.80~0.96（高信心是 basketball player）
- 人工看標框影片：**非常乾淨**，裁判/籃框/球/計時器都被排除
- **結論：concept prompt 從語義層排除非球員的假設成立 → 進阶段 3**

**驗證工具**：`nba_data_pipeline/sam3_concept_check.py`（ffmpeg 抽 jpg → SAM3 concept
propagate → 標框影片 + 每幀 obj 數統計）。

**SAM3 video API（實跑確認）**：
```python
from sam3.model_builder import build_sam3_video_predictor
predictor = build_sam3_video_predictor(gpus_to_use=[0])
r = predictor.handle_request(dict(type="start_session", resource_path=jpg_dir_or_mp4))
sid = r["session_id"]
predictor.handle_request(dict(type="add_prompt", session_id=sid,
                              frame_index=0, text="basketball player"))
for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
    outputs_per_frame[r["frame_index"]] = r["outputs"]
```
每幀 `outputs` dict 結構（鍵名）：
- `out_obj_ids`：array[int]，追蹤物體 id
- `out_probs`：array[float]，是該 concept 的信心
- `out_boxes_xywh`：歸一化 xywh 框
- `out_binary_masks`：每物體 binary mask
- `frame_stats`

負範例 refine（若 concept 漏抓/誤抓）：`add_prompt` 帶 `points` + `point_labels`
（1=正/前景, 0=負/去除）+ `obj_id`。

### 阶段 2 環境搭建（踩過的坑，重跑必看）

**根因坑：GB10 太新（compute capability sm_121, ARM64 Grace Blackwell），舊 CUDA
編不出它的 kernel。**

- ❌ 在舊 container `nvcr.io/nvidia/pytorch:25.03-py3`（CUDA 12.8）裝 sam3：
  `nvrtc: error: invalid value for --gpu-architecture` — nvrtc 太舊不認 sm_121。
  手動裝 torch 2.10 也沒用（底層 CUDA/nvrtc 還是 image 的 12.8）。
- ❌ `TORCH_CUDA_ARCH_LIST="12.0"` 強制 sm_120 也沒解（nvrtc 本身太舊）。
- ✅ **改用 `nvcr.io/nvidia/pytorch:25.12-py3`（CUDA 13.1 + torch 2.10 nv 版）**：
  `torch.prod` 測試通過（能編 GB10 kernel）。DGX 上這 image 本地已有。
- **開 container**（獨立於 sam2 的 basketball_analysis container）：
  ```bash
  docker run -d --name sam3_env --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/vipl/timwu/basketball_analysis:/workspace -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 sleep infinity
  ```
- **裝依賴**：`pip install -e .`（在挂載的 /workspace/sam3）用 image 自帶 torch，
  不重裝 torch（會把 numpy 降到 1.26.4，無害）；另 `pip install
  opencv-python-headless` + `apt install ffmpeg`（image 沒帶）。
- **HF 認證**：image 的 `hf` CLI 壞了（typer 版本衝突），改用環境變數
  `export HF_TOKEN="hf_xxx"`（huggingface_hub 直接讀，繞過 CLI）。
- **影片讀取**：container cv2 沒 FFMPEG（同 CLAUDE.md 坑10），腳本改用 ffmpeg
  抽 jpg 資料夾餵 SAM3（SAM3 支援 jpg 資料夾或 mp4）。

### 阶段 3：重寫追蹤層整合 SAM3（阶段 2 過關才做）

把現有 `run_sam2_tracks`（YOLO init → box prompt → propagate → Pass2 scan → Pass3）
重寫成 SAM3 concept-based 流程。這是大工程，僅在阶段 2 驗證 SAM3 有效後才投入。

**影響範圍**（重寫時評估）：
- `inference.py` 的 `run_sam2_tracks` / `verify_sam2_with_yolo` / `_pick_init_frame`
  / `_filter_player_like_bboxes` 可能整組被 concept prompt 取代
- 下游 `court_coords` / `player_assignment` / handler 配對介面盡量保持不變，
  讓 `tactics_tools.py` 分析層不用改

## 附帶未解問題（獨立追蹤，不阻塞 SAM3）

1. **v125-130 座標系統性偏移**：擋拆接觸距離被 homography 量成 5-8ft（正常應 <3.5ft），
   `contact_thr=3.5` 系統性太嚴。這是 homography 校準問題，跟非球員污染獨立。
   SAM3 換上後仍需查（因為 court_coords 換算不受追蹤模型影響）。
2. **handoff 偵測**（已有 spec/plan：2026-09-03）：等 SAM3 穩定後再驗，因為它也吃
   被污染的 ball_acquisition / court_coords。

## 檔案

- 驗證腳本：`nba_data_pipeline/verify_referee_embedding.py`（重跑 SAM2 + 抽 embedding +
  標框影片 + CSV 報告）
- 裁判標注：`nba_data_pipeline/referee_labels.json`（舊 id）+ 本文件的重跑後 id 表
- SAM3 論文：`專題資料/2511.16719v2.pdf`

### 重跑後（對得上 verify CSV）的非球員標注

| 影片 | 裁判 | 籃框 | 球 | 計時器 |
|------|------|------|-----|--------|
| v116 | 1, 13, 11(裁判衣橘色部分) | | | |
| v117 | 6, 19 | | | |
| v118 | 6, 15 | 9 | | |
| v119 | 9, 10 | 15 | | |
| v121 | 7, 13, 15 | 14 | 11 | |
| v125 | 13 | | | |
| v127 | 7, 9, 15 | 12 | | 14 |
| v128 | 3, 5, 8 | 10 | | |
| v129 | 8, 15 | | | |
| v130 | 6, 11, 13 | 12 | | |
