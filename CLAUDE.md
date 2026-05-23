# 籃球擋拆偵測 (HoopScout / CourtMind) — 進度紀錄

## 專案目標

對籃球比賽影片自動偵測 **Pick-and-Roll（擋拆）** 事件：
- 找出 **handler**（持球者）
- 找出 **screener**（掩護者）
- 標註兩者的對應關係（誰幫誰擋）

---

## 目前 Pipeline（最新）

```
影片 input
  ↓
read_video（cv2 失敗自動 fallback imageio + pyav）
  ↓
YOLO 第一幀偵測 + SAM2 跨幀傳播  ←── 穩定 player_id（取代舊 ByteTrack）
  ↓
handler_detector.pt（multi-class YOLO）每幀偵測 ball_handler
  ↓ IOU 配對到 SAM2 player_id
  ↓
CourtKeypointDetector → Homography → 球員場上座標 (feet)
  ↓
TeamAssigner（sports.TeamClassifier，SigLIP + UMAP + K-means）
  ↓
ScreenNetPairwise（10 幀軌跡 → 預測 screen pair）
  ↓
影片輸出（HANDLER 綠框 + SCREENER 紅框）
```

---

## 訓練流程（已完成）

| 階段 | 資料 | 輸出 ckpt |
|------|------|-----------|
| **Step 1: Pretrain** | NBA tracking 公開資料 (`nba-movement-data`)，自監督軌跡預測 | `pretrain_best.pt` |
| **Step 2: Weak finetune** | `extract_sequences.py` 規則自動標的 weak label | `finetune_weak_best.pt` |
| **Step 3: Manual finetune** | 人工標 164 個 clip → `convert_manual.py` → npz | `finetune_manual_best.pt` (v1) / `finetune_manual_v2_best.pt` (v2) |

---

## ScreenNet 訓練結果

| 模型 | AUC | F1 @ 0.5 | Best F1 | Active pairs |
|------|-----|----------|---------|--------------|
| **v1** (single screener) | 0.9217 | **0.8889** | 0.9031 @ 0.70 | 1/20 |
| **v2** (multi screener) | **0.9633** | 0.5524 | 0.5852 @ 0.73 | 3/20 |

- **v1** 任務簡單（只判斷 slot 1）→ F1 高，但只能抓 1 個 screener
- **v2** 任務難（要判斷 slot 1~3）→ AUC 提升，能抓 double screen

---

## 訓練資料改進

### A. SAM2 取代 ByteTrack
- **問題**：ByteTrack 在球員被遮擋時 ID 跳動，screener 變成兩個不同 tid
- **解法**：用 SAM2 video predictor 做穩定追蹤
- **結果**：訓練資料 pos 大幅增加（2451 pos / 1212 neg）

### B. 多 screener 支援
- **發現**：26%（43/164）的 clip 同時有 2-3 個 screener（horns / staggered / double drag）
- **舊版**：只保留得票最多的 1 個 screener
- **新版（v2）**：依「最擁擠那幀」決定保留幾個 screener tid
- **結果**：`Active pairs=1/20 → 3/20`

---

## Inference Pipeline 改進（已完成）

### 1. PlayerTracker → SAM2
- 移掉 `PlayerTracker (ByteTrack)`
- 改用 `run_sam2_tracks()`：YOLO 第一幀 → SAM2 mask propagate → 穩定 id
- 跟 `convert_manual.py` 流程一致

### 2. BallAquisitionDetector → handler_detector.pt
- 移掉 `BallTracker + BallAquisitionDetector`（球距離猜法太不準）
- 改用 multi-class YOLO `handler_detector.pt`（class 0 = ball_handler）
- 每幀偵測 handler bbox → IOU 配對 SAM2 player_id → 穩定的 handler id
- **94% accuracy**（vs 舊方法的不穩定）

### 3. TeamAssigner: fashion-CLIP → sports.TeamClassifier
- 舊版：fashion-CLIP zero-shot 配 prompt（"white shirt" vs "red shirt"）
- 新版：sports.TeamClassifier（**SigLIP + UMAP + K-means k=2**）
  - 不需要 prompt（無監督）
  - 每 30 幀抓 crops（縮 40% 取球衣中央區域）
  - 一次性 fit 整段影片
  - 每幀 batch predict

### 4. video_utils.py 加 imageio fallback
- Container 內 OpenCV 沒 FFMPEG/GStreamer 支援，無法讀影片
- `read_video()` 先試 cv2，失敗自動 fallback `imageio + pyav`
- `save_video()` 同樣，失敗 fallback imageio（`.avi` 改存 `.mp4`）

---

## YOLO Handler Detector 訓練

### 觀察
Roboflow 公開 notebook 不追蹤球，改用 **multi-class 偵測**直接判斷「誰持球」。

### 訓練資料
- `coco_to_yolo.py` 把現有 COCO 標注（`ball_handler`、`screener`、`defender`、`others`）轉成 YOLO 格式
- 4 class 多類別偵測器
- Fine-tune from `player_detector.pt`

### 訓練結果
- mAP50（整體）：**0.885**
- Precision: 0.897
- Recall: 0.803

**Confusion matrix（normalized）**：

| Class | 正確率 |
|-------|--------|
| **ball_handler** | **0.94** ⭐ |
| screener | 0.90 |
| defender | 0.87 |
| others | 0.78 |

### 驗證
`test_ball_acquisition.py --yolo_model models/handler_detector.pt` 看影片：handler 偵測非常穩，綠框跟著實際運球者。

---

## 下一步

### 短期
- 跑改好的 inference（SAM2 + handler_detector + sports.TeamClassifier）在 `screen_13.mp4` / `screen_100.mp4` 驗證
- 看 handler、team、screener 預測是否都穩定
- 比較 v1 / v2 ckpt 在實際影片上的效果

### 中期
- handler_detector 偵測到的 screener / defender 也接進來當輔助訊號（多模型 fusion）
- 重新標多 screener 的 clip 補資料

### 長期
- Jersey number OCR（Roboflow notebook 有用）→ 認球員身份不只認隊伍
- 整合 fast break / shot quality 等其他偵測模組

---

## 檔案結構

```
basketball_analysis/
├── nba_data_pipeline/
│   ├── convert_manual.py        # COCO → 訓練 npz（v2 多 screener）
│   ├── finetune.py              # ScreenNet 訓練
│   ├── inference.py             # 完整 inference（SAM2 + handler_detector）
│   ├── model.py                 # ScreenNetPairwise 定義
│   ├── coco_to_yolo.py          # COCO → YOLO 格式轉換
│   ├── test_ball_acquisition.py # 驗證 handler 偵測（舊 + YOLO 模式）
│   ├── PROGRESS.md              # 詳細進度筆記
│   ├── checkpoints/
│   │   ├── pretrain_best.pt
│   │   ├── finetune_weak_best.pt
│   │   ├── finetune_manual_best.pt       # v1
│   │   └── finetune_manual_v2_best.pt    # v2
│   └── output_manual/
│       ├── positive_sequences.npz
│       └── negative_sequences.npz
├── team_assigner/
│   └── team_assigner.py         # 用 sports.TeamClassifier 無監督分隊
├── utils/
│   └── video_utils.py           # read/save with imageio fallback
├── trackers/                    # ByteTrack 版（inference.py 已不用）
├── ball_aquisition/             # 舊規則 detector（inference.py 已不用）
├── models/
│   ├── player_detector.pt           # YOLO 單 class 球員偵測
│   ├── handler_detector.pt          # multi-class YOLO ⭐ (4 class)
│   ├── court_keypoint_detector.pt
│   ├── ball_detector_model.pt       # 舊版用
│   └── sam2.1_hiera_large.pt
└── yolo_screen_dataset/             # YOLO 訓練資料
```

---

## 環境 / 套件需求

```bash
# Python 套件
pip install ultralytics supervision torch numpy opencv-python scikit-learn
pip install git+https://github.com/facebookresearch/sam2.git
pip install git+https://github.com/roboflow/sports.git@feat/basketball

# 影片解碼 fallback（OpenCV 沒 FFMPEG 時）
pip install 'imageio[ffmpeg]' av

# AV1 影片要先轉 H.264
apt install ffmpeg
```

### HuggingFace token（加速 SigLIP 下載）

```bash
# https://huggingface.co/settings/tokens 拿 Read token
export HF_TOKEN="hf_xxx"
echo 'export HF_TOKEN="hf_xxx"' >> ~/.bashrc
```

---

## SSL_CERT_FILE 問題（已遇過）

切換 conda env 後 transformers 可能找不到 SSL cert：

```bash
# 重設成當前 env 的 certifi 路徑
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
ls "$SSL_CERT_FILE"   # 確認存在

# 或清掉讓 httpx 用預設
unset SSL_CERT_FILE
```

---

## 主要踩過的坑

1. **ByteTrack ID 跳動** → 改 SAM2
2. **多 screener clip 沒被學到** → 修改投票 + slot 配置
3. **本地 GPU 不夠** → 跑 DGX Spark / AI Stack
4. **DGX 同事搶 GPU** → 切到 AI Stack
5. **YOLO 訓練 docker shm 太小** → `workers=0` 繞過
6. **YOLO dataset path 跨平台問題** → 用絕對路徑或 `path: .`
7. **GitHub push 被 100MB 上限擋** → `.gitignore` + `git rm --cached`
8. **`BallAquisitionDetector` 不準** → 改用 multi-class YOLO 直接偵測
9. **影片是 AV1 編碼** → ffmpeg 轉 H.264
10. **OpenCV docker 編譯時沒包 FFMPEG** → `video_utils.py` 加 imageio fallback
11. **fashion-CLIP team assigner 分不穩** → 改用 sports.TeamClassifier (SigLIP + UMAP + K-means)
12. **SigLIP HF 下載限速** → 設 `HF_TOKEN`
13. **DGX docker 沒掛 utils/trackers 等 root 模組** → 在本地跑 inference

---

## 常用指令

### 跑 inference（最新版）

```bash
python nba_data_pipeline/inference.py \
    --video         input_videos/screen/screen_13.mp4 \
    --ckpt          nba_data_pipeline/checkpoints/finetune_manual_best.pt \
    --handler_model models/handler_detector.pt \
    --output_video  output_videos/screens_pred.avi \
    --threshold     0.70 \
    --debug
```

### 重跑訓練資料生成

```bash
python nba_data_pipeline/convert_manual.py
```

### 重訓 ScreenNet

```bash
python nba_data_pipeline/finetune.py \
    --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
    --data_dir      nba_data_pipeline/output_manual \
    --out_dir       nba_data_pipeline/checkpoints \
    --tag           manual_v2 \
    --pairwise \
    --epochs        30 \
    --lr            3e-4 \
    --batch         16
```

### AV1 → H.264 影片轉檔

```bash
ffmpeg -y -err_detect ignore_err -fflags +discardcorrupt \
    -i input_videos/screen/screen_13.mp4 \
    -c:v libx264 -crf 23 -preset fast \
    input_videos/screen/screen_13_h264.mp4
```

### 從本地 scp 到 AI Stack

```bash
# 注意 port（每個 pod 不同）
scp -P 31042 ~/source/basketball_analysis/path/file \
    root@140.119.163.112:/mnt/data/basketball_analysis/path/
```
