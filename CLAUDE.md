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

## TeamSiamese 分隊模型（ResNet18 pairwise）

分隊的**主要**模型，取代 sports.TeamClassifier（SigLIP）當首選，SigLIP 淪為 fallback。

- **架構**：ResNet18 backbone（ImageNet pretrained，共享權重）→ 兩個 crop 各 encode 成 512-d → concat → MLP head → 同隊機率
- **推論邏輯**：以每幀 handler 為 anchor，其他 player 跟 handler 做 pairwise 比較，`prob >= 0.5` → 同隊(offense)，否則 defense
- **相關檔案**：`nba_data_pipeline/team_classifier/`（extract_pairs.py / train.py / model.py / inference.py）
- **ckpt**：`models/team_siamese.pt`

### 訓練資料（重要坑：defender 沒標 → 模型塌掉）

- **踩到的大坑**：舊 COCO（`screen_v2.coco`）**108 號影片之後完全沒標 defender**（106-107 只標一半）。Siamese 只看過 positive pair（handler↔screener），從沒看過 negative（異隊） → 退化成「無論輸入什麼都輸出 prob=1.00」→ 新角度影片（114+）**整段抓不到 defense**。
- **解法**：
  1. **補標 defender** → 新資料集 `screen_data/screen_full_defender/`（108+ 全補齊，2087 圖 / 8395 標注）
  2. **extract_pairs.py 改「同圖任兩人全配對」**（不再只用 handler 當 anchor）：
     - offense↔offense（正）、defense↔defense（正）、offense↔defense（負）
     - crop 每個實例只存一次（省磁碟）
     - pairs 2910 → **10655**，negative 583 → **6245**（pos:neg 從 4:1 → 0.7:1）
  3. **train.py 改按 image_id 切 train/val**（原本隨機切 pair 會讓同圖 crop 洩漏到 val，虛高分數）

### 訓練結果

| 版本 | 資料 | Total pairs | pos:neg | val F1 | 症狀 |
|------|------|-------------|---------|--------|------|
| 舊 | screen_v2.coco | ~2910 | 4:1 | 虛高 | 114+ prob 全 1.00，抓不到 defense |
| **新** | screen_full_defender | **10655** | 0.7:1 | **0.932**（誠實）| screen_116 defense pool 恢復 5 人 |

- 驗證：screen_116 重訓後 `[SIAM]` prob 不再全 1.00（異隊明確 0.00、同隊 1.00），偵測到 7 個擋拆事件

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
│   ├── inference.py             # 完整 inference（Siamese 分隊, baseline pool）
│   ├── inference_defscore.py    # Siamese + YOLO defender_ratio 補救 pool（取法3）
│   ├── inference_siglip.py      # 強制走 SigLIP fallback（Siamese 塌時對照用）
│   ├── model.py                 # ScreenNetPairwise 定義
│   ├── coco_to_yolo.py          # COCO → YOLO 格式轉換
│   ├── test_ball_acquisition.py # 驗證 handler 偵測（舊 + YOLO 模式）
│   ├── PROGRESS.md              # 詳細進度筆記
│   ├── team_classifier/         # ⭐ TeamSiamese 分隊模型（ResNet18 pairwise）
│   │   ├── extract_pairs.py     #   COCO → pair crops（同圖全配對）
│   │   ├── train.py             #   訓練（image_id split）
│   │   ├── model.py             #   TeamSiamese 定義
│   │   └── inference.py         #   TeamAssignerSiamese（handler anchor）
│   ├── checkpoints/
│   │   ├── pretrain_best.pt
│   │   ├── finetune_weak_best.pt
│   │   ├── finetune_manual_best.pt       # v1
│   │   └── finetune_manual_v2_best.pt    # v2
│   └── output_manual/
│       ├── positive_sequences.npz
│       └── negative_sequences.npz
├── team_assigner/
│   └── team_assigner.py         # sports.TeamClassifier 無監督分隊（fallback）
├── screen_data/
│   ├── screen_v2.coco/          # 舊標注（108+ 缺 defender，已棄用）
│   └── screen_full_defender/    # ⭐ 補齊 defender 的新標注（重訓 Siamese 用）
├── team_classifier_data/        # extract_pairs 產出：crops/ + pairs.json
├── utils/
│   └── video_utils.py           # read/save with imageio fallback
├── trackers/                    # ByteTrack 版（inference.py 已不用）
├── ball_aquisition/             # 舊規則 detector（inference.py 已不用）
├── models/
│   ├── player_detector.pt           # YOLO 單 class 球員偵測
│   ├── handler_detector.pt          # multi-class YOLO ⭐ (4 class)
│   ├── team_siamese.pt              # ⭐ TeamSiamese 分隊 ckpt
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
14. **114+ 影片整段抓不到 defense** → COCO 108+ 沒標 defender，TeamSiamese 只學過 positive pair 塌成 prob 全 1.00 → 補標 `screen_full_defender` + extract_pairs 全配對 + 重訓
15. **Siamese train/val 隨機切 pair 洩漏** → 同圖 crop 同時進 train+val，val 分數虛高 → 改按 image_id 切
16. **VIPL Spark GPU OOM（SAM2 載不進）** → `nvidia-smi` 看到 vLLM 佔 84GB；GB10 統一記憶體共用，先確認 process 是誰的再決定 kill/等/換卡
17. **stub cache 陳舊** → 換模型/改分隊後必須 `rm stubs/player_assignment_stub.pkl`，否則讀到舊結果（權限不足時在 docker 內刪）

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

### 重訓 TeamSiamese 分隊模型

```bash
# 1. COCO → pair crops（同圖全配對）
python nba_data_pipeline/team_classifier/extract_pairs.py
# 2. 訓練（image_id split，存 models/team_siamese.pt）
python nba_data_pipeline/team_classifier/train.py
# ⚠ 換模型後記得刪 stub：rm stubs/player_assignment_stub.pkl
```

VIPL Spark（docker `basketball_analysis`，workdir `/workspace`）：
```bash
ssh vipl-spark "docker exec -w /workspace basketball_analysis bash -c '
  cp models/team_siamese.pt models/team_siamese_old.pt && \
  python nba_data_pipeline/team_classifier/extract_pairs.py && \
  python nba_data_pipeline/team_classifier/train.py'"
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
