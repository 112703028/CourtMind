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

**泛化（3 支，60 幀 each）**：
| 影片 | 結果 |
|------|------|
| v130 | 完美，只框 10 球員，非球員全排除 |
| v129 | 只漏 1 裁判（3 個中的 1 個，看不出差異）+ 抓到 1 替補（可接受，是球員）|
| v128 | 只漏 1 裁判（**在動的那個**，不動的沒抓）+ 抓到 1 替補（可接受）|

模式一致：SAM3 從 14 個混亂 obj（SAM2）→ 收斂到「10 球員 + 偶爾 1 裁判 + 偶爾 1 替補」。
漏網裁判疑似「跑動姿態更像球員」（v128 支持，v129 不明顯）→ 阶段3 用負範例 refine 補救。
替補是真球員，下游擋拆分析會自然忽略（不參與擋拆），可接受。

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

### 阶段 3：追蹤層整合 SAM3 ✅ 核心完成（2026-09-18）

實作 plan：`docs/superpowers/plans/2026-09-18-sam3-tracking-integration.md`（範圍A：只換追蹤）。
**兩階段解耦**（沒有「重寫」現有 SAM2 邏輯，而是加平行路徑）：

1. **`run_sam3_tracks.py`**（sam3_env）：SAM3 concept `"basketball player"` → 產
   `player_tracks`（格式同 SAM2 `list[dict[pid:{'bbox'}]]`）dump 成 pkl。
2. **inference.py 加 `--sam3_tracks`**（basketball_analysis）：給定則讀 pkl 當
   player_tracks、跳過 SAM2 + verify + predictor/player_model 載入；下游不變。

**驗證結果（v130，2026-09-18）**：
- SAM3 tracked 360/360 frames，**追到剛好 10 個純球員 [0-9]**（SAM2 版追 14 個含
  裁判 #6/#11/#13/#15 + 籃框）→ **非球員污染徹底解決** ✅
- 下游整套（court/分隊/handler 77.5%/擋拆 16 events）**無痛吃下 SAM3 tracks，不用改**。
- 兩 container 透過掛載的 /workspace 共用 pkl。

**後續工作（獨立 plan）**：
- 刪 SAM2 污染的清非球員補救邏輯（inference.py ~1001-1056）— 本 plan 只加平行路徑
  不動它，確認 SAM3 穩定後再開 plan 刪。
- verify 去留：SAM3 追蹤乾淨，暫不需要 verify_sam2_with_yolo（跑更多支確認）。
- 範圍B：SAM3 追球 + v1 持球判定取代 handler_detector。
- off-ball 掩護偵測（依賴乾淨追蹤，**現在解鎖了**）。

## 附帶發現：持球偵測抓不到傳球/接球（根因＝訓練資料標註偏差）

**現象**：v117 射手 #9 繞出來接球持球「不到 2 秒」（~30-60 幀），但
handler_detector 只標到 3 幀 → 平滑後被當噪音清掉 → 擋拆分析看不到接球者。

**排查過程**：
1. 先改 `smooth_ball_acquisition`（min_total_frames 一刀切 → 連續段邏輯），
   傳球者能保留了，但射手 #9（3 幀）跟噪音 #8（3 幀）幀數一樣，平滑區分不了。
2. 加 `[RAW]` 診斷：raw 分布 `{6:160, 8:3, 9:3, 3:2, 10:1}` — #9 真的只有 3 幀。
3. **直接查訓練資料 COCO（`screen_data/screen_v2.coco`）**：
   - 每幀幾乎只有 1 個 ball_handler 標註（「多於1個bh的幀」幾乎全 0）
   - ball_handler 中心 x 座標**逐幀平滑連續**（一路跟同一個人），傳球時**沒有跳位**
   - → **標註原則是「框持球發起者滿全程」，傳球後沒改框接球者**

**根因**：handler_detector 學到的是「這球的主角/發起者是誰」，**不是「此刻誰持球」**。
接球射手從沒被標成 ball_handler → 模型認不出 → 傳球/接球抓不到。這不是平滑、
不是 conf、不是 IOU 配對問題，是訓練資料的標註定義。

**解法（已定：先 B 再 A）**：
- **方向 B（先試）**：handler_detector 繼續用（抓主要持球者，94% 準），**另用 SAM3
  追球 "basketball" → 球在誰手上 → 補傳球/接球換手偵測**。不用重標，跟 SAM3 整合合流。
  兩訊號各司其職。**前置未驗證**：SAM3 追球在我方影片準不準、能否穩定判「球在誰手上」
  → 需比照球員驗證先小成本試（可加進 sam3_concept_check.py，prompt 換 "basketball"）。
- **方向 A（B 沒用才做）**：重標 153 支影片的 ball_handler 為「每幀此刻持球者」+ 重訓。

**已改但待驗**：`smooth_ball_acquisition` 已改連續段邏輯（`min_run`/`fill_gap`），
本身是對的改善（傳球者保留、噪音/漏幀處理更好），單元測試通過。與方向 B 不衝突，
可保留。

**這也直接關係 handoff 偵測**（2026-09-03 spec）：handoff = 換手，同樣需要能看到
「球從 A 到 B」。SAM3 追球若可行，handoff 也受益。

## 戰術判斷延伸：持球偵測 + off-ball 掩護（2026-09-18 調查）

目的釐清：最終要用「球員軌跡 + 擋拆訊號 + 持球資訊」判斷命名戰術（西班牙擋拆等），
持球和 off-ball 掩護都是**戰術判斷的輸入特徵**，不是最終目的。

### 持球偵測（SAM3 追球 → 誰持球）：v1 vs v2

用 SAM3 追球(100%) + 球員位置判持球。試了兩種後處理，5 支對照 GT（v126-130）：
- **v1（遲滯 overlap + confirm_frames）**：接球者命中率高（5 支 GT 持球者全抓到），
  但誤標多（貼近誤判、飛球掠過）。
- **v2（球速度事件：球高速飛=傳球 → 落回誰身上=接球）**：誤標少、乾淨，但會漏
  接球者（尤其快攻長傳 v129，球飛後接球者立刻運球推進，速度降不下來）。
- **相對速度**（位移/球框對角線）讓速度門檻在遠近鏡頭通用（spd≈3 球身/幀）。
- **結論**：對「戰術特徵」用途，**v1 較安全**（接球者不漏，誤標可在戰術層多數決濾）。
  持球「夠用就好」，不追求逐幀完美。快攻長傳當已知限制。
- 工具：`sam3_handler_check.py`（跑 SAM3 + 判持球，存 cache pkl）、
  `sam3_handler_sweep.py`（讀 cache 本機掃參數，不碰 GPU）。**cache 機制讓調參數
  秒級迭代**（同 CLAUDE.md dump-pkl 哲學）。

### off-ball 掩護偵測（西班牙擋拆的關鍵）

西班牙擋拆 = on-ball 擋拆 + **off-ball 背擋**（第三人擋 on-ball screener 的防守者）。
偵測西班牙擋拆 = 偵測那個 off-ball 背擋。

**關鍵發現（不用重標！）**：
1. 訓練資料把 on-ball / off-ball 掩護者**都標成 screener**（混標）→ ScreenNet 學過
   off-ball，會輸出它。
2. 但 `find_screen_instances` 的 **near-handler filter（screener 離 handler <20ft）
   把 off-ball 濾掉了**（背擋遠離持球者）。
3. 驗證：掃 12 支 events.pkl，**10 支都有「離 handler 遠(>20ft)+ 高信心(>0.8)」的
   screener**（共 45 筆）= off-ball 掩護被 ScreenNet 抓到、被 filter 濾掉的證據。
4. GT 對照：v129（on-ball #2 / off-ball #7）、v117（off-ball #9）— #7/#9 都在
   ScreenNet 輸出裡。

**區分 on-ball / off-ball / double screen 的原則**：用 `screened_defender` 算「擋誰的
防守者」——擋 handler 的防守者=on-ball；擋別人的=off-ball；兩個都擋 handler 防守者
=double。距離（離 handler 遠近）是輔助訊號。

**但目前算不準的根源（已查明）**：`screened_defender` 用「整段單幀最緊接觸」，被
**追蹤漏幀**污染。實例：v117 #9 實際貼身擋 #13（4.6ft），但 #13 在 frames 203-204
從 court_coords 消失（SAM2 漏幀）→ screened_defender 取到持續存在但較遠的 #7（19ft）
→ off-ball 分類看錯防守者。**不是 off-ball 本質難分，是追蹤漏幀 + screened_defender
不 robust。**

**決策：off-ball 偵測等 SAM3 整合完（追蹤乾淨、漏幀少）再做（治本）**。SAM3 追蹤上線
→ screened_defender 資料完整 → off-ball 自然變準。off-ball 精度綁在追蹤品質上。

### 複雜戰術判斷還缺什麼（論文調查）

LLM 用「規則算好的原子事實 + 戰術知識庫」判斷命名戰術（已定架構）。原子事實盤點：
- ✅ 已有：on-ball 擋拆、roll/pop、coverage、多screener分組、持球者(v1)
- ❌ 缺：**off-ball 掩護偵測**（西班牙擋拆必需，等 SAM3）、傳球/handoff 明確事實
  （v1 有基礎）、spacing/站位（次要，先看 LLM 夠不夠）
- 時序鏈（先擋→再背擋→pop）**不用規則算，交給 LLM 用知識庫組合**
- 參考：HoopTransformer（需大資料，不適用）、SGA-INTERACT（3D skeleton+動捕，沒有）
  → 有限資料下走「規則原子事實 + LLM 組合」是對的。

### 篩「場上真 5v5」：排裁判/替補（2026-09-18 調查）

**問題**：SAM3 `"basketball player"` 大部分乾淨（v130 剛好 10 人），但偶爾漏 1 個裁判
（v129 混入裁判#6 + 替補#11 → 進攻方變 7 人 → 防守配對亂）。戰術判斷/off-ball 需要
乾淨的場上 5v5。

**防守配對驗證（off-ball 的地基）**：對每個防守者算「整段最近進攻者多數決 = 守誰」。
- v130（乾淨 5v5）：**完美一對一，一致度 57-75%** ✅ 邏輯成立
- v129（混裁判/替補）：亂（多人守一人、有人沒守）→ **不是配對邏輯錯，是混入非球員**
- 結論：防守配對邏輯對，但依賴「乾淨 5v5」前提。

**排裁判試過的方法**：
| 方法 | 結果 |
|------|------|
| TeamSiamese embedding 距離（outlier） | ❌ 球員#9 比裁判#6 更 outlier，無乾淨分界 |
| sports.TeamClassifier（SigLIP，roboflow 用的） | ❌ 裁判一致度 100%（最穩），完全分不出 |
| 行為特徵（孤立度/最近鄰距） | ⚠️ 裁判#6 最孤立(15.2ft) 但無乾淨斷層，球員也有孤立的 |
| **SAM3 `"referee"` concept** | ✅ **8 支驗證 7 乾淨**（抓全部裁判+教練不誤傷），只 v128 誤標 3 球員（疑直條紋球衣像裁判服）|

**替補**：#11 只 40 幀 → **已被上游 deflicker/merge 濾掉**，不用特別處理。

**論文主流（查證）**：DeepPlayer-Track 等把裁判當「偵測 class」訓練（不是事後過濾），
呼應 SAM3 concept 的語義做法。外觀/球衣顏色分裁判在他們有效是因裁判服鮮明區別，
我方裁判服不夠區別（故 embedding/SigLIP 失敗）。

**決策**：用 **SAM3 `"referee"` concept 抓裁判 → 對照排除球員裡的裁判**（全自動、符合
論文主流）。v128 直條紋誤標是低機率邊界 case，**先接受不特別處理**（需要時那支手動）。
未整合進 run_sam3_tracks（第二個 concept + 對照排除），列後續工作。若之後 v128 類誤標
變多，再加「行為交叉驗證」（referee 候選須無防守配對/不參與擋拆才排除）。

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
