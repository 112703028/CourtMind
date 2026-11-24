# 骨架版手遞手判斷 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用骨架特徵 + 小分類器判斷候選是不是手遞手，按片段交叉驗證比較現在上線的做法，回答「骨架有沒有額外資訊」。

**Architecture:** 新檔 `dho_skeleton.py`：純函式算骨架特徵（可單元測試）→ 候選 = 擋拆事件 ∪ 持球者換手候選 → 每個候選一列特徵 + 標記 → sklearn 按片段分組交叉驗證。不改任何正式流程；成功才另外接上線。

**Tech Stack:** Python、numpy、scikit-learn 1.9、pytest 9.1（本機 CPU）。

**Spec:** `docs/superpowers/specs/2026-11-24-skeleton-dho-design.md`

## Global Constraints

- 不改 ScreenNet、不改持球者、不需要 GPU、不需要新標記。
- 標記對應容許：同一對（A, B）、候選時間 − 標記交接張在 −0.6 ~ +2.1 秒內。
- 特徵視窗：候選時間前後 1 秒、10fps；距離用身高（畫面框高中位數）正規化。
- 門檻只用舊 476 支交叉驗證挑：準確率 ≥ 95% 下抓到最多。
- 新來源 27 個只在最後看一次，報告註明「之前看過結果，不完全乾淨」。
- 成功標準：舊資料全部特徵 F1 > 現在上線的做法且準確率 ≥ 95%；新來源抓到 ≥ 16/27、誤報 ≤ 1。

## Review Focus

- 骨架整段缺（被擋住 / 出畫面）→ 特徵是 NaN + `has_pose=0`，不能當成 0（0 會被當成「手很近」）。
- 給球者或接球者對不到 SAM3 編號（`map_ids` 沒有）→ 該候選骨架特徵全 NaN，不丟掉候選。
- 同一個手遞手同時是擋拆事件和換手候選 → 去重成一個候選（同一對、時間差 ≤ 0.5 秒），否則一個標記被算兩次。
- 球那幾張沒追到（touch 中心 None）→ 「兩人手在球上」「球移動距離」是 NaN，其他骨架特徵照算。
- 交叉驗證按片段分組：同一支片段的候選不能同時在訓練和驗證。

---

### Task 1: 骨架特徵（純函式）

**Files:**
- Create: `nba_data_pipeline/dho_skeleton.py`
- Test: `tests/test_dho_skeleton.py`

**Interfaces:**
- Produces: `skeleton_features(kp_a: list, kp_b: list, ball: list, H: float) -> dict`
  - `kp_a`, `kp_b`：前後 1 秒每張 10fps 的 (17, 3) 陣列（x, y, 信心）或 None；`ball`：每張球中心 (x, y) 或 None；COCO 關節：5/6 肩、7/8 肘、9/10 腕，信心 < 0.3 當沒有。
  - 回傳鍵：`both_hands`、`hand_to_hand`、`arm_angle`（度，0 = A 手臂指向 B）、`arm_ext`（肘→腕長度 / H）、`ball_travel`、`turn`（度）、`has_pose`（0/1）。

- [ ] **Step 1: 寫失敗的測試**

```python
import numpy as np, math
from dho_skeleton import skeleton_features

def kp(**pts):                       # 只放指定關節，其他信心 0
    a = np.zeros((17, 3)); 
    for j, (x, y) in pts.items(): a[int(j[1:])] = (x, y, 0.9)
    return a

def test_hand_to_hand_normalized():
    f = skeleton_features([kp(j9=(0, 0))], [kp(j10=(10, 0))], [None], 100)
    assert math.isclose(f['hand_to_hand'], 0.1, abs_tol=1e-6)

def test_arm_points_to_receiver():
    a = kp(j7=(0, 0), j9=(10, 0), j5=(0, -5), j6=(4, -5)); b = kp(j5=(50, 0), j6=(54, 0))
    assert skeleton_features([a], [b], [None], 100)['arm_angle'] < 5

def test_arm_points_away():
    a = kp(j7=(0, 0), j9=(-10, 0), j5=(0, -5), j6=(4, -5)); b = kp(j5=(50, 0), j6=(54, 0))
    assert skeleton_features([a], [b], [None], 100)['arm_angle'] > 170

def test_ball_travel_normalized():
    f = skeleton_features([None] * 3, [None] * 3, [(0, 0), (25, 0), (50, 0)], 100)
    assert math.isclose(f['ball_travel'], 0.5, abs_tol=1e-6)

def test_turn_degrees():
    a0 = kp(j5=(0, 0), j6=(10, 0)); a1 = kp(j5=(0, 0), j6=(0, 10))
    assert abs(skeleton_features([a0, a1], [None, None], [None, None], 100)['turn'] - 90) < 1

def test_missing_pose_is_nan_not_zero():
    f = skeleton_features([None] * 5, [None] * 5, [None] * 5, 100)
    assert f['has_pose'] == 0 and all(math.isnan(f[k]) for k in ('hand_to_hand', 'arm_angle', 'turn', 'both_hands'))
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `PYTHONPATH=nba_data_pipeline python -m pytest tests/test_dho_skeleton.py -v`
Expected: FAIL（`ModuleNotFoundError: dho_skeleton`）

- [ ] **Step 3: 實作 `skeleton_features` 於 `nba_data_pipeline/dho_skeleton.py`**

`both_hands` = 每張「兩人手腕離球比較遠的那個」/ H 的最小值（同 `dho.hands_feature`）；`hand_to_hand` = A 任一手腕到 B 任一手腕的最小距離 / H；
`arm_angle` = A 每隻手臂（肘→腕）跟 A→B（兩人肩中點連線）的夾角，取全部張的最小值；`arm_ext` = 對應那張肘→腕長度 / H；
`ball_travel` = 有球的張裡，第一顆到最後一顆球中心距離 / H；`turn` = A 第一張與最後一張有肩線的肩線方向夾角（0~180）；
任一特徵沒有可用資料 → NaN；`has_pose` = A 有任何一張有效關節。

- [ ] **Step 4: 跑測試確認通過**

Run: `PYTHONPATH=nba_data_pipeline python -m pytest tests/test_dho_skeleton.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add nba_data_pipeline/dho_skeleton.py tests/test_dho_skeleton.py
git commit -m "feat: 手遞手骨架特徵（純函式 + 單元測試）"
```

### Task 2: 候選 + 特徵列 + 標記

**Files:**
- Modify: `nba_data_pipeline/dho_skeleton.py`
- Test: `tests/test_dho_skeleton.py`

**Interfaces:**
- Consumes: `skeleton_features`（Task 1）；`dho.candidates(d)`、`dho.from_screens`、`dho_model.probe_scores(d)`、`dho_model.labels_old()`、`dho_model.labels_new()`、`shot_detect.map_ids`。
- Produces:
  - `merge_candidates(events: list[tuple], handoffs: list[tuple], fps: float) -> list[dict]`：events / handoffs 都是 `(giver, receiver, frame)`；同一對時間差 ≤ 0.5 秒併成一個，回 `{giver, receiver, frame, from_event, from_handoff}`。
  - `clip_rows(pk: Path, need: list[Path], labels: list, assume_neg: bool) -> list[dict]`：每個候選一列 = 骨架特徵 + 現有特徵（`approach_ft`、`v_giver`、`hold_b_s`、`probe`、`from_screens`、`from_event`）+ `label`（True/False/None）+ `clip`。
  - `all_rows(split: str) -> list[dict]`：`split` = `'old'`（476 支，`stubs/ball_batch` + `stubs/pose_hands`）或 `'new'`（29 段，`stubs/pipeline`）。

- [ ] **Step 1: 寫失敗的測試**

```python
from dho_skeleton import merge_candidates

def test_merge_same_pair_close_in_time():
    c = merge_candidates([(3, 6, 72)], [(3, 6, 62)], 30)
    assert len(c) == 1 and c[0]['from_event'] and c[0]['from_handoff']

def test_keep_different_pairs():
    assert len(merge_candidates([(3, 6, 72)], [(4, 6, 70)], 30)) == 2

def test_keep_same_pair_far_apart():
    assert len(merge_candidates([(3, 6, 72)], [(3, 6, 150)], 30)) == 2
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `PYTHONPATH=nba_data_pipeline python -m pytest tests/test_dho_skeleton.py -v -k merge`
Expected: FAIL（`ImportError: merge_candidates`）

- [ ] **Step 3: 實作 `merge_candidates`、`clip_rows`、`all_rows`**

骨架：候選時間前後 1 秒的 10fps 張（`shot_sam3['frame_idx']`），A / B 用 `map_ids` 反查 SAM3 編號（查不到 → 該人骨架 None）；H = SAM3 框高中位數。
標記對應用 Global Constraints 的容許；`assume_neg=True`（新來源）時沒對到的候選 = False，舊片段沒對到的 = False（設計：同片段其他候選當不是）。

- [ ] **Step 4: 跑測試確認通過 + 實跑一次**

Run: `PYTHONPATH=nba_data_pipeline python -m pytest tests/test_dho_skeleton.py -v`
Expected: 9 passed
Run: `python -c "import sys; sys.path.insert(0,'nba_data_pipeline'); import dho_skeleton as D; r=D.all_rows('old'); print(len(r), sum(x['label'] is True for x in r))"`
Expected: 候選數 > 500、`label=True` 約 18（容許少 1~2 個對不到的，印出來）

- [ ] **Step 5: Commit**

```bash
git add nba_data_pipeline/dho_skeleton.py tests/test_dho_skeleton.py
git commit -m "feat: 手遞手骨架候選 + 特徵列 + 標記"
```

### Task 3: 交叉驗證比較 + 新來源最後測試 + 紀錄

**Files:**
- Modify: `nba_data_pipeline/dho_skeleton.py`
- Modify: `docs/experiments/2026-10-23-screen-coverage-experiments.md`（新增一節）

**Interfaces:**
- Consumes: `all_rows`（Task 2）。
- Produces: CLI `python nba_data_pipeline/dho_skeleton.py eval`；`evaluate()` 印出四組比較 + 特徵重要性 + 新來源結果。

- [ ] **Step 1: 實作 `evaluate()`**

四組：(a) 現在上線的做法（`with_dho` 有沒有標 → 當預測，不訓練）、(b) 現有特徵、(c) 骨架特徵、(d) 全部；
分類器：`HistGradientBoostingClassifier(max_depth=3)`（吃 NaN）和羅吉斯迴歸（`SimpleImputer` 中位數 + 缺值旗標）各跑一次；
`GroupKFold(n_splits=5)` 按 `clip` 分組；門檻掃 0.05~0.95，取舊資料「準確率 ≥ 95% 下抓到最多」；
(d) 用全部舊資料重訓 → 套新來源，用舊資料挑的門檻；印特徵重要性（`permutation_importance`）。

- [ ] **Step 2: 實跑**

Run: `PYTHONIOENCODING=utf-8 python nba_data_pipeline/dho_skeleton.py eval`
Expected: 印出四組在舊資料的準確率 / 抓到 / F1、(d) vs (b) 差距、新來源抓到 x/27、誤報 y。

- [ ] **Step 3: 寫進實驗紀錄**

新增「### 41. 骨架版手遞手（方法 A）」：四組表格、骨架特徵重要性、新來源結果（註明不完全乾淨）、是否達成功標準、結論（上線 / 不上線）。

- [ ] **Step 4: Commit**

```bash
git add nba_data_pipeline/dho_skeleton.py docs/experiments/2026-10-23-screen-coverage-experiments.md
git commit -m "exp: 骨架版手遞手交叉驗證結果"
```

（上線接進 `dho.add` / `with_dho` 只在達成成功標準時另開任務。）
