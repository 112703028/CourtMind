# 常用指令速查

環境:
- 本地(Windows)：`c:\Users\timwu\source\basketball_analysis`
- VIPL Spark：`ssh vipl-spark`，docker container `basketball_analysis`，workdir `/workspace`
- Spark 主機路徑：`/home/vipl/timwu/basketball_analysis`

> ⚠ Windows 本地跑分析腳本要加 `python -X utf8`，否則中文輸出會亂碼(cp950)。

---

## 1. Inference（產生 pkl + 標框影片）— 在 VIPL Spark docker

擋拆偵測 → 會 dump `<output_video 去副檔名>_events.pkl`（給分析用）。

```bash
# ⚠ 換模型/改分隊後先刪 stub，否則讀到舊分隊結果
ssh vipl-spark "docker exec basketball_analysis rm -f /workspace/stubs/player_assignment_stub.pkl"

ssh vipl-spark "docker exec -w /workspace basketball_analysis bash -c '
python nba_data_pipeline/inference.py \
  --video         input_videos/screen/screen_117.mp4 \
  --ckpt          nba_data_pipeline/checkpoints/finetune_manual_best.pt \
  --handler_model models/handler_detector.pt \
  --output_video  output_videos/screens_v117_retrain.avi \
  --threshold     0.70 \
  --debug 2>&1 | tee output_videos/v117.log'"
# → 產生 output_videos/screens_v117_retrain_events.pkl
```

---

## 2. 把 pkl / 影片 / log 抓回本地

```bash
# 若 /workspace 沒掛到主機，先從容器 copy 到主機
ssh vipl-spark "docker cp basketball_analysis:/workspace/output_videos/screens_v117_retrain_events.pkl /home/vipl/timwu/basketball_analysis/output_videos/"
```
```powershell
# 主機 → 本地（在專案根目錄跑）
scp vipl-spark:/home/vipl/timwu/basketball_analysis/output_videos/screens_v117_retrain_events.pkl output_videos/
scp vipl-spark:/home/vipl/timwu/basketball_analysis/output_videos/screens_v117_retrain.mp4     output_videos/
scp vipl-spark:/home/vipl/timwu/basketball_analysis/output_videos/v117.log                     output_videos/
```

---

## 3. 把改過的檔案傳上 Spark

```powershell
scp nba_data_pipeline/inference.py       vipl-spark:/home/vipl/timwu/basketball_analysis/nba_data_pipeline/
scp nba_data_pipeline/tactics_tools.py   vipl-spark:/home/vipl/timwu/basketball_analysis/nba_data_pipeline/
scp utils/video_utils.py                 vipl-spark:/home/vipl/timwu/basketball_analysis/utils/
```

---

## 4. 分析事實（tactics_tools.py）— 本地，不用 GPU

讀 pkl → de-flicker + ID-merge → 接觸式偵測 → roll/pop、coverage、擋誰。

```bash
python -X utf8 nba_data_pipeline/tactics_tools.py output_videos/screens_v117_retrain_events.pkl

# 可調參數
python -X utf8 nba_data_pipeline/tactics_tools.py output_videos/screens_v117_retrain_events.pkl \
    --contact_thr 3.5 \   # screener 撞防守者 < 此距離才算真擋
    --time_gap    25      # 時間相近的擋湊成同一事件(雙擋)
```

---

## 5. Level 1 戰術解說（describe_screens.py）— 本地

事實 → LLM 生成解說。需要 `.env` 裡的 `OPENAI_API_KEY`。

```bash
python -X utf8 nba_data_pipeline/describe_screens.py output_videos/screens_v117_retrain_events.pkl

# 只跑 template 不呼叫 LLM（先驗證流程、不花錢）
python -X utf8 nba_data_pipeline/describe_screens.py output_videos/screens_v117_retrain_events.pkl --no_llm

# 換 model
python -X utf8 nba_data_pipeline/describe_screens.py xxx_events.pkl --model gpt-4o-mini
```

---

## 6. Level 2 Agent（tactics_agent.py）— 本地，選配

LLM 用工具自己調查一次擋拆(ReAct)。需要 `OPENAI_API_KEY`。

```bash
python -X utf8 nba_data_pipeline/tactics_agent.py \
    output_videos/screens_v117_retrain_events.pkl \
    --handler 6 --screener 8 --frame 74 \
    --model gpt-4o-mini          # 先用便宜的試

# 自訂問題
python -X utf8 nba_data_pipeline/tactics_agent.py xxx_events.pkl \
    --question "分析第 133 幀 #9 對 #7 的擋拆"
```

---

## 7. 重訓模型（參考，在 Spark docker 跑）

```bash
# TeamSiamese 分隊（換模型後記得刪 stub）
ssh vipl-spark "docker exec -w /workspace basketball_analysis bash -c '
  cp models/team_siamese.pt models/team_siamese_old.pt && \
  python nba_data_pipeline/team_classifier/extract_pairs.py && \
  python nba_data_pipeline/team_classifier/train.py'"

# ScreenNet
ssh vipl-spark "docker exec -w /workspace basketball_analysis bash -c '
  python nba_data_pipeline/finetune.py \
    --pretrain_ckpt nba_data_pipeline/checkpoints/finetune_weak_best.pt \
    --data_dir nba_data_pipeline/output_manual \
    --out_dir nba_data_pipeline/checkpoints \
    --tag manual_v2 --pairwise --epochs 30 --lr 3e-4 --batch 16'"
```

---

## 環境檢查（agent/解說跑不動時）

```bash
# openai 套件 + key 在不在（不印出 key）
python -c "import openai; print('openai', openai.__version__)"
python -c "from dotenv import load_dotenv,find_dotenv; load_dotenv(find_dotenv()); import os; print('key' if os.environ.get('OPENAI_API_KEY') else 'no key')"

# 換模型/改分隊後一定刪 stub
ssh vipl-spark "docker exec basketball_analysis rm -f /workspace/stubs/player_assignment_stub.pkl"
```

---

## 一條龍流程（從影片到解說）

```
1. (Spark) inference.py         → screens_vNNN_retrain_events.pkl + 標框影片
2. (本地)  scp 抓 pkl 回來
3. (本地)  tactics_tools.py      → 看事實對不對(roll/pop、擋誰)
4. (本地)  describe_screens.py   → LLM 生成戰術解說
5. (選配)  tactics_agent.py      → 疑難案例讓 agent 調查
```
