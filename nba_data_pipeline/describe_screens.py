"""
describe_screens.py — Level 1：LLM 判斷戰術 + 生成解說。

流程：
  events.pkl → postprocess_events → analyze_event（幾何 facts）
            → 組 prompt（結構化事實 + 原始數字）
            → OpenAI 判斷戰術類型/防守應對 + 寫中文解說

設計原則：
  - 幾何工具（tactics_tools）先算好特徵，LLM 專心做「判斷 + 語言」
  - LLM 對規則標 low-confidence 的 coverage 有覆蓋權（給它原始數字自己判）
  - 沒有 OPENAI_API_KEY 時 graceful degrade：只印 template facts（等於 Level 0）

用法：
  export OPENAI_API_KEY=sk-...
  python nba_data_pipeline/describe_screens.py output_videos/xxx_events.pkl
  python nba_data_pipeline/describe_screens.py xxx_events.pkl --no_llm   # 只跑 template
"""

import os
import json
import pickle
import argparse

# 從 .env 讀 OPENAI_API_KEY（沒裝 python-dotenv 就跳過，仍可用系統環境變數）
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

from tactics_tools import postprocess_events, analyze_event, deflicker_court_coords


# ── 中文對照 ──────────────────────────────────────────────────────────────────
ACTION_ZH = {"roll": "順下(roll)", "pop": "外拉(pop)", "unclear": "不明顯"}


def fmt_time(fr, fps):
    return f"f{fr}({fr / fps:.1f}s)"


# ── Level 0：純模板（不需 LLM，也當 graceful degrade 的輸出）────────────────────

def render_template(facts, fps):
    """把 facts 填成一段確定性的事實描述。"""
    s, e = facts['frames']
    return (
        f"第 {fmt_time(s, fps)}~{fmt_time(e, fps)}："
        f"#{facts['handler_id']} 持球，於「{facts['region']}」利用 #{facts['screener_id']} 的掩護。"
        f"掩護後 #{facts['screener_id']} {ACTION_ZH.get(facts['action'])}"
        f"（離籃框變化 {facts['action_delta_ft']:+.1f}ft）。"
        f"防守推測：{facts['coverage']}（信心 {facts['coverage_confidence']}）。"
    )


# ── Level 1：組 prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是專業籃球戰術分析師。你會收到一次擋拆(pick-and-roll)的結構化幾何數據，
這些數據由電腦視覺系統從比賽影片自動抽取。請根據數據做戰術判斷。

注意：
- 幾何系統對「掩護後動作(roll/pop)」通常可靠。
- 對「防守應對(coverage)」的判斷有時信心不足(confidence=low)，此時請根據提供的原始數字
  （防守者移動、與持球者距離、與籃框距離）自行判斷，可以推翻系統的初步結論。
- 若數據不足以判斷，誠實說「無法確定」，不要編造。

只輸出 JSON，格式：
{
  "play_type": "戰術類型（如 高位擋拆/側翼擋拆/手遞手/西班牙擋拆/stack pick and roll/double screen/...）",
  "screener_action": "roll | pop | unclear",
  "defensive_coverage": "switch | hedge | drop | over | under | unclear",
  "confidence": "high | medium | low",
  "description": "一段 2-3 句的繁體中文戰術解說"
}"""


def build_user_prompt(facts, fps):
    s, e = facts['frames']
    return f"""擋拆事件數據：
- 時間：第 {s}~{e} 幀（{s/fps:.1f}s ~ {e/fps:.1f}s）
- 持球者 ID：{facts['handler_id']}
- 掩護者 ID：{facts['screener_id']}
- 偵測信心分數：{facts['score']:.2f}
- 擋拆接觸幀：{facts['screen_frame']}
- 擋拆發生位置：{facts['region']}
- 掩護後掩護者動作（幾何判斷）：{facts['action']}
- 掩護者離籃框距離變化：{facts['action_delta_ft']:+.1f} ft（負=靠近籃框）
- 防守應對（幾何初判）：{facts['coverage']}
- 防守判斷信心：{facts['coverage_confidence']}
- 防守判斷依據：{facts['coverage_detail']}

請判斷戰術並輸出 JSON。"""


# ── Level 1：呼叫 OpenAI ──────────────────────────────────────────────────────

def llm_classify(facts, fps, model="gpt-4o-mini"):
    """呼叫 OpenAI 判斷戰術。回 dict；失敗或無 key 回 None。"""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("  ⚠ 未安裝 openai 套件（pip install openai），跳過 LLM")
        return None

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(facts, fps)},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as ex:
        print(f"  ⚠ LLM 呼叫失敗：{ex}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl', help='inference dump 的 events.pkl')
    ap.add_argument('--merge_gap', type=int, default=15)
    ap.add_argument('--min_frames', type=int, default=5)
    ap.add_argument('--model', default='gpt-4o-mini', help='OpenAI model')
    ap.add_argument('--no_llm', action='store_true', help='只跑 template，不呼叫 LLM')
    args = ap.parse_args()

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    # 先修正 homography 左右鏡像閃爍（否則座標每隔幾幀翻一次，戰術判斷全錯）
    data['court_coords'], nflip = deflicker_court_coords(data['court_coords'])

    fps = data.get('fps', 30.0)
    events = postprocess_events(data['events'], args.merge_gap, args.min_frames)
    print(f"讀入 {len(data['events'])} 原始事件 → 後處理 {len(events)} 個 "
          f"(fps={fps:.1f}, de-flicker 修正 {nflip} 幀)\n")

    use_llm = not args.no_llm and bool(os.environ.get("OPENAI_API_KEY"))
    if not use_llm:
        print("（未用 LLM：無 OPENAI_API_KEY 或 --no_llm，只輸出 template）\n")

    for i, ev in enumerate(events, 1):
        facts = analyze_event(ev, data)
        print(f"══ 事件 #{i} ══════════════════════════════════════")
        print("【幾何事實】", render_template(facts, fps))

        if use_llm:
            result = llm_classify(facts, fps, args.model)
            if result:
                print(f"【LLM 判斷】")
                print(f"  戰術類型：{result.get('play_type')}")
                print(f"  掩護動作：{result.get('screener_action')}")
                print(f"  防守應對：{result.get('defensive_coverage')} "
                      f"(信心 {result.get('confidence')})")
                print(f"  解說：{result.get('description')}")
        print()


if __name__ == '__main__':
    main()
