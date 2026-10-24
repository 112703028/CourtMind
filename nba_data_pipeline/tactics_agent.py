"""
tactics_agent.py — Level 2 Agentic：讓 LLM 當分析師，自己呼叫工具調查一次擋拆。

架構（ReAct）:
  LLM 想(Thought) → 呼叫工具(Action) → 看結果(Observation) → 再想 … → 下結論

  - 工具 = 包一層 tactics_tools 的幾何函式（回精確數字，LLM 不用自己算）
  - LLM 決定看誰、看哪幾幀（推理）；工具回真實數字（可靠）

用法:
  export OPENAI_API_KEY=sk-...   （或放 .env）
  python nba_data_pipeline/tactics_agent.py xxx_events.pkl \
      --handler 6 --screener 8 --frame 74
"""

import os
import json
import pickle
import argparse

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

from tactics_tools import (deflicker_court_coords, merge_swapped_ids,
                           infer_target_basket, nearest_defender, get_pos, _dist)


# ── 工具實作（每個吃 ctx + 參數，回精簡結果）──────────────────────────────────

def t_list_players(ctx):
    """列出所有球員 + 進攻/防守（用整段多數決判隊）。"""
    from collections import Counter
    cc, pa = ctx['cc'], ctx['pa']
    teams = {}

    for f in range(len(cc)):                             # 掃每一幀
        for pid in cc[f]:                                # 那幀的每個球員
            t = pa[f].get(pid) if f < len(pa) else None  # 他的隊伍
            if t:
                teams.setdefault(pid, Counter())[t] += 1

    out = {}
    for pid, c in teams.items():
        maj = c.most_common(1)[0][0]
        out[str(pid)] = "offense" if maj == ctx['handler_team'] else "defense"

    return out


def t_get_players_at(ctx, frame):
    """某幀所有球員的位置 + 隊伍 + 離籃框距離。"""
    cc, pa, basket = ctx['cc'], ctx['pa'], ctx['basket']

    if not (0 <= frame < len(cc)):
        return {"error": f"frame {frame} 超出範圍 (0~{len(cc) - 1})"}

    out = {}
    for pid, (x, y) in cc[frame].items():
        out[str(pid)] = {
            "x": round(x, 1), "y": round(y, 1),
            "team": "offense" if pa[frame].get(pid) == ctx['handler_team'] else "defense",
            "dist_to_basket": round(_dist((x, y), basket), 1),
        }

    return out


def t_get_trajectory(ctx, pid, start, end):
    """某球員一段時間的移動路徑（最多取 ~10 個取樣點）+ 離籃框距離。"""
    cc, basket = ctx['cc'], ctx['basket']
    n = len(cc)
    start, end = max(0, start), min(n - 1, end)

    if end <= start:
        return {"error": "start 要小於 end"}

    step = max(1, (end - start) // 10)

    path = []
    for f in range(start, end + 1, step):
        p = get_pos(cc, pid, f, max_search=2)
        if p:
            path.append({"frame": f, "x": round(p[0], 1), "y": round(p[1], 1),
                         "dist_to_basket": round(_dist(p, basket), 1)})
    if not path:
        return {"error": f"#{pid} 這段時間都沒座標"}
    
    return {"pid": pid, "path": path}


def t_distance(ctx, pid_a, pid_b, frame):
    """兩球員某幀的距離(ft)。"""
    cc = ctx['cc']
    pa_, pb_ = get_pos(cc, pid_a, frame, 2), get_pos(cc, pid_b, frame, 2)
    if not pa_ or not pb_:
        return {"error": "某球員該幀附近無座標"}
    return {"dist_ft": round(_dist(pa_, pb_), 1)}


def t_nearest_defender(ctx, pid, frame):
    """某球員在某幀最近的防守者 + 距離。"""
    d, dist = nearest_defender(ctx['cc'], ctx['pa'], pid, frame, ctx['handler_team'])
    return {"defender": d, "dist_ft": round(dist, 1) if dist is not None else None}


def t_basket_info(ctx):
    """進攻籃框位置 + 座標系說明。"""
    bx, by = ctx['basket']
    return {"basket_xy": [round(bx, 1), round(by, 1)],
            "note": "座標 x:0~91.8(球場長), y:0~49.2(球場寬)。離籃框距離越小越靠近籃下。"}


# ── 精確判斷工具（包一層規則，精確處查這些，不要自己猜）──────────────────────

def t_list_screens(ctx):
    """列出這球所有偵測到的擋拆 + 規則算好的事實（掩護者/擋誰/roll-pop/coverage）。"""
    from tactics_tools import (find_screen_instances, group_screen_events,
                               analyze_screen_instance)
    inst = find_screen_instances(ctx['cc'], ctx['pa'], ctx['events'])
    groups = group_screen_events(inst)
    data_like = {'court_coords': ctx['cc'], 'player_assignment': ctx['pa'], 'fps': ctx['fps']}
    out = []
    for i, g in enumerate(groups, 1):
        screens = []
        for it in g:
            fa = analyze_screen_instance(it, data_like, ctx['handler_team'], ctx['basket'])
            screens.append({"screener": fa["screener_id"], "defender": fa["screened_defender"],
                            "action": fa["action"], "coverage": fa["coverage"],
                            "coverage_confidence": fa["coverage_confidence"],
                            "region": fa["region"], "frame": fa["screen_frame"]})
        # 結構標籤：讓 LLM 直接有正確的戰術詞，不用自己編
        frames = [it['frame'] for it in g]
        if len(g) == 1:
            ptype = "single (單擋 pick-and-roll)"
        elif max(frames) - min(frames) <= 10:   # 接觸幀很近 = 同時
            ptype = "double (雙擋，兩人幾乎同時擋)"
        else:
            ptype = "staggered (連續擋，兩人先後接力擋)"
        out.append({"event": i,
                    "time_s": round(min(frames) / ctx['fps'], 1),
                    "handler": g[0]['handler_id'],   # 持球者，避免 LLM 亂猜
                    "play_type": ptype, "n_screens": len(g), "screens": screens})
    return out


def t_classify_roll_pop(ctx, screener, screen_frame):
    """規則精確判 roll/pop（用長窗，比自己看軌跡準）。"""
    from tactics_tools import classify_roll_pop
    action, delta = classify_roll_pop(ctx['cc'], screener, screen_frame,
                                      ctx['basket'], ctx['fps'])
    return {"action": action, "delta_ft": round(delta, 1),
            "note": "delta<0 往籃框=roll, >0 往外=pop"}


def t_classify_coverage(ctx, handler, screener, screen_frame):
    """模型判防守方式（switch/blitz/drop[含延誤]/unclear）。"""
    from tactics_tools import classify_coverage_ml
    cov = classify_coverage_ml(ctx['cc'], ctx['pa'], handler, screener, screen_frame,
                               ctx['fps'], ctx['handler_team'], basket=ctx['basket'])
    return {"coverage": cov['coverage'], "confidence": cov['confidence'],
            "detail": cov['detail']}


def t_screened_defender(ctx, screener, frame):
    """規則精確判掩護者擋的是哪個防守者（最緊接觸）。"""
    from tactics_tools import screened_defender
    fps = ctx['fps']
    s = max(0, frame - int(fps))
    e = min(len(ctx['cc']) - 1, frame + int(fps))
    d, dist = screened_defender(ctx['cc'], ctx['pa'], screener, ctx['handler'],
                                s, e, ctx['handler_team'])
    return {"defender": d, "contact_ft": round(dist, 1) if dist is not None else None}


DISPATCH = {
    "list_players": t_list_players,
    "get_players_at": t_get_players_at,
    "get_trajectory": t_get_trajectory,
    "distance": t_distance,
    "nearest_defender": t_nearest_defender,
    "basket_info": t_basket_info,
    # 精確判斷工具
    "list_screens": t_list_screens,
    "classify_roll_pop": t_classify_roll_pop,
    "classify_coverage": t_classify_coverage,
    "screened_defender": t_screened_defender,
}


# ── 工具 schema（給 LLM 看的說明）────────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "list_players",
        "description": "列出場上所有球員 id 和他們是進攻(offense)還是防守(defense)。先用這個知道有誰。",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_players_at",
        "description": "取得某一幀所有球員的位置(x,y)、隊伍、離籃框距離。用來看擋拆當下的佈局。",
        "parameters": {"type": "object", "properties": {
            "frame": {"type": "integer", "description": "幀編號"}}, "required": ["frame"]},
    }},
    {"type": "function", "function": {
        "name": "get_trajectory",
        "description": "取得某球員從 start 到 end 幀的移動路徑和離籃框距離。用來判斷掩護後往籃框(roll)還是往外(pop)。",
        "parameters": {"type": "object", "properties": {
            "pid": {"type": "integer"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
            "required": ["pid", "start", "end"]},
    }},
    {"type": "function", "function": {
        "name": "distance",
        "description": "兩個球員在某一幀的距離(英尺)。用來看誰貼著誰、掩護者有沒有撞到防守者。",
        "parameters": {"type": "object", "properties": {
            "pid_a": {"type": "integer"}, "pid_b": {"type": "integer"}, "frame": {"type": "integer"}},
            "required": ["pid_a", "pid_b", "frame"]},
    }},
    {"type": "function", "function": {
        "name": "nearest_defender",
        "description": "某球員在某一幀最近的防守者是誰、距離多少。用來找持球者/掩護者的防守者、判斷防守應對。",
        "parameters": {"type": "object", "properties": {
            "pid": {"type": "integer"}, "frame": {"type": "integer"}}, "required": ["pid", "frame"]},
    }},
    {"type": "function", "function": {
        "name": "basket_info",
        "description": "進攻籃框在哪、座標系怎麼看。判斷方向前先確認基準。",
        "parameters": {"type": "object", "properties": {}},
    }},
    # ── 精確判斷工具（精確的戰術判斷用這些，不要自己看軌跡猜）──
    {"type": "function", "function": {
        "name": "list_screens",
        "description": "列出這球所有偵測到的擋拆事件，每個含掩護者、擋的防守者、roll/pop、防守方式(coverage)、位置。開場先用這個掌握全局。",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "classify_roll_pop",
        "description": "用規則精確判斷掩護者擋完是 roll(順下往籃框) 還是 pop(外拉)。判 roll/pop 一定用這個，不要自己看軌跡猜(你猜會錯)。",
        "parameters": {"type": "object", "properties": {
            "screener": {"type": "integer"}, "screen_frame": {"type": "integer"}},
            "required": ["screener", "screen_frame"]},
    }},
    {"type": "function", "function": {
        "name": "classify_coverage",
        "description": "用模型判斷防守方對擋拆的應對：switch換防 / blitz包夾 / drop退防（含延誤hedge，兩者分不出）/ unclear不確定。判 coverage 用這個。",
        "parameters": {"type": "object", "properties": {
            "handler": {"type": "integer"}, "screener": {"type": "integer"},
            "screen_frame": {"type": "integer"}},
            "required": ["handler", "screener", "screen_frame"]},
    }},
    {"type": "function", "function": {
        "name": "screened_defender",
        "description": "用規則精確判斷掩護者擋的是哪個防守者(用最緊接觸)。判擋誰用這個，比看誰在附近準。",
        "parameters": {"type": "object", "properties": {
            "screener": {"type": "integer"}, "frame": {"type": "integer"}},
            "required": ["screener", "frame"]},
    }},
]


# ── Agent 迴圈（ReAct）─────────────────────────────────────────────────────────

SYSTEM = """你是專業籃球轉播主播兼戰術分析師。你要針對一球的進攻，做出生動、
詳細、有洞察的戰術解說，就像現場轉播一樣。

你有兩種工具：
【精確判斷工具】list_screens, classify_roll_pop, classify_coverage, screened_defender
  → 凡是精確的戰術判斷(roll還是pop、防守是switch/hedge/drop、擋的是誰)，
     一定呼叫這些工具查證，絕對不要自己看座標猜(你猜會錯)。
【調查工具】get_players_at, get_trajectory, nearest_defender, distance, list_players, basket_info
  → 用來理解「流程和脈絡」：這球怎麼發起、球員怎麼跑位、擋拆前後發生什麼、
     空間怎麼變化。這部分你自己判讀。

流程建議：
  1. 先 list_screens 掌握這球有幾次擋、大致結構。
  2. 針對關鍵擋拆，用精確工具確認 roll/pop、防守方式、擋誰。
  3. 用調查工具看流程脈絡(誰啟動、擋完的空間、後續)。
  4. 【最重要】先綜合所有擋拆，判斷「這是什麼擋拆戰術」，用戰術名稱開頭，
     再展開解說。不要只是一次一次機械地念每個擋。

命名戰術（步驟4的核心）：
  把偵測到的擋拆 pattern 綜合成一個戰術概念，例如：
    - 單擋 pick-and-roll：一個掩護者為持球者擋一次
    - 雙擋 double screen：兩個掩護者同時為持球者擋（同一時間點）
    - 連續擋拆 staggered / 拖車擋拆 double drag：兩個掩護者先後接力擋
    - 西班牙擋拆 Spain PnR：擋完又有人幫掩護者擋（screen the screener）
    - 手遞手 DHO 等
  用「這是一次◯◯戰術」開場，說明持球者吃了誰的掩護、目的是什麼，
  再依時間順序解說每個擋的 roll/pop 和防守應對。

防守方式（coverage）的說法：
  - switch = 換防；blitz = 包夾（兩人夾持球者）；drop = 退防或延誤（資料分不出這兩種，
    說「掩護者的防守者沒有一路跟出來包夾或換防」即可，不要斷言是哪一種）
  - unclear = 模型沒把握：就說防守應對不明確，不要自己從座標推測或硬給答案
  - confidence 是 medium 時用比較保留的語氣（「看起來是…」）

風格：專業、精準、有畫面感。生動但每句都要有事實根據，不要編造球員姿勢、
出手、得分或心理（你只看到擋拆，看不到出手結果）。
每次呼叫工具前先用一句話說明為什麼要查（讓推理可見）。"""


def run_agent(question, ctx, model="gpt-4o", max_steps=10, verbose=True):
    from openai import OpenAI
    client = OpenAI()

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        messages.append(msg)                       # ① 先記下 LLM 的回應（含 tool_calls）

        # LLM 說話了（Thought / 最終答案）
        if msg.content and verbose:
            print(f"\n💭 {msg.content}")

        if not msg.tool_calls:                     # 沒有要呼叫工具 = 最終答案
            return msg.content

        for tc in msg.tool_calls:                  # 執行每個工具呼叫
            args = json.loads(tc.function.arguments)   # ② arguments 是字串，要 parse
            if verbose:
                print(f"  🔧 {tc.function.name}({args})")
            try:
                result = DISPATCH[tc.function.name](ctx, **args)
            except Exception as ex:
                result = {"error": str(ex)}
            if verbose:
                print(f"     → {json.dumps(result, ensure_ascii=False)[:200]}")
            messages.append({                      # ③ 結果綁回 tool_call_id
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "（達到最大步數，未收斂）"


# ── Main ──────────────────────────────────────────────────────────────────────

def build_ctx(data, frame_hint=None, handler=None):
    """前置：de-flicker + ID-merge，算籃框，打包成工具用的 context。"""
    cc, _ = deflicker_court_coords(data['court_coords'])
    cc, pa, events, _ = merge_swapped_ids(cc, data['player_assignment'], data['events'])
    n = len(cc)
    # 籃框：用 frame 附近一段推（半場進攻的重心決定哪個籃框）
    if frame_hint is not None:
        s, e = max(0, frame_hint - 60), min(n - 1, frame_hint + 60)
    else:
        s, e = 0, n - 1
    basket = infer_target_basket(cc, s, e)
    return {'cc': cc, 'pa': pa, 'basket': basket, 'handler_team': 1,
            'fps': data.get('fps', 30.0), 'events': events, 'handler': handler}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl', help='inference dump 的 events.pkl')
    ap.add_argument('--handler', type=int, help='持球者 id')
    ap.add_argument('--screener', type=int, help='掩護者 id')
    ap.add_argument('--frame', type=int, help='擋拆大約發生的幀')
    ap.add_argument('--question', help='自訂問題（不給就用 handler/screener/frame 自動組）')
    ap.add_argument('--model', default='gpt-4o')
    ap.add_argument('--max_steps', type=int, default=10)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("錯誤：沒有 OPENAI_API_KEY（放環境變數或 .env）")
        return

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    ctx = build_ctx(data, frame_hint=args.frame, handler=args.handler)

    if args.question:
        question = args.question
    else:
        focus = ""
        if args.handler is not None:
            focus += f"持球者是 #{args.handler}。"
        if args.screener is not None:
            focus += f"特別留意 #{args.screener} 的掩護。"
        question = (f"請像轉播主播一樣，詳細解說這球的進攻戰術。{focus}"
                    f"先用 list_screens 掌握全局，再對關鍵擋拆用精確工具查證 roll/pop、"
                    f"防守方式、擋誰，並看流程脈絡，最後給一段生動的解說。")

    print(f"問題：{question}\n{'─' * 60}")
    answer = run_agent(question, ctx, model=args.model, max_steps=args.max_steps)
    print(f"\n{'─' * 60}\n📋 最終結論：\n{answer}")


if __name__ == '__main__':
    main()
