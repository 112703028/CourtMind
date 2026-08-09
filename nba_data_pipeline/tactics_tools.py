"""
tactics_tools.py — 擋拆事件的幾何分析工具（純計算，不碰 LLM）。

三個 Level 共用的地基：
  Level 1（特徵→分類）：這裡算特徵，丟一次 LLM 分類
  Level 2（agentic）  ：把這裡每個函式包成 tool 給 LLM function-calling 呼叫
  Level 3（VLM）      ：同上 + 影片幀

座標系（跟 inference.py 一致）：
  x: 0~91.8 ft（球場長軸），兩籃框在 x≈5 和 x≈87
  y: 0~49.2 ft（球場寬軸），中線 y≈24.6
  player_assignment: team 1 = 進攻（handler 同隊），team 2 = 防守

獨立測試：
  python nba_data_pipeline/tactics_tools.py output_videos/screens_v116_retrain_events.pkl
"""

import math
import pickle

# ── 場地常數 ──────────────────────────────────────────────────────────────────
COURT_W_FT = 91.8   # 長軸
COURT_H_FT = 49.2   # 寬軸
CENTER_Y   = COURT_H_FT / 2       # 24.6
BASKET_INSET = 5.0                # 籃框離底線約 5 ft
BASKET_LEFT  = (BASKET_INSET, CENTER_Y)            # (5, 24.6)
BASKET_RIGHT = (COURT_W_FT - BASKET_INSET, CENTER_Y)  # (86.8, 24.6)


# ── 基本幾何 helper ───────────────────────────────────────────────────────────

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_pos(court_coords, pid, frame, max_search=12):
    """
    取某 pid 在某幀的座標；該幀沒有就往前後找最近的有效幀（homography 可能漏）。
    回 (x, y) 或 None。
    """
    n = len(court_coords)
    if 0 <= frame < n and pid in court_coords[frame]:
        return court_coords[frame][pid]
    for d in range(1, max_search + 1):
        for f in (frame - d, frame + d):
            if 0 <= f < n and pid in court_coords[f]:
                return court_coords[f][pid]
    return None


def get_team(player_assignment, pid, frame, max_search=12):
    """取某 pid 在某幀的隊伍（1=進攻/2=防守），漏就找鄰近幀。"""
    n = len(player_assignment)
    if 0 <= frame < n and pid in player_assignment[frame]:
        return player_assignment[frame][pid]
    for d in range(1, max_search + 1):
        for f in (frame - d, frame + d):
            if 0 <= f < n and pid in player_assignment[f]:
                return player_assignment[f][pid]
    return None


# ── 0. 修正 homography 左右鏡像閃爍 ───────────────────────────────────────────

def deflicker_court_coords(court_coords, court_w=COURT_W_FT):
    """
    修正 court keypoint / homography 的左右鏡像閃爍（球場對稱 → 某些幀被投成鏡像）。

    做法：逐幀跟「前一個已接受幀」比對共同球員的 x。若把這幀 x 鏡像
    （x → court_w − x）後更貼近前一幀，就翻回。只要求「幀間一致」，不保證
    絕對方向——但一致就足夠（roll/pop 用距離、region 用 y，都不受一致的 x 翻影響）。

    鏡像位移 ~半場(45ft)，真實移動 <3ft/幀，訊號極明確不會誤翻。
    回 (修正後的 court_coords, 翻轉的幀數)。
    """
    out = []
    prev = None       # 前一個已接受幀 {pid:(x,y)}
    n_flipped = 0
    for frame in court_coords:
        if not frame:
            out.append(frame)
            continue
        if prev is None:                      # 第一個有效幀 → 當基準方向
            out.append(dict(frame))
            prev = dict(frame)
            continue

        common = set(frame) & set(prev)
        if len(common) >= 2:
            as_is   = sum(abs(frame[p][0] - prev[p][0]) for p in common)
            flipped = sum(abs((court_w - frame[p][0]) - prev[p][0]) for p in common)
            if flipped < as_is:               # 鏡像後更連續 → 翻回
                frame = {p: (court_w - x, y) for p, (x, y) in frame.items()}
                n_flipped += 1

        out.append(dict(frame))
        prev = dict(frame)
    return out, n_flipped


# ── 1. 推斷進攻籃框 ───────────────────────────────────────────────────────────

def infer_target_basket(court_coords, start, end):
    """
    半場進攻時，所有球員都在進攻的那半場 → 用所有球員重心離哪個籃框近判斷。
    回 (bx, by)。
    """
    xs = []
    for f in range(start, end + 1):
        if f < 0 or f >= len(court_coords):
            continue
        for pos in court_coords[f].values():
            xs.append(pos[0])
    if not xs:
        return BASKET_RIGHT   # 沒資料就給預設，不 crash
    avg_x = sum(xs) / len(xs)
    # 重心偏哪半場 → 進攻該半場的籃框
    return BASKET_LEFT if avg_x < COURT_W_FT / 2 else BASKET_RIGHT


# ── 2. 找擋拆接觸幀（handler 的防守者離 screener 最近的那幀）────────────────────

def find_screen_frame(court_coords, handler_id, screener_id, start, end,
                      player_assignment=None, handler_team=1, basket=None):
    """
    回擋拆接觸幀 = handler 的防守者離 screener 最近的那幀。
    （擋拆的定義：handler 的防守者撞上 screener 被擋到。）

    先做前場過濾排除中場過場的誤判，再在候選幀裡取「handler 防守者 ↔ screener」
    距離最小的。需要 player_assignment 才能認 handler 的防守者；沒有（或認不到）
    就 fallback 回「handler ↔ screener 最近」。
    """
    if basket is None:
        basket = infer_target_basket(court_coords, start, end)
    midcourt_x = COURT_W_FT / 2

    # 候選幀（handler/screener 都有座標）
    cands = []
    for f in range(start, end + 1):
        hp = get_pos(court_coords, handler_id, f, max_search=0)
        sp = get_pos(court_coords, screener_id, f, max_search=0)
        if hp is None or sp is None:
            continue
        cands.append((f, hp, sp))
    if not cands:
        return start

    # 前場過濾：handler 要跟籃框同半場（過場/後場剔除）
    def in_frontcourt(hp):
        if basket[0] < midcourt_x:      # 進攻左籃 → 前場是 x < 中線
            return hp[0] <= midcourt_x + 3
        return hp[0] >= midcourt_x - 3   # 進攻右籃 → 前場是 x > 中線
    front = [(f, hp, sp) for f, hp, sp in cands if in_frontcourt(hp)]
    pool = front if front else cands     # 沒前場幀就退回全部

    # 找 handler 的防守者（用第一個前場幀認人）
    h_def = None
    if player_assignment is not None:
        h_def, _ = nearest_defender(court_coords, player_assignment,
                                    handler_id, pool[0][0], handler_team)

    # 主判斷：handler 的防守者離 screener 最近的幀
    if h_def is not None:
        best_f, best_d = pool[0][0], float('inf')
        for f, hp, sp in pool:
            ddp = get_pos(court_coords, h_def, f, max_search=0)
            if ddp is None:
                continue
            d = _dist(ddp, sp)
            if d < best_d:
                best_d, best_f = d, f
        return best_f

    # fallback：沒防守者資訊 → handler ↔ screener 最近
    best_f, best_d = pool[0][0], float('inf')
    for f, hp, sp in pool:
        d = _dist(hp, sp)
        if d < best_d:
            best_d, best_f = d, f
    return best_f


# ── 3. 場上區域分類 ───────────────────────────────────────────────────────────

def classify_region(pos, basket):
    """
    把一個座標相對進攻籃框分類成戰術區域。
    回中文區域名。
    """
    if pos is None:
        return "未知"
    dist = _dist(pos, basket)
    dy = pos[1] - CENTER_Y     # 相對中線的橫向偏移（±）
    side = "左" if dy < -3 else ("右" if dy > 3 else "")

    if dist < 8:
        return "籃下/禁區"
    
    if dist < 16:
        return f"罰球線附近{('（' + side + '側）') if side else ''}"

    # 外圍：用橫向偏移分 弧頂 / 翼側 / 底角
    if abs(dy) < 8:
        return "弧頂（正面）"
    
    if abs(dy) < 17:
        return f"{side}側翼"
    
    return f"{side}底角"


# ── 4. roll vs pop ────────────────────────────────────────────────────────────

def classify_roll_pop(court_coords, screener_id, screen_frame, basket, fps,
                      window_sec=3.6, thresh=3.0):
    """
    擋完後 screener 往籃框靠 = roll（順下），往外拉 = pop（外拉）。

    參考點用「擋拆後固定 window_sec 秒」而非事件結尾——事件太短時結尾離
    screen_frame 太近，screener 還沒動就測不出 roll/pop。用完整 court_coords，
    不受事件邊界限制。

    回 ('roll'|'pop'|'unclear', delta_ft)。delta<0 表示更靠近籃框。
    """
    p_screen = get_pos(court_coords, screener_id, screen_frame)  # 擋拆當下的位置
    if p_screen is None:
        return "unclear", 0.0

    # 參考點：擋拆後 window_sec 秒「內」，screener 最後一個有座標的幀。
    # 不用固定的 screen_frame+window——那可能落在 screener 已從追蹤消失的幀，
    # 拿到 None 就只能放棄。改從窗尾往前找最後有效位置，screener 中途消失也 robust。
    n = len(court_coords)
    w = max(1, int(window_sec * fps))
    f_end = min(screen_frame + w, n - 1)
    p_ref = None
    for f in range(f_end, screen_frame, -1):                   # 從窗尾往前找
        p = get_pos(court_coords, screener_id, f, max_search=0)
        if p is not None:
            p_ref = p
            break
    if p_ref is None:                                          # 擋拆後整段都沒座標
        return "unclear", 0.0

    d_screen = _dist(p_screen, basket)
    d_ref    = _dist(p_ref, basket)
    delta = d_ref - d_screen     # 負=靠近籃框

    if delta < -thresh:
        return "roll", delta

    if delta > thresh:
        return "pop", delta

    return "unclear", delta


# ── 5. 找某人的防守者（最近的異隊球員）────────────────────────────────────────

def nearest_defender(court_coords, player_assignment, offense_pid, frame, handler_team=1):
    """回 (defender_pid, dist) — 離 offense_pid 最近的防守方球員。找不到回 (None, None)。"""
    op = get_pos(court_coords, offense_pid, frame)

    if op is None:
        return None, None
    
    best_pid, best_d = None, float('inf')
    n = len(court_coords)

    # 用 frame 附近有座標的球員
    f = frame if 0 <= frame < n else max(0, min(frame, n - 1))
    for pid, pos in court_coords[f].items():
        if pid == offense_pid:
            continue

        team = get_team(player_assignment, pid, f)

        if team is None or team == handler_team:
            continue   # 只找防守方

        d = _dist(op, pos)

        if d < best_d:
            best_d, best_pid = d, pid

    if best_pid is None:
        return None, None

    return best_pid, best_d


# ── 6. 防守 coverage 分類（啟發式，標明信心）──────────────────────────────────

def classify_coverage(court_coords, player_assignment, handler_id, screener_id,
                      screen_frame, end, basket, handler_team=1):
    """
    判斷防守方對擋拆的應對方式（heuristic，非 100% 準）：
      - switch（換防）：screener 的防守者擋完換去守 handler
      - hedge/show（包夾/延誤）：screener 的防守者短暫上前擾亂 handler 再回收
      - drop（退防）：screener 的防守者留在籃框附近不上前
      - unclear：訊號不足

    回 dict：{coverage, confidence, detail}
    """
    # 擋拆當下：handler 的防守者 / screener 的防守者
    h_def, _ = nearest_defender(court_coords, player_assignment, handler_id,
                                screen_frame, handler_team)
    s_def, _ = nearest_defender(court_coords, player_assignment, screener_id,
                                screen_frame, handler_team)

    if h_def is None or s_def is None:
        return {"coverage": "unclear", "confidence": "low",
                "detail": "擋拆當下抓不到雙方防守者"}

    if h_def == s_def:
        return {"coverage": "unclear", "confidence": "low",
                "detail": "handler 與 screener 的最近防守者是同一人，訊號模糊"}

    # 觀察 screener 的防守者(s_def) 在擋拆到事件尾之間的行為
    hp_screen = get_pos(court_coords, handler_id, screen_frame)
    # s_def 最靠近 handler 的距離（有沒有上前延誤）

    min_sdef_to_handler = float('inf')
    
    for f in range(screen_frame, end + 1):
        sdp = get_pos(court_coords, s_def, f, max_search=0)
        hp  = get_pos(court_coords, handler_id, f, max_search=0)
        if sdp is None or hp is None:
            continue
        min_sdef_to_handler = min(min_sdef_to_handler, _dist(sdp, hp))

    # s_def 尾端離籃框距離（有沒有退防）
    sdef_end = get_pos(court_coords, s_def, end)
    sdef_end_to_basket = _dist(sdef_end, basket) if sdef_end else None

    # 尾端誰在守 handler
    end_h_def, _ = nearest_defender(court_coords, player_assignment, handler_id,
                                    end, handler_team)

    # ── 判斷邏輯 ──
    STEP_OUT = 6.0   # s_def 上前到 handler 6ft 內 → 視為上前干擾
    DROP_ZONE = 14.0 # s_def 尾端仍在籃框 14ft 內 → 視為退防

    if end_h_def == s_def:
        # 擋完換人守 handler → 換防
        return {"coverage": "switch", "confidence": "medium",
                "detail": f"screener 的防守者(#{s_def})擋完接手守 handler"}

    if min_sdef_to_handler <= STEP_OUT and end_h_def == h_def:
        # 上前干擾過又交回原防守者 → hedge/show
        return {"coverage": "hedge", "confidence": "medium",
                "detail": f"#{s_def} 一度上前到 handler {min_sdef_to_handler:.0f}ft 再回收"}

    if sdef_end_to_basket is not None and sdef_end_to_basket <= DROP_ZONE \
       and min_sdef_to_handler > STEP_OUT:
        # 沒上前、尾端仍守在籃框附近 → drop
        return {"coverage": "drop", "confidence": "medium",
                "detail": f"#{s_def} 未上前延誤，退守籃框附近({sdef_end_to_basket:.0f}ft)"}

    return {"coverage": "unclear", "confidence": "low",
            "detail": f"s_def=#{s_def} 行為不明確 "
                      f"(min→handler={min_sdef_to_handler:.0f}ft)"}


# ── 事件後處理：合併碎片 + 濾雜訊（分析層調校，不動 inference）────────────────

def postprocess_events(events, merge_gap=15, min_frames=5, blip_frames=2):
    """
    對 inference dump 的原始事件做清理：
      1. 先移除極短 blip（<= blip_frames 幀）→ 這種 1-2 幀的雜訊若 screener 不同，
         會卡在中間打斷合併鏈。必須「合併前」先清掉。門檻比 min_frames 低，
         避免誤殺會併進長事件的正常碎片。
      2. 合併「同一 handler+screener、間隔 <= merge_gap」的相鄰事件
         → 補追蹤斷訊造成的單次擋拆被切碎（不同 screener 不會被合併，保留 double/staggered screen）
      3. 濾掉短於 min_frames 的事件（合併後仍太短 = 誤判雜訊）

    events: [{start, end, handler_id, screener_id, score, ...}]（需已按 start 排序）
    回：清理後的 events list。
    """
    if not events:
        return []

    # 1. 合併前先清掉極短 blip（會打斷合併鏈的雜訊）
    events = [e for e in events if (e['end'] - e['start'] + 1) > blip_frames]
    if not events:
        return []

    # 2. 合併同 screener 的小間隔碎片
    merged = []
    cur = dict(events[0])
    for ev in events[1:]:
        same_pair = (ev['handler_id'] == cur['handler_id']
                     and ev['screener_id'] == cur['screener_id'])
        gap = ev['start'] - cur['end'] - 1
        if same_pair and gap <= merge_gap:
            cur['end'] = ev['end']
            cur['score'] = max(cur['score'], ev['score'])
        else:
            merged.append(cur)
            cur = dict(ev)
    merged.append(cur)

    # 3. 濾掉合併後仍太短的事件
    kept = [ev for ev in merged if (ev['end'] - ev['start'] + 1) >= min_frames]
    return kept


# ── 頂層：分析一個事件 → facts dict ───────────────────────────────────────────

def analyze_event(event, data):
    """
    吃一個 event + dump 的資料，回結構化戰術事實 dict（給模板/LLM 用）。
    """
    cc = data['court_coords']
    pa = data['player_assignment']
    start, end = event['start'], event['end']
    handler_id = event['handler_id']
    screener_id = event['screener_id']

    handler_team = get_team(pa, handler_id, start) or 1
    basket = infer_target_basket(cc, start, end)
    screen_f = find_screen_frame(cc, handler_id, screener_id, start, end,
                                 player_assignment=pa, handler_team=handler_team,
                                 basket=basket)

    fps = data.get('fps', 30.0)
    screen_pos = get_pos(cc, screener_id, screen_f)
    region = classify_region(screen_pos, basket)
    action, delta = classify_roll_pop(cc, screener_id, screen_f, basket, fps)
    coverage = classify_coverage(cc, pa, handler_id, screener_id,
                                 screen_f, end, basket, handler_team)

    return {
        "frames": (start, end),
        "handler_id": handler_id,
        "screener_id": screener_id,
        "score": event.get('score'),
        "screen_frame": screen_f,
        "region": region,
        "action": action,             # roll / pop / unclear
        "action_delta_ft": round(delta, 1),
        "coverage": coverage['coverage'],
        "coverage_confidence": coverage['confidence'],
        "coverage_detail": coverage['detail'],
    }


# ── 獨立測試：讀 dump pkl，印每個事件的 facts ─────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl', help='inference dump 的 events.pkl')
    ap.add_argument('--merge_gap', type=int, default=15,
                    help='同 screener、間隔 <= 此幀數的事件合併（補斷訊）')
    ap.add_argument('--min_frames', type=int, default=5,
                    help='短於此幀數的事件濾掉')
    ap.add_argument('--no_deflicker', action='store_true',
                    help='關閉 homography 鏡像修正（debug 用）')
    args = ap.parse_args()

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    # 先修正 homography 左右鏡像閃爍（否則座標每隔幾幀翻一次，戰術判斷全錯）
    if not args.no_deflicker:
        data['court_coords'], nflip = deflicker_court_coords(data['court_coords'])
        print(f"de-flicker: 修正 {nflip}/{len(data['court_coords'])} 幀鏡像\n")

    raw = data['events']
    fps = data.get('fps', 30.0)   # 舊 pkl 沒存 fps 就 fallback 30
    events = postprocess_events(raw, args.merge_gap, args.min_frames)
    print(f"讀入 {len(raw)} 個原始事件 → 後處理後 {len(events)} 個  "
          f"(fps={fps:.1f}, merge_gap={args.merge_gap}, min_frames={args.min_frames})\n")
    action_zh = {"roll": "順下(roll)", "pop": "外拉(pop)", "unclear": "不明顯"}

    def _fmt(fr):
        return f"f{fr}({fr / fps:.1f}s)"

    for i, ev in enumerate(events, 1):
        facts = analyze_event(ev, data)
        s, e = facts['frames']
        sf = facts['screen_frame']
        dur = (e - s) / fps
        print(f"── 事件 #{i}  score={facts['score']:.2f} ──")
        print(f"  開始：{_fmt(s)}   結束：{_fmt(e)}   持續：{dur:.1f}s")
        print(f"  持球 #{facts['handler_id']} / 掩護 #{facts['screener_id']}")
        print(f"  擋拆接觸幀：{_fmt(sf)}")
        print(f"  擋拆位置：{facts['region']}")
        print(f"  掩護後動作：{action_zh.get(facts['action'])} "
              f"(離籃框變化 {facts['action_delta_ft']:+.1f}ft)")
        print(f"  防守應對：{facts['coverage']} "
              f"[{facts['coverage_confidence']}] — {facts['coverage_detail']}")
        print()


if __name__ == '__main__':
    main()
