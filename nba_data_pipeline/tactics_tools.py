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
import collections
import pickle
import statistics

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


# ── 0b. 合併 SAM2 ID swap（同一實體球員被切成兩個 id）────────────────────────

def _median(vals):
    s = sorted(vals)
    m = len(s)
    return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2


def _is_same_player(court_coords, a, b,
                    max_copresent_ratio=0.1, pos_tol=8.0, gap_tol=3):
    """
    判斷 id a、b 是否為同一實體球員被 SAM2 切成兩個 id：
      1. 幾乎不共存（同時出現的幀數 / 較少者總幀數 <= max_copresent_ratio）
      2. 交替處位置連續（a 出現幀 對到最近的 b 出現幀，位置中位差 <= pos_tol ft）
    """
    n = len(court_coords)
    frames_a = [f for f in range(n) if a in court_coords[f]]
    frames_b = [f for f in range(n) if b in court_coords[f]]
    if len(frames_a) < 3 or len(frames_b) < 3:
        return False

    fb_set = set(frames_b)
    co = sum(1 for f in frames_a if f in fb_set)
    if co > max_copresent_ratio * min(len(frames_a), len(frames_b)):
        return False   # 常常共存 → 不同人

    # 位置連續性：a 每個出現幀，找 <= gap_tol 內最近的 b 出現幀比位置
    gaps = []
    for f in frames_a:
        for d in range(gap_tol + 1):
            hit = None
            for g in (f - d, f + d):
                if g in fb_set:
                    hit = g
                    break
            if hit is not None:
                gaps.append(_dist(court_coords[f][a], court_coords[hit][b]))
                break
    if len(gaps) < 3:
        return False
    return _median(gaps) <= pos_tol


def merge_swapped_ids(court_coords, player_assignment, events):
    """
    偵測並合併 SAM2 ID swap（如 #9/#10 其實是同一人）。
    把被交換的 id 統一改成同一個（取較小的 id 當代表），順帶改 player_assignment
    與 events 的 handler_id/screener_id。

    回 (court_coords, player_assignment, events, merge_pairs)。
    """
    all_ids = set()
    for fr in court_coords:
        all_ids.update(fr)
    all_ids = sorted(all_ids)

    # 建立 remap：old id → 代表 id（union-find 式，代表取最小）
    remap = {}
    merge_pairs = []
    for i, a in enumerate(all_ids):
        for b in all_ids[i + 1:]:
            ca, cb = remap.get(a, a), remap.get(b, b)
            if ca == cb:
                continue
            if _is_same_player(court_coords, a, b):
                lo, hi = sorted((ca, cb))
                # 把所有指向 hi 的都改指向 lo
                for k, v in list(remap.items()):
                    if v == hi:
                        remap[k] = lo
                remap[hi] = lo
                merge_pairs.append((hi, lo))

    if not remap:
        return court_coords, player_assignment, events, []

    def _remap_frame(frame):
        out = {}
        for pid, val in frame.items():
            np = remap.get(pid, pid)
            if np not in out:           # 罕見共存幀：保留先出現的
                out[np] = val
        return out

    cc = [_remap_frame(fr) for fr in court_coords]
    pa = [_remap_frame(fr) for fr in player_assignment]
    evs = []
    for ev in events:
        ev = dict(ev)
        ev['handler_id'] = remap.get(ev['handler_id'], ev['handler_id'])
        ev['screener_id'] = remap.get(ev['screener_id'], ev['screener_id'])
        evs.append(ev)
    return cc, pa, evs, merge_pairs


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


def clean_court_tracks(court_coords, court_w=COURT_W_FT, jump=8.0, max_run=3, fill_gaps=0):
    """
    球場座標清理（訓練 / 交叉驗證 / 推論共用同一套，5fps 下 jump=8ft ≈ 0.2 秒內不可能的位移）：
      1. 左右鏡像閃爍 → deflicker_court_coords
      2. 壞影格：某段（≤ max_run 張）所有人一起跳走、之後又跳回來（homography 暫時算壞）
         → 那段用前後好影格線性內插；兩邊沒都有的人直接拿掉（build_windows 會沿用最後位置）
      3. 單人尖刺：某人跳出去、下一張又跳回來（腳點抓錯）→ 內插
    「跳走之後沒有跳回來」的持續偏移不動（分不出是哪一段才是對的）。
      4. fill_gaps > 0：某人中間連續缺 ≤ fill_gaps 張座標（前後都有）→ 用前後兩張線性內插補上
         （2026-10-24：撞上那刻持球者 / 掩護者有框沒座標佔 ~12%，其中 2/3 連續缺 ≥3 張；
          下游原本只沿用最後一次的位置 → 撞上時持球者等於停在半秒前不動）。預設 0 = 不補（舊行為）。
    回 (清理後的 court_coords, {'flip', 'bad_frames', 'spikes', 'filled'})。
    """
    cc, n_flip = deflicker_court_coords(court_coords, court_w)
    cc = [dict(f) if f else f for f in cc]
    n = len(cc)

    def med_disp(a, b):
        com = [p for p in a if p in b] if a and b else []
        if len(com) < 2:
            return None
        return statistics.median(math.dist(a[p], b[p]) for p in com)

    # 2. 壞影格（整張一起跳）
    n_bad, f = 0, 1
    while f < n:
        m = med_disp(cc[f - 1], cc[f])
        if m is None or m <= jump:
            f += 1
            continue
        g_ok = None
        for g in range(f + 1, min(n, f + max_run + 1)):
            back, ref = med_disp(cc[g - 1], cc[g]), med_disp(cc[f - 1], cc[g])
            if back is not None and back > jump and ref is not None and ref <= jump:
                g_ok = g
                break
        if g_ok is None:
            f += 1
            continue
        a, b = cc[f - 1], cc[g_ok]
        span = g_ok - (f - 1)
        for k in range(f, g_ok):
            t = (k - (f - 1)) / span
            cc[k] = {p: (a[p][0] + (b[p][0] - a[p][0]) * t, a[p][1] + (b[p][1] - a[p][1]) * t)
                     for p in cc[k] if p in a and p in b}
            n_bad += 1
        f = g_ok + 1

    # 3. 單人尖刺
    n_spike = 0
    for f in range(1, n - 1):
        a, b, d = cc[f - 1], cc[f], cc[f + 1]
        if not (a and b and d):
            continue
        for p in list(b):
            if p in a and p in d and math.dist(a[p], b[p]) > jump and math.dist(b[p], d[p]) > jump \
                    and math.dist(a[p], d[p]) <= jump:
                b[p] = ((a[p][0] + d[p][0]) / 2, (a[p][1] + d[p][1]) / 2)
                n_spike += 1

    # 4. 個人短缺口內插
    n_fill = 0
    if fill_gaps > 0:
        seen = {}                                  # pid → 最後一次有座標的影格
        for f in range(n):
            fd = cc[f] or {}
            for p, xy in list(fd.items()):
                g = seen.get(p)
                if g is not None and 1 < f - g <= fill_gaps + 1:
                    a = cc[g][p]
                    for k in range(g + 1, f):
                        t = (k - g) / (f - g)
                        if not cc[k]:
                            cc[k] = {}
                        cc[k][p] = (a[0] + (xy[0] - a[0]) * t, a[1] + (xy[1] - a[1]) * t)
                        n_fill += 1
                seen[p] = f
    return cc, {'flip': n_flip, 'bad_frames': n_bad, 'spikes': n_spike, 'filled': n_fill}


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


# ── 5b. 逐時刻防守配對（HMM / Viterbi）────────────────────────────────────────

def _franks_mu(off_pos, ball_pos, basket, g_off=0.5, g_ball=0.2, g_hoop=0.3):
    """Franks 加權點：防守者「該站」的位置 = 進攻者 + 球 + 籃框的加權。
    球不明時用進攻者位置代（等於把球權重併給進攻者）。"""
    bx = ball_pos[0] if ball_pos else off_pos[0]
    by = ball_pos[1] if ball_pos else off_pos[1]
    return (g_off * off_pos[0] + g_ball * bx + g_hoop * basket[0],
            g_off * off_pos[1] + g_ball * by + g_hoop * basket[1])


def defensive_matchup_hmm(court_coords, player_assignment, handler_id, basket,
                          start=None, end=None, handler_team=None,
                          sigma=5.0, switch_pen=8.0, min_seg=30,
                          g_off=0.5, g_ball=0.2, g_hoop=0.3, blend=0.5):
    """逐時刻防守配對（每個防守者一條 Viterbi）。

    多數決只能給「整段誰守誰」的單一答案，抹平時間 → 抓不到：
      (1) 換防：擋拆後防守者交換守備對象（狀態隨時間變）
      (2) 兩人守一：包夾/hedge/trap（兩個防守者同時指向同一進攻者）
    HMM 對每個防守者獨立解「它逐幀守誰」的序列，兩者都抓得到。

    模型：
      隱藏狀態 = 這一刻守哪個進攻者
      發射機率 = 防守者離該進攻者的 Franks 加權點越近越高（高斯，sigma 控寬窄）
      轉移機率 = 維持同對象機率高、換低但不為零（switch_pen 控換防門檻）
      Viterbi 解最可能序列，再把短於 min_seg 幀的段併入相鄰長段（去抖動）

    參數（在 v129/v130 兩支不同角度影片同組驗證過）：
      sigma=5.0    高斯寬度(ft)，窄=要「明顯更近」才算換
      switch_pen=8.0  換守備對象的 log 懲罰，高=不易亂換（真換防才壓得過）
      min_seg=30   段落最短幀數(~1秒@30fps)，短段當雜訊併掉
      blend=0.5    發射距離 = blend·(離進攻者本人) + (1-blend)·(離 μ 加權點)。
                   純 μ（blend=0）在掩護/貼身時會失真——防守者被擋住貼著被掩護者，
                   不在理論質心，且球的權重把 μ 拉歪 → 誤判守錯人（v118 f81 實證）。
                   混入「離本人」校正。0.5 在 v118/v129/v130 標記集驗過。

    回 dict：{defender_id: [(offender_id, f_start, f_end), ...]}（逐段配對）。
    段落序列允許：同一防守者多段（換防）、不同防守者的段指向同一 offender（兩人守一）。
    """
    import collections as _c
    n = len(court_coords)
    s = 0 if start is None else max(0, start)
    e = n - 1 if end is None else min(n - 1, end)

    # 整段多數隊伍分進攻/防守（robust to 單幀分隊雜訊，同 find_screen_instances 慣例）
    _votes = {}
    for fr in player_assignment:
        for pid, t in fr.items():
            _votes.setdefault(pid, _c.Counter())[t] += 1
    maj_team = {pid: cnt.most_common(1)[0][0] for pid, cnt in _votes.items()}
    if handler_team is None:
        handler_team = maj_team.get(handler_id, 1)

    # 出現太少幀的 id（污染/替補）不納入，門檻 = 觀察窗的 1/3
    span = max(1, e - s + 1)
    appear = {pid: sum(1 for f in range(s, e + 1) if pid in court_coords[f])
              for pid in maj_team}
    offense = sorted(p for p in maj_team
                     if maj_team[p] == handler_team and appear[p] >= span // 3)
    defense = sorted(p for p in maj_team
                     if maj_team[p] != handler_team and appear[p] >= span // 3)
    if not offense or not defense:
        return {}

    def _emit_logp(d_pos, off_pos, ball_pos):
        # 距離 = blend·離進攻者本人 + (1-blend)·離 μ 加權點。
        # 純 μ 在掩護貼身時失真（防守者被擋、不在質心），混本人距離校正。
        d_mu = _dist(d_pos, _franks_mu(off_pos, ball_pos, basket, g_off, g_ball, g_hoop))
        d_real = _dist(d_pos, off_pos)
        dd = blend * d_real + (1.0 - blend) * d_mu
        return -(dd * dd) / (2.0 * sigma * sigma)   # 高斯 log（未正規化，比較用足夠）

    def _viterbi(D):
        frames = [f for f in range(s, e + 1) if D in court_coords[f]]
        if not frames:
            return {}, []
        S, T = offense, len(frames)
        dp = [{o: -1e18 for o in S} for _ in range(T)]
        bp = [{o: None for o in S} for _ in range(T)]
        f0 = frames[0]
        ball0 = court_coords[f0].get(handler_id)
        for o in S:
            dp[0][o] = (_emit_logp(court_coords[f0][D], court_coords[f0][o], ball0)
                        if o in court_coords[f0] else -1e18)
        for i in range(1, T):
            f = frames[i]
            ball = court_coords[f].get(handler_id)
            for o in S:
                em = (_emit_logp(court_coords[f][D], court_coords[f][o], ball)
                      if o in court_coords[f] else -1e18)
                best, barg = -1e30, None
                for op in S:
                    sc = dp[i - 1][op] + (0.0 if op == o else -switch_pen)
                    if sc > best:
                        best, barg = sc, op
                dp[i][o] = best + em
                bp[i][o] = barg
        last = max(S, key=lambda o: dp[T - 1][o])
        seq = [last]
        for i in range(T - 1, 0, -1):
            last = bp[i][last]
            seq.append(last)
        seq.reverse()
        return dict(zip(frames, seq)), frames

    def _segment(seq, frames):
        segs = []
        for f in frames:
            o = seq[f]
            if segs and segs[-1][0] == o:
                segs[-1][2] = f
            else:
                segs.append([o, f, f])
        # 反覆把短段併進較長鄰居（去單幀抖動）
        changed = True
        while changed and len(segs) > 1:
            changed = False
            for i, (o, a, b) in enumerate(segs):
                if b - a + 1 < min_seg:
                    left = segs[i - 1] if i > 0 else None
                    right = segs[i + 1] if i < len(segs) - 1 else None
                    cand = [x for x in (left, right) if x]
                    if not cand:
                        break
                    pick = max(cand, key=lambda x: x[2] - x[1])
                    pick[1] = min(pick[1], a)
                    pick[2] = max(pick[2], b)
                    segs.pop(i)
                    changed = True
                    break
        # 併相鄰同對象
        merged = []
        for o, a, b in segs:
            if merged and merged[-1][0] == o:
                merged[-1][2] = b
            else:
                merged.append([o, a, b])
        return [(o, a, b) for o, a, b in merged]

    out = {}
    for D in defense:
        seq, frames = _viterbi(D)
        if frames:
            out[D] = _segment(seq, frames)
    return out


def matchup_at_frame(matchup, frame):
    """從 defensive_matchup_hmm 的輸出取某一幀的配對快照。
    回 {defender_id: offender_id}（該幀有段落覆蓋的防守者）。"""
    snap = {}
    for D, segs in matchup.items():
        for o, a, b in segs:
            if a <= frame <= b:
                snap[D] = o
                break
    return snap


def double_teams(matchup, frame):
    """從 matchup 找某一幀「兩人以上守同一進攻者」的情況（包夾/hedge/trap 訊號）。
    回 {offender_id: [defender_id, ...]}，只列 >=2 人守的。"""
    import collections as _c
    snap = matchup_at_frame(matchup, frame)
    by_off = _c.defaultdict(list)
    for D, o in snap.items():
        by_off[o].append(D)
    return {o: sorted(ds) for o, ds in by_off.items() if len(ds) >= 2}


def _matchup_mode(matchup, D, f_start, f_end):
    """D 在 [f_start,f_end] 窗內守得最多的對象（多數決），None=無段落覆蓋。"""
    import collections as _c
    cnt = _c.Counter()
    for o, a, b in matchup.get(D, []):
        lo, hi = max(a, f_start), min(b, f_end)
        if hi >= lo:
            cnt[o] += hi - lo + 1
    return cnt.most_common(1)[0][0] if cnt else None


def classify_coverage_hmm(court_coords, matchup, handler_id, screener_id,
                          defender_id, contact_frame, basket, fps=30.0,
                          pre_win=0.6, post_win=1.2, hedge_dist=5.0,
                          drop_back_ft=4.0):
    """用 HMM 配對判一次擋拆的 coverage：switch / hedge / drop / unclear。

    規則負責看得準——讀 HMM 的逐時刻配對 + 擋拆前後的位置變化，不靠單幀最近。
    角色（擋拆是 screener 擋「守 handler 的防守者」，好讓 handler 甩開他）：
      Dh = 被擋的防守者（= defender_id，find_screen_instances 給的就是守 handler 被擋的人）
      Ds = 守 screener 的防守者（擋拆前另外找：matchup 裡守 screener 的人）
    coverage 看的是 Ds（守 screener 的人）怎麼應對這次擋拆。

    判斷順序（擋拆接觸幀 cf 前後各開一個窗）：
      hedge（先判）：原守 handler 的 Dh 沒走 + 守 screener 的 Ds 也上前貼近 handler
                    → 後窗變「兩人夾 handler」。這是 hedge/show 的定義，不是 switch。
                    用原始距離補判 Ds 是否貼近（HMM 包夾抓不全，Ds 去夾時常仍判它守 screener）。
      switch：Dh 與 Ds 對調（Ds 改守 handler、Dh 改守 screener），且非兩人守 handler。
      drop  ：Ds 維持守 screener 沒上前，且往籃框側退（離籃框更近 drop_back_ft 以上）。
      unclear：都不符。
    關鍵教訓：switch vs hedge 的差別在「原守 handler 的人有沒有走」——沒走+對方來夾=hedge（兩人守
    handler）；對調且仍一對一=switch。先判 hedge 才不會把「兩人夾」誤當 switch（v124/v127/v130 實證）。

    回 dict：{coverage, confidence, detail}
    """
    n = len(court_coords)
    cf = contact_frame
    pre_a, pre_b = max(0, cf - int(pre_win * fps)), cf
    post_a, post_b = cf, min(n - 1, cf + int(post_win * fps))

    # 不依賴 defender_id 當固定角色（實測它有時是 Dh 有時是 Ds）。
    # 直接用 HMM 配對找擋拆「前」誰守 handler(Dh)、誰守 screener(Ds)。
    Dh = Ds = None
    for D in matchup:
        pre = _matchup_mode(matchup, D, pre_a, pre_b)
        if pre == handler_id and Dh is None:
            Dh = D
        elif pre == screener_id and Ds is None:
            Ds = D
    # Ds 補救：配對沒抓到就用「擋拆前離 screener 最近的防守者」
    if Ds is None:
        best = None
        for D in matchup:
            if D == Dh:
                continue
            dp = get_pos(court_coords, D, pre_b, max_search=5)
            sp = get_pos(court_coords, screener_id, pre_b, max_search=5)
            if dp and sp:
                dd = _dist(dp, sp)
                if best is None or dd < best[1]:
                    best = (D, dd)
        if best:
            Ds = best[0]
    if Dh is None or Ds is None:
        return {"coverage": "unclear", "confidence": "low",
                "detail": "擋拆前找不到守 handler 或守 screener 的防守者"}

    Dh_post = _matchup_mode(matchup, Dh, post_a, post_b)
    Ds_post = _matchup_mode(matchup, Ds, post_a, post_b)

    # Ds 擋拆後有沒有一度貼近 handler（原始距離，繞過 HMM 包夾抓不全）
    min_ds_to_h = float('inf')
    hedge_frame = None
    for f in range(post_a, post_b + 1):
        dp = get_pos(court_coords, Ds, f, max_search=0)
        hp = get_pos(court_coords, handler_id, f, max_search=0)
        if dp and hp:
            dd = _dist(dp, hp)
            if dd < min_ds_to_h:
                min_ds_to_h, hedge_frame = dd, f
    dh_stays = (Dh_post == handler_id)   # 原守 handler 的沒走

    # 1) hedge（先判）：原守 handler 的沒走 + 守 screener 的也上來夾 handler
    #    → 後窗變成「兩人守 handler」。這是 hedge 的定義（包夾/show）。
    if dh_stays and min_ds_to_h < hedge_dist:
        return {"coverage": "hedge", "confidence": "high",
                "detail": f"#{Dh} 續守 handler#{handler_id}，#{Ds} 於 f{hedge_frame} "
                          f"上前夾（最近 {min_ds_to_h:.1f}ft）→ 兩人夾 handler"}

    # 2) switch：Dh 與 Ds 對調（Ds 去守 handler、Dh 去守 screener），且非兩人守 handler
    if Ds_post == handler_id and Dh_post == screener_id:
        return {"coverage": "switch", "confidence": "high",
                "detail": f"#{Dh}↔#{Ds} 交換：#{Ds}改守handler#{handler_id}、"
                          f"#{Dh}改守screener#{screener_id}"}

    # 3) drop：Ds 維持守 screener 且往籃框側退
    if Ds_post == screener_id or Ds_post is None:
        ds_pre = get_pos(court_coords, Ds, pre_b, max_search=5)
        ds_post_pos = get_pos(court_coords, Ds, post_b, max_search=5)
        if ds_pre and ds_post_pos:
            d_pre = _dist(ds_pre, basket)
            d_post = _dist(ds_post_pos, basket)
            if d_pre - d_post >= drop_back_ft:
                return {"coverage": "drop", "confidence": "medium",
                        "detail": f"#{Ds} 不上前，退向籃框（距籃框 "
                                  f"{d_pre:.0f}→{d_post:.0f}ft）"}
        return {"coverage": "drop", "confidence": "low",
                "detail": f"#{Ds} 維持守 screener#{screener_id}，未見上前夾擊"}

    return {"coverage": "unclear", "confidence": "low",
            "detail": f"#{Dh}守handler→#{Dh_post}、#{Ds}守screener→#{Ds_post}，不符模式"}


# ── 6. 防守 coverage 分類（啟發式，標明信心）──────────────────────────────────

def classify_coverage(court_coords, player_assignment, handler_id, screener_id,
                      screen_frame, end, basket, handler_team=1,
                      start=0, fps=30.0):
    """
    判斷防守方對擋拆的應對方式（heuristic，非 100% 準）：
      - switch（換防）：screener 的防守者擋完換去守 handler
      - hedge/show（包夾/延誤）：screener 的防守者短暫上前擾亂 handler 再回收
      - drop（退防）：screener 的防守者留在籃框附近不上前
      - unclear：訊號不足

    回 dict：{coverage, confidence, detail}
    """
    # 認防守者用「擋拆前 ~0.5 秒」而非擋拆瞬間——擋拆瞬間 handler/screener 貼在一起，
    # 兩人的最近防守者常算成同一人。擋拆前兩人還分開，各自的防守者分得清楚。
    pre_frame = max(start, screen_frame - int(0.5 * fps))
    h_def, _ = nearest_defender(court_coords, player_assignment, handler_id,
                                pre_frame, handler_team)
    s_def, _ = nearest_defender(court_coords, player_assignment, screener_id,
                                pre_frame, handler_team)

    if h_def is None or s_def is None:
        return {"coverage": "unclear", "confidence": "low",
                "detail": "擋拆前抓不到雙方防守者"}

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


def classify_coverage_ml(court_coords, player_assignment, handler_id, screener_id,
                         screen_frame, fps=30.0, handler_team=1, fallback_end=None,
                         basket=None):
    """
    防守方式（人工標記訓練的隨機森林，coverage_model.py）：switch / blitz / drop（含延誤）/ unclear。
    信心不足回 unclear。models/coverage_rf.pkl 不存在時退回規則版 classify_coverage。
    """
    try:
        from coverage_model import predict_coverage
        r = predict_coverage(court_coords, player_assignment, handler_id, screener_id,
                             screen_frame, fps, handler_team=handler_team)
    except ImportError:
        r = None
    if r is not None:
        return r
    end = fallback_end if fallback_end is not None else min(screen_frame + int(1.5 * fps),
                                                            len(court_coords) - 1)
    if basket is None:
        basket = infer_target_basket(court_coords, max(0, screen_frame - int(2 * fps)), end)
    return classify_coverage(court_coords, player_assignment, handler_id, screener_id,
                             screen_frame, end, basket, handler_team,
                             start=screen_frame, fps=fps)


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
    coverage = classify_coverage_ml(cc, pa, handler_id, screener_id, screen_f, fps,
                                    handler_team, fallback_end=end, basket=basket)

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


# ── 多 screener：分組 + 分析（double / staggered screen）──────────────────────

def screened_defender(court_coords, player_assignment, screener_id, handler_id,
                      s_start, s_end, handler_team=1, exclude_ids=None):
    """
    這個 screener 擋的是哪個防守者：整段跨度內「最靠近接觸」的防守者（方法2）。
    擋拆 = 近距離接觸，最靠近 screener 的防守者就是被擋的。
    回 (defender_id, min_contact_dist)；接觸距離也當信心/noise 訊號（太大=假 screener）。

    exclude_ids: 已知進攻方 id（handler + 其他 screener）→ 一律排除，避免 team
                 assignment 把進攻球員誤標成防守而被當成被擋防守者。
    """
    exclude = set(exclude_ids or ())
    exclude.update((screener_id, handler_id))
    min_by_def = {}
    for f in range(s_start, s_end + 1):
        sp = get_pos(court_coords, screener_id, f, max_search=0)
        if sp is None:
            continue
        for pid, pos in court_coords[f].items():
            if pid in exclude:
                continue
            if get_team(player_assignment, pid, f) == handler_team:
                continue   # 只看防守方
            d = _dist(sp, pos)
            if d < min_by_def.get(pid, float('inf')):
                min_by_def[pid] = d
    if not min_by_def:
        return None, None
    best = min(min_by_def, key=min_by_def.get)
    return best, min_by_def[best]


def group_multiscreener_events(events, merge_gap=15, min_screener_frames=5):
    """
    把單 screener 事件按 handler + 時間分組（不管 screener），收集這波所有 screener。
    → 天然支援 double/staggered screen，也讓 screener 跳動的碎裂消失。

    回 [{start, end, handler_id, screeners:[{screener_id, start, end, score, frames}]}]
    events 需已按 start 排序。
    """
    if not events:
        return []

    # 1. 按 handler + 時間相鄰聚叢
    clusters = []
    cur = None
    for ev in events:
        if (cur is None or ev['handler_id'] != cur['handler_id']
                or ev['start'] - cur['end'] - 1 > merge_gap):
            if cur:
                clusters.append(cur)
            cur = {'start': ev['start'], 'end': ev['end'],
                   'handler_id': ev['handler_id'], 'raw': [ev]}
        else:
            cur['end'] = max(cur['end'], ev['end'])
            cur['raw'].append(ev)
    if cur:
        clusters.append(cur)

    # 2. 叢內收集每個 screener 的跨度 + 總幀數
    out = []
    for cl in clusters:
        by_s = {}
        for ev in cl['raw']:
            s = ev['screener_id']
            dur = ev['end'] - ev['start'] + 1
            if s not in by_s:
                by_s[s] = {'screener_id': s, 'start': ev['start'], 'end': ev['end'],
                           'score': ev['score'], 'frames': dur}
            else:
                by_s[s]['start'] = min(by_s[s]['start'], ev['start'])
                by_s[s]['end'] = max(by_s[s]['end'], ev['end'])
                by_s[s]['score'] = max(by_s[s]['score'], ev['score'])
                by_s[s]['frames'] += dur
        # 3. 濾掉總幀數太少的 screener（noise）
        screeners = [v for v in by_s.values() if v['frames'] >= min_screener_frames]
        if not screeners:
            continue
        screeners.sort(key=lambda s: s['start'])
        out.append({'start': cl['start'], 'end': cl['end'],
                    'handler_id': cl['handler_id'], 'screeners': screeners})
    return out


def _classify_play_type(screeners, overlap_ratio=0.3):
    """依 screener 跨度重疊判 double（同時）vs staggered（先後）vs single。"""
    if len(screeners) <= 1:
        return "single"
    # 取跨度最大的兩個比重疊
    a, b = screeners[0], screeners[1]
    overlap = min(a['end'], b['end']) - max(a['start'], b['start'])
    span = max(a['end'], b['end']) - min(a['start'], b['start'])
    if span <= 0:
        return "double_screen"
    return "double_screen" if overlap > overlap_ratio * span else "staggered_screen"


def analyze_multiscreener_event(event, data):
    """對一個多 screener 事件，每個 screener 各自算 facts。"""
    cc = data['court_coords']
    pa = data['player_assignment']
    fps = data.get('fps', 30.0)
    handler_id = event['handler_id']
    start, end = event['start'], event['end']
    handler_team = get_team(pa, handler_id, start) or 1
    basket = infer_target_basket(cc, start, end)

    # 這事件裡所有進攻方 id（handler + 全部 screener）→ 找被擋防守者時排除
    offense_ids = {handler_id} | {sc['screener_id'] for sc in event['screeners']}

    facts = []
    for sc in event['screeners']:
        sid, ss, se = sc['screener_id'], sc['start'], sc['end']
        sf = find_screen_frame(cc, handler_id, sid, ss, se,
                               player_assignment=pa, handler_team=handler_team,
                               basket=basket)
        region = classify_region(get_pos(cc, sid, sf), basket)
        action, delta = classify_roll_pop(cc, sid, sf, basket, fps)
        cov = classify_coverage(cc, pa, handler_id, sid, sf, se, basket,
                                handler_team, start=ss, fps=fps)
        sdef, sdist = screened_defender(cc, pa, sid, handler_id, ss, se,
                                        handler_team, exclude_ids=offense_ids)
        facts.append({
            "screener_id": sid,
            "screened_defender": sdef,
            "contact_dist": round(sdist, 1) if sdist is not None else None,
            "screen_frame": sf,
            "region": region,
            "action": action,
            "action_delta_ft": round(delta, 1),
            "coverage": cov['coverage'],
            "coverage_confidence": cov['confidence'],
            "coverage_detail": cov['detail'],
        })

    return {
        "frames": (start, end),
        "handler_id": handler_id,
        "n_screeners": len(facts),
        "play_type": _classify_play_type(event['screeners']),
        "screeners": facts,
    }


# ── 接觸式擋拆偵測（screener × 被擋防守者 × 時間）─────────────────────────────

def find_screen_instances(court_coords, player_assignment, events,
                          contact_thr=3.5, gap_merge=20, handler_team=1,
                          near_handler_thr=20.0):
    """
    找所有擋拆實例：(screener, 被擋防守者, 接觸幀, 接觸距離, handler)。

    四個訊號組合：
      1. ScreenNet 擋拆窗：只在 ScreenNet 認定 screener 擋拆的時間內找（濾開場假接觸）
      2. 幾何最緊接觸：screener 撞上防守者的瞬間 = 擋拆時間點
      3. 用防守者區分 + 保留多次：同一 (screener,defender) 若時間分開接觸多次，
         各算一次擋（gap_merge 內視為同一次，取最緊）→ 支援同一人擋多次
      4. near-handler：擋拆接觸時 screener 要離 handler < near_handler_thr。
         擋是「為持球者」設的，一定在球附近 → 濾掉 pop 投籃、遠處路過等假擋。

    回 list[dict]，按 frame 排序。
    """
    # 每個 screener 的 ScreenNet 窗 + handler
    windows, handler_of = {}, {}
    for e in events:
        s = e['screener_id']
        windows.setdefault(s, set()).update(range(e['start'], e['end'] + 1))
        handler_of[s] = e['handler_id']   # 同 screener 的 handler 一致
    offense = set(windows) | set(handler_of.values())

    # 每個球員整段的多數隊伍（robust to 單幀分隊雜訊）。
    # 之前每幀查隊伍：擋拆那幀防守者剛好被誤分成進攻，就被排掉 → 抓錯擋誰。
    import collections as _c
    _votes = {}
    for fr in player_assignment:
        for pid, t in fr.items():
            _votes.setdefault(pid, _c.Counter())[t] += 1
    maj_team = {pid: cnt.most_common(1)[0][0] for pid, cnt in _votes.items()}

    instances = []
    for S in windows:
        H = handler_of[S]

        def add_instance(bf, bd, D):
            hp = get_pos(court_coords, H, bf, max_search=3)
            sp = get_pos(court_coords, S, bf, max_search=3)
            s2h = round(_dist(sp, hp), 1) if (hp and sp) else None
            instances.append({'frame': bf, 'screener_id': S, 'defender_id': D,
                              'contact_dist': bd, 'handler_id': H,
                              'screener_to_handler': s2h})

        # 收集窗內 screener 對每個防守者的 (frame, dist)
        # near-handler filter 放這裡：只看 screener 離 handler 近的幀，
        # 「最緊接觸」才會落在真正的擋拆時機（而非遠處的偶然貼近）
        by_def = {}
        for f in sorted(windows[S]):
            sp = court_coords[f].get(S) if 0 <= f < len(court_coords) else None
            if sp is None:
                continue
            hp = get_pos(court_coords, H, f, max_search=3)
            if hp is None or _dist(sp, hp) > near_handler_thr:
                continue   # 這幀 screener 離持球者太遠 → 不是擋拆時機
            for D, dp in court_coords[f].items():
                if D in offense or maj_team.get(D) == handler_team:
                    continue   # 用多數決隊伍，不看單幀（避免雜訊誤排除防守者）
                by_def.setdefault(D, []).append((f, _dist(sp, dp)))
        # 每個防守者：取 <thr 的接觸，按時間 gap 分段，每段取最緊 = 一次擋
        for D, seq in by_def.items():
            below = sorted((f, d) for f, d in seq if d < contact_thr)
            if not below:
                continue
            run = [below[0]]
            for f, d in below[1:]:
                if f - run[-1][0] > gap_merge:
                    bf, bd = min(run, key=lambda x: x[1])
                    add_instance(bf, bd, D)
                    run = [(f, d)]
                else:
                    run.append((f, d))
            bf, bd = min(run, key=lambda x: x[1])
            add_instance(bf, bd, D)

    # peak 模式（ScreenNet 輸出撞上時間點，start/end 只是前後 ±0.3 秒）：上面的幾何規則在這麼短的窗內
    # 常因座標誤差找不到 < contact_thr 的接觸 → 整個擋拆被丟掉（2026-10-26 端到端 10 支只剩 4 支）。
    # ScreenNet 說有擋拆就保留：幾何找不到時用 peak_frame + 當下離掩護者最近的防守者，標 contact_geom=False。
    for e in events:
        pf = e.get('peak_frame')
        if pf is None:
            continue
        S, H = e['screener_id'], e['handler_id']
        if any(it['screener_id'] == S and abs(it['frame'] - pf) <= max(gap_merge, e['end'] - e['start'])
               for it in instances):
            continue
        sp = get_pos(court_coords, S, pf, max_search=3)
        best = (None, None)
        if sp is not None and 0 <= pf < len(court_coords):
            cands = [(round(_dist(sp, dp), 1), D) for D, dp in court_coords[pf].items()
                     if D not in offense and maj_team.get(D) != handler_team]
            if cands:
                best = min(cands)
        hp = get_pos(court_coords, H, pf, max_search=3)
        instances.append({'frame': pf, 'screener_id': S, 'defender_id': best[1],
                          'contact_dist': best[0] if best[0] is not None else float('nan'), 'handler_id': H,
                          'screener_to_handler': round(_dist(sp, hp), 1) if (sp and hp) else None,
                          'contact_geom': False})

    instances.sort(key=lambda x: x['frame'])
    return instances


def group_screen_events(instances, time_gap=25):
    """把時間相近的擋拆實例湊成事件（同時 2+ 個 = double/staggered）。回 list[list[instance]]。

    同事件內若同一 (screener, 被擋防守者) 出現多次（一次擋被切成多筆），
    去重只留接觸最緊的一筆。
    """
    if not instances:
        return []
    events, cur = [], [instances[0]]
    for it in instances[1:]:
        if it['frame'] - cur[-1]['frame'] <= time_gap:
            cur.append(it)
        else:
            events.append(cur)
            cur = [it]
    events.append(cur)

    # 事件內去重：同一 screener 只留接觸最緊的一筆
    # （一個掩護者在一次擋拆事件裡只擋一個人；多筆是同一次被切碎或抓到雜訊防守者）
    deduped = []
    for g in events:
        best = {}
        for it in g:
            key = it['screener_id']
            if key not in best or it['contact_dist'] < best[key]['contact_dist']:
                best[key] = it
        deduped.append(sorted(best.values(), key=lambda x: x['frame']))
    return deduped


KIND_ZH = {'jump-shot': '跳投', 'layup': '上籃 / 灌籃', 'unknown': '投籃'}


def screen_outcome(data, screen_frame, handler_id, screener_id, handler_team, basket, max_s=8.0):
    """擋拆之後這波的出手結果（shots 來自 inference.py 的投籃偵測，shot_detect.py）。
    同一段鏡頭、擋拆後 -0.5 ~ max_s 秒內，進攻方的第一次出手；中間先看到防守方出手 = 球權已換，算沒有出手。
    回 None = 沒有投籃偵測資料（舊 pkl / 沒跑 --ball）。"""
    shots = data.get('shots')
    if shots is None:
        return None
    fps = data.get('fps', 30.0)
    cuts = data.get('cuts') or []
    seg_end = next((c for c in cuts if c > screen_frame), len(data['court_coords']))
    seg_start = max([0] + [c for c in cuts if c <= screen_frame])
    lo, hi = screen_frame - int(0.5 * fps), min(seg_end, screen_frame + int(max_s * fps))
    for sh in sorted(shots, key=lambda x: x['release_frame']):
        f = sh['release_frame']
        if not (max(lo, seg_start) <= f < hi):
            continue
        # 隊伍用整段多數決（單張分隊有雜訊：screen_286 / 342 / 407 持球者自己出手被判成防守方）
        votes = collections.Counter(fr[sh['shooter']] for fr in data['player_assignment'] if sh['shooter'] in fr)
        team = votes.most_common(1)[0][0] if votes else None
        if sh['shooter'] != handler_id and team is not None and team != handler_team:
            return {"shot": False, "note": f"{(f - screen_frame) / fps:.1f} 秒後防守方出手（球權已換）"}
        pos = get_pos(data['court_coords'], sh['shooter'], f)
        dist = round(_dist(pos, basket), 1) if pos else None
        role = "持球者" if sh['shooter'] == handler_id else ("掩護者" if sh['shooter'] == screener_id else "其他進攻球員")
        return {"shot": True, "seconds_after_screen": round((f - screen_frame) / fps, 1),
                "shooter_id": sh['shooter'], "shooter_role": role,
                "kind": KIND_ZH.get(sh['kind'], '投籃'),
                "zone": classify_region(pos, basket) if pos else "未知",
                "three_point": (dist >= 22.0) if dist is not None else None,
                "shot_dist_ft": dist,
                "made": sh['made'],          # True / False / None（看不出來）
                "confident": bool(sh['strong'])}
    if hi >= seg_end and seg_end < screen_frame + int(max_s * fps):
        return {"shot": None, "note": f"擋拆後 {(seg_end - screen_frame) / fps:.1f} 秒鏡頭就切走，沒拍到結果"}
    return {"shot": False, "note": f"擋拆後 {max_s:.0f} 秒內沒有偵測到出手"}


def analyze_screen_instance(inst, data, handler_team, basket):
    """對一個擋拆實例算 region / roll-pop / coverage。"""
    cc = data['court_coords']
    pa = data['player_assignment']
    fps = data.get('fps', 30.0)
    sid, sf, H = inst['screener_id'], inst['frame'], inst['handler_id']
    # coverage 需要一個「擋拆後」的觀察區間
    cov_end = min(sf + int(1.5 * fps), len(cc) - 1)
    region = classify_region(get_pos(cc, sid, sf), basket)
    action, delta = classify_roll_pop(cc, sid, sf, basket, fps)
    cov = classify_coverage_ml(cc, pa, H, sid, sf, fps, handler_team,
                               fallback_end=cov_end, basket=basket)
    return {
        "screener_id": sid,
        "screened_defender": inst['defender_id'],
        "contact_dist": round(inst['contact_dist'], 1),
        "screen_frame": sf,
        "region": region,
        "action": action,
        "action_delta_ft": round(delta, 1),
        "coverage": cov['coverage'],
        "coverage_confidence": cov['confidence'],
        "coverage_detail": cov['detail'],
        "outcome": screen_outcome(data, sf, H, sid, handler_team, basket),
    }


# ── 獨立測試：讀 dump pkl，印每個事件的 facts ─────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl', help='inference dump 的 events.pkl')
    ap.add_argument('--contact_thr', type=float, default=3.5,
                    help='接觸距離門檻（ft）：screener 撞到防守者 < 此距離才算真擋')
    ap.add_argument('--time_gap', type=int, default=25,
                    help='時間相近的擋拆湊成同一事件（雙擋）的幀數門檻')
    ap.add_argument('--no_deflicker', action='store_true',
                    help='關閉 homography 鏡像修正（debug 用）')
    args = ap.parse_args()

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    fps = data.get('fps', 30.0)   # 舊 pkl 沒存 fps 就 fallback 30
    action_zh = {"roll": "順下(roll)", "pop": "外拉(pop)", "unclear": "不明顯"}

    def _fmt(fr):
        return f"f{fr}({fr / fps:.1f}s)"

    # 1. 修正 homography 鏡像
    if not args.no_deflicker:
        data['court_coords'], nflip = deflicker_court_coords(data['court_coords'])
    else:
        nflip = 0
    # 2. 合併 SAM2 ID swap（#9/#10 這種同一人兩 id）
    data['court_coords'], data['player_assignment'], data['events'], pairs = \
        merge_swapped_ids(data['court_coords'], data['player_assignment'], data['events'])
    # 3. 接觸式偵測：找所有擋拆實例 → 按時間湊成事件
    instances = find_screen_instances(data['court_coords'], data['player_assignment'],
                                      data['events'], contact_thr=args.contact_thr)
    groups = group_screen_events(instances, time_gap=args.time_gap)

    print(f"原始 {len(data['events'])} 事件 → {len(instances)} 次擋 → {len(groups)} 個擋拆事件  "
          f"(de-flicker {nflip}幀, ID-merge {pairs}, thr={args.contact_thr}ft, fps={fps:.1f})\n")

    for i, g in enumerate(groups, 1):
        f0 = min(it['frame'] for it in g)
        f1 = max(it['frame'] for it in g)
        H = g[0]['handler_id']
        handler_team = get_team(data['player_assignment'], H, f0) or 1
        basket = infer_target_basket(data['court_coords'], max(0, f0 - 15), f1 + 15)
        kind = "雙擋" if len(g) >= 2 else "單擋"
        print(f"── 事件 #{i}  [{_fmt(f0)}~{_fmt(f1)}]  持球 #{H}  【{kind}, {len(g)}次擋】──")
        for j, it in enumerate(g, 1):
            fa = analyze_screen_instance(it, data, handler_team, basket)
            print(f"  擋{j}: #{fa['screener_id']} 擋防守者#{fa['screened_defender']}"
                  f"(接觸{fa['contact_dist']}ft)  @{_fmt(fa['screen_frame'])}")
            print(f"        {fa['region']} | {action_zh.get(fa['action'])}"
                  f"({fa['action_delta_ft']:+.1f}ft) | 防守={fa['coverage']}"
                  f"[{fa['coverage_confidence']}]")
        oc = analyze_screen_instance(g[-1], data, handler_team, basket)['outcome']   # 最後一次擋之後的結果
        if oc is not None:
            if oc.get('shot'):
                made = {True: '進', False: '沒進', None: '進沒進看不出來'}[oc['made']]
                print(f"  結果：{oc['seconds_after_screen']} 秒後 #{oc['shooter_id']}（{oc['shooter_role']}）"
                      f"{oc['zone']}{'三分' if oc['three_point'] else ''}{oc['kind']}，{made}"
                      f"{'' if oc['confident'] else '（沒看到籃框，較不確定）'}")
            else:
                print(f"  結果：{oc['note']}")
        print()


if __name__ == '__main__':
    main()
