"""
NBA Movement Data → 訓練序列

從 .7z 檔案讀取 NBA SportVU 追蹤資料，
套用論文 NETS (Appendix A) 的弱標籤規則產生：
  - 正樣本：Pick & Roll 序列
  - 負樣本：兩人靠近但沒有掩護的序列

輸出格式（每個序列）：
  trajectories: [3人, SEQ_LEN幀, 2座標]  (screener, handler, defender)
  roles:        [3]                        (0=screener, 1=handler, 2=defender)
  label:        0 or 1

使用方式：
  python extract_sequences.py --data_dir ../nba-movement-data/data --out_dir ./output --max_games 10
"""

import os
import json
import glob
import argparse
import numpy as np
import py7zr
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque

# ── 常數 ──────────────────────────────────────────────────────────────────────
SEQ_LEN        = 10          # 序列長度（幀），論文設定
STEP           = 5           # 滑動窗口步長（重疊取樣）
FPS            = 25          # SportVU 取樣率

# 論文 Appendix A 的閾值（英尺）
BALL_PLAYER_MAX_DIST   = 5.0   # 球距球員最大距離才算持球
BALL_MAX_HEIGHT        = 10.0  # 球高度上限（radius 欄位）
BALL_MAX_SPEED         = 25.0  # 球速上限 ft/s
POSSESSION_MIN_FRAMES  = 5     # 連續持球最少幀數

SCREEN_HANDLER_DIST    = 6.0   # δa：掩護者↔持球者
SCREEN_DEF_HANDLER_DIST= 6.0   # δd1：防守者↔持球者
SCREEN_DEF_SCREENER_DIST= 3.0  # δd2：防守者↔掩護者（最嚴格的條件）

# 負樣本觸發：兩名同隊球員靠近但沒掩護
NEG_SAMPLE_DIST        = 8.0   # 比正樣本閾值稍大，捕捉「快要掩護但還沒」

# ── 工具函式 ──────────────────────────────────────────────────────────────────

def dist(a, b):
    """兩點歐幾里得距離。"""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def parse_moment(moment):
    """
    解析單幀 moment 為結構化字典。

    Returns:
        dict with keys:
          'ball': {'pos': (x,y), 'height': radius}
          'players': {player_id: {'team': team_id, 'pos': (x,y)}}
    """
    period, ms, shot_clock, game_clock, _, player_list = moment

    ball_entry = player_list[0]   # [-1, -1, x, y, radius]
    ball = {
        'pos':    (ball_entry[2], ball_entry[3]),
        'height':  ball_entry[4],
    }

    players = {}
    for entry in player_list[1:]:
        team_id, player_id, x, y, _ = entry
        players[player_id] = {'team': team_id, 'pos': (x, y)}

    return {'ball': ball, 'players': players}


def find_ball_handler(parsed_moments, home_team_id, visitor_team_id):
    """
    論文 Appendix A 的持球判定：
      - 與球距離最近
      - 球高度 < BALL_MAX_HEIGHT ft
      - 球速 < BALL_MAX_SPEED ft/s
      - 連續 POSSESSION_MIN_FRAMES 幀

    Returns:
        list of (handler_id, handler_team_id) or None，長度 = len(parsed_moments)
    """
    n = len(parsed_moments)
    raw_handler = []   # 每幀的「最近球員」，未考慮連續性

    prev_ball_pos = None
    for pm in parsed_moments:
        ball_pos    = pm['ball']['pos']
        ball_height = pm['ball']['height']

        # 球速（ft/frame → ft/s）
        if prev_ball_pos is not None:
            ball_speed = dist(ball_pos, prev_ball_pos) * FPS
        else:
            ball_speed = 0.0
        prev_ball_pos = ball_pos

        # 球太高或太快 → 飛行中，無人持球
        if ball_height > BALL_MAX_HEIGHT or ball_speed > BALL_MAX_SPEED:
            raw_handler.append(None)
            continue

        # 找最近的球員
        best_pid, best_d = None, float('inf')
        for pid, info in pm['players'].items():
            d = dist(ball_pos, info['pos'])
            if d < best_d:
                best_d, best_pid = d, pid

        if best_pid is not None and best_d <= BALL_PLAYER_MAX_DIST:
            raw_handler.append((best_pid, pm['players'][best_pid]['team']))
        else:
            raw_handler.append(None)

    # 套用連續幀過濾
    handlers = [None] * n
    streak = 0
    current = None
    for i, h in enumerate(raw_handler):
        if h == current:
            streak += 1
        else:
            current = h
            streak = 1
        if streak >= POSSESSION_MIN_FRAMES and current is not None:
            handlers[i] = current

    return handlers


def find_defensive_assignments(handler_id, handler_team, parsed_moment):
    """
    論文 Appendix A：用線性分配（匈牙利算法）讓每名防守者對應一名進攻者。
    最小化總距離。

    Returns:
        dict: {offensive_player_id: defensive_player_id}
    """
    players = parsed_moment['players']

    offense = [pid for pid, info in players.items() if info['team'] == handler_team]
    defense = [pid for pid, info in players.items() if info['team'] != handler_team]

    if not offense or not defense:
        return {}

    # 成本矩陣：offense × defense
    cost = np.zeros((len(offense), len(defense)))
    for i, o in enumerate(offense):
        for j, d_pid in enumerate(defense):
            cost[i, j] = dist(players[o]['pos'], players[d_pid]['pos'])

    row_ind, col_ind = linear_sum_assignment(cost)
    return {offense[r]: defense[c] for r, c in zip(row_ind, col_ind)}


def check_pick_and_roll(handler_id, handler_pos,
                         candidate_id, candidate_pos,
                         def_of_handler_pos):
    """
    論文的三角形條件：
      δa  = dist(掩護者, 持球者)     < 6ft
      δd1 = dist(防守者, 持球者)     < 6ft
      δd2 = dist(防守者, 掩護者)     < 3ft  ← 防守者被卡住
    """
    da  = dist(candidate_pos, handler_pos)
    dd1 = dist(def_of_handler_pos, handler_pos)
    dd2 = dist(def_of_handler_pos, candidate_pos)

    return (da  < SCREEN_HANDLER_DIST      and
            dd1 < SCREEN_DEF_HANDLER_DIST  and
            dd2 < SCREEN_DEF_SCREENER_DIST)


# ── 主要提取邏輯 ──────────────────────────────────────────────────────────────

N_OFFENSE = 5
N_DEFENSE = 5
N_PLAYERS = N_OFFENSE + N_DEFENSE   # 10人，不含球


def extract_sequences_from_event(event, home_team_id):
    """
    從單個 event 提取全隊10人的帶標籤序列（對應 NETS 論文架構）。

    輸出格式（每筆序列）：
      trajectories: (10, SEQ_LEN, 2)
        slot 0      = ball handler（進攻方持球者）
        slot 1~4    = 其他進攻球員
        slot 5~9    = 防守球員

      roles: (10,)
        0 = ball handler
        1 = other offense
        2 = defense

      label: 0 or 1
        以「窗口內超過半數幀滿足三角形掩護條件」為正樣本

    標籤邏輯（論文 Appendix A）：
      逐幀對每位進攻隊友檢查三角形條件，
      只要有任何一對滿足即該幀為掩護幀，
      超過 SEQ_LEN//2 個掩護幀 → 整段序列標 1。
    """
    moments = event['moments']
    if len(moments) < SEQ_LEN:
        return []

    parsed = [parse_moment(m) for m in moments]
    handlers = find_ball_handler(parsed, home_team_id, None)

    sequences = []

    for start in range(0, len(parsed) - SEQ_LEN + 1, STEP):
        window         = parsed[start: start + SEQ_LEN]
        window_handlers = handlers[start: start + SEQ_LEN]

        # 以中間幀決定隊伍和防守分配（穩定的基準幀）
        mid        = SEQ_LEN // 2
        mid_pm     = window[mid]
        mid_handler = window_handlers[mid]

        if mid_handler is None:
            continue

        handler_id, handler_team = mid_handler
        if handler_id not in mid_pm['players']:
            continue

        # 確認這個 event 有完整的10名球員
        all_players = mid_pm['players']
        offense_ids = [pid for pid, info in all_players.items()
                       if info['team'] == handler_team]
        defense_ids = [pid for pid, info in all_players.items()
                       if info['team'] != handler_team]

        if len(offense_ids) < N_OFFENSE or len(defense_ids) < N_DEFENSE:
            continue

        # 固定球員順序：handler 放 slot 0，其他進攻按 id 排序
        offense_ids = ([handler_id] +
                       sorted(p for p in offense_ids if p != handler_id))
        defense_ids = sorted(defense_ids)

        # 防守分配（找每位進攻球員的防守者）
        assignments = find_defensive_assignments(handler_id, handler_team, mid_pm)

        # 逐幀提取10人軌跡 + 計算 pairwise 掩護條件
        # pair_screen_frames[i][j] = offense[j] 幫 offense[i] 做掩護的幀數
        player_order = offense_ids[:N_OFFENSE] + defense_ids[:N_DEFENSE]
        traj = np.zeros((N_PLAYERS, SEQ_LEN, 2), dtype=np.float32)
        pair_screen_frames = np.zeros((N_OFFENSE, N_OFFENSE), dtype=np.int32)
        valid = True

        for t, pm in enumerate(window):
            players = pm['players']

            if any(pid not in players for pid in player_order):
                valid = False
                break

            for i, pid in enumerate(player_order):
                traj[i, t] = players[pid]['pos']

            # 對每一對 (i, j) 進攻球員檢查三角形條件
            # i = 被掩護者（有防守者在旁），j = 掩護者
            for i in range(N_OFFENSE):
                off_i_id  = offense_ids[i]
                def_of_i  = assignments.get(off_i_id)
                if def_of_i is None or def_of_i not in players:
                    continue
                for j in range(N_OFFENSE):
                    if i == j:
                        continue
                    off_j_id = offense_ids[j]
                    if off_j_id not in players:
                        continue
                    if check_pick_and_roll(off_i_id, players[off_i_id]['pos'],
                                           off_j_id, players[off_j_id]['pos'],
                                           players[def_of_i]['pos']):
                        pair_screen_frames[i][j] += 1

        if not valid:
            continue

        # 多數幀投票（per pair）
        pairwise_labels = (pair_screen_frames >= SEQ_LEN // 2).astype(np.int64)  # pair_screen_frames 是 (5, 5) 的 numpy array，>= SEQ_LEN // 2 會對每個元素做比較，回傳 (5, 5) 的 bool array，.astype(np.int64) 再把 True/False 轉成 1/0。

        # 展平成 (20,)，順序與 PairwiseScreenHead 一致
        pairwise_flat = np.array([
            pairwise_labels[i][j]
            for i in range(N_OFFENSE)
            for j in range(N_OFFENSE)
            if i != j
        ], dtype=np.int64)

        # 向後相容的單一標籤
        is_screen = bool(pairwise_labels.any())

        # 負樣本篩選：保留進攻球員有互相靠近的片段
        if not is_screen:
            h_pos_mid = traj[0, mid]
            has_close_teammate = any(
                dist(traj[i, mid], h_pos_mid) < NEG_SAMPLE_DIST
                for i in range(1, N_OFFENSE)
            )
            if not has_close_teammate:
                continue

        # 正規化：以持球者起始位置為原點
        origin = traj[0, 0].copy()
        traj -= origin

        roles = np.array(
            [0] + [1] * (N_OFFENSE - 1) + [2] * N_DEFENSE,
            dtype=np.int64
        )

        sequences.append({
            'trajectories':       traj,                        # (10, SEQ_LEN, 2)
            'roles':              roles,                       # (10,)
            'label':              np.int64(1 if is_screen else 0),
            'pairwise_labels':    pairwise_flat,              # (20,)
            'screen_frame_ratio': pair_screen_frames.max() / SEQ_LEN,
        })

    return sequences


def process_game(archive_path):
    """
    解壓並處理單場比賽。

    Returns:
        list of sequence dicts
    """
    all_sequences = []

    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                z.extractall(tmp)
            json_files = [f for f in os.listdir(tmp) if f.endswith('.json')]
            if not json_files:
                return []
            with open(os.path.join(tmp, json_files[0])) as f:
                game_data = json.load(f)
    except Exception as e:
        print(f"  Error reading {archive_path}: {e}")
        return []

    home_team_id = game_data['events'][0]['home']['teamid']

    for event in game_data['events']:
        seqs = extract_sequences_from_event(event, home_team_id)
        all_sequences.extend(seqs)

    return all_sequences


# ── 平行處理 ─────────────────────────────────────────────────────────────────

def _worker(archive_path):
    """單場比賽的處理函式（給 multiprocessing 用）。"""
    seqs = process_game(archive_path)
    pos  = sum(1 for s in seqs if s['label'] == 1)
    neg  = sum(1 for s in seqs if s['label'] == 0)
    name = os.path.basename(archive_path).replace('.7z', '')
    return seqs, pos, neg, name


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../nba-movement-data/data',
                        help='Directory containing .7z game files')
    parser.add_argument('--out_dir', default='./output',
                        help='Output directory for .npz files')
    parser.add_argument('--max_games', type=int, default=0,
                        help='Max number of games to process (0 = all)')
    parser.add_argument('--workers', type=int,
                        default=max(1, mp.cpu_count() - 1),
                        help='Number of parallel workers')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    archives = sorted(glob.glob(os.path.join(args.data_dir, '*.7z')))
    if args.max_games > 0:
        archives = archives[:args.max_games]

    total_games = len(archives)
    print(f"Processing {total_games} games with {args.workers} workers...")

    all_pos, all_neg = [], []
    done = 0

    with mp.Pool(processes=args.workers) as pool:
        for seqs, pos, neg, name in pool.imap_unordered(_worker, archives):
            all_pos.extend(s for s in seqs if s['label'] == 1)
            all_neg.extend(s for s in seqs if s['label'] == 0)
            done += 1
            print(f"[{done}/{total_games}] {name}  pos={pos} neg={neg} "
                  f"| total pos={len(all_pos):,} neg={len(all_neg):,}",
                  flush=True)

    # 儲存結果
    print(f"\nSaving to {args.out_dir}/ ...")

    def save_split(sequences, label_str):
        if not sequences:
            print(f"  No {label_str} sequences to save.")
            return
        trajs    = np.stack([s['trajectories']      for s in sequences])
        roles    = np.stack([s['roles']              for s in sequences])
        labels   = np.stack([s['label']              for s in sequences])
        pw_labels= np.stack([s['pairwise_labels']    for s in sequences])
        ratios   = np.array([s['screen_frame_ratio'] for s in sequences],
                            dtype=np.float32)
        path = os.path.join(args.out_dir, f'{label_str}_sequences.npz')
        np.savez_compressed(path,
                            trajectories=trajs,
                            roles=roles,
                            labels=labels,
                            pairwise_labels=pw_labels,
                            screen_frame_ratio=ratios)
        print(f"  Saved {len(sequences):,} → {path}")
        print(f"  trajectories: {trajs.shape}  pairwise_labels: {pw_labels.shape}")

    save_split(all_pos, 'positive')
    save_split(all_neg, 'negative')

    total = len(all_pos) + len(all_neg)
    if total > 0:
        print(f"\n── 統計 ──")
        print(f"  正樣本 (Pick & Roll): {len(all_pos):,}  ({100*len(all_pos)/total:.1f}%)")
        print(f"  負樣本 (靠近非掩護):  {len(all_neg):,}  ({100*len(all_neg)/total:.1f}%)")
        print(f"  總計:                {total:,}")


if __name__ == '__main__':
    main()
