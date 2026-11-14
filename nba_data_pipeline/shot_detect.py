"""
shot_detect.py — 推論用的投籃偵測（把 shot_owner.py 的規則包成一個函式）。

輸入：
  shot_sam3：run_sam3_tracks.py --ball 存的 10fps 抽樣影格 球員遮罩 + 球遮罩（pkl 的 'shot_sam3'）
  frames：原片影格（切鏡頭偵測用）
  rf：Roboflow NBA 偵測（籃框 / 跳投 / 上籃 / 球進框；run_rf() 產生，格式同 shot_detect_rf.py）
規則（2026-10-26，84 支延長片段人工核對，63 支有答案：出手時間 + 出手者全對 48）：
  一定要 10fps：5fps 追球，球飛起來那 0.2 秒會跟丟（screen_286）；球員遮罩借 5fps 的也掉到 45。
  球在空中（沒人碰）的時段 → 出手者 = 離手前 1.5 秒碰球最多的人；投籃動作幅度 ≥ 0.9 倍出手者身高；
  球離籃框 ≤ 1.0 倍身高或飛出畫面上方 = 強候選。類型 / 進球用 Roboflow（跳投 / 上籃、短暫的球進框）。
輸出：list[dict]（每次出手）：release_frame（原片影格）、t、shooter（已對回 player_tracks 的編號；
      shooter_raw = 投籃那次 SAM3 的原編號）、
      kind（jump-shot / layup / unknown）、made（True / False / None = 看不出來）、strong、amp、rim、segment
"""
from pathlib import Path

import numpy as np

import shot_owner as so
from shot_events import find_cuts, analyze, IN_BASKET

ROOT = Path(__file__).parent.parent
RF_WEIGHTS = ROOT / 'models' / 'rf_basketball_det_v4.onnx'


def run_rf(frames, fps, target_fps=10, weights=RF_WEIGHTS):
    """Roboflow NBA 偵測（ONNX，本機權重，不需金鑰）每 fps/target_fps 張跑一次 → shot_events 用的格式。"""
    from shot_detect_rf import OnnxDetector, KEEP
    det = OnnxDetector(weights)
    stride = max(1, int(round(fps / target_fps)))
    out = [(f, [d for d in det.infer(frames[f], conf=0.25, iou=0.9) if d[0] in KEEP])
           for f in range(0, len(frames), stride)]
    return dict(fps=fps, n=len(frames), stride=stride, size=frames[0].shape[1::-1], frames=out)


# ── 進球判斷：籃框放大連續截圖給 VLM（2026-11-01：42 次出手，Roboflow 67% → VLM 83~86%，兩次重跑 4/42 題答案不同）──
VLM_MODEL, VLM_EFFORT = 'gpt-5.5', 'medium'
VLM_SYSTEM = ('你是籃球比賽影像判讀員。給你一張由 12 格組成的圖：同一次投籃、籃框附近的放大連續截圖，'
              '時間由左到右、由上到下（每格左上角有秒數）。判斷這次投籃有沒有進。'
              '進＝球從籃框上方穿過籃框、往下穿過籃網（籃網常被往下拉）；沒進＝球打框彈出、在框上轉出、沒碰框、被蓋火鍋。'
              '注意轉播是側面高角度：球從籃框前面或後面掉下去，看起來也可能像穿過，要看球是不是真的進到籃網裡。'
              '看不清楚就回 null，不要猜。只輸出 JSON：{"made": true/false/null, "confidence": "high"|"medium"|"low", "reason": "一句話"}')
GRID_TIMES = (-0.3, -0.2, -0.13, -0.07, 0.0, 0.07, 0.13, 0.2, 0.3, 0.45, 0.6, 0.9)   # 球到籃框前後；每 0.2 秒會漏掉球在網裡那一刻


def _rim_at(rf, f):
    at = dict(rf['frames'])
    for d in (0, -1, 1, -2, 2, -3, 3):
        dets = [x for x in at.get(f + d, []) if x[0] == 9 and x[1] >= 0.4]
        if dets:
            return max(dets, key=lambda x: x[1])[2]
    return None


def made_grid(s, tch, frames, rf, release, out_path, win=2.5):
    """球最接近籃框那刻前後 12 張、以籃框為中心放大（籃框寬 ±1.8 倍、上 1.5 下 2.2 倍）→ 3×4 圖存檔。
    win = 離手後幾秒內找「球最接近籃框」：判斷是不是投籃要短（1.5），不然假候選會拍到後面那次真投籃（screen_79）。"""
    import cv2
    fps_s, fps_o = s['fps'], s['orig_fps']
    r0 = int(round(release * fps_s)); end = min(len(tch), r0 + int(win * fps_s))
    best = None
    for i in range(r0, end):
        c, rim = tch[i][1], _rim_at(rf, s['frame_idx'][i])
        if c is None or rim is None:
            continue
        w = rim[2] - rim[0]
        d = np.hypot(c[0] - (rim[0] + rim[2]) / 2, c[1] - (rim[1] + rim[3]) / 2) / w
        if best is None or d < best[0]:
            best = (d, s['frame_idx'][i])
    fk = best[1] if best and best[0] < 3 else int((release + min(1.0, win / 2)) * fps_o)   # 球沒追到籃框附近 → 出手後 1 秒
    tiles = []
    for t in GRID_TIMES:
        fo = min(len(frames) - 1, max(0, int(fk + t * fps_o)))
        rim = _rim_at(rf, fo) or _rim_at(rf, fk)
        im = frames[fo]; H, W = im.shape[:2]
        if rim is not None:
            w = rim[2] - rim[0]; cx = (rim[0] + rim[2]) / 2; cy = (rim[1] + rim[3]) / 2
            im = im[int(max(0, cy - 1.5 * w)):int(min(H, cy + 2.2 * w)), int(max(0, cx - 1.8 * w)):int(min(W, cx + 1.8 * w))]
        crop = cv2.resize(im, (400, 360))
        cv2.putText(crop, f'{(fo - fk) / fps_o:+.2f}s', (6, 26), 0, 0.8, (0, 255, 255), 2)
        tiles.append(crop)
    cv2.imwrite(str(out_path), np.vstack([np.hstack(tiles[k:k + 4]) for k in (0, 4, 8)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


VLM_SYSTEM_SHOT = ('你是籃球比賽影像判讀員。給你一張由 12 格組成的圖：同一次「疑似投籃」、籃框附近的放大連續截圖，'
                   '時間由左到右、由上到下（每格左上角有秒數）。'
                   '第一題：這是不是一次投籃？投籃＝球離開球員的手、朝籃框飛去（打框、進網、沒碰框、被蓋火鍋都算），'
                   '或球員把球帶到籃框放進去 / 灌進去（上籃、灌籃）。不是投籃＝球一直在球員手上（運球、持球、做假動作沒出手）、'
                   '球被傳給隊友（包括高吊傳球在籃框附近被接住）、或球只是從籃框前面或下面經過、沒有往籃框飛。'
                   '注意轉播是側面高角度：籃框下方站著的人、經過的球看起來都會貼著籃框，要看球是不是真的往籃框上方飛、碰到籃框或籃網。'
                   '第二題：是投籃的話，有沒有進？進＝球從籃框上方穿過籃框、往下穿過籃網（籃網常被往下拉）；'
                   '沒進＝球打框彈出、在框上轉出、沒碰框、被蓋火鍋。不是投籃就回 null。'
                   '看不清楚就回 null，不要猜。只輸出 JSON：'
                   '{"shot": true/false/null, "made": true/false/null, "confidence": "high"|"medium"|"low", "reason": "一句話"}')


def vlm_made(img_path, client, cache_path=None, system=None, ask='這次投籃有沒有進？只輸出 JSON。'):
    """→ {'made': True/False/None, 'confidence', 'reason'}（system=VLM_SYSTEM_SHOT 時多 'shot'）；同一張圖 + 題目讀快取。"""
    import base64, hashlib, json
    system = system or VLM_SYSTEM
    data = open(img_path, 'rb').read()
    h = hashlib.md5((system + VLM_MODEL).encode() + data).hexdigest()
    cache_path = Path(cache_path or ROOT / 'stubs' / 'vlm_made_cache_infer.jsonl')
    if cache_path.exists():
        for line in open(cache_path, encoding='utf-8'):
            j = json.loads(line)
            if j['h'] == h:
                return j['resp']
    msgs = [{'role': 'system', 'content': system},
            {'role': 'user', 'content': [{'type': 'text', 'text': ask},
                                         {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode(),
                                                                             'detail': 'high'}}]}]
    r = client.chat.completions.create(model=VLM_MODEL, reasoning_effort=VLM_EFFORT,
                                       response_format={'type': 'json_object'}, messages=msgs)
    resp = json.loads(r.choices[0].message.content)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'a', encoding='utf-8') as fo:
        fo.write(json.dumps({'h': h, 'img': str(img_path), 'resp': resp}, ensure_ascii=False) + '\n')
    return resp


def apply_vlm_made(data, pkl_path):
    """讀 events.pkl 後呼叫（tactics_tools / tactics_agent，本機有 LLM 金鑰）：有 made_grid 圖的出手問 VLM，
    答 true / false 就取代 Roboflow 的進球判斷；null / 沒金鑰 / 失敗 → 保留 Roboflow。改的是 data['shots'] 本身。"""
    shots = data.get('shots') or []
    todo = [sh for sh in shots if sh.get('made_grid') and 'made_source' not in sh]
    if not todo:
        return 0
    try:
        from coverage_model import _llm_client
        client = _llm_client()
    except Exception:
        client = None
    n = 0
    for sh in todo:
        sh['made_rf'], sh['made_source'] = sh['made'], 'roboflow'
        img = Path(pkl_path).parent / sh['made_grid']
        if client is None or not img.exists():
            continue
        try:
            v = vlm_made(img, client)
        except Exception as e:
            print(f'  [VLM 進球判斷失敗，用 Roboflow] {e}')
            continue
        if v.get('made') in (True, False):
            sh['made'], sh['made_source'], sh['made_reason'] = v['made'], 'vlm', v.get('reason', '')
            n += 1
    return n


def map_ids(shot_sam3, player_tracks):
    """投籃偵測那次 SAM3（10fps）的球員編號 → 擋拆偵測用的 player_tracks（5fps 內插回原片）編號：
    同一張原片影格 bbox IoU ≥ 0.5，整段多數決。對不到的保留原編號（會在戰術分析對不到位置 / 隊伍）。"""
    from collections import Counter
    votes = {}
    for i, f in enumerate(shot_sam3['frame_idx']):
        if f >= len(player_tracks):
            break
        ref = {p: (v.get('bbox') if isinstance(v, dict) else v) for p, v in player_tracks[f].items()}
        for pid, b in shot_sam3['players'][i].items():
            best = max(((so.iou(b, rb), q) for q, rb in ref.items() if rb is not None), default=(0, None))
            if best[0] >= 0.5:
                votes.setdefault(pid, Counter())[best[1]] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


def detect_shots(shot_sam3, frames, rf, player_tracks=None, grid_prefix=None, start_frame=0):
    """grid_prefix：給了就為每次出手存籃框放大圖 <prefix>_shot<i>.jpg（本機 tactics 用 VLM 判進球）。
    start_frame：片頭剪掉的保留點（run_sam3_tracks 的 start_frame）→ 之前的切點（前一段畫面接進來那刀）不算。"""
    cuts = [c for c in find_cuts(frames) if c > (start_frame or 0)]
    shots, tch = detect_core(shot_sam3, rf, cuts, frames[0].shape[:2], len(frames), player_tracks)
    if grid_prefix is not None:
        for i, sh in enumerate(shots):
            gp = Path(f'{grid_prefix}_shot{i}.jpg')
            try:
                made_grid(shot_sam3, tch, frames, rf, sh['t'], gp)
                sh['made_grid'] = gp.name                        # 跟 events.pkl 同資料夾
            except Exception as e:
                print(f'  [籃框圖存檔失敗] {e}')
    return shots, cuts


def ball_touches(shot_sam3, rf, shape):
    """選球 + 每張碰球的人（解遮罩，最慢的一步；shot_eval.py 存檔重用）。"""
    chosen, _ = so.pick_ball(shot_sam3, rf)
    return so.smooth_touches(so.touches(shot_sam3, shape, chosen))


def detect_core(shot_sam3, rf, cuts, shape, n_frames, player_tracks=None, tch=None, cand_fn=None):
    """detect_shots 不需要畫面的部分（shot_eval.py 用存好的 Roboflow 結果 + 切點 + 碰球重算）→ (shots, touches)。"""
    s = shot_sam3
    bounds = [0] + list(cuts) + [n_frames]
    tch = tch if tch is not None else ball_touches(s, rf, shape)
    if cand_fn is None:
        # 2026-11-10 換成第 2 版候選（shot_v2.py）：仲裁（1 秒內被別人接住、接住前球沒到籃框 = 傳球）+ 拿掉保底
        # （弱候選要 Roboflow 跳投 / 上籃）。真傳球被當投籃 13/56 → 1/56、延長 63 支全對 41 → 47、抽查 30 支假投籃 11 → 9、抓到率不變
        from shot_v2 import make_cand_fn
        cand_fn = make_cand_fn(None, cuts)
    fps_o = s['orig_fps']
    idmap = map_ids(s, player_tracks) if player_tracks is not None else {}
    shots = []
    for seg, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if b - a < fps_o * 1.0:                         # 太短的鏡頭（轉場 / 重播片頭）不看
            continue
        rf_events, _, _ = analyze(None, rf, cuts, rng=(a, b))      # Roboflow：跳投 / 上籃候選、進球
        for c in cand_fn(s, tch, rf, (a, b)):
            near = [e for e in rf_events if -0.6 <= e['t'] - c['release'] <= 1.2]
            e = min(near, key=lambda e: abs(e['t'] - c['release'])) if near else None
            made = any(x['made'] and -0.6 <= x['t'] - c['release'] <= 2.0 for x in rf_events)
            saw_rim_area = any(x[0] == IN_BASKET for f, dets in rf['frames']
                               if c['release'] * fps_o <= f <= (c['release'] + 2.0) * fps_o for x in dets) or c['rim'] is not None
            shots.append(dict(release_frame=int(round(c['release'] * fps_o)), t=c['release'],
                              shooter=idmap.get(c['shooter'], c['shooter']), shooter_raw=c['shooter'],
                              kind=e['kind'] if e else 'unknown',
                              made=True if made else (False if saw_rim_area else None),
                              strong=c['strong'], amp=c['amp'], rim=c['rim'], segment=seg))
    return shots, tch
