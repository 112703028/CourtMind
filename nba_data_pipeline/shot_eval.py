"""
shot_eval.py — 投籃偵測評估（2026-11-09，修「傳球當投籃 / 出手者認錯」用）。

答案三組：
  ext63   output_videos/shot_check2/shot_gt.json（63 支延長片段，出手者 + 時間；回歸測試，原本 48 支全對）
  batch   output_videos/shot_check_batch/shot_check_batch.csv（476 支裡抽 30 支，使用者標偵測到的出手對不對 + 漏掉的）
  pass    傳球檢查表標「對」的傳球 → 那個時間傳球者不該被判出手（免費的假投籃答案）
Roboflow 每張的偵測很慢（本地 CPU ~45 秒 / 支）→ 先 cache 存 stubs/rf_cache/<片段>.pkl，改規則重算只要幾秒。

用法：python nba_data_pipeline/shot_eval.py cache ext63 batch pass   # 存 Roboflow 結果
     python nba_data_pipeline/shot_eval.py report                    # 正式版在三組答案上的成績
"""
import sys
import csv
import json
import pickle
import collections
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
RF_CACHE = ROOT / 'stubs' / 'rf_cache'


def clip_set(name):
    """→ [(片段名, 影片, shot_sam3 pkl)]"""
    if name == 'ext63':
        gt = json.load(open(ROOT / 'output_videos' / 'shot_check2' / 'shot_gt.json', encoding='utf-8'))
        return [(c, ROOT / 'input_videos' / 'screen_ext' / f'{c}.mp4', ROOT / 'stubs' / 'shot_sam3' / f'{c}_shot_sam3.pkl', 'ext')
                for c in gt]
    if name == 'batch':
        clips = json.load(open(ROOT / 'output_videos' / 'shot_check_batch' / 'clips.json'))
    elif name == 'pass':
        clips = set()
        for p, col in (('output_videos/framenum_boxes/pass_check_pose.csv', 6), ('output_videos/pass_check_other/pass_check_other.csv', 7)):
            for r in list(csv.reader(open(ROOT / p, encoding='utf-8-sig')))[1:]:
                if r[col].strip().lower().startswith('t'):
                    clips.add(r[1])
        clips = sorted(clips | {'screen_54', 'screen_136'}, key=lambda x: int(x.split('_')[1]))
    else:
        raise ValueError(name)
    return [(c, ROOT / 'input_videos' / 'screen' / f'{c}.mp4', ROOT / 'stubs' / 'ball_batch' / f'{c}_ball.pkl', 'batch') for c in clips]


def rf_cached(key, video, fps):
    """Roboflow 每張偵測（shot_detect.run_rf），存檔重用。key 要分 ext / batch（同名片段影片不同）。"""
    p = RF_CACHE / f'{key}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    from utils import read_video
    from shot_detect import run_rf
    rf = run_rf(read_video(str(video)), fps)
    RF_CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(rf, open(p, 'wb'))
    return rf


def cache(names):
    for name in names:
        for c, video, sp, src in clip_set(name):
            if (RF_CACHE / f'{src}_{c}.pkl').exists() or not video.exists() or not sp.exists():
                continue
            s = pickle.load(open(sp, 'rb'))
            s = s.get('shot_sam3', s)
            rf_cached(f'{src}_{c}', video, s['orig_fps'])
            print(f'{name} {c}', flush=True)
    print('CACHE_DONE')


def clip_meta(c, video, src):
    """(切點, 畫面大小, 總張數)：批次片段讀 events.pkl（backfill_shots 補過切點）；延長片段讀一次影片存檔。"""
    import cv2
    p = RF_CACHE / f'{src}_{c}_meta.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    if src == 'batch':
        d = pickle.load(open(ROOT / 'output_videos' / 'batch' / f'{c}_events.pkl', 'rb'))
        cap = cv2.VideoCapture(str(video)); ok, im = cap.read()
        meta = (list(d.get('cuts') or []), im.shape[:2], len(d['court_coords']))
    else:
        from utils import read_video
        from shot_events import find_cuts
        frames = read_video(str(video))
        meta = (find_cuts(frames), frames[0].shape[:2], len(frames))
    pickle.dump(meta, open(p, 'wb'))
    return meta


def touches_cached(key, s, rf, shape):
    p = RF_CACHE / f'{key}_tch.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    from shot_detect import ball_touches
    tch = ball_touches(s, rf, shape)
    pickle.dump(tch, open(p, 'wb'))
    return tch


def load_pose(c, src):
    p = ROOT / 'stubs' / ('pose_hands' if src == 'batch' else 'pose_hands_ext') / f'{c}.pkl'
    return pickle.load(open(p, 'rb')) if p.exists() else None


def run_detect(name, detect=None):
    """一組片段全部重算投籃 → {片段: shots}。
    detect(s, rf, cuts, shape, n, tracks, tch, pose) 預設 = 正式版 detect_core（不用 pose）。"""
    from shot_detect import detect_core
    detect = detect or (lambda s, rf, cuts, shape, n, tracks, tch, pose: detect_core(s, rf, cuts, shape, n, tracks, tch)[0])
    out = {}
    for c, video, sp, src in clip_set(name):
        if not (RF_CACHE / f'{src}_{c}.pkl').exists():
            continue
        s = pickle.load(open(sp, 'rb'))
        s = s.get('shot_sam3', s)
        rf = pickle.load(open(RF_CACHE / f'{src}_{c}.pkl', 'rb'))
        cuts, shape, n = clip_meta(c, video, src)
        tracks = pickle.load(open(ROOT / 'output_videos' / 'batch' / f'{c}_events.pkl', 'rb'))['player_tracks'] if src == 'batch' else None
        tch = touches_cached(f'{src}_{c}', s, rf, shape)
        out[c] = detect(s, rf, cuts, shape, n, tracks, tch, load_pose(c, src))
    return out


def score_ext63(shots):
    """延長片段 63 支：出手者（SAM3 編號）對 + 時間 ±1 秒 = 抓對；全對 = 抓對而且沒有多的出手。"""
    gt = json.load(open(ROOT / 'output_videos' / 'shot_check2' / 'shot_gt.json', encoding='utf-8'))
    hit = allok = extra = 0
    for c, g in gt.items():
        ss = shots.get(c, [])
        ok = [x for x in ss if x['shooter_raw'] == g['shooter'] and (g['t'] is None or abs(x['t'] - g['t']) <= 1.0)]   # 21 支答案只有出手者
        hit += bool(ok); extra += len(ss) - bool(ok); allok += bool(ok) and len(ss) == 1
    return dict(n=len(gt), hit=hit, all_ok=allok, extra=extra)


def pass_labels():
    """使用者確認的真傳球 → [(片段, 離手影格, 接球影格, 傳球者)]：這時傳球者被判出手 = 把傳球當投籃。"""
    from pass_gt_probe import GT
    out = []
    for p, col in (('output_videos/framenum_boxes/pass_check_pose.csv', 6), ('output_videos/pass_check_other/pass_check_other.csv', 7)):
        for r in list(csv.reader(open(ROOT / p, encoding='utf-8-sig')))[1:]:
            if r[col].strip().lower().startswith('t'):
                out.append((r[1], int(r[2]), int(r[3]), int(r[4][1:])))
    out += [(f'screen_{c}', min(f0, f1), max(f0, f1), A) for c, f0, f1, A, B in GT]
    return out


def score_pass(shots):
    """真傳球被判成傳球者出手（離手前 0.3 秒 ~ 接球）的有幾個。"""
    bad, n = [], 0
    for c, rel, cat, A in pass_labels():
        if c not in shots:
            continue
        n += 1
        fps = pickle.load(open(ROOT / 'output_videos' / 'batch' / f'{c}_events.pkl', 'rb'))['fps']
        if any(x['shooter'] == A and rel - 0.3 * fps <= x['release_frame'] <= cat for x in shots[c]):
            bad.append((c, rel, A))
    return dict(n=n, pass_as_shot=len(bad), cases=bad)


def score_batch(shots):
    """抽查 30 支（shot_labels.json：使用者標的真投籃 [影格, 出手者] 含漏掉的）：偵測到的出手跟真投籃 ±0.5 秒對上 = 抓對。"""
    lab = json.load(open(ROOT / 'output_videos' / 'shot_check_batch' / 'shot_labels.json'))
    tp = fp = fn = who = 0
    for c, g in lab.items():
        fps = pickle.load(open(ROOT / 'output_videos' / 'batch' / f'{c}_events.pkl', 'rb'))['fps']
        ss, used = shots.get(c, []), set()
        for x in ss:
            m = [k for k, (f, _) in enumerate(g['true']) if k not in used and abs(f - x['release_frame']) <= 0.5 * fps]
            if m:
                used.add(m[0]); tp += 1; who += x['shooter'] == g['true'][m[0]][1]
            else:
                fp += 1
        fn += len(g['true']) - len(used)
    return dict(真投籃=tp + fn, 抓到=tp, 出手者對=who, 假投籃=fp, 漏掉=fn,
                準確率=f'{tp / max(tp + fp, 1):.0%}', 抓到率=f'{tp / max(tp + fn, 1):.0%}')


def report(detect=None, label='正式版'):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        e = run_detect('ext63', detect)
        b = run_detect('batch', detect)
        p = run_detect('pass', detect)
    print(f'[{label}] 延長 63 支：{score_ext63(e)}')
    sp = score_pass(p)
    print(f'[{label}] 真傳球被當投籃：{sp["pass_as_shot"]}/{sp["n"]}  {sp["cases"]}')
    if (ROOT / 'output_videos' / 'shot_check_batch' / 'shot_labels.json').exists():
        try:
            print(f'[{label}] 抽查 30 支：{score_batch(b)}')
        except Exception as ex:
            print(f'[{label}] 抽查表還沒標好（{ex}）')


def vlm_eval(sets=('batch', 'ext63')):
    """B1：正式版的每次出手 → 籃框放大圖 → VLM 三選一（是不是投籃 + 進沒進）→ stubs/rf_cache/vlm_shot.json。"""
    import io, contextlib
    from utils import read_video
    from shot_detect import made_grid, vlm_made, VLM_SYSTEM_SHOT
    from coverage_model import _llm_client
    client = _llm_client()
    gd = RF_CACHE / 'grids'
    gd.mkdir(parents=True, exist_ok=True)
    outp = RF_CACHE / 'vlm_shot.json'
    res = json.load(open(outp, encoding='utf-8')) if outp.exists() else {}
    for name in sets:
        with contextlib.redirect_stdout(io.StringIO()):
            shots = run_detect(name)
        for c, video, sp, src in clip_set(name):
            ss = shots.get(c) or []
            todo = [x for x in ss if f'{src}_{c}_{x["release_frame"]}' not in res]
            if not todo:
                continue
            s = pickle.load(open(sp, 'rb'))
            s = s.get('shot_sam3', s)
            rf = pickle.load(open(RF_CACHE / f'{src}_{c}.pkl', 'rb'))
            tch = pickle.load(open(RF_CACHE / f'{src}_{c}_tch.pkl', 'rb'))
            frames = read_video(str(video))
            for x in todo:
                key = f'{src}_{c}_{x["release_frame"]}'
                img = gd / f'{key}.jpg'
                if not img.exists():
                    made_grid(s, tch, frames, rf, x['t'], img, win=1.5)
                res[key] = vlm_made(img, client, system=VLM_SYSTEM_SHOT, ask='這是不是一次投籃？有沒有進？只輸出 JSON。')
                json.dump(res, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                print(key, res[key], flush=True)
    print('VLM_DONE')


if __name__ == '__main__':
    if sys.argv[1] == 'cache':
        cache(sys.argv[2:])
    elif sys.argv[1] == 'report':
        report()
    elif sys.argv[1] == 'vlm':
        vlm_eval()
