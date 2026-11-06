"""
camera_calib.py — 3D 第 1 步：從球場 homography 解出完整鏡頭（焦距、旋轉、位置），可以把「有高度的點」投到畫面。

homography（court_v2.homography）是「畫面 → 戰術板像素（300×161）」。換成真實公尺（NBA 28.65 × 15.24 m）後反過來
G = 「地面（X, Y, 0）→ 畫面」。假設像素正方形、主點在畫面中心（轉播鏡頭的標準假設），K = [[f,0,cx],[0,f,cy],[0,0,1]]：
  K⁻¹G = λ[r1 r2 t]，r1 ⟂ r2、|r1| = |r2| 兩個條件 → 解 1/f²（最小平方）→ r1, r2, t → r3 = r1 × r2（Z 朝上：鏡頭在地板上方）。
投影：x = K(R·X + t)，X = (X, Y, Z) 公尺，Z = 離地高度。

驗證（main）：籃框中心（離底線 1.6 m、場寬中間、離地 3.05 m）投回畫面 vs Roboflow 偵測的籃框；
對照組：籃框正下方的地板點（Z = 0）——如果鏡頭解對了，3.05 m 的投影應該明顯比地板點更接近偵測到的籃框。

用法：python nba_data_pipeline/camera_calib.py
"""
import sys
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

COURT_L, COURT_W = 28.65, 15.24          # NBA 94 × 50 ft（公尺）
TAC_W, TAC_H = 300.0, 161.0              # convert_manual.TARGET_KPS 的戰術板像素
RIM_H, RIM_X = 3.05, 1.60                # 籃框離地、籃框中心離底線（公尺）
S = np.diag([TAC_W / COURT_L, TAC_H / COURT_W, 1.0])   # 公尺 → 戰術板像素


def calibrate(H_img2tac, w, h):
    """→ dict(K, R, t, f, C)（C = 鏡頭位置，公尺）或 None。"""
    if H_img2tac is None:
        return None
    G = np.linalg.inv(H_img2tac) @ S                   # 地面（公尺）→ 畫面
    cx, cy = w / 2.0, h / 2.0
    A = np.linalg.inv(np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])) @ G
    a = A
    p1, q1 = a[0, 0] * a[0, 1] + a[1, 0] * a[1, 1], a[2, 0] * a[2, 1]
    p2, q2 = a[0, 0] ** 2 + a[1, 0] ** 2 - a[0, 1] ** 2 - a[1, 1] ** 2, a[2, 0] ** 2 - a[2, 1] ** 2
    den = p1 * p1 + p2 * p2
    if den <= 0:
        return None
    u = -(p1 * q1 + p2 * q2) / den                     # u = 1 / f²
    if u <= 0:
        return None
    f = 1.0 / np.sqrt(u)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    B = np.linalg.inv(K) @ G
    lam = (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1])) / 2
    r1, r2, t = B[:, 0] / lam, B[:, 1] / lam, B[:, 2] / lam
    if t[2] < 0:                                       # 球場要在鏡頭前面
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    R = U @ Vt
    C = -R.T @ t
    if C[2] < 0:                                       # Z 朝上：鏡頭要在地板上方
        R[:, 2] = -R[:, 2]
        C = -R.T @ t
    return dict(K=K, R=R, t=t, f=f, C=C)


def project(cam, X):
    """3D 點（公尺，Z = 離地）→ 畫面像素 (u, v)。"""
    x = cam['K'] @ (cam['R'] @ np.asarray(X, float) + cam['t'])
    return x[:2] / x[2]


def kp3d():
    """18 個球場關鍵點的 3D 位置（公尺，地板 Z = 0）。"""
    import convert_manual as cm
    return np.array([[x / TAC_W * COURT_L, y / TAC_H * COURT_W, 0.0] for x, y in cm.TARGET_KPS])


def _rot(rv):
    import cv2
    return cv2.Rodrigues(np.asarray(rv, float))[0]


def _proj(f, rv, C, X, cx, cy):
    R = _rot(rv)
    Xc = (np.asarray(X, float) - C) @ R.T                # 鏡頭座標
    return np.column_stack([f * Xc[:, 0] / Xc[:, 2] + cx, f * Xc[:, 1] / Xc[:, 2] + cy])


def refine_clip(obs, w, h, rim_obs=None, shared_C=False, w_rim=3.0, s_f=0.03, s_r=0.01, s_c=0.3):
    """一支片段所有影格一起解鏡頭（非線性最小平方）：
      關鍵點重投影誤差（像素 / 畫面寬 × 100）+ 籃框（已知 3D 點）重投影誤差 × w_rim（rim_obs 有給的影格才算）
      + 相鄰影格平滑：焦距相對變化 / s_f、旋轉變化 / s_r（弧度）、鏡頭位置變化 / s_c（公尺）。
    shared_C = True：整支片段同一個鏡頭位置（轉播鏡頭架在固定位置，只轉 / 變焦）。
    obs：list（每張）(cam0 或 None, [(關鍵點編號, u, v)], left)；rim_obs：{影格: (u, v)}。回 list（每張）cam 或 None。"""
    import cv2
    from scipy.optimize import least_squares
    n = len(obs)
    idx = [i for i in range(n) if obs[i][0] is not None]
    if len(idx) < 2:
        return [None] * n
    cx, cy = w / 2.0, h / 2.0
    K3 = kp3d()
    init = {}
    for i in range(n):                                   # 解不出來的影格用最近的
        j = min(idx, key=lambda k: abs(k - i))
        c0 = obs[j][0]
        init[i] = (np.log(c0['f']), cv2.Rodrigues(c0['R'])[0].ravel(), c0['C'])
    Cmed = np.median([init[i][2] for i in range(n)], axis=0)
    def pack():
        v = []
        for i in range(n):
            v += [init[i][0], *init[i][1]] + ([] if shared_C else list(init[i][2]))
        return np.array(v + (list(Cmed) if shared_C else []))
    per = 4 if shared_C else 7
    def unpack(v, i):
        lf, rv = v[i * per], v[i * per + 1:i * per + 4]
        C = v[n * per:n * per + 3] if shared_C else v[i * per + 4:i * per + 7]
        return np.exp(lf), rv, C
    sc = 100.0 / w
    def resid(v):
        r = []
        for i in range(n):
            f, rv, C = unpack(v, i)
            pts = obs[i][1]
            if pts:
                P = _proj(f, rv, C, K3[[k for k, _, _ in pts]], cx, cy)
                r += list(((P - np.array([[u, vv] for _, u, vv in pts])) * sc).ravel())
            if rim_obs and i in rim_obs:
                P = _proj(f, rv, C, [rim_point(obs[i][2])], cx, cy)[0]
                r += list((P - np.array(rim_obs[i])) * sc * w_rim)
            if i + 1 < n:
                f2, rv2, C2 = unpack(v, i + 1)
                r += [(np.log(f2) - np.log(f)) / s_f, *((rv2 - rv) / s_r)]
                if not shared_C:
                    r += list((C2 - C) / s_c)
        return np.array(r)
    # 稀疏結構：每張的重投影誤差只跟自己的參數有關、平滑項只跟相鄰兩張 → 告訴最佳化器，數值微分快幾十倍
    from scipy.sparse import lil_matrix
    nv = len(pack())
    rows = []
    def cols(i):
        c = list(range(i * per, i * per + per))
        return c + (list(range(n * per, n * per + 3)) if shared_C else [])
    for i in range(n):
        m = 2 * len(obs[i][1]) + (2 if rim_obs and i in rim_obs else 0)
        rows += [cols(i)] * m
        if i + 1 < n:
            rows += [cols(i) + cols(i + 1)] * (4 + (0 if shared_C else 3))
    J = lil_matrix((len(rows), nv), dtype=int)
    for r, cs in enumerate(rows):
        J[r, cs] = 1
    sol = least_squares(resid, pack(), loss='soft_l1', f_scale=1.0, max_nfev=100, jac_sparsity=J, x_scale='jac')
    out = []
    for i in range(n):
        f, rv, C = unpack(sol.x, i)
        R = _rot(rv)
        out.append(dict(K=np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), R=R, t=-R @ C, f=f, C=C))
    return out


def ray_at(cam, uv, xy):
    """從鏡頭穿過畫面像素 uv 的射線，經過地面點 xy（公尺）那條鉛直線時：(水平距離 m, 那裡的離地高度 m)。
    不用知道球的深度：球一定在這條射線上 → 射線在某球員位置的高度 = 「如果球在他那裡，球離地多高」。"""
    d = cam['R'].T @ np.linalg.inv(cam['K']) @ np.array([uv[0], uv[1], 1.0])
    C = cam['C']
    dxy = d[:2]
    s_ = float(np.dot(np.asarray(xy) - C[:2], dxy) / max(np.dot(dxy, dxy), 1e-9))
    Q = C + s_ * d
    return float(np.hypot(*(Q[:2] - np.asarray(xy)))), float(Q[2])


def floor_point(cam, uv):
    """畫面像素 → 射線打到地板（Z = 0）的位置（公尺）；打不到（往上看）回 None。"""
    d = cam['R'].T @ np.linalg.inv(cam['K']) @ np.array([uv[0], uv[1], 1.0])
    if d[2] >= -1e-9:
        return None
    s_ = -cam['C'][2] / d[2]
    Q = cam['C'] + s_ * d
    return float(Q[0]), float(Q[1])


def clip_cams(clip, cache_dir=ROOT / 'stubs' / 'cam_cache'):
    """訓練片段（stubs/court_kp，5fps）→ 每張 5fps 的鏡頭（時間平滑）；快取。"""
    import court_v2
    p = Path(cache_dir) / f'{clip}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    kp = pickle.load(open(ROOT / 'stubs' / 'court_kp' / f'{clip}.pkl', 'rb'))
    w, h = kp['size']
    obs = []
    for i in range(len(kp['xy'])):
        xy, conf = kp['xy'][i], kp['conf'][i]
        left = conf[court_v2.LEFT].sum() >= conf[court_v2.RIGHT].sum()
        side = court_v2.LEFT if left else court_v2.RIGHT
        pts = [(k, float(xy[k, 0]), float(xy[k, 1])) for k in side + court_v2.MID
               if conf[k] >= court_v2.THR and 0 <= xy[k, 0] < w and 0 <= xy[k, 1] < h]
        obs.append((calibrate(court_v2.homography(xy, conf, w, h), w, h), pts, left))
    cams = refine_clip(obs, w, h)
    p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cams, open(p, 'wb'))
    return cams


def rim_point(left, z=RIM_H):
    return (RIM_X if left else COURT_L - RIM_X, COURT_W / 2, z)


def main():
    import glob
    import court_v2
    rf_all = {}
    for p in ('rf_dets.pkl', 'rf_dets_more.pkl', 'rf_dets_all.pkl'):
        q = ROOT / 'output_videos' / 'shot_check' / p
        if q.exists():
            rf_all.update(pickle.load(open(q, 'rb')))
    errs_rim, errs_floor, fs, Cs, n_ok, n_try = [], [], [], [], 0, 0
    for clip, rf in sorted(rf_all.items()):
        kpp = ROOT / 'stubs' / 'court_kp' / f'{clip}.pkl'
        cache = ROOT / 'stubs' / 'train_contact_cache' / f'{clip}.pkl'
        if not kpp.exists() or not cache.exists():
            continue
        kp = pickle.load(open(kpp, 'rb'))
        fps = pickle.load(open(cache, 'rb'))['fps']
        w, h = kp['size']
        at = dict(rf['frames'])
        for i in range(len(kp['xy'])):
            xy, conf = kp['xy'][i], kp['conf'][i]
            H = court_v2.homography(xy, conf, w, h)
            f_orig = int(round(i * fps / 5.0))
            dets = next((at[g] for g in (f_orig, f_orig - 1, f_orig + 1, f_orig - 2, f_orig + 2) if g in at), None)
            rims = [d for d in (dets or []) if d[0] == 9 and d[1] >= 0.5]
            if H is None or not rims:
                continue
            n_try += 1
            cam = calibrate(H, w, h)
            if cam is None:
                continue
            n_ok += 1
            left = conf[court_v2.LEFT].sum() >= conf[court_v2.RIGHT].sum()   # 跟 homography 用的同一邊
            b = max(rims, key=lambda d: d[1])[2]
            rc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); rw = b[2] - b[0]
            errs_rim.append(np.hypot(*(project(cam, rim_point(left)) - rc)) / rw)
            errs_floor.append(np.hypot(*(project(cam, rim_point(left, 0.0)) - rc)) / rw)
            fs.append(cam['f']); Cs.append(cam['C'])
    er, ef = np.array(errs_rim), np.array(errs_floor)
    Cs = np.array(Cs)
    print(f'有籃框偵測的 5fps 影格 {n_try} 張，鏡頭解得出來 {n_ok} 張（{n_ok / max(1, n_try):.0%}）')
    print(f'籃框（離地 3.05 m）投影誤差：中位數 {np.median(er):.2f} 倍籃框寬，< 0.5 倍 {np.mean(er < 0.5):.0%}、< 1 倍 {np.mean(er < 1):.0%}')
    print(f'對照：籃框正下方地板點        中位數 {np.median(ef):.2f} 倍籃框寬，< 0.5 倍 {np.mean(ef < 0.5):.0%}、< 1 倍 {np.mean(ef < 1):.0%}')
    print(f'焦距中位數 {np.median(fs):.0f} px；鏡頭高度中位數 {np.median(Cs[:, 2]):.1f} m（10%~90% {np.percentile(Cs[:, 2], 10):.1f}~{np.percentile(Cs[:, 2], 90):.1f}）；'
          f'離球場中心 {np.median(np.hypot(Cs[:, 0] - COURT_L / 2, Cs[:, 1] - COURT_W / 2)):.1f} m')


def refine_eval():
    """比較：只用 homography / 時間平滑 / + 偶數張籃框 / + 鏡頭位置固定；全部在「奇數張」（沒拿去校正）的籃框上量誤差。"""
    import court_v2
    rf_all = {}
    for p in ('rf_dets.pkl', 'rf_dets_more.pkl', 'rf_dets_all.pkl'):
        q = ROOT / 'output_videos' / 'shot_check' / p
        if q.exists():
            rf_all.update(pickle.load(open(q, 'rb')))
    E = {k: [] for k in ('只用 homography', '時間平滑', '時間平滑 + 偶數張籃框', '時間平滑 + 籃框 + 鏡頭位置固定')}
    for clip, rf in sorted(rf_all.items()):
        kpp = ROOT / 'stubs' / 'court_kp' / f'{clip}.pkl'
        cache = ROOT / 'stubs' / 'train_contact_cache' / f'{clip}.pkl'
        if not kpp.exists() or not cache.exists():
            continue
        kp = pickle.load(open(kpp, 'rb')); fps = pickle.load(open(cache, 'rb'))['fps']
        w, h = kp['size']; at = dict(rf['frames'])
        obs, rims = [], {}
        for i in range(len(kp['xy'])):
            xy, conf = kp['xy'][i], kp['conf'][i]
            left = conf[court_v2.LEFT].sum() >= conf[court_v2.RIGHT].sum()
            side = court_v2.LEFT if left else court_v2.RIGHT
            pts = [(k, float(xy[k, 0]), float(xy[k, 1])) for k in side + court_v2.MID
                   if conf[k] >= court_v2.THR and 0 <= xy[k, 0] < w and 0 <= xy[k, 1] < h]
            obs.append((calibrate(court_v2.homography(xy, conf, w, h), w, h), pts, left))
            f_o = int(round(i * fps / 5.0))
            dets = next((at[g] for g in (f_o, f_o - 1, f_o + 1, f_o - 2, f_o + 2) if g in at), None)
            r = [d for d in (dets or []) if d[0] == 9 and d[1] >= 0.5]
            if r:
                b = max(r, key=lambda d: d[1])[2]
                rims[i] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0])
        if not rims or all(o[0] is None for o in obs):
            continue
        even = {i: v[:2] for i, v in rims.items() if i % 2 == 0}
        variants = {'只用 homography': [o[0] for o in obs],
                    '時間平滑': refine_clip(obs, w, h),
                    '時間平滑 + 偶數張籃框': refine_clip(obs, w, h, even),
                    '時間平滑 + 籃框 + 鏡頭位置固定': refine_clip(obs, w, h, even, shared_C=True)}
        print(f'  {clip} 完成', flush=True)
        for name, cams in variants.items():
            for i, (u, v, rw) in rims.items():
                if i % 2 == 1 and cams[i] is not None:
                    E[name].append(np.hypot(*(project(cams[i], rim_point(obs[i][2])) - (u, v))) / rw)
    for name, e in E.items():
        e = np.array(e)
        print(f'{name:22s} 奇數張籃框誤差 中位數 {np.median(e):.2f} 倍籃框寬，< 0.5 倍 {np.mean(e < 0.5):.0%}、< 1 倍 {np.mean(e < 1):.0%}（{len(e)} 張）')


if __name__ == '__main__':
    if sys.argv[1:2] == ['refine']:
        refine_eval()
    else:
        main()
