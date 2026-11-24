"""dho_skeleton 的單元測試（骨架特徵純函式 + 候選合併）。執行：PYTHONPATH=nba_data_pipeline python -m pytest tests/test_dho_skeleton.py -v"""
import math

import numpy as np

from dho_skeleton import skeleton_features


def kp(**pts):
    """只放指定關節（例 j9=(x, y) = 第 9 號左手腕），其他信心 0。"""
    a = np.zeros((17, 3))
    for j, (x, y) in pts.items():
        a[int(j[1:])] = (x, y, 0.9)
    return a


def test_hand_to_hand_normalized():
    f = skeleton_features([kp(j9=(0, 0))], [kp(j10=(10, 0))], [None], 100)
    assert math.isclose(f['hand_to_hand'], 0.1, abs_tol=1e-6)


def test_arm_points_to_receiver():
    a = kp(j7=(0, 0), j9=(10, 0), j5=(0, -5), j6=(4, -5))
    b = kp(j5=(50, -5), j6=(54, -5))
    assert skeleton_features([a], [b], [None], 100)['arm_angle'] < 5


def test_arm_points_away():
    a = kp(j7=(0, 0), j9=(-10, 0), j5=(0, -5), j6=(4, -5))
    b = kp(j5=(50, -5), j6=(54, -5))
    assert skeleton_features([a], [b], [None], 100)['arm_angle'] > 170


def test_ball_travel_normalized():
    f = skeleton_features([None] * 3, [None] * 3, [(0, 0), (25, 0), (50, 0)], 100)
    assert math.isclose(f['ball_travel'], 0.5, abs_tol=1e-6)


def test_turn_degrees():
    a0 = kp(j5=(0, 0), j6=(10, 0))
    a1 = kp(j5=(0, 0), j6=(0, 10))
    assert abs(skeleton_features([a0, a1], [None, None], [None, None], 100)['turn'] - 90) < 1


def test_missing_pose_is_nan_not_zero():
    f = skeleton_features([None] * 5, [None] * 5, [None] * 5, 100)
    assert f['has_pose'] == 0
    assert all(math.isnan(f[k]) for k in ('hand_to_hand', 'arm_angle', 'turn', 'both_hands'))
