"""
A module for reading and writing video files.

優先用 cv2（快），如果 cv2 沒 FFMPEG 支援（某些 docker container 編譯版本不含），
自動 fallback 到 imageio（需要 imageio[ffmpeg] 或 imageio[pyav]）。

save_video 也一樣：cv2.VideoWriter 失敗就用 imageio。
"""

import os
import cv2
import numpy as np


def _cv2_can_open(video_path):
    cap = cv2.VideoCapture(video_path)
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    cap.release()
    return ok


def _read_video_cv2(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _read_video_imageio(video_path):
    """imageio 讀的是 RGB，OpenCV 用 BGR，要轉。"""
    try:
        import imageio.v3 as iio
    except ImportError:
        raise RuntimeError(
            "OpenCV 沒 FFMPEG 支援，需要 imageio：pip install 'imageio[ffmpeg]'")
    frames = []
    for frame in iio.imiter(video_path, plugin='pyav'):
        frames.append(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    return frames


def read_video(video_path):
    """讀影片返回 BGR numpy frame 的 list。"""
    if _cv2_can_open(video_path):
        frames = _read_video_cv2(video_path)
        # AV1 等 codec：cv2 可以「打開」但實際讀不到 frame → 也要 fallback
        if len(frames) > 0:
            return frames
        print(f"  [video_utils] cv2 開了 {video_path} 但 0 frames，改用 imageio")
    else:
        print(f"  [video_utils] cv2 無法讀 {video_path}，改用 imageio")
    return _read_video_imageio(video_path)


def _save_video_cv2(output_video_frames, output_video_path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h, w = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    if not out.isOpened():
        return False
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    if os.path.getsize(output_video_path) == 0:
        return False
    return True


def _save_video_imageio(output_video_frames, output_video_path, fps=24):
    try:
        import imageio.v3 as iio
    except ImportError:
        raise RuntimeError(
            "OpenCV 沒 FFMPEG 支援，需要 imageio：pip install 'imageio[ffmpeg]'")
    # imageio 預期 RGB，cv2 是 BGR，要轉
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in output_video_frames]
    # imageio 對 .avi 可能不支援，存成 .mp4 比較通用
    out_path = output_video_path
    if out_path.lower().endswith('.avi'):
        out_path = os.path.splitext(out_path)[0] + '.mp4'
        print(f"  [video_utils] imageio 不支援 .avi，改存成 {out_path}")
    iio.imwrite(out_path, np.stack(rgb_frames), fps=fps, plugin='pyav', codec='h264')


def save_video(output_video_frames, output_video_path, fps=24):
    if not output_video_frames:
        print("  [video_utils] 沒有 frame 可儲存")
        return

    out_dir = os.path.dirname(output_video_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if _save_video_cv2(output_video_frames, output_video_path, fps=fps):
        return
    print(f"  [video_utils] cv2 VideoWriter 失敗，改用 imageio")
    _save_video_imageio(output_video_frames, output_video_path, fps=fps)
