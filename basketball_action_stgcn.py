import cv2
import numpy as np
from ultralytics import YOLO
from mmaction.apis import init_recognizer, inference_recognizer

from trackers.player_tracker import PlayerTracker 
from trackers.ball_tracker import BallTracker
from ball_aquisition import BallAquisitionDetector


# ---------------- 初始化模型 ----------------

pose_model = YOLO("yolov8n-pose.pt")

stgcn_model = init_recognizer(
    "mmaction2/configs/skeleton/stgcn/stgcn_basketball.py",
    "mmaction2/work_dirs/stgcn_basketball/best_acc_top1_epoch_78.pth",
    device="cpu"
)

player_tracker = PlayerTracker("models/player_detector.pt")
ball_tracker = BallTracker("models/ball_detector_model.pt")

ball_acquisition_detector = BallAquisitionDetector()


# ---------------- 影片設定 ----------------

cap = cv2.VideoCapture("input_videos/video_2.mp4")

width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(
    "output_result.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (width, height)
)


# ---------------- STGCN設定 ----------------

window = 4

player_sequences = {}

action_labels = [
    "Dribble",
    "Layup",
    "Shoot"
]


# ---------------- 讀取所有 frames ----------------

frames = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frames.append(frame)


# ---------------- Player Tracking ----------------

player_tracks = player_tracker.get_object_tracks(
    frames,
    read_from_stub=False,
    stub_path="stubs/player_tracks_stub.pkl"
)


# ---------------- Ball Tracking ----------------

ball_tracks = ball_tracker.get_object_tracks(
    frames,
    read_from_stub=False,
    stub_path="stubs/ball_tracks_stub.pkl"
)

ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)


# ---------------- Ball Possession ----------------

ball_acquisition = ball_acquisition_detector.detect_ball_possession(
    player_tracks,
    ball_tracks
)


# ---------------- 主迴圈 ----------------

for frame_num, frame in enumerate(frames):

    tracks = player_tracks[frame_num]

    # ---------------- 取得持球者 ----------------

    ball_player = ball_acquisition[frame_num]

    if ball_player not in tracks:

        out.write(frame)
        cv2.imshow("result", frame)

        if cv2.waitKey(1) == 27:
            break

        continue


    info = tracks[ball_player]

    track_id = ball_player

    print("ball holder:", track_id)

    if track_id in player_sequences:
        print("sequence:", len(player_sequences[track_id]))

    x1, y1, x2, y2 = map(int, info["bbox"])


    # ★ FIX 1: 避免 ROI 超出畫面
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    roi = frame[y1:y2, x1:x2]

    # ★ FIX 2: 避免空 ROI
    if roi.size == 0:
        continue

    # ---------------- Pose Detection ----------------

    pose_res = pose_model(roi, verbose=False)[0]

    if len(pose_res.keypoints.xy) == 0:
        if track_id in player_sequences and len(player_sequences[track_id]) > 0:
            keypoints = player_sequences[track_id][-1]  # 用上一幀 skeleton
        else:
            continue
    else:
        keypoints = pose_res.keypoints.xy[0].cpu().numpy()


    # ★ FIX 3: ROI 座標轉回全圖座標 (非常重要)
    keypoints[:, 0] += x1
    keypoints[:, 1] += y1


    # ---------------- Skeleton Buffer ----------------

    if track_id not in player_sequences:

        player_sequences[track_id] = []

    player_sequences[track_id].append(keypoints)

    # ★ FIX 4: 只保留 window 長度
    if len(player_sequences[track_id]) > window:
        player_sequences[track_id] = player_sequences[track_id][-window:]


    # ---------------- STGCN Action Recognition ----------------

    if len(player_sequences[track_id]) == window:

        skeleton = np.array(player_sequences[track_id])  # (30,17,2)

        data = dict(
            frame_dir='',
            label=-1,
            img_shape=(height, width),
            original_shape=(height, width),
            total_frames=window,
            keypoint=skeleton[None],                # (1,30,17,2)
            keypoint_score=np.ones((1, window, 17))
        )

        result = inference_recognizer(stgcn_model, data)

        scores = result.pred_score.cpu().numpy()

        action_id = scores.argmax()

        confidence = scores[action_id]

        action_name = action_labels[action_id]


        # ---------------- 畫結果 ----------------

        cv2.putText(
            frame,
            f"BallHolder ID:{track_id} {action_name} ({confidence:.2f})",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    else:

        cv2.putText(
            frame,
            f"BallHolder ID:{track_id} collecting...",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )


    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)


    # ---------------- Output ----------------

    out.write(frame)

    cv2.imshow("result", frame)

    if cv2.waitKey(1) == 27:
        break


# ---------------- 結束 ----------------

cap.release()
out.release()
cv2.destroyAllWindows()