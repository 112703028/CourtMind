import cv2
import numpy as np
import torch
from mmaction.apis import init_recognizer, inference_recognizer

# 假設這些是你原本的自定義模組
from trackers.player_tracker import PlayerTracker 
from trackers.ball_tracker import BallTracker
from ball_aquisition import BallAquisitionDetector

# ---------------- 1. 初始化模型 ----------------

# 載入你剛練好的 ACANet (94.6% 準確率版本)
config_file = "mmaction2/configs/recognition/tsn/acanet.py"
checkpoint_file = "mmaction2/work_dirs/acanet/best_acc_top1_epoch_9.pth"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

acanet_model = init_recognizer(config_file, checkpoint_file, device=device)
acanet_model.cfg.test_pipeline = [
    t for t in acanet_model.cfg.test_pipeline 
    if t['type'] not in ['DecordInit', 'SampleFrames', 'DecordDecode', 'OpenCVInit', 'OpenCVDecode']
]

# 初始化 Tracker 與 持球偵測器
player_tracker = PlayerTracker("models/player_detector.pt")
ball_tracker = BallTracker("models/ball_detector_model.pt")
ball_acquisition_detector = BallAquisitionDetector()

# ---------------- 2. 影片設定 ----------------

cap = cv2.VideoCapture("input_videos/shoot_7.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(
    "output_acanet_result7.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (width, height)
)

# ---------------- 3. ACANet 參數設定 ----------------

# Window 代表一次推理需要參考幾幀影像 (建議設為 8 或 16，視你的訓練設定而定)
window_size = 8
# 用來存放不同玩家影像序列的字典
player_video_buffer = {}

action_labels = ["Dribble", "Layup", "Shoot"]

# ---------------- 4. 預處理：讀取影片與 Tracking ----------------

frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)

print(f"Total frames: {len(frames)}, Start Tracking...")

# 執行玩家與球的追蹤
player_tracks = player_tracker.get_object_tracks(frames, read_from_stub=False, stub_path="stubs/player_tracks_stub.pkl")
ball_tracks = ball_tracker.get_object_tracks(frames, read_from_stub=False, stub_path="stubs/ball_tracks_stub.pkl")
ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

# 偵測每一幀是誰持球
ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

# ---------------- 5. 主迴圈：影像剪裁與推理 ----------------

for frame_num, frame in enumerate(frames):
    display_frame = frame.copy()
    tracks = player_tracks[frame_num]
    ball_player_id = ball_acquisition[frame_num]

    # 如果這幀沒人持球，直接寫入並跳過
    if ball_player_id not in tracks:
        out.write(display_frame)
        cv2.imshow("ACANet Result", display_frame)
        if cv2.waitKey(1) == 27: break
        continue

    # 取得持球者的 BBox
    info = tracks[ball_player_id]
    x1, y1, x2, y2 = map(int, info["bbox"])

    # --- 影像處理：剪裁持球者的 ROI ---
    # 邊界檢查防止超出畫面
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    roi = frame[y1:y2, x1:x2]
    
    if roi.size != 0:
        # 1. 轉為 RGB (MMAction2 標準)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # 2. 縮放到模型輸入大小 (ACANet 通常是 224x224)
        roi_resized = cv2.resize(roi_rgb, (224, 224))
        
        # 3. 放入該玩家的 Buffer
        if ball_player_id not in player_video_buffer:
            player_video_buffer[ball_player_id] = []
        
        player_video_buffer[ball_player_id].append(roi_resized)
        
        # 限制 Buffer 長度
        if len(player_video_buffer[ball_player_id]) > window_size:
            player_video_buffer[ball_player_id].pop(0)

    # --- ACANet 推理 ---
    if ball_player_id in player_video_buffer and len(player_video_buffer[ball_player_id]) == window_size:
        imgs_8 = np.stack(player_video_buffer[ball_player_id])
    
        # 2. 暴力複製：[1, 2, 3...] 變成 [1, 1, 2, 2, 3, 3...]
        # 這樣 Shape 會變成 (16, 224, 224, 3)，模型就不會噴 Shape Mismatch 了
        video_input = np.repeat(imgs_8, 2, axis=0)


        # 封裝成 MMAction2 要求的字典格式
        data = dict(
            imgs=video_input,
            label=-1,
            modality='RGB',
            sample_interval=1
        )
        
        # 執行模型推理
        result = inference_recognizer(acanet_model, data)
        
        # 取得最高分與動作標籤
        scores = result.pred_score.cpu().numpy()
        action_id = scores.argmax()
        confidence = scores[action_id]
        action_name = action_labels[action_id]

        # 畫出結果
        color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
        text = f"ID:{ball_player_id} {action_name} ({confidence:.2f})"
        cv2.putText(display_frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
    else:
        # Buffer 還沒滿，顯示收集資料中
        cv2.putText(display_frame, f"ID:{ball_player_id} Buffering...", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # 寫入影片與顯示
    out.write(display_frame)
    cv2.imshow("ACANet Result", display_frame)
    if cv2.waitKey(1) == 27: break

# ---------------- 6. 資源釋放 ----------------

cap.release()
out.release()
cv2.destroyAllWindows()
print("Process Finished. Result saved as output_acanet_result.avi")