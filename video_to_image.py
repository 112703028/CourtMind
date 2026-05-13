import cv2
import os
from tqdm import tqdm # 建議安裝這個庫，可以看到進度條：pip install tqdm

def batch_extract_frames(input_dir, base_output_dir, fps=5):
    # 支援的影片格式
    valid_extensions = ('.mp4', '.mov', '.avi', '.mkv')
    
    # 取得所有影片檔案路徑
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"找到 {len(video_files)} 個影片，準備開始抽幀...")

    for video_name in tqdm(video_files):
        video_path = os.path.join(input_dir, video_name)
        
        # 為每個影片建立專屬資料夾 (拿掉副檔名當資料夾名稱)
        video_id = os.path.splitext(video_name)[0]
        output_dir = os.path.join(base_output_dir, video_id)
        
        if os.path.exists(output_dir) and os.listdir(output_dir):
            continue   # 已有 frames，跳過
        os.makedirs(output_dir, exist_ok=True)
            
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 防呆：如果抓不到影片 FPS，預設給 30
        if video_fps == 0: video_fps = 30
        
        hop = round(video_fps / fps)
        
        count = 0
        extracted_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % hop == 0:
                # 檔名格式：影片名_張數.jpg
                frame_name = f"{video_id}_{extracted_count:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)
                extracted_count += 1
            count += 1
        
        cap.release()

# 執行：把 screen 資料夾裡的所有影片，轉成圖片存到 screen_frames
batch_extract_frames("input_videos/screen", "input_videos/screen_frames", fps=5)