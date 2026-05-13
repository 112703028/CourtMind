from roboflow import Roboflow
import os

# 1. 初始化 (填入你的資訊)
rf = Roboflow(api_key="GexJ6GGAUGS1RcLL7Dha")
workspace = rf.workspace("tim-y1dkn")
project = workspace.project("screen-jt8c4")

# 2. 設定抽好幀的圖片主資料夾
# 假設你的資料夾結構是：input_videos/screen_frames/video_1/*.jpg
base_frame_dir = "input_videos/screen_frames"

# 3. 遍歷所有子資料夾並上傳
print("開始掃描資料夾並準備上傳...")

import re

START_CLIP = 138

for folder_name in os.listdir(base_frame_dir):
    m = re.search(r'screen_(\d+)', folder_name)
    if not m or int(m.group(1)) < START_CLIP:
        continue

    folder_path = os.path.join(base_frame_dir, folder_name)

    if os.path.isdir(folder_path):
        print(f"\n正在處理影片片段: {folder_name}")
        
        # 取得該影片片段的所有圖片
        images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            
            print(f"  上傳中: {img_name}...", end="\r")
            
            # 執行上傳
            project.upload(
                image_path=img_path,
                batch_name=f"batch_{folder_name}", # 以影片片段命名 Batch，方便後續管理
                split="train",
                num_retry_uploads=3
            )

print("\n恭喜！所有圖片已上傳至 Roboflow。")