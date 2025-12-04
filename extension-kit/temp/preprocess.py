'''
import os
import subprocess

# 경로 설정
BASE_PATH = "/home/hyhy/Datasets/FR_Dataset/LaSOT"
OUTPUT_PATH = "/home/hyhy/Datasets/FR_Dataset/LaSOT/video"

# 출력 경로가 없으면 생성
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 모든 객체 폴더 순회
for object_folder in os.listdir(BASE_PATH):
    object_path = os.path.join(BASE_PATH, object_folder)
    
    if os.path.isdir(object_path):
        print(f"Processing object folder: {object_folder}")
        
        # 1️⃣ **하위 폴더가 있는 경우**
        has_subfolders = False
        for sub_folder in os.listdir(object_path):
            sub_path = os.path.join(object_path, sub_folder)
            img_path = os.path.join(sub_path, "img")
            
            if os.path.isdir(img_path):
                has_subfolders = True
                output_video = os.path.join(OUTPUT_PATH, f"{sub_folder}.mp4")
                print(f"Converting images in: {img_path}")
                
                cmd = [
                    "ffmpeg",
                    "-framerate", "60",
                    "-i", os.path.join(img_path, "%08d.jpg"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    output_video
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    print(f"Video created: {output_video}")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {img_path}: {e}")
        
        # 2️⃣ **하위 폴더가 없고 img 폴더가 직접 있는 경우**
        if not has_subfolders:
            img_path = os.path.join(object_path, "img")
            if os.path.isdir(img_path):
                output_video = os.path.join(OUTPUT_PATH, f"{object_folder}.mp4")
                print(f"Converting images in: {img_path}")
                
                cmd = [
                    "ffmpeg",
                    "-framerate", "60",
                    "-i", os.path.join(img_path, "%04d.jpg"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    output_video
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    print(f"Video created: {output_video}")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {img_path}: {e}")
            else:
                print(f"No 'img' folder found in: {object_path}")
'''
import os
import subprocess

def convert_single_image_folder_to_video(img_path, output_video):
    # ffmpeg command to create video
    FR= 120
    output_video = output_video.replace(".mp4", f"_{FR}fps.mp4")

    cmd = [
        "ffmpeg",
        "-framerate", str(FR),
        "-i", os.path.join(img_path, "%08d.jpg"),  # Modify if filenames are different
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Video created: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {img_path}: {e}")

# Example usage
BASE_PATH = "/home/hyhy/Datasets/FR_Dataset/LaSOT/person"
OUTPUT_PATH = "/home/hyhy/Datasets/FR_Dataset/LaSOT/video"

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Single folder example
object_folder = "person-10"  # Example folder name
img_path = os.path.join(BASE_PATH, object_folder, "img")
output_video = os.path.join(OUTPUT_PATH, f"{object_folder}.mp4")

convert_single_image_folder_to_video(img_path, output_video)