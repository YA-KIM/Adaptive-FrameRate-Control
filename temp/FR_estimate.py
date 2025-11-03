
import os
import datetime

# 이미지 프레임 경로 설정
frame_folder = '/home/hyhy/Datasets/FR_Dataset/LaSOT/person/person-11/img'
frame_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')])

# 타임스탬프 분석
timestamps = []
for frame in frame_files:
    timestamp = os.path.getmtime(frame)
    timestamps.append(timestamp)

# 프레임 간 시간 간격 계산
time_intervals = [j - i for i, j in zip(timestamps[:-1], timestamps[1:])]
average_interval = sum(time_intervals) / len(time_intervals) if time_intervals else 0

# FPS 추정
fps = 1 / average_interval if average_interval > 0 else None
print(f"Average Frame Interval: {average_interval:.4f} seconds")
print(f"Estimated FPS: {fps:.2f}")
'''
#####################################################################################
import cv2
import os

frame_folder = '/home/hyhy/Datasets/FR_Dataset/tracker_benchmark/seq/DragonBaby/img'
output_video = 'reconstructed_video.mp4'

frame_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')])
frame = cv2.imread(frame_files[0])
h, w, _ = frame.shape

# 다양한 FPS로 동영상 생성
for fps in [24, 30, 60]:
    video_writer = cv2.VideoWriter(f'reconstructed_{fps}fps.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved at reconstructed_{fps}fps.mp4")

##############################################################
import cv2
import os

# 이미지 프레임 경로 설정
frame_files = sorted([os.path.join('/home/hyhy/Datasets/FR_Dataset/LaSOT/Basketball/img', f) 
                      for f in os.listdir('/home/hyhy/Datasets/FR_Dataset/LaSOT/Basketball/img') 
                      if f.endswith('.jpg')])

# 프레임 파일 확인
print(f"Found {len(frame_files)} image frames.")


frame_files = sorted([os.path.join('/home/hyhy/Datasets/FR_Dataset/LaSOT/Basketball/img', f) for f in os.listdir('/home/hyhy/Datasets/FR_Dataset/LaSOT/Basketball/img') if f.endswith('.jpg')])

prev_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
for i in range(1, len(frame_files)):
    curr_frame = cv2.imread(frame_files[i], cv2.IMREAD_GRAYSCALE)
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print(f"Frame {i}: Average Flow Magnitude: {magnitude.mean():.2f}")
    prev_frame = curr_frame
'''