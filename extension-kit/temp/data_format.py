import os
import pandas as pd

# 파일 경로 설정
tracker_dir = "/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/MOT"

# 경로가 존재하는지 확인
if not os.path.exists(tracker_dir):
    print(f"경로가 존재하지 않습니다: {tracker_dir}")
else:
    print(f"경로가 존재합니다: {tracker_dir}")

# tracker.txt 파일을 찾아서 정렬 후 덮어쓰는 작업 수행
found_files = False  # 파일이 발견되었는지 확인하는 변수

for root, dirs, files in os.walk(tracker_dir):
    for file in files:
        if file.endswith("trackers.txt"):  # MOT17-01-FRCNN.txt 형식 파일 찾기
            file_path = os.path.join(root, file)
            
for root, dirs, files in os.walk(tracker_dir):
    for file in files:
        if file.endswith("trackers.txt"):  # MOT17-01-FRCNN.txt 형식 파일 찾기
            file_path = os.path.join(root, file)
            
            df = pd.read_csv(file_path, header=None)

            # 컬럼 이름 설정 (원하는 경우 생략 가능)
            df.columns = ['frame_number', 'object_id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']

            # object_id 기준 오름차순 정렬, 그 안에서 frame_number 기준 오름차순 정렬
            df_sorted = df.sort_values(by=['object_id', 'frame_number'])
 
            # 정렬된 데이터를 기존 파일에 덮어쓰기
            df_sorted.to_csv(file_path, sep=" ", header=False, index=False, lineterminator="\n")
            
            print(f"{file} 파일 정렬 완료!")

# 만약 아무 파일도 발견되지 않았다면
if not found_files:
    print("trackers.txt 파일을 찾을 수 없습니다.")
