import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from utility.model import *
from utility.tools import *
from utility.agent_MOT import Agent
from utility.moment import *
from yolov7_object_tracking.DnT_by_frame import *
from yolov7_object_tracking.utils.download_weights import download
import random
from dataclasses import dataclass
from typing import List, Optional
import gc
import matplotlib.pyplot as plt
from yolov7_object_tracking.utils.datasets import letterbox
import psutil
from collections import Counter
import time
import pandas as pd

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools.generate_detections import ImageEncoder


@dataclass
class Options:
    # 모델 및 경로 관련
    weights: str = '/home/hyhy/Desktop/yolov7.pt'         # 모델 경로
    source: str = ''                                       # 입력 소스 (파일/폴더 경로)
    project: str = '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/MOT_ds'  # 결과 저장 폴더
    name: str = 'exp'                                      # 프로젝트 하위 폴더명
    exist_ok: bool = False                                 # 기존 폴더 덮어쓰기 허용 여부

    # 모델 설정 및 연산
    img_size: int = 640                                    # 입력 이미지 크기
    imgsz: int = 640                                       # 입력 이미지 크기
    conf_thres: float = 0.2                                # 객체 탐지 신뢰도 임계값
    iou_thres: float = 0.45                                 # NMS에서 IOU 임계값
    device: str = 'cuda'                                   # 사용 디바이스 (cpu/cuda)
    augment: bool = False                                  # 증강 추론 여부
    no_trace: bool = False                                 # 모델 트레이싱 비활성화
    update: bool = False                                   # 모델 업데이트 여부

    # 결과 저장 및 시각화
    view_img: bool = False                                 # 결과 시각화 여부
    save_txt: bool = True                                  # 탐지 결과 txt 저장
    save_conf: bool = False                                # txt에 신뢰도 저장 여부
    nosave: bool = False                                   # 이미지/비디오 저장 안 함
    save_bbox_dim: bool = False                            # 바운딩 박스 크기 저장
    save_with_object_id: bool = False                      # 객체 ID와 함께 저장
    classes: Optional[List[int]] = None                    # 특정 클래스 필터링
    agnostic_nms: bool = False                             # 클래스 무관 NMS
    colored_trk: bool = False                              # 각 트랙에 색상 지정
    download: bool = True                                  # 모델 다운로드 여부
    half: bool = False 
    fps: int = 30                                          # 초당 프레임 수

def draw_boxes(img, bbox, identities=None, names=None, velocities=None, accelerations=None, angular_velocities=None, sim=None, save_with_object_id=False, path=None,offset=(0, 0)):
    #print(f"==> draw_boxes: 이미지 크기 {img.shape}, 바운딩 박스 수: {len(bbox)}")
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(b) for b in box]
        #print(f"  - Box {i}: ({x1},{y1}) to ({x2},{y2}) → W: {x2-x1}, H: {y2-y1}")
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        vel = velocities[i] if velocities is not None else (0, 0)
        acc = accelerations[i] if accelerations is not None else (0, 0)
        ang_vel = angular_velocities[i] if angular_velocities is not None else 0.0

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # 속도 벡터 끝점 계산 (스케일링 적용)
        scale = 5  # 속도 벡터 크기 스케일링
        end_x = int(center_x + vel[0] * scale)
        end_y = int(center_y + vel[1] * scale)

        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = (f"V:[{vel[0]:.2f},{vel[1]:.2f}] "
                f"Acc:[{acc[0]:.2f},{acc[1]:.2f}] "
                f"AngV:{ang_vel:.5f}")
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_x = max(0, min(x1, img.shape[1] - w - 5))
        text_y = y1 - 5 

        if text_y - h < 0:  # 텍스트가 위로 넘어가는 경우
            text_y = y1 + h + 5 

        if text_x + w > img.shape[1]:  # 텍스트가 오른쪽으로 넘어가는 경우
            text_x = img.shape[1] - w - 5   

        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.1)

        txt_str = ""
        if save_with_object_id:
            txt_str = (f"{box[0]/img.shape[1]:.6f} {box[1]/img.shape[0]:.6f} "
                    f"{box[2]/img.shape[1]:.6f} {box[3]/img.shape[0]:.6f} "
                    f"{(box[0] + box[2]/2)/img.shape[1]:.6f} "
                    f"{(box[1] + box[3]/2)/img.shape[0]:.6f}\n")
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img

def convert_bbox_format(temp: np.ndarray) -> np.ndarray:
    """    
    입력: [x1, y1, x2, y2, vx, vy, ax, ay, ang_vel]
    출력: [cx, cy, h, w, vx, vy, ax, ay, ang_vel]

    Args:
        temp (np.ndarray): (1, 9) 형태의 입력 배열
    Returns:
        np.ndarray: (1, 9) 형태의 변환된 배열
    """
    if not isinstance(temp, np.ndarray) or temp.shape != (1, 9):
        raise ValueError(f"입력은 (1, 9) 형태의 numpy 배열이어야 합니다. 현재 입력 shape: {temp.shape}")
    
    x1, y1, x2, y2, vx, vy, ax, ay, ang_vel = temp.flatten()

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    converted = np.array([[cx, cy, h, w, vx, vy, ax, ay, ang_vel]], dtype=np.float32)
    return converted

def get_next_frame_index(current_index, fps, total_images):
    fps_map = {30: 1, 15: 2, 10: 3, 5:6}
    increment = fps_map.get(fps, 1)  # 기본값 1

    next_index = int(current_index) + increment
    next_index = f"{next_index:08d}"

    return None if int(next_index) >= total_images else int(next_index)

def cat_His_OTID(On_target_ID_List, Supervised_History: History_Supervisor):
    Hisories = []
    for track_id in On_target_ID_List:
        inpu = Supervised_History.get_state_history(track_id)
        Hisories.append(inpu)
    
    return Hisories

class SOT_with_DRL_Test:
    def __init__(self, agent: Agent, dataset_path: Path, yolo_model, opt: Options):
        self.Agent = agent
        self.DataPath = dataset_path
        self.yolo_model = yolo_model
        self.device = torch.device(opt.device if opt.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.opt = opt
        self.prevFr = 30
        
    def Test_MOT(self):
        sdp_folders = [f for f in self.DataPath.iterdir() if f.is_dir() and 'FRCNN' in f.name]
        sort_tracker=None
        fr_count = {5: 0, 10: 0, 15: 0, 30: 0}
        durations = [] 
        self.image_encoder = ImageEncoder(
            '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/deep_sort/model_data/mars-small128.pb',
            'images',    # 그래프의 Placeholder op 이름
            'features')

        for i_episode, subfolder in enumerate(sdp_folders):
            if i_episode >= 7:
                break
            '''
            # === Memory Replier buffer Reset ===
            if hasattr(self.Agent, "memory"):
                del self.Agent.memory
            
            self.Agent.memory = ReplayMemory(10000)
            '''
            # === Start Texts ===
            print(f"Episode {i_episode + 1} 시작: {subfolder.name}")
            img_folder = subfolder / "img1"
            if not img_folder.exists():
                print(f"\t이미지 폴더 없음: {img_folder}")
                continue

            # === Load GT (1회만)
            det_path = subfolder / "det" / "det.txt"
            det_data = pd.read_csv(det_path, header=None)
            det_data.columns = ["frame", "id", "x", "y", "w", "h", "conf"]

            image_files = sorted(list(img_folder.glob("*.jpg")))
            total_img_num = len(image_files)
            if not image_files:
                print(f"\tNo images found in {img_folder}")
                continue

            # === Initialize components for the episode ===
            # --- DeepSORT 초기화 (metric, tracker) ---
            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine",               # appearance metric
                matching_threshold=0.2, # cosine 게이팅 임계값
                budget=50             # feature 버짓 없슴
            )
            ds_tracker = Tracker(metric)

            #History 초기화화
            Episode_History = History_Supervisor(History_Length = self.Agent.history_length)
            Episode_History.clear()

            # Parameter Declare for ith_Episode
            done = False
            self.currentFr, self.prevFr = 30, 30  # Initialize frame rates
            predicted_fr =30
            state =None
            best_target = None
            temp = None
            ds_tracker = Tracker(metric)
            done = False
            current_img_index = 0
 
           # === log file settings ===
            gt_name = subfolder.name
            log_file_path = Path(self.opt.project) / gt_name / f"{gt_name}.txt"
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file_path, 'w') as log_file:
                log_file.write("")
            
            log_file = open(log_file_path, "a") 

            tracker_result_path = Path(self.opt.project) / gt_name / "trackers.txt"
            tracker_result_path.parent.mkdir(parents=True, exist_ok=True)
            log_file_eval = open(tracker_result_path, "w")

            #Loop starts       
            while not done and current_img_index is not None:
                img_path = image_files[current_img_index]
                frame_num = int(img_path.stem) 
                frame_id = str(frame_num)
                img0 = cv2.imread(str(img_path))
                if img0 is None:
                    print(f"\t이미지를 불러올 수 없음: {img_path}")
                    break
                
                # === GT 기반 detections (mark==1) ===
                frame_det = det_data[det_data["frame"] == frame_num]

                # 1) 매 프레임마다 새 리스트로 시작
                det_list = []
                for _, row in frame_det.iterrows():
                    x1 = row["x"]
                    y1 = row["y"]
                    x2 = x1 + row["w"]
                    y2 = y1 + row["h"]
                    score = 1.0
                    det_list.append([x1, y1, x2, y2, score, 1])

                # 2) 리스트 → numpy array 변환
                if len(det_list) == 0:
                    dets_to_sort = np.zeros((0, 6), dtype=np.float32)
                else:
                    dets_to_sort = np.array(det_list, dtype=np.float32)
                    # 1D 배열이 된 경우 (원소가 하나일 때) 강제 reshape
                    if dets_to_sort.ndim == 1:
                        dets_to_sort = dets_to_sort.reshape(1, -1)
                start_time = time.time()
                
                raw_boxes = dets_to_sort[:, :4].tolist()
                scores    = dets_to_sort[:, 4].tolist()

                h0, w0 = img0.shape[:2]
                clipped_boxes, clipped_scores = [], []
                for box, score in zip(raw_boxes, scores):
                    x1, y1, x2, y2 = box
                    # int 변환 & 이미지 경계 클리핑
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2      = min(w0-1, int(x2))
                    y2      = min(h0-1, int(y2))
                    # 유효 박스만
                    if x2 <= x1 or y2 <= y1:
                        continue
                    clipped_boxes.append([x1, y1, x2, y2])
                    clipped_scores.append(score)

                # 2) (x1,y1,x2,y2) → tlwh 변환
                tlwh_boxes = []
                for x1, y1, x2, y2 in clipped_boxes:
                    w, h = x2 - x1, y2 - y1
                    tlwh_boxes.append([x1, y1, w, h])

                # 3) patch 생성 → feature 추출 (이미 작성된 image_encoder 사용)
                patches = []
                for (x1, y1, w, h) in tlwh_boxes:
                    patch = img0[y1:y1+h, x1:x1+w]
                    if patch.size == 0:
                        continue
                    patch_resized = cv2.resize(patch, (64, 128))
                    patches.append(patch_resized)

                if patches:
                    features = self.image_encoder.session.run(
                        self.image_encoder.output_var,
                        feed_dict={self.image_encoder.input_var: np.stack(patches, axis=0)}
                    )
                else:
                    features = np.zeros((0, self.image_encoder.output_var.shape[-1]), dtype=np.float32)

                # 4) Detection 리스트 생성
                deep_dets = [
                    Detection(tlwh, score, feature)
                    for tlwh, score, feature in zip(tlwh_boxes, clipped_scores, features)
                ]

                ds_tracker.predict(frame_rate=predicted_fr)
                ds_tracker.update(deep_dets)

                # --- 트랙 정보 추출 ---
                identities, boxes = [], []
                velocities, accelerations, ang_vels = [], [], []
                track_ids = []

                for track in ds_tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > ds_tracker.max_age:
                        continue
                    track_id = track.track_id
                    x, y, w, h = track.to_tlwh()
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    vx, vy = track.velocities[-1] if track.velocities else (0.0, 0.0)
                    ax, ay = track.accelerations[-1] if track.accelerations else (0.0, 0.0)
                    ang_vel = track.ang_vels[-1] if track.ang_vels else 0.0

                    box_state = np.array([[x1, y1, x2, y2, vx, vy, ax, ay, ang_vel]], dtype=np.float32)
                    Episode_History.update(track_id,convert_bbox_format(box_state),predicted_fr)

                    identities.append(track_id)
                    track_ids.append(track_id)  # 이제 track_ids에도 ID를 저장합니다
                    boxes.append((x1, y1, x2, y2))
                    velocities.append((vx, vy))
                    accelerations.append((ax, ay))
                    ang_vels.append(ang_vel)

                    log_file.write(
                        f"Frame: {frame_id}, ID: {track_id}, "
                        f"BBox: [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}], "
                        f"Vel: [{vx:.2f},{vy:.2f}], "
                        f"Acc: [{ax:.2f},{ay:.2f}], "
                        f"AngV: {ang_vel:.4f}\n"
                    )
                    log_file_eval.write(
                        f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,1,1,1\n"
                    )

                if track_ids:
                    inpu_tensor = cat_His_OTID(track_ids, Episode_History)   # concat inputs and transform to tensor
                    state_batch = self.Agent.get_features_Test(track_ids, inpu_tensor) # tensor of N , #of On_target_ID
                    _, fr_list = self.Agent.get_best_next_action4MOT_Test(state_batch)
                    #predicted_fr = max(fr_list)
                    counter = Counter(fr_list)
                    predicted_fr = counter.most_common(1)[0][0]
                    #predicted_fr = 30
                    fr_count[predicted_fr] += 1

                self.prevFr = predicted_fr
                current_img_index = get_next_frame_index(current_img_index, predicted_fr, total_img_num)
                if current_img_index is None:
                    done = True
                #print(f"running time : {end_time - start_time}")
                end_time = time.time()
                #durations.append(end_time - start_time)

                # === 결과 이미지 저장 ===
                if not self.opt.nosave:
                    img_with_boxes = draw_boxes(
                        img0.copy(), boxes, identities, velocities=velocities,
                        accelerations=accelerations, angular_velocities=ang_vels,
                        save_with_object_id=self.opt.save_with_object_id, path=str(log_file_path))
                    save_dir = Path(self.opt.project) / gt_name / f"frames"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / img_path.name
                    cv2.imwrite(str(save_path), img_with_boxes)
                

        print(f"\n\tFPS 선택 분포: {fr_count}")
        '''
        if durations:
            avg_time = sum(durations) / len(durations)
            print(f"평균 처리 시간: {avg_time:.4f} 초 (프레임당)")
        else:
            print("측정된 프레임이 없습니다.")
        '''
        log_file.close()
        gc.collect()
        torch.cuda.empty_cache()

        print("\n모든 시퀀스 테스트 종료")


def main():
    # === 설정 ===
    dataset_path = Path("/home/hyhy/Datasets/FR_Dataset/MOT17/train")  # 데이터셋 경로
    opt = Options(
        source=str(dataset_path),
        name='LaSOT_Training',
        img_size=640,
        conf_thres=0.3,
        iou_thres=0.2,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load('/home/hyhy/Desktop/yolov7.pt', map_location=device)
    agent = Agent(num_episodes=1, load=True, n_actions=4, device=device)  # 모델 로드 모드 설정

    if device.type == "cuda":
        model.half()  
    else:
        model.float() 

    # === SOT_with_DRL_Tr 인스턴스 생성 ===
    tester = SOT_with_DRL_Test(agent=agent, dataset_path=dataset_path, yolo_model=model, opt=opt)

    # === 학습 시작 ===
    print("평가 시작...")
    tester.Test_MOT()
    print("평가 완료!")
    print(
        f"Parameters\n"
        f"  Gamma: {agent.GAMMA}, "
        f"  EPS:   {agent.EPS}, "
        f"  IOUW:  {agent.w_iou}, "
        f"  thW:   {agent.w_theta}, "
        f"  FRW:   {agent.w_FR},"
        f"  His_Length: {agent.history_length},"
            )

if __name__ == "__main__":
    main()


