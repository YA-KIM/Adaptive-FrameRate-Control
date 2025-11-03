import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from utility.model import *
from utility.tools import *
from utility.agent import Agent
from utility.moment import *
from yolov7_object_tracking.DnT_ranking import * # Changable
from yolov7_object_tracking.utils.download_weights import download
import random
from dataclasses import dataclass
from typing import List, Optional
import gc
import psutil
import tracemalloc
import time

# === memory Usage Checking ===
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / 1024**2:.5f} MB, VMS: {mem_info.vms / 1024**2:.5f} MB")

def get_memory_usage(obj):
    """객체와 참조된 모든 객체들의 총 메모리 사용량을 계산"""
    seen_ids = set()
    size = 0

    # 참조된 객체를 재귀적으로 탐색
    def get_size(o):
        nonlocal size
        obj_id = id(o)
        if obj_id in seen_ids:
            return
        seen_ids.add(obj_id)
        size += sys.getsizeof(o)
        
        # 내부 요소까지 탐색
        for ref in gc.get_referents(o):
            get_size(ref)

    get_size(obj)
    print(f"aa:{size / 1024}")  # KB 단위 변환


def free_unused_tensors():
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.grad is None:
                del obj

@dataclass
class Options:
    # 모델 및 경로 관련
    weights: str = '/home/hyhy/Desktop/yolov7.pt'          # 모델 경로
    source: str = ''                                       # 입력 소스 (파일/폴더 경로)
    project: str = '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/DnT_by_frame'  # 결과 저장 폴더
    name: str = 'exp'                                      # 프로젝트 하위 폴더명
    exist_ok: bool = False                                 # 기존 폴더 덮어쓰기 허용 여부

    # 모델 설정 및 연산
    img_size: int = 640                                    # 입력 이미지 크기
    conf_thres: float = 0.2                                # 객체 탐지 신뢰도 임계값
    iou_thres: float = 0.3                                 # NMS에서 IOU 임계값
    device: str = ''                                       # 사용 디바이스 (cpu/cuda)
    augment: bool = False                                  # 증강 추론 여부
    no_trace: bool = False                                 # 모델 트레이싱 비활성화
    update: bool = False                                   # 모델 업데이트 여부

    # 결과 저장 및 시각화
    view_img: bool = False                                 # 결과 시각화 여부
    save_txt: bool = True                                 # 탐지 결과 txt 저장
    save_conf: bool = False                                # txt에 신뢰도 저장 여부
    nosave: bool = False                                   # 이미지/비디오 저장 안 함
    save_bbox_dim: bool = False                            # 바운딩 박스 크기 저장
    save_with_object_id: bool = False                      # 객체 ID와 함께 저장
    classes: Optional[List[int]] = None                    # 특정 클래스 필터링
    agnostic_nms: bool = False                             # 클래스 무관 NMS
    colored_trk: bool = False                              # 각 트랙에 색상 지정
    download: bool = True                                  # 모델 다운로드 여부
    fps: int = 60                                          # 초당 프레임 수

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
    fps_map = {60: 1, 30: 2, 20: 3}
    increment = fps_map.get(fps, 1)  # 기본값 1

    next_index = int(current_index) + increment
    next_index = f"{next_index:08d}"

    return None if int(next_index) >= total_images else int(next_index)

# === Train 수행 ===
class SOT_with_DRL_Tr():
    def __init__(self, agent: Agent, DataSetPath, yolo_model, opt: Options):
        self.Agent = agent
        self.DataPath = DataSetPath  # 이미지 시퀀스들의 상위폴더더
        self.currentFr = None
        self.prevFr = None
        self.yolo_model = yolo_model
        self.device = torch.device(opt.device if opt.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.current_obj_id = None
        self.prev_obj_id = None
        self.opt = opt

    def Train(self):
        category_folders = [f for f in self.DataPath.iterdir() if f.is_dir()]
        subfolders = []  # 모든 객체 폴더 리스트
        total_episodes = 0
        SoT=None

        # **각 카테고리 내부 객체 폴더 탐색**
        for category_folder in category_folders:
            object_folders = [obj for obj in category_folder.iterdir() if obj.is_dir()]
            subfolders.extend(object_folders)     
        total_episodes += len(subfolders)
        self.Agent.num_episodes = total_episodes
        print(f"총 {total_episodes}개의 episode")
        random.shuffle(subfolders)

        for i_episode, subfolder in enumerate(subfolders): #Training Start
            if i_episode >= self.Agent.num_episodes: 
                break

            # === Memory Replier buffer Reset ===
            if hasattr(self.Agent, "memory"):
                del self.Agent.memory
            
            self.Agent.memory = ReplayMemory(10000)

            print(f"Episode {i_episode + 1} 시작: {subfolder}")
            img_folder = os.path.join(subfolder, 'img')
            if not os.path.exists(img_folder):
                print(f"이미지 폴더 없음: {img_folder}")
                continue

            gt_folder_name = Path(subfolder).name
            log_file_path = Path(self.opt.project) / gt_folder_name / f"{gt_folder_name}.txt"
            if not log_file_path.exists():
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file_path, 'w') as log_file:
                log_file.write("")

            image_files = sorted(list(Path(img_folder).glob("*.jpg")))  # 이미지 파일 리스트
            total_img_num = len(image_files)

            if total_img_num == 0:
                print(f"No images found in: {img_folder}")
                continue

            # === Initialize components for the episode ===
            #SORT 초기화
            if 'SoT' in locals() and SoT is not None:
                SoT.reset()
                del SoT
                gc.collect()
                torch.cuda.empty_cache()
            SoT = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

            #History 초기화화
            Episode_History = History_Supervisor(History_Length = self.Agent.history_length)
            Episode_History.clear()

            current_img_indx = 0
            done = False
            self.currentFr, self.prevFr = 60, 60  # Initialize frame rates
            predicted_fr =60
            initial_features = []
            state =None
            best_target = None
            first_hist = None
            temp = None
            gt_format = None
            fr_count = {20: 0, 30: 0, 60: 0}

            while not done and current_img_indx is not None:
                if current_img_indx < 5: 
                    #Start Episode
                    cur_img = image_files[current_img_indx]

                    # === SORT[0] ====
                    with torch.no_grad():
                        kalman_predict, kalman_states, initial_features, best_target, SoT, first_hist, _ = detect_frame(
                            model=self.yolo_model, device=self.device, half=True, opt=self.opt, source=cur_img, 
                            gt_folder=subfolder, initial_features=initial_features, prev_best_target=best_target, 
                            sort_tracker=SoT, first_hist=first_hist, gt_format=gt_format, frame_rate=self.prevFr)


                    print(f"kalman_predict = {kalman_predict}")
                    # Kalman tracker 정보 추출 
                    if kalman_states and isinstance(kalman_states, list):  # 리스트이면서 비어있지 않은 경우
                        tracker = kalman_states[0]  

                        if hasattr(tracker, 'get_state'):
                            x1, y1, x2, y2 = tracker.get_state()[0][:4]  # bbox 정보 추출
                            velocity = tracker.velocities[-1] if tracker.velocities else (0.0, 0.0)
                            acceleration = tracker.accelerations[-1] if tracker.accelerations else (0.0, 0.0)
                            ang_vel = tracker.ang_velocities[-1] if tracker.ang_velocities else 0.0

                            bbox_state = np.array([x1, y1, x2, y2, *velocity, *acceleration, ang_vel], dtype=np.float32).reshape(1, 9)
                            temp = convert_bbox_format(bbox_state)

                        if kalman_predict is not None and isinstance(kalman_predict, np.ndarray) and kalman_predict.size > 0:
                            self.current_obj_id = 1.0
                        else:
                            self.current_obj_id = 1.0  # 이전 객체 ID 유지
                            temp = np.zeros((1, 9), dtype=np.float32) #temp = None
                    #print(f"obj_id: {self.current_obj_id}")

                    # === State Consturction ===  
                    if temp is not None:  
                        Episode_History.update(self.current_obj_id, temp, self.currentFr)
                        current_moment = Episode_History[self.current_obj_id][-1]
                        inpu = Episode_History.get_state_history(self.current_obj_id)

                        # === Feature Extraction ===
                        state = self.Agent.get_features(self.current_obj_id, inpu)
                        # === DQN[t=0] ===
                        index, predicted_fr = self.Agent.select_action(state)
                    self.prevFr = self.currentFr
                    current_img_indx = get_next_frame_index(current_img_indx, 60, total_img_num)

                else : 
                # === Episode Steps ===
                    # tracemalloc.start()
                    # Checking Memory
                    #print_memory_usage()
                    #print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
                    #print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB")

                    # === state changes[t+1] ===
                    cur_img = image_files[current_img_indx]
                    # === Sort[t+1] ===
                    self.currentFr = predicted_fr
                    #print(f"obj_id: {self.current_obj_id}") #지워
                    with torch.no_grad():
                        kalman_predict, kalman_states, initial_features, best_target, SoT, first_hist, gt_format = detect_frame(
                            model=self.yolo_model, device=self.device, half=True, opt=self.opt, source=cur_img, 
                            gt_folder=subfolder, initial_features=initial_features, prev_best_target=best_target, 
                            sort_tracker=SoT, first_hist=first_hist, gt_format=gt_format, frame_rate=self.prevFr)
                        
                    # Kalman tracker 정보 추출
                    if kalman_states and isinstance(kalman_states, list):  # 리스트이면서 비어있지 않은 경우
                        tracker = kalman_states[0]  

                        if hasattr(tracker, 'get_state'):
                            x1, y1, x2, y2 = tracker.get_state()[0][:4]  # bbox 정보 추출
                            velocity = tracker.velocities[-1] if tracker.velocities else (0.0, 0.0)
                            acceleration = tracker.accelerations[-1] if tracker.accelerations else (0.0, 0.0)
                            ang_vel = tracker.ang_velocities[-1] if tracker.ang_velocities else 0.0

                            bbox_state = np.array([x1, y1, x2, y2, *velocity, *acceleration, ang_vel], dtype=np.float32).reshape(1, 9)
                            temp = convert_bbox_format(bbox_state)

                        if kalman_predict is not None and isinstance(kalman_predict, np.ndarray) and kalman_predict.size > 0:
                            self.current_obj_id = kalman_predict[0, -1]  # 새로운 객체 ID 업데이트
                        else:
                            self.current_obj_id = self.prev_obj_id  # 이전 객체 ID 유지
                            temp = np.zeros((1, 9), dtype=np.float32)  #temp = None

                    if temp is not None:  
                        Episode_History.update(self.current_obj_id, temp, self.currentFr)
                        current_moment = Episode_History[self.current_obj_id][-1]
                        prev_moment = Episode_History[self.current_obj_id][-2] if len(Episode_History[self.current_obj_id]) >= 2 else current_moment
                    #print(f"current_moment = {current_moment}, prev_moment={prev_moment}")

                    # === Reward Computation ===
                    # print(best_target)
                    if best_target is None:
                        reward = -20
                    elif self.current_obj_id == self.prev_obj_id and self.current_obj_id is not None:
                        reward = self.Agent.compute_reward(
                            prev_moment.current_vector, prev_moment.previous_vector, current_moment.current_vector, self.prevFr, self.currentFr)
                    else: # obj_id가 바뀜뀜
                        reward = -10 
                    print(f"Reward: {reward}")

                    # === t+1 state extract ===
                    next_inpu = Episode_History.get_state_history(self.current_obj_id)
                    next_state = self.Agent.get_features(self.current_obj_id, next_inpu)

                    # === Construct Buffer ===
                    if state is not None and index is not None and next_state is not None and reward is not None:
                        self.Agent.memory.push(state, index, next_state, reward)

                    # === DQN ===
                    state = next_state
                    self.prev_obj_id = self.current_obj_id
                    self.prevFr = self.currentFr 
                    index, predicted_fr = self.Agent.select_action(state)
                    current_img_indx = get_next_frame_index(current_img_indx, predicted_fr, total_img_num)
                    if predicted_fr in fr_count:
                        fr_count[predicted_fr] += 1
                    else:
                        print(f"Unexpected value: {predicted_fr}")  # 예외 처리

                    # === Back Propagation When there is enough experiment in buffer ===
                    self.Agent.optimize_model(verbose=True)

                    #print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    #print(f"Reserved Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                    #print(f"Max Reserved Memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB"

                    if current_img_indx is None:
                        done = True
                    #current, peak = tracemalloc.get_traced_memory()
                    #print(f"현재 메모리 사용량: {current / 1024:.2f} KB, 최대 메모리 사용량: {peak / 1024:.2f} KB")
                    #tracemalloc.stop()
                    cv2.destroyAllWindows() # window plot 제거..
                    cv2.waitKey(1)

            print(f"selected_fr : {fr_count}\n")
            # update target net every TARGET_UPDATE episodes
            # Traget Network는 Episdoe(one Video)마다 변화시키는게 안정적임.
            if (subfolders.index(subfolder) + 1) % self.Agent.TARGET_UPDATE == 0:
                with torch.no_grad():
                    self.Agent.target_net.load_state_dict(self.Agent.policy_net.state_dict())
            self.Agent.EPS = max(self.Agent.EPS_min, self.Agent.EPS * (0.95 ** i_episode))
            SoT.reset()
            free_unused_tensors()
            
            del state, next_state, kalman_predict, kalman_states, initial_features, Episode_History ,SoT
            del self.Agent.memory
            self.Agent.memory = None
            self.current_obj_id = None
            SoT = None
            Episode_History = None
        
            gc.collect()
            torch.cuda.empty_cache()       
        self.Agent.save_network()
        print("Training completed.")


def main():
    # === 설정 ===
    dataset_path = Path("/home/hyhy/Datasets/FR_Dataset/LaSOT2/training")  # 데이터셋 경로
    opt = Options(
        source=str(dataset_path),
        name='LaSOT_Training',
        img_size=640,
        conf_thres=0.3,
        iou_thres=0.2,
        device='cuda'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load('/home/hyhy/Desktop/yolov7.pt', map_location=device)
    agent = Agent(load=False, n_actions=3, device=device)

    if device.type == "cuda":
        model.half()  
    else:
        model.float() 

    # === SOT_with_DRL_Tr 인스턴스 생성 ===
    trainer = SOT_with_DRL_Tr(agent=agent, DataSetPath=dataset_path, yolo_model=model, opt=opt)

    # === 학습 시작 ===
    print("학습 시작...")
    trainer.Train()
    print("학습 완료!")
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