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
from yolov7_object_tracking.DnT_ranking import * # Changable
import random
from dataclasses import dataclass
from typing import List, Optional
import gc
import psutil
import tracemalloc
import time

from yolov7_object_tracking.utils.download_weights import download
from yolov7_object_tracking.utils.datasets import letterbox

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
    project: str = '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/MOT'  # 결과 저장 폴더
    name: str = 'exp'                                      # 프로젝트 하위 폴더명
    exist_ok: bool = False                                 # 기존 폴더 덮어쓰기 허용 여부

    # 모델 설정 및 연산
    img_size: int = 640                                    # 입력 이미지 크기
    conf_thres: float = 0.5                                # 객체 탐지 신뢰도 임계값
    iou_thres: float = 0.5                                 # NMS에서 IOU 임계값
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
    half: bool = False 
    fps: int = 30                                          # 초당 프레임 수
def draw_boxes(img, bbox, identities=None, names=None, velocities=None, accelerations=None, angular_velocities=None, sim=None, save_with_object_id=False, path=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
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

def sqb2cb(temp: np.ndarray) -> np.ndarray:
    if not isinstance(temp, np.ndarray) or temp.shape != (1, 4):
        raise ValueError(f"입력은 (1, 4) 형태의 numpy 배열이어야 합니다. 현재 입력 shape: {temp.shape}")
    
    x1, y1, x2, y2 = temp.flatten()

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    converted = np.array([[cx, cy, h, w]], dtype=np.float32)
    return converted

def cb2sqb(cx,cy,h,w):

    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    return x1, y1, x2, y2

# letterbox의 pad, ratio 고려하여 bbox 원래 좌표계로 복원
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """YOLOv5 utils에서 가져온 함수 — padding/resize 보정"""
    if ratio_pad is None:
        # calculate from img0 shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0::2] = coords[:, 0::2].clip(0, img0_shape[1])  # x 좌표 clip
    coords[:, 1::2] = coords[:, 1::2].clip(0, img0_shape[0])  # y 좌표 clip
    return coords


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    #print("box1:", box1, "shape:", np.shape(box1), "dtype:", np.array(box1).dtype)
    #print("box2:", box2, "shape:", np.shape(box2), "dtype:", np.array(box2).dtype)

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def get_next_frame_index(current_index, fps, total_images):
    fps_map = {30: 1, 15: 2, 10: 3, 5:6}
    increment = fps_map.get(fps, 1)  # 기본값 1

    next_index = int(current_index) + increment
    next_index = f"{next_index:08d}"

    return None if int(next_index) >= total_images else int(next_index)

def sort_state_to_xyxy(cx, cy, s, r):
    w = (s * r)**0.5
    h = s / w
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


# === Train 수행 ===
class SOT_with_DRL_Tr():
    def __init__(self, agent: Agent, DataSetPath, yolo_model, opt: Options):
        self.Agent = agent
        self.DataPath = DataSetPath  # 이미지 시퀀스들의 상위폴더더
        self.currentFr = None
        self.prevFr = None
        self.yolo_model = yolo_model
        self.device = torch.device(opt.device if opt.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.opt = opt
        self.uncertain_frame_count = 0

    def Train(self):
        frcnn_folders = [f for f in self.DataPath.iterdir() if f.is_dir() and 'FRCNN' in f.name]
        total_episodes = 0
        sort_tracker=None

        for i_episode, subfolder in enumerate(frcnn_folders): #Training Start
            if i_episode >= 7:
                break

            # === Memory Replier buffer Reset ===
            if hasattr(self.Agent, "memory"):
                del self.Agent.memory
            
            self.Agent.memory = ReplayMemory(10000)

            print(f"Episode {i_episode + 1} 시작: {subfolder}")
            img_folder = subfolder / "img1"
            if not img_folder.exists():
                print(f"이미지 폴더 없음: {img_folder}")
                continue

            image_files = sorted(list(Path(img_folder).glob("*.jpg")))  # 이미지 파일 리스트
            total_img_num = len(image_files)

            if total_img_num == 0:
                print(f"No images found in: {img_folder}")
                continue

            # === Initialize components for the episode ===
            #SORT 초기화
            if 'sort_tracker' in locals() and sort_tracker is not None:
                sort_tracker.reset()
                del sort_tracker
                gc.collect()
                torch.cuda.empty_cache()
            sort_tracker = Sort(max_age=5, min_hits=0, iou_threshold=0.3)

            #History 초기화화
            Episode_History = History_Supervisor(History_Length = self.Agent.history_length)
            Episode_History.clear()

            done = False
            self.currentFr, self.prevFr = 30, 30  # Initialize frame rates
            predicted_fr =30
            initial_features = []
            state =None
            best_target = None
            temp = None
            object_lifetime = {}
            fr_count = {5: 0, 10: 0, 15: 0, 30: 0}
 
            initial_image_index = 0 # 50frame씩 재반복 하며 단일 객체를 추적
            while initial_image_index is not None :
                current_img_indx = initial_image_index

                gt_folder_name = Path(subfolder).name
                log_file_path = Path(self.opt.project) / gt_folder_name/ f"{initial_image_index}" / f"{gt_folder_name}.txt"
                if not log_file_path.exists():
                    log_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file_path, 'w') as log_file:
                    log_file.write("")
                
                log_file = open(log_file_path, "a") 

                cur_img = image_files[current_img_indx]
                img0 = cv2.imread(str(cur_img))
                if img0 is None:
                    raise RuntimeError(f"이미지를 불러올 수 없습니다: {cur_img}")
                
                # letterbox 처리 (비율 유지하며 리사이징 + 패딩)
                img, ratio, pad = letterbox(img0, new_shape=640)
                imgf = img
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device).detach()
                img = img.half() if next(self.yolo_model.parameters()).dtype == torch.float16 else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                

                # Warmup
                if self.device.type != 'cpu':
                    self.yolo_model(torch.zeros(1, 3, self.opt.img_size, self.opt.img_size).to(self.device).type_as(next(self.yolo_model.parameters())))
                    old_img_b, old_img_h, old_img_w = 1, self.opt.img_size, self.opt.img_size
                    if old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]:
                        old_img_b, old_img_h, old_img_w = img.shape[0], img.shape[2], img.shape[3]
                        for _ in range(3):
                            self.yolo_model(img, augment=self.opt.augment)[0]

                with torch.no_grad():
                    pred = self.yolo_model(img, augment=self.opt.augment)[0]
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

                dets_to_sort = np.empty((0, 6))
                for det in pred:
                    if det is not None and len(det):
                        for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                            if int(detclass) == 0:
                                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                person_num = dets_to_sort.shape[0]
                print(f"MOT_Object_Number: {person_num}")

                for i in range(person_num):

                    # === reset for next brench episode ===
                    # === Initialize components for the episode ===
                    #SORT 초기화
                    if 'sort_tracker' in locals() and sort_tracker is not None:
                        sort_tracker.reset()
                        del sort_tracker
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    current_img_indx = initial_image_index
                    sort_tracker = Sort(max_age=5, min_hits=0, iou_threshold=0.3)
                    done = False
                    print(f"i: {i}")
                    log_file.write(f"\n{i}th object\n")
                    while not done and current_img_indx is not None:
                        cur_img = image_files[current_img_indx]
                        self.currentFr = predicted_fr
                        max_IOU = 0.1
                        best_target = None

                        # === 이미지 로딩 및 전처리 ===
                        img0 = cv2.imread(str(cur_img))
                        if img0 is None:
                            print(f"이미지를 불러올 수 없습니다: {cur_img}")
                            break

                        # letterbox 처리 (비율 유지하며 리사이징 + 패딩)
                        img, ratio, pad = letterbox(img0, new_shape=640)
                        imgf = img #plot용 이미지 복사 
                        # YOLO 전처리 
                        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
                        img = np.ascontiguousarray(img)
                        img = torch.from_numpy(img).to(self.device).detach()
                        img = img.half() if next(self.yolo_model.parameters()).dtype == torch.float16 else img.float()
                        img /= 255.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)
                        

                        # Warmup
                        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                            old_img_b = img.shape[0]
                            old_img_h = img.shape[2]
                            old_img_w = img.shape[3]
                            for j in range(3):
                                self.yolo_model(img, augment=self.opt.augment)[0]

                        # Inference
                        with torch.no_grad():
                            pred = self.yolo_model(img, augment=self.opt.augment)[0]
                            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

                        if current_img_indx == initial_image_index:
                            dets_to_sort = np.empty((0, 6))
                            for det in pred:
                                if det is not None and len(det):
                                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                                        if int(detclass) == 0:
                                            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                            person_num = dets_to_sort.shape[0]

                            gt = dets_to_sort[i][0:4].reshape(1, 4)
                            dets_to_sort[i][0:4] = gt.flatten()

                            kalman_predict = sort_tracker.update(dets_to_sort[i].reshape(1, -1))  # (1, 6) shape으로 reshape

                            kalman_states = sort_tracker.getTrackers()

                            if len(kalman_states) > 0:
                                tracker = kalman_states[0]
                                if hasattr(tracker, 'get_state'):
                                    cx, cy, h, w = tracker.get_state()[0][:4]
                                    velocity = tracker.velocities[-1] if tracker.velocities else (0.0, 0.0)
                                    acceleration = tracker.accelerations[-1] if tracker.accelerations else (0.0, 0.0)
                                    ang_vel = tracker.ang_velocities[-1] if tracker.ang_velocities else 0.0
                                    bbox_state = np.array([cx, cy, h, w, *velocity, *acceleration, ang_vel], dtype=np.float32).reshape(1, 9)
                                    temp = convert_bbox_format(bbox_state)

                                    # === 로그 저장 ===
                                    track_id = int(kalman_predict[0, 8])
                                    category = int(dets_to_sort[i][-1])
                                    bbox_formatted = [f"{coord:.2f}" for coord in dets_to_sort[i][:4]]

                                    log_file.write(
                                        f"Frame: {cur_img.name}, TrackID: {track_id}, "
                                        f"BBox: [{', '.join(bbox_formatted)}], "
                                        f"Vel: [{velocity[0]:.2f}, {velocity[1]:.2f}], "
                                        f"Acc: [{acceleration[0]:.2f}, {acceleration[1]:.2f}], "
                                        f"AngVel: {ang_vel:.5f}\n"
                                    )
                                                
                                # dets_to_sort[i][0:4] 는 letterbox된 이미지 좌표 기준이므로 원본 좌표계로 복원 필요
                                box = dets_to_sort[i, 0:4].copy().reshape(1, 4)  # (1,4) 형태 유지
                                box = scale_coords(img.shape[1:], box, img0.shape, ratio_pad=(ratio, pad)).round()

                                # draw_boxes는 원본 이미지에 직접 그려야 하므로 img0 사용
                                img_with_boxes = draw_boxes(
                                    img0.copy(),  # 원본 이미지에 그리기
                                    [(box[0][0], box[0][1], box[0][2], box[0][3])],  # 좌표
                                    identities=[1],
                                    names=None,
                                    velocities=[velocity],
                                    accelerations=[acceleration],
                                    angular_velocities=[ang_vel],
                                    save_with_object_id=self.opt.save_with_object_id,
                                    path=str(log_file_path)
                                )

                                # === 결과 이미지 저장 ===
                                save_img = not self.opt.nosave
                                if save_img:
                                    save_dir = Path(self.opt.project) / gt_folder_name / f"{initial_image_index}" / f'img1_{i}'
                                    save_dir.mkdir(parents=True, exist_ok=True)
                                    save_path = str(save_dir / cur_img.name)
                                    cv2.imwrite(save_path, img_with_boxes)
                                    print(f"Frame saved to {save_path}")

                            temp = np.zeros((1, 9), dtype=np.float32) if kalman_predict is None else temp

                            if temp is not None:
                                Episode_History.update(track_id, temp, self.currentFr)
                                current_moment = Episode_History[track_id][-1]
                                inpu = Episode_History.get_state_history(track_id)
                                state = self.Agent.get_features(track_id, inpu)
                                index, predicted_fr = self.Agent.select_action(state)

                            self.prevFr = self.currentFr
                            prev_track_id = track_id
                            current_img_indx = get_next_frame_index(current_img_indx, predicted_fr, total_img_num)
                        else : # 이후 프레임 처리
                            for det in pred:
                                if det is not None and len(det):
                                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                                        if int(detclass) == 0:
                                            cx, cy, h, w = kalman_predict[0][:4]
                                            iou = calculate_iou([cx, cy, h, w ], [x1, y1, x2, y2])
                                            #print(f"{iou} calculated  with {cx, cy, h, w } AND {x1}, {y1}, {x2}, {y2}" )
                                            if iou > max_IOU:
                                                max_IOU = iou
                                                best_target = np.array([x1, y1, x2, y2, conf, detclass])

                                    print(f"best target: {best_target} with {max_IOU}")
                                    if max_IOU < 0.4:
                                        self.uncertain_frame_count += 1

                                        if self.uncertain_frame_count >= 5:
                                            print(f" 종료")
                                            log_file.write("종료")
                                            done = True
                                    else:
                                        self.uncertain_frame_count = 0 # 확실한 타겟이 있으므로 카운터 리셋
                                            
                            if best_target is not None:
                                best_target = best_target.reshape(1, -1)
                                kalman_predict = sort_tracker.update(best_target)
                                kalman_states = sort_tracker.getTrackers()
                                #check if predict BB is still inside of frame
                                cx, cy, h, w = kalman_predict[0][:4]
                                
                                if any(val in (0, 640) for val in [cx, cy, h, w ]):
                                    done = True
                                    print(f"Object on target is out of frame: {i}_th person")

                                else:
                                    if len(kalman_states) > 0:
                                        tracker = kalman_states[0]
                                        if hasattr(tracker, 'get_state'):
                                            x1, y1, x2, y2 = best_target[0, 0:4]
                                            velocity = tracker.velocities[-1] if tracker.velocities else (0.0, 0.0)
                                            acceleration = tracker.accelerations[-1] if tracker.accelerations else (0.0, 0.0)
                                            ang_vel = tracker.ang_velocities[-1] if tracker.ang_velocities else 0.0
                                            bbox_state = np.array([x1, y1, x2, y2, *velocity, *acceleration, ang_vel], dtype=np.float32).reshape(1, 9)
                                            temp = convert_bbox_format(bbox_state)

                                            # === 로그 저장 ===
                                            track_id = int(kalman_predict[0, 8])
                                            category = int(best_target[0, -1])  # dets_to_sort[i] → best_target 기준으로 변경
                                            bbox_formatted = [f"{coord:.2f}" for coord in best_target[0, :4]]

                                            log_file.write(
                                                f"Frame: {cur_img.name}, TrackID: {track_id}, "
                                                f"BBox: [{', '.join(bbox_formatted)}], "
                                                f"Vel: [{velocity[0]:.2f}, {velocity[1]:.2f}], "
                                                f"Acc: [{acceleration[0]:.2f}, {acceleration[1]:.2f}], "
                                                f"AngVel: {ang_vel:.5f}\n"
                                            )

                                            # === 원래 좌표계로 복원 ===
                                            box = best_target[0, 0:4].copy().reshape(1, 4)
                                            box = scale_coords(img.shape[1:], box, img0.shape, ratio_pad=(ratio, pad)).round()

                                            # === 바운딩 박스 그리기 및 저장 ===
                                            img_with_boxes = draw_boxes(
                                                img0.copy(),
                                                [(box[0][0], box[0][1], box[0][2], box[0][3])],
                                                identities=[1],
                                                names=None,
                                                velocities=[velocity],
                                                accelerations=[acceleration],
                                                angular_velocities=[ang_vel],
                                                save_with_object_id=self.opt.save_with_object_id,
                                                path=str(log_file_path)
                                            )

                                            save_img = not self.opt.nosave
                                            if save_img:
                                                save_dir = Path(self.opt.project) / gt_folder_name/ f"{initial_image_index}" / f'img1_{i}'
                                                save_dir.mkdir(parents=True, exist_ok=True)
                                                save_path = str(save_dir / cur_img.name)
                                                cv2.imwrite(save_path, img_with_boxes)
                                                print(f"Frame saved to {save_path}")

                                temp = np.zeros((1, 9), dtype=np.float32) if kalman_predict is None else temp
                            else:
                                save_img = not self.opt.nosave
                                if save_img:
                                    save_dir = Path(self.opt.project) / gt_folder_name/ f"{initial_image_index}" / f'img1_{i}'
                                    save_dir.mkdir(parents=True, exist_ok=True)
                                    save_path = str(save_dir / cur_img.name)
                                    cv2.imwrite(save_path, img_with_boxes)
                                    print(f"Frame saved to {save_path}")
                                print("No best target tracked")
                                log_file.write(f"Frame: {cur_img.name}, No Best Target\n")

                            #print(temp)

                            if temp is not None:
                                Episode_History.update(track_id, temp, self.currentFr)
                                current_moment = Episode_History[track_id][-1]
                                prev_moment = Episode_History[track_id][-2] if len(Episode_History[track_id]) >= 2 else current_moment

                                if best_target is None:
                                    reward = -10
                                elif track_id == prev_track_id and track_id is not None:
                                    reward = self.Agent.compute_reward(
                                        prev_moment.current_vector, prev_moment.previous_vector,
                                        current_moment.current_vector, self.prevFr, self.currentFr)
                                else:
                                    reward = -5

                                print(f"Reward: {reward}")

                                next_input = Episode_History.get_state_history(track_id)
                                next_state = self.Agent.get_features(track_id, next_input)
                                if state is not None and index is not None and next_state is not None and reward is not None:
                                    self.Agent.memory.push(state, index, next_state, reward)

                                state = next_state
                                prev_track_id = track_id
                                self.prevFr = self.currentFr
                                index, predicted_fr = self.Agent.select_action(state)
                                current_img_indx = get_next_frame_index(current_img_indx, predicted_fr, total_img_num)
                                if predicted_fr in fr_count:
                                    fr_count[predicted_fr] += 1
                                else:
                                    print(f"Unexpected frame rate: {predicted_fr}")

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
                    self.Agent.EPS = max(self.Agent.EPS_min, self.Agent.EPS * (0.95))
                    #print(self.Agent.EPS )
                
                if (initial_image_index+50) < len(image_files):
                    initial_image_index +=50
                else:
                    initial_image_index = None
            

            print(f"selected_fr : {fr_count}\n")
            # update target net every TARGET_UPDATE episodes
            # Traget Network는 Episdoe(one Video)마다 변화시키는게 안정적임.
            with torch.no_grad():
                self.Agent.target_net.load_state_dict(self.Agent.policy_net.state_dict())
            sort_tracker.reset()
            free_unused_tensors()
            
            # del state, kalman_predict, kalman_states, Episode_History ,sort_tracker
            del self.Agent.memory
            self.Agent.memory = None
            track_id = None
            prev_track_id = None
            sort_tracker = None
            Episode_History = None
        
            gc.collect()
            torch.cuda.empty_cache()       
        self.Agent.save_network()
        print("Training completed.")


def main():
    # === 설정 ===
    dataset_path = Path("/home/hyhy/Datasets/FR_Dataset/MOT17/test")  # 데이터셋 경로
    opt = Options(
        source=str(dataset_path),
        name='MOT17_Training',
        img_size=640,
        conf_thres=0.3,
        iou_thres=0.2,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load('/home/hyhy/Desktop/SYD_DtoS/yolov7.pt', map_location=device)
    agent = Agent(load=False, n_actions=4, device=device)

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