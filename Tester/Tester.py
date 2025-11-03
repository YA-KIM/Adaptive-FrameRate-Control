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
from yolov7_object_tracking.DnT_by_frame import *
from yolov7_object_tracking.utils.download_weights import download
import random
from dataclasses import dataclass
from typing import List, Optional
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import auc


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
    conf_thres: float = 0.1                                # 객체 탐지 신뢰도 임계값
    iou_thres: float = 0.2                                 # NMS에서 IOU 임계값
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

def load_groundtruth(gt_file, bbox_format, sep=','):
    """
    groundtruth.txt 파일에서 바운딩 박스 정보 로드
    bbox_format에 따라 (x1, y1, x2, y2) 또는 (x1, y1, width, height) 형식에 맞게 처리
    """
    gt_bboxes = []
    with open(gt_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split(sep)))  # 각 값을 float으로 변환

            if bbox_format == 'xyxy':
                if len(values) == 4:  # (x1, y1, x2, y2)
                    x1, y1, x2, y2 = values
                else:
                    raise ValueError(f"Unexpected format in line: {line}")

            else:
                if len(values) == 4:  # (x1, y1, width, height)
                    x1, y1, w, h = values
                    x2, y2 = x1 + w, y1 + h
                else:
                    raise ValueError(f"Unexpected format in line: {line}")

            # 바운딩 박스 정보와 confidence를 리스트에 추가
            gt_bboxes.append([x1, y1, x2, y2])

    return gt_bboxes  # 각 프레임에 대한 바운딩 박스 정보 리스트 반환


def compute_auc(x_values, y_values):
    return auc(x_values, y_values)

# 평가 클래스
class TrackingEvaluator:
    def __init__(self, distance_threshold_set=None, iou_threshold_set=None):
        self.distance_thresholds = distance_threshold_set if distance_threshold_set is not None else [5, 10, 20, 30, 50]
        self.iou_thresholds = iou_threshold_set if iou_threshold_set is not None else [0.3, 0.5, 0.7]

        self.precision_stats = {thr: {'correct': 0, 'wrong': 0} for thr in self.distance_thresholds}
        self.success_stats = {thr: {'correct': 0, 'wrong': 0} for thr in self.iou_thresholds}

    def update(self, pred_bbox, gt_bbox):
        # 중심 좌표 거리 계산
        pred_center = np.array([(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2])
        gt_center = np.array([(gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2])
        distance = np.linalg.norm(pred_center - gt_center)

        for thr in self.distance_thresholds:
            if distance <= thr:
                self.precision_stats[thr]['correct'] += 1
            else:
                self.precision_stats[thr]['wrong'] += 1

        iou = calculate_iou(pred_bbox, gt_bbox)
        for thr in self.iou_thresholds:
            if iou >= thr:
                self.success_stats[thr]['correct'] += 1
            else:
                self.success_stats[thr]['wrong'] += 1

    def get_result(self):
        precision_curve = []
        success_curve = []

        for thr in self.distance_thresholds:
            stats = self.precision_stats[thr]
            total = stats['correct'] + stats['wrong']
            precision_curve.append(stats['correct'] / total if total > 0 else 0)

        for thr in self.iou_thresholds:
            stats = self.success_stats[thr]
            total = stats['correct'] + stats['wrong']
            success_curve.append(stats['correct'] / total if total > 0 else 0)

        return {
            'precision_curve': precision_curve,
            'success_curve': success_curve
        }

    def get_average(self):
        precision_values = []
        success_values = []

        for thr in self.distance_thresholds:
            stats = self.precision_stats[thr]
            total = stats['correct'] + stats['wrong']
            if total > 0:
                precision_values.append(stats['correct'] / total)

        for thr in self.iou_thresholds:
            stats = self.success_stats[thr]
            total = stats['correct'] + stats['wrong']
            if total > 0:
                success_values.append(stats['correct'] / total)

        avg_precision = np.mean(precision_values) if precision_values else 0
        avg_success = np.mean(success_values) if success_values else 0

        return {
            'average_precision': avg_precision,
            'average_success_rate': avg_success
        }

    def get_auc(self):
        result = self.get_result()
        raw_precision_auc = compute_auc(self.distance_thresholds, result['precision_curve'])
        success_auc = compute_auc(self.iou_thresholds, result['success_curve'])
        precision_auc = raw_precision_auc / (self.distance_thresholds[-1] - self.distance_thresholds[0])
    
        return {
            'precision_auc': precision_auc,
            'success_auc': success_auc
        }

    def plot_curves(self, save_path=None):
        result = self.get_result()
        auc_result = self.get_auc()

        precision_curve = result['precision_curve']
        success_curve = result['success_curve']
        precision_auc = auc_result['precision_auc']
        success_auc = auc_result['success_auc']

        plt.figure(figsize=(10, 6))
        plt.plot(self.distance_thresholds, precision_curve, label=f'Precision (AUC={precision_auc:.3f})', color='b', marker='o')
        plt.plot(self.iou_thresholds, success_curve, label=f'Success Rate (AUC={success_auc:.3f})', color='g', marker='s')
        plt.xlabel('Threshold')
        plt.ylabel('Performance')
        plt.title('Precision and Success Rate vs Threshold')
        plt.legend()
        plt.grid(True)

        # 그래프 내부에 텍스트로 AUC 표시
        plt.text(0.05, 0.05, f'Precision AUC: {precision_auc:.3f}\nSuccess AUC: {success_auc:.3f}',
                 transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.6))

        if save_path:
            plt.savefig(save_path)
            print(f"그래프 저장 완료: {save_path}")
        else:
            plt.show()

        plt.close()


# === Test 수행 ===
class SOT_with_DRL_Test:
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

    def Test(self):
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

        # 학습된 모델 로드
        self.Agent.load_network()
        model = torch.load("/home/hyhy/Desktop/SYD_DtoS/DRL_FR/models/q_network_feature.pth")
        #model = torch.load("/home/hyhy/Desktop/SYD_DtoS/DRL_FR/models/q_network_feature.pth", map_location=torch.device('cpu'))
        torch.save(model.state_dict(), "/home/hyhy/Desktop/SYD_DtoS/DRL_FR/models/q_network_feature_fixed.pth")
        #self.Agent.feature_extractor.load_state_dict(torch.load("/home/hyhy/Desktop/SYD_DtoS/DRL_FR/models/q_network_feature.pth"))  
        self.Agent.feature_extractor.eval() 
        self.Agent.EPS = 0  # 테스트 시 Exploration 없이 활용(Exploitation)만 수행
        gt_format = None
        evaluator = TrackingEvaluator(distance_threshold_set=np.arange(0, 51, 1), iou_threshold_set=np.arange(0, 1.05, 0.05))

        for i_episode, test_folder in enumerate(subfolders):
            img_folder = os.path.join(test_folder, 'img')
            gt_file = os.path.join(test_folder, 'groundtruth.txt')  # GT 파일 경로

            if not os.path.exists(img_folder) or not os.path.exists(gt_file):
                print(f"필수 파일 없음: {img_folder} 또는 {gt_file}")
                continue

            image_files = sorted(list(Path(img_folder).glob("*.jpg")))
            total_img_num = len(image_files)
            if total_img_num == 0:
                print(f"No images found in: {img_folder}")
                continue

            # GT 바운딩 박스 로드
            bbox_format = gt_format
            gt_bboxes = load_groundtruth(gt_file, bbox_format, sep=',')

            print(f"Episode {i_episode + 1} 시작: {img_folder}")

            # SORT 초기화
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
            self.currentFr, self.prevFr = 60, 60  # 초기 프레임 속도 (초반 5프레임은 무조건 60)
            predicted_fr = 60
            initial_features = []
            state = None
            best_target = None
            first_hist = None
            temp = None
            fr_count = {20: 0, 30: 0, 60: 0}    

            while not done and current_img_indx is not None:
                cur_img = image_files[current_img_indx]
                frame_id = current_img_indx  # 현재 프레임 ID

                # YOLO + SORT 실행 (객체 탐지 및 추적)
                with torch.no_grad():
                    kalman_predict, kalman_states, initial_features, best_target, SoT, first_hist, gt_format = detect_frame(
                        model=self.yolo_model, device=self.device, half=True, opt=self.opt, source=cur_img, 
                        gt_folder=test_folder, initial_features=initial_features, prev_best_target=best_target, 
                        sort_tracker=SoT, first_hist=first_hist, gt_format=gt_format, frame_rate=60)  # 초반 5프레임 고정

                # Kalman tracker 정보 추출
                if kalman_states and isinstance(kalman_states, list):
                    tracker = kalman_states[0]  

                    if hasattr(tracker, 'get_state'):
                        x1, y1, x2, y2 = tracker.get_state()[0][:4]  # bbox 정보 추출
                        velocity = tracker.velocities[-1] if tracker.velocities else (0.0, 0.0)
                        acceleration = tracker.accelerations[-1] if tracker.accelerations else (0.0, 0.0)
                        ang_vel = tracker.ang_velocities[-1] if tracker.ang_velocities else 0.0

                        bbox_state = np.array([x1, y1, x2, y2, *velocity, *acceleration, ang_vel], dtype=np.float32).reshape(1, 9)
                        temp = convert_bbox_format(bbox_state)
                        predicted_bbox = np.array([x1, y1, x2, y2])  # 예측된 bbox 저장

                    if kalman_predict is not None and isinstance(kalman_predict, np.ndarray) and kalman_predict.size > 0:
                        self.current_obj_id = kalman_predict[0, -1]  # 객체 ID 업데이트
                    else:
                        self.current_obj_id = self.prev_obj_id  # 이전 객체 ID 유지
                        temp = np.zeros((1, 9), dtype=np.float32)  # temp = None

                # 초반 5 프레임은 목표 객체 식별만 수행 (DRL 미적용)
                if current_img_indx < 5:
                    #print(f"초반 탐색 단계 (Frame {current_img_indx+1}/5): 객체 탐색 중...")
                    self.prevFr = self.currentFr  # 60 고정
                    current_img_indx = get_next_frame_index(current_img_indx, 60, total_img_num)  # 60fps 유지
                    continue  # DRL 적용 X

                # === GT와 IOU 비교 ===
                
                if frame_id < len(gt_bboxes):
                    gt_bbox = np.array(gt_bboxes[frame_id])  # 현재 프레임의 GT 바운딩 박스
                    if 'predicted_bbox' not in locals():  # predicted_bbox가 정의되지 않았으면
                        predicted_bbox = np.array([0, 1, 0, 1])
                
                    evaluator.update(predicted_bbox, gt_bbox)

                '''
                    averages = evaluator.get_average()
                    print("Average Precision:", averages['average_precision'])
                    print("Average Success Rate:", averages['average_success_rate'])

                    iou = calculate_iou(predicted_bbox, gt_bbox)
                    if iou >= 0.5:
                        success_count += 1
                    else:
                        fail_count += 1

                    if current_img_indx % 5 == 0:
                        print(f"IOU = {iou:.3f}, {'SUCCESS' if iou >= 0.5 else 'FAIL'}")
                '''

                # === State Construction (5프레임 이후) ===  
                if temp is not None:
                    Episode_History.update(self.current_obj_id, temp, self.currentFr)
                    current_moment = Episode_History[self.current_obj_id][-1]
                    inpu = Episode_History.get_state_history(self.current_obj_id)
                    # === Feature Extraction ===
                    state = self.Agent.get_features(self.current_obj_id, inpu)
                    # === DQN (Q-Network) ===
                    index, predicted_fr = self.Agent.select_action(state)

                # === 프레임 업데이트 ===
                self.prevFr = self.currentFr
                self.currentFr = predicted_fr
                current_img_indx = get_next_frame_index(current_img_indx, predicted_fr, total_img_num)
                if predicted_fr in fr_count:
                        fr_count[predicted_fr] += 1

                # 프레임 종료 조건
                if current_img_indx is None:
                    done = True

            print(f"테스트 {i_episode + 1} 완료!")
            print(f"selected_fr : {fr_count}")
            auc_result = evaluator.get_auc() 
            print("Success AUC:", auc_result['success_auc'],"AND Precision AUC:", auc_result['precision_auc'],"\n")

            SoT.reset()
            del state, kalman_predict, kalman_states, initial_features, Episode_History, SoT
            gc.collect()
            torch.cuda.empty_cache()

        # 최종 결과 출력
        print(f"\n=== 최종 테스트 결과 ===")
        averages = evaluator.get_average()
        print("Average Success Rate:", averages['average_success_rate']," AND Average Precision:", averages['average_precision'] )

        auc_result = evaluator.get_auc() 
        print( "Success AUC:", auc_result['success_auc'],"AND Precision AUC:", auc_result['precision_auc'])
        print("=== Testing Completed ===")

        # 결과 그래프 저장
        evaluator.plot_curves('/home/hyhy/Desktop/SYD_DtoS/DRL_FR/result/performance_plot.png')


def main():
    # === 설정 ===
    dataset_path = Path("/home/hyhy/Datasets/FR_Dataset/LaSOT2/val")  # 테스트할 데이터셋 경로
    opt = Options(
        source=str(dataset_path),
        name='LaSOT_Testing',
        project='/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/Testing',
        img_size=640,
        conf_thres=0.2,
        iou_thres=0.3,
        device='cuda'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load('/home/hyhy/Desktop/yolov7.pt', map_location=device)
    agent = Agent(num_episodes=1, load=True, n_actions=3, device=device)  # 모델 로드 모드 설정

    if device.type == "cuda":
        model.half()  
    else:
        model.float() 

    # SOT_with_DRL_Test 인스턴스 생성
    tester = SOT_with_DRL_Test(agent=agent, DataSetPath=dataset_path, yolo_model=model, opt=opt)
    '''
    # 네트워크 로드
    agent.load_network()
    agent.load_extractor()'
    '''

    # 테스트 시작
    tester.Test()

if __name__ == "__main__":
    main()
