import os
import sys
# 프로젝트 경로를 sys.path에 추가
sys.path.insert(0, '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking')
project_root = '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking'
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn as nn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download

#For SORT tracking
import skimage
from sort import *
from models.yolo import extract_roi_feature, visualize_roi, compute_similarity
import torch.nn.functional as F
from PIL import Image


#............................... Bounding Boxes Drawing ............................
"""Function to Draw Bounding boxes"""
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
        sim = sim[i] if sim is not None else 0.0

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

def process_and_log(log_file, img_name, track_id, bbox, velocity, acceleration, angular_velocity, similarity, im0s):
    """
    바운딩 박스 시각화 및 로그 작성 함수

    Args:
        log_file: 로그 파일 객체.
        img_name: 현재 프레임 이미지 이름.
        track_id: 객체 ID.
        bbox: 바운딩 박스 좌표 (x1, y1, x2, y2).
        velocity: 속도 (vx, vy).
        acceleration: 가속도 (ax, ay).
        angular_velocity: 각속도.
        im0s: 이미지 (시각화를 위한 입력 이미지).
    """
    x1, y1, x2, y2 = bbox
    bbox_formatted = [f"{x1}", f"{y1}", f"{x2}", f"{y2}"]

    # 로그 작성
    log_file.write(
        f"Frame: {img_name}, TrackID: {track_id}, "
        f"BBox: [{', '.join(bbox_formatted)}], "
        f"Vel: [{velocity[0]:.2f}, {velocity[1]:.2f}], "
        f"Acc: [{acceleration[0]:.2f}, {acceleration[1]:.2f}], "
        f"AngVel: {angular_velocity:.5f}, "
    )

    # DRAW_BOXES 호출
    return draw_boxes(
        im0s,
        [[x1, y1, x2, y2]],  # bbox 리스트로 전달
        identities=[track_id],
        velocities=[velocity],
        accelerations=[acceleration],
        angular_velocities=[angular_velocity],
        sim=[similarity]
    )

#............................... Frame Rate Filtering Function ............................
def get_next_frame_index(current_index, fps, total_images):
    """
    Args:
        current_index (int): 현재 이미지의 인덱스 (0-based).
        fps (int): 현재 선택된 frame rate (10, 20, 30, 60 중 하나).
        total_images (int): 전체 이미지 파일의 개수.

    Returns:
        int: 다음 프레임의 인덱스. 범위를 초과하면 None 반환.
    """
    # Frame increment 계산
    if fps == 60:
        increment = 1  # 60fps는 모든 프레임을 사용
    elif fps == 30:
        increment = 2  # 매 2번째 프레임
    elif fps == 20:
        increment = 3  # 매 3번째 프레임
    elif fps == 10:
        increment = 6  # 매 6번째 프레임
    else:
        raise ValueError("Invalid fps value. Choose from 60, 30, 20, or 10.")

    # 다음 프레임 인덱스 계산
    next_index = current_index + increment

    # 인덱스가 범위를 초과하는지 확인
    if next_index >= total_images:
        return None  # 끝에 도달했으면 None 반환
    return next_index

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def update_object_info(object_info, object_id, current_center):
    """
    객체의 위치, 속도, 가속도, 각속도를 업데이트
    """
    if object_id in object_info:
        # 이전 정보 가져오기
        positions = object_info[object_id]['positions']
        velocities = object_info[object_id]['velocities']
        angular_velocities = object_info[object_id]['angular_velocities']

        # 속도 계산
        prev_center = positions[-1]
        vx = (current_center[0] - prev_center[0])
        vy = (current_center[1] - prev_center[1])
        current_velocity = (vx, vy)

        # 가속도 계산
        if velocities:
            prev_velocity = velocities[-1]
            ax = (current_velocity[0] - prev_velocity[0])
            ay = (current_velocity[1] - prev_velocity[1])
            acceleration = (ax, ay)
        else:
            acceleration = (0, 0)

        # 각속도 계산
        if len(positions) > 1:
            cx1, cy1 = positions[-2]
            cx2, cy2 = positions[-1]
            cx3, cy3 = current_center

            vec1 = np.array([cx2 - cx1, cy2 - cy1])
            vec2 = np.array([cx3 - cx2, cy3 - cy2])
            angle_change = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
            angular_velocity = angle_change
        else:
            angular_velocity = 0.0

        # 정보 업데이트
        positions.append(current_center)
        velocities.append(current_velocity)
        angular_velocities.append(angular_velocity)

        object_info[object_id].update({
            'positions': positions,
            'velocities': velocities,
            'acceleration': acceleration,
            'angular_velocities': angular_velocities
        })
    else:
        # 새로운 객체 초기화
        object_info[object_id] = {
            'positions': [current_center],
            'velocities': [],
            'acceleration': (0, 0),
            'angular_velocities': []
        }

object_history = {} 
initial_features = []
def add_to_initial_features(id_counter, roi_feature):
    global object_history

    if id_counter not in object_history:
        object_history[id_counter] = 1
    else:
        object_history[id_counter] += 1

    if object_history[id_counter] >= 3: 
        initial_features.append(roi_feature)
        object_history[id_counter] = 0

def adjust_threshold(iou):
    global max_similarity1, max_similarity2
    if iou > 0.7: 
        max_similarity1 = 0.997  
        max_similarity2 = 0.995  
    elif iou > 0.5:
        max_similarity1 = 0.995  
        max_similarity2 = 0.992
    else:  
        max_similarity1 = 0.992 
        max_similarity2 = 0.990  
    print(max_similarity1,max_similarity2)

def get_image_size(path):
    with Image.open(path) as img:
        width, height = img.size
    return width, height
        
def sot(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id, fps = (
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, 
        not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id, opt.fps
    )
    opt.fps = 60
    possible_gt_files = ["groundtruth.txt", "groundtruth_rect.txt"]
    gt_path = None
    for filename in possible_gt_files:
        gt_paths = Path(source_path).parent / filename
        if gt_paths.exists():
            gt_path = gt_paths
            break
    feature_maps = []
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')
    )

    # Initialize SORT tracker
    sort_tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.3)

    # Directories
    save_dir = Path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = save_dir / f"{opt.name}.txt"

    initial_detections = []
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        line_count = 0  # 저장된 라인 수를 추적
        for line in lines:
            line = line.strip()
            separators = [',', '\t', '\t\t']
            for sep in separators:
                if sep in line:
                    try:
                        x1, y1, w, h = map(float, line.split(sep))
                        confidence = 1.0  # 고정된 신뢰도
                        initial_detections.append([x1, y1, x1 + w, y1 + h, confidence])
                        line_count += 1
                        break 
                    except ValueError:
                        print(f"Invalid line with separator '{sep}': {line}")
                    break
                print(initial_detections)
            if line_count >= 5:  # 최대 10줄까지만 저장
                break

    initial_detections = np.array(initial_detections)    
    print(f"Loaded {len(initial_detections)} ground truth detections.")

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load YOLO model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    def save_features(module, input, output):
        """Hook for saving feature maps."""
        feature_maps.append(output.clone().detach())

    # Hook 등록
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):  # 특정 레이어 선택 가능
            layer.register_forward_hook(save_features)

    # Data loader
    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        image_files = list(Path(source).rglob("*.jpg"))
        total_images = len(image_files)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    image_count = 0
    t0 = time.time()
    
    with open(log_file_path, 'w') as log_file:
            initial_features = []
            yolo_features = []
            detected_boxes = [] 
            frame_count = 1
            object_info = {}
            sort=0
            yolo=0
            fail=0
            max_similarity1 = 0.994
            max_similarity2 = 0.991
            id_counter=0
            log_file.write(f"{max_similarity1}/{max_similarity2}\n")
            for path, img, im0s, vid_cap in dataset:
                image_count += 1
                img_name = Path(path).stem
                width, height = get_image_size(path)

                # Preprocess image
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                feature_maps.clear()
                _ = model(img)  # Hook를 통해 feature maps 저장
                current_feature_map = feature_maps[0]

                if frame_count < 5:  # 초기 프레임 처리
                    target_bbox = initial_detections[frame_count]
                    x1, y1, x2, y2 = map(int, target_bbox[:4])
                    confidence = target_bbox[4]
                    cls = 0
                    initial_detection = [[x1, y1, x2, y2, confidence, cls]]

                    # SORT 초기화
                    tracked_dets = sort_tracker.update(np.array(initial_detection))
                    if len(tracked_dets) > 0:
                        print(f"초기 추적 대상 객체 바운딩 박스=[{x1}, {y1}, {x2}, {y2}]")
                    else:
                        print("초기 프레임에서 대상 추적 실패.")
                        continue

                    roi_feature = extract_roi_feature([current_feature_map], x1, y1, x2, y2, width, height)
                    initial_features.append(roi_feature)

                    visualize_roi(im0s, x1, y1, x2, y2, label=f"ID {frame_count}")

                    # Draw bounding boxes and save results
                    if len(tracked_dets) > 0:
                        for det in tracked_dets:
                            x1, y1, x2, y2, track_id = map(int, det[:5])
                            label = f"ID {track_id}"
                            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            update_object_info(object_info, track_id, current_center)
                            cv2.rectangle(im0s, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(im0s, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            bbox_formatted = [f"{x1}", f"{y1}", f"{x2}", f"{y2}"]

                            # 객체 정보 가져오기
                            if track_id in object_info:
                                velocity = object_info[track_id]['velocities'][-1] if object_info[track_id]['velocities'] else (0, 0)
                                acceleration = object_info[track_id]['acceleration']
                                angular_velocity = object_info[track_id]['angular_velocities'][-1] if object_info[track_id]['angular_velocities'] else 0.0

                                # 텍스트 정보 저장
                                log_file.write(
                                    f"Frame: {img_name}, TrackID: {track_id}, "
                                    f"BBox: [{', '.join(bbox_formatted)}], "
                                    f"Vel: [{velocity[0]:.2f}, {velocity[1]:.2f}], "
                                    f"Acc: [{acceleration[0]:.2f}, {acceleration[1]:.2f}], "
                                    f"AngVel: {angular_velocity:.5f}\n")

                        save_path = save_dir / f"{img_name}.jpg"
                        cv2.imwrite(str(save_path), im0s)
                        print(f"Frame saved to {save_path}")
                    else:
                        print("No tracked detections. Skipping save.")

                    frame_count += 1
                    continue
                
                Max_len = 1
                matched = False
                if frame_count >= 5:  # 이후 프레임 처리
                    pred = model(img, augment=opt.augment)[0] # YOLO 수행
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                    max_similarity = max_similarity2
                    best_target = None

                    if pred[0] is not None:
                        pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], im0s.shape).round()
                        for saved_feature in initial_features:
                            for det in pred[0]:
                                x1, y1, x2, y2, conf, cls = det.cpu().detach().numpy()
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                roi_feature = extract_roi_feature([current_feature_map], x1, y1, x2, y2, width, height)
                                similarity = compute_similarity(saved_feature, roi_feature, 0)
                                
                                if similarity > max_similarity:
                                    max_similarity = similarity
                                    best_target = (x1, y1, x2, y2, conf, roi_feature)
                                    matched = True
                        
                    if best_target:
                        x1, y1, x2, y2, conf, roi_feature = best_target
                        print(f"Target found by YOLO: {x1, y1, x2, y2} | Similarity: {max_similarity:.3f}")
                        current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        update_object_info(object_info, track_id, current_center)
                        if track_id in object_info:
                            velocity = object_info[track_id]['velocities'][-1] if object_info[track_id]['velocities'] else (0, 0)
                            acceleration = object_info[track_id]['acceleration']
                            angular_velocity = object_info[track_id]['angular_velocities'][-1] if object_info[track_id]['angular_velocities'] else 0.0

                            im0s = process_and_log(log_file, img_name, track_id, [x1, y1, x2, y2],velocity, acceleration, angular_velocity, max_similarity, im0s)
                            log_file.write("yolo\n")
                                        
                        if len(detected_boxes) >= Max_len:
                            detected_boxes.pop(0) 

                        detected_boxes.append([x1, y1, x2, y2, conf, cls])
                        tracked_dets = sort_tracker.update(np.array(detected_boxes))
                        if similarity>0.994:
                            if len(initial_detections)>5 : 
                                initial_detections.pop(0)
                            initial_features.append(roi_feature)
                        yolo += 1
                    else:           # SORT 수행
                        #print("YOLO failed, switching to SORT.")
                        max_similarity = max_similarity1
                        if len(detected_boxes) is not None:
                            tracked_dets = sort_tracker.update(np.array(detected_boxes))
                            for saved_feature in initial_features:
                                for det in tracked_dets:
                                    x1, y1, x2, y2, track_id = map(int, det[:5])
                                    roi_feature = extract_roi_feature([current_feature_map], x1, y1, x2, y2, width, height)
                                    similarity = compute_similarity(saved_feature, roi_feature, 0)
                                        
                                    if similarity > max_similarity:
                                        max_similarity = similarity
                                        best_target = (x1, y1, x2, y2, conf, similarity)
                                        
                            if best_target:
                                x1, y1, x2, y2, conf, similarity = best_target
                                detected_boxes.append([x1, y1, x2, y2, 1, 0])
                                print(f"SORT tracking detected: {x1, y1, x2, y2} | Similarity: {similarity:.3f}")
                                current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                update_object_info(object_info, track_id, current_center)

                                # 객체 정보 가져오기
                                if track_id in object_info:
                                    velocity = object_info[track_id]['velocities'][-1] if object_info[track_id]['velocities'] else (0, 0)
                                    acceleration = object_info[track_id]['acceleration']
                                    angular_velocity = object_info[track_id]['angular_velocities'][-1] if object_info[track_id]['angular_velocities'] else 0.0

                                    im0s = process_and_log(log_file, img_name, track_id, [x1, y1, x2, y2], velocity, acceleration, angular_velocity, max_similarity, im0s)
                                    log_file.write("sort\n")
                                    
                                if len(detected_boxes) >= Max_len:
                                    detected_boxes.pop(0) 
                                detected_boxes.append([x1, y1, x2, y2, conf, cls]) 
                                tracked_dets = sort_tracker.update(np.array(detected_boxes))
                                sort+=1 

                            else: 
                                log_file.write(f"Frame: {img_name}, Failed to find a matched object\n")
                                fail += 1


                    # 결과 저장
                    save_path = save_dir / f"{img_name}.jpg"
                    cv2.imwrite(str(save_path), im0s)
                    print(f"Frame saved to {save_path}")
                    frame_count += 1

    print(f"Results saved to {save_dir}")
    print(f"Done. ({time.time() - t0:.3f}s)")
    print(f"Tracked : {sort}, YOLO : {yolo}, No detection : {fail}")
    

if __name__ == '__main__':
    source_path = '/home/hyhy/Datasets/FR_Dataset/LaSOT/dog/dog-1/img'
    gt_path = Path(source_path).parent / "groundtruth_rect.txt"
    folder_name = os.path.basename(os.path.dirname(source_path))

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/hyhy/Desktop/yolov7.pt', help='model.pt path')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for image sampling (60, 30, 20, 10)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default=source_path, help='source file/folder path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1,help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/sot3', help='project save folder')
    parser.add_argument('--name', default=folder_name, help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='dont trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    parser.set_defaults(download=True)

    opt = parser.parse_args()
    print(opt)

    if opt.download and not os.path.exists(''.join(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')
    
    with torch.no_grad():
        sot(opt)