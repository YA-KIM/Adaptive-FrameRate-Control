import os
import sys
# 프로젝트 경로를 sys.path에 추가
sys.path.insert(0, '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking')
project_root = '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking'
if project_root not in sys.path:
    sys.path.append(project_root)
import argparse

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
from utils.datasets import LoadStreams, LoadImages, LoadImage
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.download_weights import download

#For SORT tracking
import skimage
from sort import *
from models.yolo import extract_roi_feature, visualize_roi, compute_similarity, get_histogram, compute_histogram_similarity
import torch.nn.functional as F
from PIL import Image

from dataclasses import dataclass
from typing import List, Optional
import gc

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
    conf_thres: float = 0.3                                # 객체 탐지 신뢰도 임계값
    iou_thres: float = 0.2                                 # NMS에서 IOU 임계값
    device: str = ''                                       # 사용 디바이스 (cpu/cuda)
    augment: bool = False                                  # 증강 추론 여부
    no_trace: bool = False                                 # 모델 트레이싱 비활성화
    update: bool = False                                   # 모델 업데이트 여부

    # 결과 저장 및 시각화
    view_img: bool = False                                 # 결과 시각화 여부
    save_txt: bool = False                                 # 탐지 결과 txt 저장
    save_conf: bool = False                                # txt에 신뢰도 저장 여부
    nosave: bool = False                                   # 이미지/비디오 저장 안 함
    save_bbox_dim: bool = False                            # 바운딩 박스 크기 저장
    save_with_object_id: bool = False                      # 객체 ID와 함께 저장
    classes: Optional[List[int]] = None                    # 특정 클래스 필터링
    agnostic_nms: bool = False                             # 클래스 무관 NMS
    colored_trk: bool = False                              # 각 트랙에 색상 지정
    download: bool = True                                  # 모델 다운로드 여부
    fps: int = 60                                          # 초당 프레임 수

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

def draw_box(img, bbox, identity=None, name=None, velocity=None, acceleration=None, angular_velocity=None, sim=None, save_with_object_id=False, path=None, offset=(0, 0)):
    """
    단일 객체만 처리하도록 draw_boxes() 함수 수정.
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]

    id = int(identity) if identity is not None else 0
    vel = velocity if velocity is not None else (0, 0)
    acc = acceleration if acceleration is not None else (0, 0)
    ang_vel = angular_velocity if angular_velocity is not None else 0.0

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # 속도 벡터 끝점 계산 (스케일링 적용)
    scale = 5  # 속도 벡터 크기 스케일링
    end_x = int(center_x + vel[0] * scale)
    end_y = int(center_y + vel[1] * scale)

    data = (int((x1 + x2) / 2), int((y1 + y2) / 2))
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

    cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.1)

    if save_with_object_id and path is not None:
        txt_str = (f"{x1 / img.shape[1]:.6f} {y1 / img.shape[0]:.6f} "
                   f"{x2 / img.shape[1]:.6f} {y2 / img.shape[0]:.6f} "
                   f"{(x1 + x2 / 2) / img.shape[1]:.6f} "
                   f"{(y1 + y2 / 2) / img.shape[0]:.6f}\n")
        with open(path + '.txt', 'a') as f:
            f.write(txt_str)

    return img


#............................... Calculate iou ............................
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

#............................... Determine GT Bbox Format ............................
def determine_bbox_format(gt_bbox, predicted_bbox):
    """
    Determine if the GT bounding box is in 'xyxy' or 'xywh' format.
    """
    # If the predicted bbox format is `xyxy`, and GT bbox is matched, check if it can be `xywh`
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted_bbox

    x1, y1, x2, y2 = gt_bbox
    iou_xyxy = calculate_iou([pred_x1, pred_y1, pred_x2, pred_y2], [x1, y1, x2, y2])

    x, y, w, h = gt_bbox
    x1, y1, x2, y2 = x, y, x+w, y+h
    iou_xywh = calculate_iou([pred_x1, pred_y1, pred_x2, pred_y2], [x1, y1, x2, y2])

    # If x2 < x1 or y2 < y1, we likely have a `xywh` format

    if iou_xywh>iou_xyxy:
        print("GT box is in 'xywh' format")
        return 'xywh'  # w, h form
    else:
        print("GT box is in 'xyxy' format")
        return 'xyxy'  # xyxy format

#............................... Check Image Size ............................
def get_image_size(path):
    with Image.open(path) as img:
        width, height = img.size
    return width, height

#............................... Yolo detection + Sort Tracking ............................
def detect_frame(model, device, half, opt: Options, source=None, gt_folder=None, initial_features=None, prev_best_target=None, sort_tracker =None, first_hist=None, gt_format = None, frame_rate =60):
    weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id, fps = (
        opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id, opt.fps)
    stride = int(model.stride.max())
    opt.img_size = check_img_size(opt.img_size, s=stride) 
    opt.fps=60

    possible_gt_files = ["groundtruth.txt", "groundtruth_rect.txt"]
    gt_path = None
    for filename in possible_gt_files:
        gt_path = Path(gt_folder) / filename
        if gt_path.exists():
            break

    # Directories
    gt_folder_name = os.path.basename(gt_folder)
    save_dir = Path(Path(opt.project) / gt_folder_name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = save_dir / f"{gt_folder_name}.txt" 
    log_file = open(log_file_path, "a")  

    save_img = not opt.nosave and not str(source).endswith('.txt')
    webcam = False

    # Initialize SORT tracker
    if sort_tracker is None:
        sort_tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

    feature_maps = []
    def save_features(module, input, output):
        """Hook for saving feature maps."""
        feature_maps.append(output.detach().cpu())

    # Hook 등록
    target_layer = None
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Conv2d):
            target_layer = layer
            break

    assert target_layer is not None, "Conv2d 레이어를 찾을 수 없습니다."
    feature_maps.clear()
    hook_handle = target_layer.register_forward_hook(save_features)

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    elif Path(source).is_file():
        dataset = LoadImage(source, img_size =imgsz, stride=stride)
    else:
        image_files = list(Path(source).rglob("*.jpg"))
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    with torch.no_grad():
        initial_features = initial_features if initial_features is not None else []
        prev_best_target = prev_best_target if prev_best_target is not None else None
        object_lifetime = {}
        path, img, im0s = dataset.load()    #img = 전처리 되는 이미지 (for Yolo) / im0s = 완전한 원본 이미지
        img_name = Path(path).stem  # 이미지 파일명 추출 (확장자 제외)
        width, height = get_image_size(path)
        
        # Inference 준비
        img = torch.from_numpy(img).to(device).detach()
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        feature_maps.clear()
        _ = model(img)  # Hook를 통해 feature maps 저장
        current_feature_map = feature_maps[0]
        hook_handle.remove()

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference + Apply NMS
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if int(img_name) <= 5:  # 초기 프레임 처리
                initial_detections = []
                tracks = []
                with open(gt_path, 'r') as f:
                    for i, line in enumerate(f):
                        if i == int(img_name) - 1:
                            line = line.strip()
                            separators = [',', '\t', '\t\t']  # 다양한 구분자 처리
                            for sep in separators:
                                if sep in line:
                                    try:
                                        values = list(map(float, line.split(sep)))
                                        x1, y1, x2, y2 = values

                                        # x2, y2가 좌표가 아닌 w, h일 가능성 체크
                                        if x2 <= x1 or y2 <= y1:
                                            gt_format = 'xywh'
                                            print("GT box is in 'xywh' format")

                                        confidence = 1.0  # 기본 confidence 값 추가
                                        initial_detections.append([x1, y1, x2, y2, confidence])
                                        break
                                    except ValueError:
                                        print(f"Invalid line with separator '{sep}': {line}")
                                    break
                target_bbox = np.array(initial_detections[0], dtype=float) 
                print(target_bbox)
                
                # target_bbox가 NumPy 배열인지 확인 후 변환
                if isinstance(target_bbox, np.ndarray) and target_bbox.shape == (5,):
                    x1, y1, x2, y2, confidence = target_bbox.astype(int).tolist()
                cls = 0
                initial_detection = [[x1, y1, x2, y2, confidence, cls]]
                best_target = (x1, y1, x2, y2, confidence, cls)

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for obj in det.cpu().detach().numpy():
                        x1, y1, x2, y2, conf, cls = obj
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        if gt_format is None or gt_format == 'xyxy': # GT 포맷을 결정
                            gt_format = determine_bbox_format(target_bbox[:4], [x1, y1, x2, y2])
                else: 
                    print("No tracked detection in the current frame.") 

                # gt_format을 사용하여 후속 처리
                if gt_format == 'xywh':
                    x1, y1, w, h = target_bbox[:4]
                    x2, y2 = x1 + w, y1 + h
                    initial_detection = [[x1, y1, x2, y2, confidence, 0]]  # xyxy 형식으로 변환
                else:
                    x1,y1,x2,y2= target_bbox[:4]
                    initial_detection = [[x1, y2, x2, y2, confidence, 0]] 

                # SORT 초기화
                tracked_dets = sort_tracker.update(np.array(initial_detection), frame_rate)

                if first_hist is None: 
                    first_hist = get_histogram(im0s, [x1, y1, x2, y2])

                roi_feature = extract_roi_feature([current_feature_map], int(x1), int(y1), int(x2), int(y2), width, height)
                if len(initial_features) > 4:
                    initial_features.pop(2) 
                initial_features.append(roi_feature)
                hook_handle.remove()
                save_path = save_dir / f"{img_name}.jpg"
                cv2.imwrite(str(save_path), im0s)
                im0=im0s
                draw_box(im0, (x1, y1, x2, y2), 0)
            
            else:
                max_similarity = 0.90
                mmax_similarity = 0.92
                best_target = None
                #print(f"1. {len(det)}")

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for obj in det.cpu().detach().numpy():
                        x1, y1, x2, y2, conf, cls = obj
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        if prev_best_target:
                            prev_x1, prev_y1, prev_x2, prev_y2, _, _ = prev_best_target
                            iou = calculate_iou([prev_x1, prev_y1, prev_x2, prev_y2], [x1, y1, x2, y2])
                        
                            if iou == 0:
                                hook_handle.remove()
                                continue 

                            roi_feature = extract_roi_feature([current_feature_map], x1, y1, x2, y2, width, height)
                            hist_sim = compute_histogram_similarity(im0, first_hist, [x1, y1, x2, y2])
                            
                            for saved_feature in initial_features:
                                similarity = compute_similarity(saved_feature, roi_feature)
                                if similarity > max_similarity:  # 가장 유사한 객체 선택
                                    max_similarity = similarity
                                    best_target = (x1, y1, x2, y2, conf, cls)
                            #print(f"feature sim : {similarity}")
                        else: 
                            roi_feature = extract_roi_feature([current_feature_map], x1, y1, x2, y2, width, height)
                            hist_sim = compute_histogram_similarity(im0, first_hist, [x1, y1, x2, y2])

                            for saved_feature in initial_features:
                                similarity = compute_similarity(saved_feature, roi_feature)
                                if similarity > mmax_similarity:  # 가장 유사한 객체 선택
                                    mmax_similarity = similarity
                                    best_target = (x1, y1, x2, y2, conf, cls)

                    # Process SORT tracker
                    dets_to_sort = np.empty((0, 6))
                    tracks =[]
                    tracked_dets = []
                    #print(f"2. {best_target}")
                    # best_target이 존재하면 SORT에 전달
                    if best_target:
                        x1, y1, x2, y2, conf, detclass = best_target 
                        dets_to_sort = np.array([[x1, y1, x2, y2, conf, detclass]])
                        tracked_dets = sort_tracker.update(dets_to_sort)
                        #print(f"3. {len(tracked_dets)}")

                        if len(tracked_dets) == 0:
                            print("No tracked detection in the current frame.") 
                            save_path = save_dir / f"{img_name}.jpg"
                            cv2.imwrite(str(save_path), im0s)
                            log_file.write(f"Frame: {img_name}, No target detected\n")
                            best_target = None
                            tracked_dets = sort_tracker.update()
                            print(f"Frame saved to {save_path}")
                            return [], [], initial_features, best_target, sort_tracker, first_hist, gt_format
                        else: 
                            tracks = sort_tracker.getTrackers()
                            x1, y1, x2, y2 = tracked_dets[0, :4]
                            track_id = int(tracked_dets[0, 8])
                            category = tracked_dets[0, 4]

                            velocity = tracks[0].velocities[-1] if hasattr(tracks[0], 'velocities') and tracks[0].velocities else (0, 0)
                            acceleration = tracks[0].accelerations[-1] if hasattr(tracks[0], 'accelerations') and tracks[0].accelerations else (0, 0)
                            angular_velocity = tracks[0].ang_velocities[-1] if hasattr(tracks[0], 'ang_velocities') and tracks[0].ang_velocities else 0.0

                            bbox_formatted = [f"{coord:.2f}" for coord in [x1, y1, x2, y2]]

                            if track_id not in object_lifetime:
                                object_lifetime[track_id] = 1
                            else:
                                object_lifetime[track_id] += 1

                            if object_lifetime[track_id] >= 5:
                                roi_feature = extract_roi_feature([current_feature_map], int(x1), int(y1), int(x2), int(y2), width, height)
                                if len(initial_features) > 4:
                                    initial_features.pop(2) 
                                initial_features.append(roi_feature)
                                hook_handle.remove()

                            # 로그 저장
                            log_file.write(
                                f"Frame: {img_name}, TrackID: {track_id}, "
                                f"BBox: [{', '.join(bbox_formatted)}], "
                                f"Vel: [{velocity[0]:.2f}, {velocity[1]:.2f}], "
                                f"Acc: [{acceleration[0]:.2f}, {acceleration[1]:.2f}], "
                                f"AngVel: {angular_velocity:.5f}\n"
                            )

                        draw_box(im0, (x1, y1, x2, y2), category,
                            velocity=velocity, acceleration=acceleration, 
                            angular_velocity=angular_velocity, 
                            save_with_object_id=save_with_object_id, path=txt_path)

                        if track_id in object_lifetime:
                            object_lifetime = {track_id: object_lifetime[track_id]}
                            del object_lifetime[track_id]

                else: #SORT should be updated even with no detections
                    print("len(det) == 0")
                    tracked_dets = sort_tracker.update()
                    tracks = [] 
                    save_path = save_dir / f"{img_name}.jpg"
                    cv2.imwrite(str(save_path), im0s)

        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                cv2.destroyAllWindows()
            raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                image = None 
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    else:  # stream
                        fps, w, h = fr=60, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
        print(f"Frame saved to {save_path}")

        for i in range(len(feature_maps)):
            if isinstance(feature_maps[i], torch.Tensor):
                feature_maps[i] = feature_maps[i].cpu()
        del feature_maps[:], object_lifetime, current_feature_map
        del img, dataset
        hook_handle.remove()
        feature_maps.clear()
        return tracked_dets, tracks, initial_features, best_target, sort_tracker, first_hist, gt_format

    print(f"Results saved to {log_file_path}")

'''
def main():
    source = '/home/hyhy/Datasets/FR_Dataset/LaSOT/dog/dog-2/img/'
    weights = '/home/hyhy/Desktop/yolov7.pt'

    # === Options 객체 생성 ===
    opt = Options(
        weights=weights,
        source=source,
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        device='cuda',
        fps=30,
        project='/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/DnT_by_frame',
        name='dog-2',
        view_img=False,
        save_txt=True
    )

    device = select_device(opt.device)
    half = device.type != 'cpu'

    # === 모델 로드 ===
    model = attempt_load(opt.weights, map_location=device)
    opt.img_size = check_img_size(opt.img_size, s=int(model.stride.max()))
    if half:
        model.half()

    # === 탐지 및 추적 실행 ===
    detect_frame(model=model,device=device,half=half,source=opt.source,opt=opt)


if __name__ == '__main__':
    main()
'''


'''
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
'''

''' 컬러링북,,
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    # draw colored tracks
                    if colored_trk:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    rand_color_list[track.id % amount_rand_color_prime], thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                    if i < len(track.centroidarr)-1 ] 
                    #draw same color tracks
                    else:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    (255,0,0), thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                    if i < len(track.centroidarr)-1 ] 
'''

'''
    #........Rand Color for every trk.......
    rand_color_list = []
    amount_rand_color_prime = 5003 # prime number
    for i in range(0,amount_rand_color_prime):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
'''
'''
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
'''
# Apply Classifier 
'''
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            print(f"Post-classifier predictions: {pred}")
'''
