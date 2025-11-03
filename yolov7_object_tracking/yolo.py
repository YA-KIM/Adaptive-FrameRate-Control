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

def yolo_only(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, save_bbox_dim, save_with_object_id, fps = (
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, 
        not opt.no_trace, opt.save_bbox_dim, opt.save_with_object_id, opt.fps
    )

    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')
    )

    # Directories
    save_dir = Path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    # Data loader
    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img_name = Path(path).stem

        # Preprocess image
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # YOLO 객체 탐지 수행
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        for det in pred:  # 감지된 객체 리스트
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                
                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=2)

        # 결과 저장
        save_path = save_dir / f"{img_name}.jpg"
        cv2.imwrite(str(save_path), im0s)
        print(f"Frame saved to {save_path}")

    print(f"Results saved to {save_dir}")
    print(f"Done. ({time.time() - t0:.3f}s)")

if __name__ == '__main__':
    source_path = '/home/hyhy/Datasets/FR_Dataset/LaSOT/car/car-1/img'
    folder_name = os.path.basename(os.path.dirname(source_path))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/hyhy/Desktop/yolov7.pt', help='model.pt path')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for image sampling (60, 30, 20, 10)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default=source_path, help='source file/folder path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
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
    parser.add_argument('--project', default='/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/yolo', help='project save folder')
    parser.add_argument('--name', default=folder_name, help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='dont trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    parser.set_defaults(download=True)

opt = parser.parse_args()

with torch.no_grad():
    yolo_only()
