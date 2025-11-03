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
import torch.nn.functional as F


#............................... Bounding Boxes Drawing ............................
"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None, velocities=None, accelerations=None, angular_velocities=None, save_with_object_id=False, path=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
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
        label = (f"{id}:{names[cat]}| "
                 f"V:[{vel[0]:.2f},{vel[1]:.2f}] "
                 f"Acc:[{acc[0]:.2f},{acc[1]:.2f}] "
                 f"AngV:{ang_vel:.5f}")
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.1)

        txt_str = ""
        if save_with_object_id:
            txt_str = (f"{id} {cat} {box[0]/img.shape[1]:.6f} {box[1]/img.shape[0]:.6f} "
                       f"{box[2]/img.shape[1]:.6f} {box[3]/img.shape[0]:.6f} "
                       f"{(box[0] + box[2]/2)/img.shape[1]:.6f} "
                       f"{(box[1] + box[3]/2)/img.shape[0]:.6f}\n")
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img

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


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id, fps = (
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, 
        not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id, opt.fps
    )
    opt.fps=60
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')
    )

    # Initialize SORT tracker
    sort_max_age = 5
    sort_min_hits = 1
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)

     #........Rand Color for every trk.......
    rand_color_list = []
    amount_rand_color_prime = 5003 # prime number
    for i in range(0,amount_rand_color_prime):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................

    # Directories
    save_dir = Path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = save_dir / f"{folder_name}.txt"  # 저장 파일 경로

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # YOLO의 객체 탐지를 더욱 정확하게 보정해주는 Classifier with Resnet-101 
    # 이거 아마 class도 input으로 넣을 때 class 정확도 올려야하니까 그때 추가될지도

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
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
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    image_count = 0
    t0 = time.time()

    with open(log_file_path, 'w') as log_file:
        

        #img= 전처리 되는 이미지 (for Yolo)
        #im0s= 완전한 원본 이미지
        for path, img, im0s, vid_cap in dataset:
            image_count += 1  # 이미지 순서

            img_name = Path(path).stem  # 이미지 파일명 추출 (확장자 제외)
            current_index=image_count   
            # Inference 준비
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

             # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1= time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2= time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3=time_synchronized()

             # Apply Classifier #윗부분에서 활성화 여부 결정가능능
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
                print(f"Post-classifier predictions: {pred}")


            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, frame = path, im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    s=""
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Process SORT tracker
                    dets_to_sort = np.empty((0, 6))

                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                    tracked_dets = sort_tracker.update(dets_to_sort)
                    if tracked_dets.size == 0:
                        print("No detections in the current frame.")
                        continue 
                    tracks = sort_tracker.getTrackers()

                    for track in tracks:
                        if len(track.history) > 0:
                            current_bbox = track.history[-1]
                            velocity = track.velocities[-1] if track.velocities else (0, 0)
                            acceleration = track.accelerations[-1] if track.accelerations else (0, 0)
                            angular_velocity = track.ang_velocities[-1] if track.ang_velocities else 0

                            # current_bbox 포맷팅
                            bbox_formatted = [f"{coord:.2f}" for coord in current_bbox.flatten()]

                            # 텍스트 정보 저장
                            log_file.write(
                                f"Name: {img_name}, TrackID: {track.id}, "
                                f"Class:{int(tracked_dets[i, 4])}, "
                                f"BBox: [{', '.join(bbox_formatted)}], "
                                f"Vel: [{velocity[0]:.2f}, {velocity[1]:.2f}], "
                                f"Acc: [{acceleration[0]:.2f}, {acceleration[1]:.2f}], "
                                f"AngVel: {angular_velocity:.5f}\n"
                            )
                        # color = compute_color_for_labels(id)
                        #draw colored tracks
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

                        if save_txt and not save_with_object_id:
                            # Normalize coordinates
                            txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                            if save_bbox_dim:
                                txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                            txt_str += "\n"
                    
                    if save_txt and not save_with_object_id:
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(txt_str)
                    '''
                    if len(tracked_dets) > 0:
                        # 첫 번째 객체만 선택
                        tracked_obj = tracked_dets[0]  # 첫 번째 객체 선택
                        bbox_xyxy = [tracked_obj[:4]]  # 바운딩 박스 좌표 (리스트 형태로 전달)
                        identities = [int(tracked_obj[8])]  # 객체 ID (리스트 형태로 전달)
                        categories = [int(tracked_obj[4])]  # 객체 클래스 (리스트 형태로 전달)

                        # 첫 번째 객체만 draw_boxes로 전달하여 그림
                        draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path)
                    '''
                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]

                        # 각 track에서 데이터를 추출
                        velocities = [track.velocities[-1] if hasattr(track, 'velocities') and track.velocities else (0, 0) for track in tracks]
                        accelerations = [track.accelerations[-1] if hasattr(track, 'accelerations') and track.accelerations else (0, 0) for track in tracks]
                        angular_velocities = [track.ang_velocities[-1] if hasattr(track, 'ang_velocities') and track.ang_velocities else 0.0 for track in tracks]

                        # draw_boxes에 데이터 전달
                        draw_boxes(im0, bbox_xyxy, identities, categories, names, 
                                velocities=velocities, accelerations=accelerations, 
                                angular_velocities=angular_velocities, 
                                save_with_object_id=save_with_object_id, path=txt_path)

                    
                else: #SORT should be updated even with no detections
                    tracked_dets = sort_tracker.update()

                    '''
                    # Save the image with bounding boxes
                    for *xyxy, conf, cls in det:
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                    save_path = save_dir / f"{img_name}.jpg"  # 이미지 저장 경로
                    cv2.imwrite(str(save_path), im0)  # 이미지 저장
                    print(f"Image saved to {save_path}")
                    '''

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
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        current_index = get_next_frame_index(current_index, fps, total_images)


    print(f"Results saved to {log_file_path}")
    print(f"Done. ({time.time() - t0:.3f}s)")

if __name__ == '__main__':
    source_path = '/home/hyhy/Datasets/FR_Dataset/MOT17/train/MOT17-04-FRCNN/img1'
    gt_path = Path(source_path).parent / "groundtruth_rect.txt"
    folder_name = os.path.basename(os.path.dirname(source_path))

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/hyhy/Desktop/yolov7.pt', help='model.pt path')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for image sampling (60, 30, 20, 10)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default=source_path, help='source file/folder path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/detect_and_track', help='project save folder')
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
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
