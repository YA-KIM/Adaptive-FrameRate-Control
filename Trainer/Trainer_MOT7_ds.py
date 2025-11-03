# -*- coding: utf-8 -*-
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path 
from yolov7_object_tracking.DnT_by_frame import * 
from yolov7_object_tracking.utils.download_weights import download  

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from utility.model import *           
from utility.tools import *           
from utility.agent_MOT import Agent
from utility.moment import History_Supervisor
from yolov7_object_tracking.utils.datasets import letterbox

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools.generate_detections import ImageEncoder


@dataclass
class Options:
    # 경로/저장
    weights: str = '/home/hyhy/Desktop/yolov7.pt'
    source: str = ''
    project: str = '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/MOT_ds'
    name: str = 'exp'
    exist_ok: bool = False

    # 탐지/추론
    img_size: int = 640
    imgsz: int = 640
    conf_thres: float = 0.2
    iou_thres: float = 0.45
    device: str = 'cuda'
    augment: bool = False
    no_trace: bool = False
    update: bool = False

    # 저장/시각화
    view_img: bool = False
    save_txt: bool = True
    save_conf: bool = False
    nosave: bool = False
    save_bbox_dim: bool = False
    save_with_object_id: bool = False
    classes: Optional[List[int]] = None
    agnostic_nms: bool = False
    colored_trk: bool = False
    download: bool = True
    half: bool = False
    fps: int = 30


def draw_boxes(
    img: np.ndarray,
    bbox: List[Tuple[int, int, int, int]],
    identities: Optional[List[int]] = None,
    velocities: Optional[List[Tuple[float, float]]] = None,
    accelerations: Optional[List[Tuple[float, float]]] = None,
    angular_velocities: Optional[List[float]] = None,
    save_with_object_id: bool = False,
    path: Optional[str] = None,
    offset: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """바운딩 박스와 궤적 특성을 영상에 그린다."""
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(b) for b in box]
        x1 += offset[0]; x2 += offset[0]
        y1 += offset[1]; y2 += offset[1]

        vel = velocities[i] if velocities else (0.0, 0.0)
        acc = accelerations[i] if accelerations else (0.0, 0.0)
        ang_vel = angular_velocities[i] if angular_velocities else 0.0

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        end_x, end_y = int(cx + 5 * vel[0]), int(cy + 5 * vel[1])  # 속도 화살표

        label = f"V:[{vel[0]:.2f},{vel[1]:.2f}] Acc:[{acc[0]:.2f},{acc[1]:.2f}] AngV:{ang_vel:.5f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        tx = max(0, min(x1, img.shape[1] - w - 5))
        ty = y1 - 5 if y1 - 5 - h >= 0 else y1 + h + 5

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 255, 0), 3, tipLength=0.1)

        if save_with_object_id and path:
            txt = (f"{box[0]/img.shape[1]:.6f} {box[1]/img.shape[0]:.6f} "
                   f"{box[2]/img.shape[1]:.6f} {box[3]/img.shape[0]:.6f} "
                   f"{(box[0] + box[2]/2)/img.shape[1]:.6f} {(box[1] + box[3]/2)/img.shape[0]:.6f}\n")
            with open(path + '.txt', 'a') as f:
                f.write(txt)
    return img


def get_person_only(pred) -> np.ndarray:
    """YOLO 결과에서 class 0(person)만 [x1,y1,x2,y2,conf,cls]로 추출."""
    dets = np.empty((0, 6))
    for det in pred:
        if det is None or not len(det):
            continue
        for x1, y1, x2, y2, conf, c in det.cpu().detach().numpy():
            if int(c) == 0:
                dets = np.vstack((dets, np.array([x1, y1, x2, y2, conf, c])))
    return dets


def convert_bbox_format(temp: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2,vx,vy,ax,ay,angV] → [cx,cy,h,w,vx,vy,ax,ay,angV]."""
    if not isinstance(temp, np.ndarray) or temp.shape != (1, 9):
        raise ValueError(f"(1,9) numpy array expected, got {getattr(temp, 'shape', None)}")
    x1, y1, x2, y2, vx, vy, ax, ay, ang_vel = temp.flatten()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = abs(x2 - x1), abs(y2 - y1)
    return np.array([[cx, cy, h, w, vx, vy, ax, ay, ang_vel]], dtype=np.float32)


def xyxy_to_tlwh(xyxy: Tuple[int, int, int, int]) -> List[int]:
    """[x1,y1,x2,y2] → [x,y,w,h]."""
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def get_next_frame_index(current_index: int, fps: int, total_images: int) -> Optional[int]:
    """선택된 fps에 따른 다음 프레임 인덱스 계산(30→+1, 15→+2, 10→+3, 5→+6)."""
    inc = {30: 1, 15: 2, 10: 3, 5: 6}.get(fps, 1)
    nxt = current_index + inc
    return None if nxt >= total_images else nxt


def cat_His_OTID(track_ids: List[int], sup: History_Supervisor):
    """각 track_id의 상태 히스토리를 수집."""
    return [sup.get_state_history(tid) for tid in track_ids]


class SOT_with_DRL_Test:
    """YOLOv7 + DeepSORT + DQN 기반 프레임레이트 제어 평가기."""

    def __init__(self, agent: Agent, dataset_path: Path, yolo_model, opt: Options):
        self.Agent = agent
        self.DataPath = dataset_path
        self.yolo_model = yolo_model
        self.device = torch.device(opt.device if opt.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.opt = opt
        self.prevFr = 30
        self.image_encoder: Optional[ImageEncoder] = None

    def _infer_yolo(self, img0: np.ndarray):
        """letterbox → tensor 변환 → 모델 추론 → NMS."""
        img, ratio, pad = letterbox(img0, new_shape=640)
        inp = img[:, :, ::-1].transpose(2, 0, 1)
        inp = np.ascontiguousarray(inp)
        inp = torch.from_numpy(inp).to(self.device)
        inp = inp.half() if next(self.yolo_model.parameters()).dtype == torch.float16 else inp.float()
        inp /= 255.0
        if inp.ndimension() == 3:
            inp = inp.unsqueeze(0)
        with torch.no_grad():
            raw = self.yolo_model(inp, augment=self.opt.augment)[0]
            pred = non_max_suppression(raw, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes)
        return pred, ratio, pad

    def _build_detections(self, img0, pred, ratio, pad):
        detections = []

        # 1) 사람(class 0)만 추출 (shape: (N,6) = x1,y1,x2,y2,conf,cls)
        dets_person = get_person_only(pred)  # numpy (N,6)
        if dets_person.size == 0:
            return detections  # 빈 리스트

        # 2) 네트워크 입력 좌표 -> 원본 이미지 좌표로 스케일링
        #    img.shape[2:] 대신 고정 입력 크기를 사용(예: 640)하여 스코프 문제 방지
        net_hw = (self.opt.imgsz, self.opt.imgsz)  # (h,w)
        xyxy = torch.from_numpy(dets_person[:, :4]).to(self.device)
        scaled = scale_coords(
            net_hw,            # 네트워크 입력 (h,w)
            xyxy,              # (N,4) Tensor[xyxy]
            img0.shape[:2],    # 원본 이미지 (h,w)
            ratio_pad=(ratio, pad)
        )
        scaled_np = scaled.round().cpu().numpy().astype(int)  # (N,4)
        confs = dets_person[:, 4]                              # (N,)

        # 3) 패치 배치 추출 (유효 박스만 유지)
        patches, boxes_kept, confs_kept = [], [], []
        for (x1, y1, x2, y2), c in zip(scaled_np, confs):
            if x2 <= x1 or y2 <= y1:
                continue
            patch = img0[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
            if patch.size == 0:
                continue
            patches.append(cv2.resize(patch, (64, 128)))
            boxes_kept.append([x1, y1, x2, y2])
            confs_kept.append(float(c))

        if not patches:
            return detections

        # 4) 배치로 appearance feature 추출 (한 번의 session.run)
        batch = np.stack(patches, axis=0)  # (N,128,64,3)
        feats = self.image_encoder.session.run(
            self.image_encoder.output_var,
            feed_dict={self.image_encoder.input_var: batch}
        )  # (N, feat_dim)

        # 5) DeepSORT Detection 생성
        for bbox, conf, feat in zip(boxes_kept, confs_kept, feats):
            tlwh = xyxy_to_tlwh(bbox)  # [x,y,w,h]
            detections.append(Detection(tlwh, conf, feat))

        return detections


    def Test_MOT(self):
        """시퀀스별로 탐지→추적→상태기록→DQN으로 FPS 선택→다음 프레임 진행."""
        seqs = [f for f in self.DataPath.iterdir() if f.is_dir() and 'FRCNN' in f.name]

        self.image_encoder = ImageEncoder(
            '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/deep_sort/model_data/mars-small128.pb',
            'images', 'features'
        )

        fr_count = {5: 0, 10: 0, 15: 0, 30: 0}
        durations: List[float] = []

        for i_episode, seq in enumerate(seqs):
            if i_episode >= 7:
                break

            img_dir = seq / "img1"
            if not img_dir.exists():
                print(f"[skip] no img1: {img_dir}")
                continue

            images = sorted(img_dir.glob("*.jpg"))
            if not images:
                print(f"[skip] empty: {img_dir}")
                continue

            metric = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold=0.2, budget=50)
            ds_tracker = Tracker(metric)

            hist = History_Supervisor(History_Length=self.Agent.history_length)
            hist.clear()

            predicted_fr = 30
            cur_idx = 0
            total = len(images)

            out_dir = Path(self.opt.project) / seq.name
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / f"{seq.name}.txt"
            trk_path = out_dir / "trackers.txt"

            with open(log_path, "w") as lf, open(trk_path, "w") as tf:
                while cur_idx is not None:
                    start = time.time()

                    img_path = images[cur_idx]
                    frame_id = int(img_path.stem)
                    img0 = cv2.imread(str(img_path))
                    if img0 is None:
                        print(f"[warn] cannot read: {img_path}")
                        break

                    pred, ratio, pad = self._infer_yolo(img0)
                    detections = self._build_detections(img0, pred, ratio, pad)

                    ds_tracker.predict()
                    ds_tracker.update(detections)

                    identities, boxes = [], []
                    velocities, accelerations, ang_vels = [], [], []
                    track_ids: List[int] = []

                    for t in ds_tracker.tracks:
                        if not t.is_confirmed() or t.time_since_update > ds_tracker.max_age:
                            continue

                        tid = t.track_id
                        x, y, w, h = t.to_tlwh()
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                        vx, vy = t.velocities[-1] if t.velocities else (0.0, 0.0)
                        ax, ay = t.accelerations[-1] if t.accelerations else (0.0, 0.0)
                        ang = t.ang_vels[-1] if t.ang_vels else 0.0

                        state = np.array([[x1, y1, x2, y2, vx, vy, ax, ay, ang]], dtype=np.float32)
                        hist.update(tid, convert_bbox_format(state), predicted_fr)

                        identities.append(tid)
                        boxes.append((x1, y1, x2, y2))
                        velocities.append((vx, vy))
                        accelerations.append((ax, ay))
                        ang_vels.append(ang)
                        track_ids.append(tid)

                        lf.write(
                            f"Frame:{frame_id}, ID:{tid}, "
                            f"BBox:[{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}], "
                            f"Vel:[{vx:.2f},{vy:.2f}], Acc:[{ax:.2f},{ay:.2f}], AngV:{ang:.4f}\n"
                        )
                        tf.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,1,1,1\n")

                    if track_ids:
                        his_list = cat_His_OTID(track_ids, hist)
                        state_batch = self.Agent.get_features_Test(track_ids, his_list)
                        _, fr_list = self.Agent.get_best_next_action4MOT_Test(state_batch)
                        predicted_fr = max(fr_list)  # 보수적 집계(Ours_M)
                        fr_count[predicted_fr] += 1

                    cur_idx = get_next_frame_index(cur_idx, predicted_fr, total)
                    durations.append(time.time() - start)

                    if not self.opt.nosave:
                        frame_dir = out_dir / "frames"
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        vis = draw_boxes(
                            img0.copy(), boxes, identities,
                            velocities=velocities, accelerations=accelerations, angular_velocities=ang_vels,
                            save_with_object_id=self.opt.save_with_object_id, path=str(log_path)
                        )
                        cv2.imwrite(str(frame_dir / images[cur_idx - 1].name if cur_idx else frame_dir / images[-1].name), vis)

        print(f"\nFPS 분포: {fr_count}")
        if durations:
            print(f"평균 처리 시간: {sum(durations)/len(durations):.4f} s/frame")
        else:
            print("측정된 프레임 없음.")

        gc.collect()
        torch.cuda.empty_cache()
        print("모든 시퀀스 테스트 종료")


def main():
    dataset_path = Path("/home/hyhy/Datasets/FR_Dataset/MOT17/train")
    opt = Options(
        source=str(dataset_path),
        name='LaSOT_Training',
        img_size=640,
        conf_thres=0.3,
        iou_thres=0.2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load('/home/hyhy/Desktop/yolov7.pt', map_location=device)
    agent = Agent(num_episodes=1, load=True, n_actions=4, device=device)

    model.half() if device.type == "cuda" else model.float()

    tester = SOT_with_DRL_Test(agent=agent, dataset_path=dataset_path, yolo_model=model, opt=opt)
    print("평가 시작...")
    tester.Test_MOT()
    print("평가 완료!")
    print(
        f"Parameters\n"
        f"  Gamma: {agent.GAMMA}, "
        f"  EPS: {agent.EPS}, "
        f"  IOUW: {agent.w_iou}, "
        f"  thW: {agent.w_theta}, "
        f"  FRW: {agent.w_FR}, "
        f"  His_Length: {agent.history_length},"
    )


if __name__ == "__main__":
    main()