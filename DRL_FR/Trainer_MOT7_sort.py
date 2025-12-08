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
from thop import profile, clever_format 

from utility.model import *           
from utility.tools import *           
from utility.agent_MOT import Agent
from utility.moment import History_Supervisor
from yolov7_object_tracking.utils.datasets import letterbox

# =========================
# Config & Small Utilities
# =========================

@dataclass
class Options:
    # Paths
    weights: str = '/home/hyhy/Desktop/yolov7.pt'
    source: str = ''
    project: str = '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/MOT'
    name: str = 'exp'
    exist_ok: bool = False

    # Inference
    img_size: int = 640
    conf_thres: float = 0.5
    iou_thres: float = 0.5
    device: str = ''               # '', 'cpu' or 'cuda'
    augment: bool = False
    no_trace: bool = False
    update: bool = False
    download: bool = True
    half: bool = False
    imgsz: int = 640 

    # Output/Save
    view_img: bool = False
    save_txt: bool = True
    save_conf: bool = False
    nosave: bool = False
    save_bbox_dim: bool = False
    save_with_object_id: bool = False
    classes: Optional[List[int]] = None
    agnostic_nms: bool = False
    colored_trk: bool = False

    fps: int = 30  # default start fps

    # Detector mode
    use_yolo: bool = False          # True: YOLO 추론, False: det/det.txt 사용
    mot_det_type: str = "FRCNN"     # "FRCNN", "SDP", "DPM"


def convert_bbox_format(temp: np.ndarray) -> np.ndarray:
    """
    입력: [x1, y1, x2, y2, vx, vy, ax, ay, ang_vel] (1,9)
    출력: [cx, cy, h, w,  vx, vy, ax, ay, ang_vel] (1,9)
    """
    if not isinstance(temp, np.ndarray) or temp.shape != (1, 9):
        raise ValueError(f"(1,9) numpy array expected, got {type(temp)} with shape {getattr(temp, 'shape', None)}")
    x1, y1, x2, y2, vx, vy, ax, ay, ang_vel = temp.flatten()
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return np.array([[cx, cy, h, w, vx, vy, ax, ay, ang_vel]], dtype=np.float32)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    letterbox로 리사이즈/패딩한 좌표를 원본 좌표계로 복원
    coords: (N,4) [x1,y1,x2,y2]
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0::2] = coords[:, 0::2].clip(0, img0_shape[1])
    coords[:, 1::2] = coords[:, 1::2].clip(0, img0_shape[0])
    return coords


def calculate_iou(box1, box2) -> float:
    """ box: [x1,y1,x2,y2] """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    a1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a2 = max(0.0, x2g - x1g) * max(0.0, y2g - y1g)
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def get_next_frame_index(current_index: int, fps: int, total_images: int) -> Optional[int]:
    """ 프레임레이트 정책에 따른 다음 인덱스 계산 """
    fps_map = {30: 1, 15: 2, 10: 3, 5: 6}
    inc = fps_map.get(fps, 1)
    nxt = current_index + inc
    return None if nxt >= total_images else nxt


def draw_boxes(
    img,
    bbox,
    identities=None,
    velocities=None,
    accelerations=None,
    angular_velocities=None,
    save_with_object_id=False,
    path=None,
    offset=(0, 0),
):
    """ 원본 이미지 위에 박스 + 속도/가속도/각속도 라벨 그리기 """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 += offset[0]; x2 += offset[0]
        y1 += offset[1]; y2 += offset[1]

        vel = velocities[i] if velocities is not None else (0.0, 0.0)
        acc = accelerations[i] if accelerations is not None else (0.0, 0.0)
        angv = angular_velocities[i] if angular_velocities is not None else 0.0

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        end_x = int(cx + vel[0] * 5)
        end_y = int(cy + vel[1] * 5)

        label = f"V:[{vel[0]:.2f},{vel[1]:.2f}] Acc:[{acc[0]:.2f},{acc[1]:.2f}] AngV:{angv:.5f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_x = max(0, min(x1, img.shape[1] - w - 5))
        text_y = y1 - 5
        if text_y - h < 0:  # 위로 넘어가면 박스 안으로
            text_y = y1 + h + 5

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
        cv2.rectangle(img, (text_x, text_y - h), (text_x + w, text_y), (255, 144, 30), -1)
        cv2.putText(img, label, (text_x, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.12)

        if save_with_object_id and path:
            # yolov5 txt 포맷과는 다름(여기선 단순 좌표 기록)
            with open(path + '.txt', 'a') as f:
                f.write(f"{x1} {y1} {x2} {y2}\n")
    return img


# =========================
# Trainer
# =========================

class SOT_with_DRL_Tr:
    def __init__(self, agent: Agent, DataSetPath: Path, yolo_model, opt: Options):
        self.Agent = agent
        self.DataPath = DataSetPath
        self.yolo_model = yolo_model
        self.opt = opt

        # Device
        self.device = torch.device(opt.device if opt.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # State
        self.currentFr = None
        self.prevFr = None
        self.uncertain_frame_count = 0

        # Warmup bookkeeping
        self._old_img_b = 1
        self._old_img_h = opt.img_size
        self._old_img_w = opt.img_size

        self.use_yolo = opt.use_yolo
        self.mot_det_type = opt.mot_det_type.upper()

    # ---------- I/O helpers ----------

    def _prepare_image(self, img0: np.ndarray) -> Tuple[torch.Tensor, Tuple, Tuple]:
        """BGR np image -> letterbox -> torch float[0,1] CHW with batch dim"""
        img, ratio, pad = letterbox(img0, new_shape=self.opt.img_size)
        im = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = (im.half() if next(self.yolo_model.parameters()).dtype == torch.float16 else im.float()) / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        return im, ratio, pad

    def _warmup_if_needed(self, im: torch.Tensor):
        """Input shape 바뀌면 몇 번 더미 추론으로 엔진 워밍업"""
        if self.device.type == 'cpu':
            return
        if (self._old_img_b != im.shape[0]) or (self._old_img_h != im.shape[2]) or (self._old_img_w != im.shape[3]):
            self._old_img_b, self._old_img_h, self._old_img_w = im.shape[0], im.shape[2], im.shape[3]
            for _ in range(3):
                self.yolo_model(im, augment=self.opt.augment)[0]

    @torch.no_grad()
    def _infer(self, im: torch.Tensor):
        pred = self.yolo_model(im, augment=self.opt.augment)[0]
        pred = non_max_suppression(
            pred, self.opt.conf_thres, self.opt.iou_thres,
            classes=self.opt.classes, agnostic=self.opt.agnostic_nms
        )
        return pred

    def _init_sort(self):
        return Sort(max_age=5, min_hits=0, iou_threshold=0.3)

    def _log_frame(self, log_file, cur_name: str, track_id: int, bbox: np.ndarray,
                   vel: Tuple[float, float], acc: Tuple[float, float], angv: float):
        b = [f"{v:.2f}" for v in bbox]
        log_file.write(
            f"Frame: {cur_name}, TrackID: {track_id}, "
            f"BBox: [{', '.join(b)}], "
            f"Vel: [{vel[0]:.2f}, {vel[1]:.2f}], "
            f"Acc: [{acc[0]:.2f}, {acc[1]:.2f}], "
            f"AngVel: {angv:.5f}\n"
        )

    def _save_frame(self, save_dir: Path, file_name: str, img: np.ndarray):
        if not self.opt.nosave:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(save_dir / file_name)
            cv2.imwrite(save_path, img)
            print(f"Frame saved to {save_path}")

    # ---------- Detector helpers (for det.txt 모드) ----------

    def _select_sequences(self):
        """
        mot_det_type 에 따라 사용할 시퀀스 선택.
        예: 'FRCNN', 'SDP', 'DPM', 'ALL'
        """
        det_type = getattr(self, "mot_det_type", "FRCNN").upper()
        if det_type == "ALL":
            return [f for f in self.DataPath.iterdir() if f.is_dir()]
        else:
            return [
                f for f in self.DataPath.iterdir()
                if f.is_dir() and det_type in f.name
            ]

    def _load_mot_detections(self, seq_dir: Path):
        """
        MOT17 'det/det.txt'를 읽어서
        frame_id -> [ [x1,y1,x2,y2,conf,cls], ... ] 형태 딕셔너리로 반환.
        cls는 여기선 전부 0(person)으로 둠.
        """
        dets_by_frame = defaultdict(list)
        det_file = seq_dir / "det" / "det.txt"

        if not det_file.exists():
            print(f"[warn] det.txt not found: {det_file}")
            return dets_by_frame

        with open(det_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue

                frame_id = int(float(parts[0]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                detclass = 0.0  # person

                dets_by_frame[frame_id].append(
                    np.array([x1, y1, x2, y2, conf, detclass], dtype=np.float32)
                )

        return dets_by_frame

    def _select_sequences(self):
        """
        mot_det_type 에 따라 사용할 시퀀스 선택.
        예: 'FRCNN', 'SDP', 'DPM', 'ALL'
        """
        det_type = getattr(self, "mot_det_type", "FRCNN").upper()
        if det_type == "ALL":
            return [f for f in self.DataPath.iterdir() if f.is_dir()]
        else:
            return [
                f for f in self.DataPath.iterdir()
                if f.is_dir() and det_type in f.name
            ]


    def _load_mot_detections(self, seq_dir: Path):
        """
        MOT17 'det/det.txt'를 읽어서
        frame_id -> [ [x1,y1,x2,y2,conf,cls], ... ] 형태 딕셔너리로 반환.
        cls는 여기선 전부 0(person)으로 둠.
        """
        dets_by_frame = defaultdict(list)
        det_file = seq_dir / "det" / "det.txt"

        if not det_file.exists():
            print(f"[warn] det.txt not found: {det_file}")
            return dets_by_frame

        with open(det_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue

                frame_id = int(float(parts[0]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                detclass = 0.0  # person

                dets_by_frame[frame_id].append(
                    np.array([x1, y1, x2, y2, conf, detclass], dtype=np.float32)
                )

        return dets_by_frame


    def _get_dets_for_frame(self, img0, frame_id: int, dets_by_frame):
        """
        한 프레임에 대해 person detection 들을 가져온다.
        - YOLO 모드: YOLO 추론 + NMS 후 class 0만 추출
        - det 모드: dets_by_frame[frame_id] 그대로 사용
        return: (dets, im, ratio, pad)
        dets: (N,6) [x1,y1,x2,y2,conf,cls]
        im/ratio/pad: YOLO 모드일 때만 유효, det 모드일 땐 (None, None, None)
        """
        if self.use_yolo:
            im, ratio, pad = self._prepare_image(img0)
            self._warmup_if_needed(im)
            pred = self._infer(im)

            dets = np.empty((0, 6), dtype=np.float32)
            for det in pred:
                if det is not None and len(det):
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        if int(detclass) == 0:  # person
                            dets = np.vstack(
                                (dets,
                                np.array([x1, y1, x2, y2, conf, detclass], dtype=np.float32))
                            )
            return dets, im, ratio, pad
        else:
            frame_dets = dets_by_frame.get(frame_id, [])
            if frame_dets:
                dets = np.stack(frame_dets, axis=0)
            else:
                dets = np.empty((0, 6), dtype=np.float32)
            return dets, None, None, None

    # ---------- Episode (sequence) ----------
    def Train(self):
        # 1) 사용할 시퀀스 선택 (FRCNN/SDP/DPM/ALL)
        seq_folders = self._select_sequences()

        for i_episode, subfolder in enumerate(seq_folders):
            if i_episode >= 7:
                break

            # Replay 메모리 초기화
            if hasattr(self.Agent, "memory"):
                del self.Agent.memory
            self.Agent.memory = ReplayMemory(10000)

            print(f"[Episode {i_episode + 1}] {subfolder}")
            img_folder = subfolder / "img1"
            if not img_folder.exists():
                print(f"이미지 폴더 없음: {img_folder}")
                continue

            # det.txt 모드라면 시퀀스별 det 로딩
            dets_by_frame = None
            if not self.use_yolo:
                dets_by_frame = self._load_mot_detections(subfolder)

            image_files = sorted(list(img_folder.glob("*.jpg")))
            total_img_num = len(image_files)
            if total_img_num == 0:
                print(f"No images in: {img_folder}")
                continue

            # SORT & History
            sort_tracker = self._init_sort()
            hist = History_Supervisor(History_Length=self.Agent.history_length)
            hist.clear()

            self.currentFr = self.prevFr = 30
            predicted_fr = 30
            fr_count = {5: 0, 10: 0, 15: 0, 30: 0}

            # 50 프레임 단위로 반복학습
            initial_image_index = 0
            while initial_image_index is not None:
                gt_folder_name = subfolder.name
                log_dir = Path(self.opt.project) / gt_folder_name / f"{initial_image_index}"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file_path = log_dir / f"{gt_folder_name}.txt"
                with open(log_file_path, 'w') as lf:
                    lf.write("")  # reset
                log_file = open(log_file_path, "a")

                # ==========================
                # 1) 앵커 프레임에서 사람 수 파악
                # ==========================
                current_img_indx = initial_image_index
                cur_img_path = image_files[current_img_indx]
                img0 = cv2.imread(str(cur_img_path))
                if img0 is None:
                    raise RuntimeError(f"이미지를 불러올 수 없습니다: {cur_img_path}")

                frame_id = int(Path(cur_img_path).stem)
                dets_anchor, _, _, _ = self._get_dets_for_frame(img0, frame_id, dets_by_frame)
                person_num = dets_anchor.shape[0]
                print(f"MOT_Object_Number: {person_num}")

                # =====================================
                # 2) 각 사람(i)을 개별 에피소드로 학습
                # =====================================
                for i in range(person_num):
                    # 개별 객체마다 SORT/상태 리셋
                    sort_tracker.reset()
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.uncertain_frame_count = 0

                    current_img_indx = initial_image_index
                    done = False
                    print(f"  └─ Target idx: {i}")
                    log_file.write(f"\n{i}th object\n")

                    # 루프 상태
                    state = None
                    prev_track_id = None
                    kalman_predict = None

                    # -------------------
                    # 프레임 루프
                    # -------------------
                    while not done and current_img_indx is not None:
                        cur_img_path = image_files[current_img_indx]
                        img0 = cv2.imread(str(cur_img_path))
                        if img0 is None:
                            print(f"이미지를 불러올 수 없습니다: {cur_img_path}")
                            break

                        frame_id = int(Path(cur_img_path).stem)
                        dets_frame, im, ratio, pad = self._get_dets_for_frame(img0, frame_id, dets_by_frame)

                        # -------------------------------
                        # 첫 프레임: i번째 박스로 SORT 시작
                        # -------------------------------
                        if current_img_indx == initial_image_index:
                            dets_first = dets_frame
                            if dets_first.shape[0] == 0 or i >= dets_first.shape[0]:
                                print("No person on first frame. skip target.")
                                break

                            # i번째 박스로 시작
                            gt = dets_first[i][0:4].reshape(1, 4)
                            dets_first[i][0:4] = gt.flatten()

                            kalman_predict = sort_tracker.update(dets_first[i].reshape(1, -1))
                            kalman_states = sort_tracker.getTrackers()

                            if len(kalman_states) > 0:
                                tracker = kalman_states[0]
                                if hasattr(tracker, 'get_state'):
                                    # 속도/가속도/각속도
                                    vel = tracker.velocities[-1] if tracker.velocities else (0.0, 0.0)
                                    acc = tracker.accelerations[-1] if tracker.accelerations else (0.0, 0.0)
                                    angv = tracker.ang_velocities[-1] if tracker.ang_velocities else 0.0

                                    # 위치
                                    x1, y1, x2, y2 = dets_first[i][0:4]
                                    bbox_state = np.array(
                                        [x1, y1, x2, y2, *vel, *acc, angv],
                                        dtype=np.float32
                                    ).reshape(1, 9)
                                    temp = convert_bbox_format(bbox_state)

                                    track_id = int(kalman_predict[0, 8])
                                    self._log_frame(
                                        log_file, cur_img_path.name, track_id,
                                        dets_first[i][:4], vel, acc, angv
                                    )

                                    # 좌표 복원 + 저장 (YOLO 모드만 scale_coords)
                                    box = dets_first[i, 0:4].copy().reshape(1, 4)
                                    if self.use_yolo:
                                        box = scale_coords(
                                            im.shape[1:], box, img0.shape, ratio_pad=(ratio, pad)
                                        ).round()
                                    x1d, y1d, x2d, y2d = box[0]

                                    img_draw = draw_boxes(
                                        img0.copy(),
                                        [(x1d, y1d, x2d, y2d)],
                                        velocities=[vel], accelerations=[acc], angular_velocities=[angv],
                                        save_with_object_id=self.opt.save_with_object_id, path=str(log_file_path)
                                    )
                                    self._save_frame(log_dir / f'img1_{i}', cur_img_path.name, img_draw)

                                    # DQN 상태/액션
                                    hist.update(track_id, temp, self.currentFr)
                                    inpu = hist.get_state_history(track_id)
                                    state = self.Agent.get_features(track_id, inpu)
                                    _, predicted_fr = self.Agent.select_action(state)

                                    prev_track_id = track_id
                                    self.prevFr = self.currentFr
                                    self.currentFr = predicted_fr
                                    current_img_indx = get_next_frame_index(
                                        current_img_indx, predicted_fr, total_img_num
                                    )
                                else:
                                    current_img_indx = get_next_frame_index(current_img_indx, 1, total_img_num)
                            else:
                                current_img_indx = get_next_frame_index(current_img_indx, 1, total_img_num)

                        # --------------------------
                        # 이후 프레임: best_target 선택
                        # --------------------------
                        else:
                            max_iou = 0.1
                            best_target = None

                            for det in dets_frame:
                                x1, y1, x2, y2, conf, detclass = det
                                if int(detclass) != 0:
                                    continue
                                cx, cy, h, w = kalman_predict[0][:4]
                                iou = calculate_iou([cx, cy, h, w], [x1, y1, x2, y2])
                                if iou > max_iou:
                                    max_iou = iou
                                    best_target = np.array(
                                        [x1, y1, x2, y2, conf, detclass],
                                        dtype=np.float32
                                    )

                            if max_iou < 0.4:
                                self.uncertain_frame_count += 1
                                if self.uncertain_frame_count >= 5:
                                    print("  └─ 종료(불확실 프레임 누적)")
                                    log_file.write("종료\n")
                                    done = True

                                current_img_indx = get_next_frame_index(
                                    current_img_indx, predicted_fr, total_img_num
                                )
                                continue
                            else:
                                self.uncertain_frame_count = 0

                            if best_target is not None:
                                best_target = best_target.reshape(1, -1)
                                kalman_predict = sort_tracker.update(best_target)
                                kalman_states = sort_tracker.getTrackers()

                                cx, cy, h, w = kalman_predict[0][:4]
                                if any(val in (0, 640) for val in [cx, cy, h, w]):
                                    done = True
                                    print("  └─ 대상 프레임 이탈")
                                else:
                                    if len(kalman_states) > 0:
                                        tracker = kalman_states[0]
                                        if hasattr(tracker, 'get_state'):
                                            x1, y1, x2, y2 = best_target[0, 0:4]
                                            vel = tracker.velocities[-1] if tracker.velocities else (0.0, 0.0)
                                            acc = tracker.accelerations[-1] if tracker.accelerations else (0.0, 0.0)
                                            angv = tracker.ang_velocities[-1] if tracker.ang_velocities else 0.0
                                            bbox_state = np.array(
                                                [x1, y1, x2, y2, *vel, *acc, angv],
                                                dtype=np.float32
                                            ).reshape(1, 9)
                                            temp = convert_bbox_format(bbox_state)

                                            track_id = int(kalman_predict[0, 8])
                                            self._log_frame(
                                                log_file, cur_img_path.name, track_id,
                                                best_target[0, :4], vel, acc, angv
                                            )

                                            box = best_target[0, 0:4].copy().reshape(1, 4)
                                            if self.use_yolo:
                                                box = scale_coords(
                                                    im.shape[1:], box, img0.shape, ratio_pad=(ratio, pad)
                                                ).round()
                                            x1d, y1d, x2d, y2d = box[0]

                                            img_draw = draw_boxes(
                                                img0.copy(),
                                                [(x1d, y1d, x2d, y2d)],
                                                velocities=[vel], accelerations=[acc], angular_velocities=[angv],
                                                save_with_object_id=self.opt.save_with_object_id, path=str(log_file_path)
                                            )
                                            self._save_frame(log_dir / f'img1_{i}', cur_img_path.name, img_draw)

                                            # ===== DQN 업데이트 =====
                                            temp = np.zeros((1, 9), dtype=np.float32) if kalman_predict is None else temp
                                            if temp is not None:
                                                hist.update(track_id, temp, self.currentFr)
                                                cur_m = hist[track_id][-1]
                                                prev_m = hist[track_id][-2] if len(hist[track_id]) >= 2 else cur_m

                                                if track_id == prev_track_id and track_id is not None:
                                                    reward = self.Agent.compute_reward(
                                                        prev_m.current_vector, prev_m.previous_vector,
                                                        cur_m.current_vector, self.prevFr, self.currentFr
                                                    )
                                                else:
                                                    reward = -5  # ID 바뀜 패널티

                                                print(f"    reward={reward}")

                                                next_in = hist.get_state_history(track_id)
                                                next_state = self.Agent.get_features(track_id, next_in)

                                                if state is not None:
                                                    action_idx, _ = self.Agent.select_action(state)
                                                    self.Agent.memory.push(state, action_idx, next_state, reward)

                                                state = next_state
                                                prev_track_id = track_id
                                                self.prevFr = self.currentFr
                                                action_idx, predicted_fr = self.Agent.select_action(state)
                                                self.currentFr = predicted_fr
                                                if predicted_fr in fr_count:
                                                    fr_count[predicted_fr] += 1

                                                self.Agent.optimize_model(verbose=True)

                                            current_img_indx = get_next_frame_index(
                                                current_img_indx, predicted_fr, total_img_num
                                            )

                            else:
                                # best_target 없음 → 현재 프레임 이미지라도 저장
                                img_draw = img0.copy()
                                self._save_frame(log_dir / f'img1_{i}', cur_img_path.name, img_draw)
                                print("No best target tracked")
                                log_file.write(f"Frame: {cur_img_path.name}, No Best Target\n")

                                # ===== 실패 스텝도 DQN transition으로 반영 =====
                                reward = -7  # 검출 실패 패널티

                                if prev_track_id is not None and state is not None:
                                    temp = np.zeros((1, 9), dtype=np.float32)
                                    hist.update(prev_track_id, temp, self.currentFr)

                                    next_in = hist.get_state_history(prev_track_id)
                                    next_state = self.Agent.get_features(prev_track_id, next_in)

                                    if next_state is not None:
                                        try:
                                            action_idx = self.Agent.Frame_Rates.index(self.currentFr)
                                        except ValueError:
                                            action_idx = 0

                                        self.Agent.memory.push(state, action_idx, next_state, reward)
                                        self.Agent.optimize_model(verbose=True)

                                        state = next_state

                                predicted_fr = 30
                                self.currentFr = 30
                                if 30 in fr_count:
                                    fr_count[30] += 1

                                current_img_indx = get_next_frame_index(
                                    current_img_indx, predicted_fr, total_img_num
                                )

                        # UI/리소스 정리
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)

                    # 에피소드(대상별) 종료 후 탐욕도 감소
                    self.Agent.EPS = max(self.Agent.EPS_min, self.Agent.EPS * 0.95)

                # 50 프레임 단위로 다음 블록
                initial_image_index = (
                    initial_image_index + 50
                    if (initial_image_index + 50) < len(image_files)
                    else None
                )

            print(f"selected_fr : {fr_count}\n")

            # Target 네트워크 동기화 (에피소드마다)
            with torch.no_grad():
                self.Agent.target_net.load_state_dict(self.Agent.policy_net.state_dict())

            sort_tracker.reset()
            del self.Agent.memory
            self.Agent.memory = None
            gc.collect()
            torch.cuda.empty_cache()

        # 최종 저장
        self.Agent.save_network()
        print("Training completed.")

# =========================
# Main
# =========================
def main():
    dataset_path = Path("/home/hyhy/Datasets/FR_Dataset/MOT17/test")

    # ★ 여기서 모드 선택
    USE_YOLO = True          # True면 YOLO, False면 det.txt
    DET_TYPE = "FRCNN"       # "FRCNN", "SDP", "DPM", "ALL" 중 택1 (det 모드일 때 사용)

    opt = Options(
        source=str(dataset_path),
        name='MOT17_Training',
        img_size=640,
        conf_thres=0.3,
        iou_thres=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_yolo=USE_YOLO,
        mot_det_type=DET_TYPE,
    )

    device = torch.device(opt.device)

    # ★ YOLO 모드일 때만 모델 로드
    if USE_YOLO:
        model = attempt_load('/home/hyhy/Desktop/SYD_DtoS/yolov7.pt', map_location=device)
        if device.type == "cuda":
            model.half()
        else:
            model.float()
    else:
        model = None  # det.txt 모드에서는 YOLO 안 씀

    agent = Agent(load=False, n_actions=4, device=device)  # {5,10,15,30}

    trainer = SOT_with_DRL_Tr(agent=agent, DataSetPath=dataset_path, yolo_model=model, opt=opt)
    print("학습 시작...")
    trainer.Train()
    print("학습 완료!")
    print(
        f"Parameters\n"
        f"  Gamma: {agent.GAMMA}, "
        f"  EPS:   {agent.EPS}, "
        f"  IOUW:  {agent.w_iou}, "
        f"  thW:   {agent.w_theta}, "
        f"  FRW:   {agent.w_FR}, "
        f"  His_Length: {agent.history_length},"
    )


if __name__ == "__main__":
    main()