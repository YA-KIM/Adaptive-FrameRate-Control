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
from collections import defaultdict, Counter
from thop import profile, clever_format

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

    use_yolo: bool = False          # True: YOLO 추론, False: det/det.txt 사용
    mot_det_type: str = "FRCNN"     # "FRCNN", "SDP", "DPM"


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

        # FPS aggregation 정책
        self.fps_agg_mode = "max"   # "max", "mode_max", "fixed" 중 하나
        self.fixed_fps = 15           # mode가 "fixed"일 때 사용할 FPS
        self.default_fps = 30         # fr_list 비었을 때 기본값

        # 모드 선택 플래그 
        self.use_yolo = opt.use_yolo
        self.mot_det_type = opt.mot_det_type.upper()

        # --- FLOPs/파라미터 및 처리 프레임 수 기록용 ---
        self.flops_per_frame = None
        self.params_count = None
        self.flops_str = "N/A"
        self.params_str = "N/A"
        self.processed_frames = 0

        # 얼마나 YOLO/DRL을 호출했는지
        self.yolo_forward_calls = 0
        self.drl_forward_calls = 0

        # YOLO를 실제로 사용할 때만 FLOPs 계산
        if self.use_yolo and self.yolo_model is not None:
            try:
                self.yolo_model.eval()
                dtype = next(self.yolo_model.parameters()).dtype
                dummy = torch.randn(
                    1, 3, self.opt.imgsz, self.opt.imgsz,
                    device=self.device, dtype=dtype
                )
                with torch.no_grad():
                    macs, params = profile(self.yolo_model, inputs=(dummy,), verbose=False)

                self.flops_per_frame = macs      # MACs ~ FLOPs
                self.params_count = params
                self.flops_str, self.params_str = clever_format([macs, params], "%.3f")

                print(
                    f"[info] YOLOv7 FLOPs (입력 {self.opt.imgsz}x{self.opt.imgsz}): "
                    f"{self.flops_str} FLOPs, Params: {self.params_str}"
                )
            except Exception as e:
                print(f"[warn] YOLO FLOPs 계산 실패: {e}")

        
    def _infer_yolo(self, img0: np.ndarray):
        """letterbox → tensor 변환 → 모델 추론 → NMS."""
        img, ratio, pad = letterbox(img0, new_shape=self.opt.imgsz)
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

        self.yolo_forward_calls += 1
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


    def _select_sequences(self):
        """
        mot_det_type 에 따라 사용할 시퀀스 선택.
        opt.mot_det_type: 'FRCNN', 'SDP', 'DPM', 'ALL'
        """
        det_type = getattr(self, "mot_det_type", "FRCNN").upper()
        if det_type == "ALL":
            seqs = [f for f in self.DataPath.iterdir() if f.is_dir()]
        else:
            seqs = [
                f for f in self.DataPath.iterdir()
                if f.is_dir() and det_type in f.name
            ]
        return seqs

    def _load_mot_detections(self, seq_dir: Path):
        """
        MOT17 det/det.txt 읽어서
        frame_id -> [ [x, y, w, h, conf], ... ] 형태 딕셔너리로 반환.
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

                dets_by_frame[frame_id].append([x, y, w, h, conf])

        return dets_by_frame

    def _build_detections_from_mot(self, img0, frame_id: int, dets_by_frame):
        """
        dets_by_frame[frame_id] 를 Detection 리스트로 변환.
        YOLO 경로(_build_detections)와 동일하게 mars encoder로 appearance feature 뽑음.
        """
        detections = []
        if dets_by_frame is None or frame_id not in dets_by_frame:
            return detections

        boxes = dets_by_frame[frame_id]  # list of [x, y, w, h, conf]
        patches, boxes_kept, confs_kept = [], [], []

        for x, y, w, h, conf in boxes:
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            patch = img0[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
            if patch.size == 0:
                continue

            patches.append(cv2.resize(patch, (64, 128)))
            boxes_kept.append([x1, y1, x2, y2])
            confs_kept.append(float(conf))

        if not patches:
            return detections

        batch = np.stack(patches, axis=0)  # (N, 128, 64, 3)
        feats = self.image_encoder.session.run(
            self.image_encoder.output_var,
            feed_dict={self.image_encoder.input_var: batch}
        )

        for bbox, conf, feat in zip(boxes_kept, confs_kept, feats):
            tlwh = xyxy_to_tlwh(bbox)
            detections.append(Detection(tlwh, conf, feat))

        return detections
    
    def Test_MOT(self):
        """시퀀스별로 탐지→추적→상태기록→DQN으로 FPS 선택→다음 프레임 진행."""
        # 1) 사용할 시퀀스 선택 (FRCNN / SDP / DPM / ALL)
        seqs = self._select_sequences()

        # 2) appearance feature encoder 생성
        self.image_encoder = ImageEncoder(
            '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/deep_sort/model_data/mars-small128.pb',
            'images', 'features'
        )

        fr_count = {5: 0, 10: 0, 15: 0, 30: 0}
        #durations: List[float] = []

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

            # YOLO를 사용하지 않을 경우, 이 시퀀스의 det.txt 미리 로딩
            dets_by_frame = None
            if not self.use_yolo:
                dets_by_frame = self._load_mot_detections(seq)

            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", matching_threshold=0.2, budget=50
            )
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

                    # 전체 처리 프레임 카운트 (FLOPs 나눌 때 사용)
                    self.processed_frames += 1

                    # --- 3) 디텍션: YOLO 모드 vs public det 모드 분기 ---
                    if self.use_yolo:
                        pred, ratio, pad = self._infer_yolo(img0)
                        detections = self._build_detections(img0, pred, ratio, pad)
                    else:
                        detections = self._build_detections_from_mot(
                            img0, frame_id, dets_by_frame
                        )

                    # --- 4) 추적 업데이트 ---
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

                    # --- 5) DRL로 FPS 결정 (또는 fixed) ---
                    if track_ids:
                        if self.fps_agg_mode == "fixed":
                            # 고정 FPS baseline: DRL 네트워크 사용 안 함
                            predicted_fr = self.fixed_fps
                        else:
                            his_list = cat_His_OTID(track_ids, hist)
                            state_batch = self.Agent.get_features_Test(track_ids, his_list)

                            # --- DRL FLOPs 1회 프로파일링 (처음 한 번만) ---
                            if getattr(self.Agent, "flops_per_forward", None) is None:
                                try:
                                    # state_batch가 numpy면 torch로 변환
                                    if isinstance(state_batch, np.ndarray):
                                        dummy_state = torch.from_numpy(state_batch).float().to(self.device)
                                    else:
                                        dummy_state = state_batch.to(self.device)

                                    # 배치가 너무 크면 한 개 샘플만 써도 됨
                                    if dummy_state.ndim > 2:
                                        dummy_state = dummy_state.view(dummy_state.size(0), -1)
                                    # 혹시 여러 개면 하나만 써도 충분
                                    if dummy_state.size(0) > 1:
                                        dummy_state = dummy_state[:1]

                                    self.Agent.policy_net.eval()
                                    with torch.no_grad():
                                        macs, params = profile(
                                            self.Agent.policy_net,
                                            inputs=(dummy_state,),
                                            verbose=False
                                        )

                                    self.Agent.flops_per_forward = macs
                                    flops_str, params_str = clever_format([macs, params], "%.3f")
                                    self.Agent.flops_str = flops_str
                                    self.Agent.params_str = params_str

                                    print(
                                        f"[info] DRL FLOPs (1 forward): {flops_str} FLOPs, "
                                        f"Params: {params_str}"
                                    )
                                except Exception as e:
                                    print(f"[warn] DRL FLOPs 계산 실패: {e}")

                            # --- DRL forward 호출 카운트 (FLOPs 계산용) ---
                            self.drl_forward_calls += 1

                            # 실제 정책 forward
                            _, fr_list = self.Agent.get_best_next_action4MOT_Test(state_batch)

                            if hasattr(fr_list, 'tolist'):
                                fr_list = fr_list.tolist()

                            if not fr_list:
                                predicted_fr = self.default_fps
                            else:
                                if self.fps_agg_mode == "max":
                                    predicted_fr = max(fr_list)
                                elif self.fps_agg_mode == "mode_max":
                                    counts = Counter(fr_list)
                                    most_common = counts.most_common()
                                    top_freq = most_common[0][1]
                                    predicted_fr = max(
                                        fr for fr, freq in most_common if freq == top_freq
                                    )
                                else:
                                    predicted_fr = max(fr_list)

                        fr_count[predicted_fr] += 1

                    # --- 6) 다음 프레임 인덱스 & 시간 ---
                    cur_idx = get_next_frame_index(cur_idx, predicted_fr, total)
                    #durations.append(time.time() - start)

                    # --- 7) 시각화/저장 ---
                    if not self.opt.nosave:
                        frame_dir = out_dir / "frames"
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        vis = draw_boxes(
                            img0.copy(), boxes, identities,
                            velocities=velocities, accelerations=accelerations, angular_velocities=ang_vels,
                            save_with_object_id=self.opt.save_with_object_id, path=str(log_path)
                        )
                        save_name = images[cur_idx - 1].name if cur_idx else images[-1].name
                        cv2.imwrite(str(frame_dir / save_name), vis)

        # --- 8) 요약 로그 + FLOPs 정보 ---
        print(f"\nFPS 분포: {fr_count}")
        '''if durations:
            print(f"평균 처리 시간: {sum(durations)/len(durations):.4f} s/frame")
        else:
            print("측정된 프레임 없음.")
        '''

        # YOLO FLOPs (YOLO 모드일 때만 의미 있음)
        if self.use_yolo and self.flops_per_frame is not None and self.yolo_forward_calls > 0:
            total_flops = self.flops_per_frame * self.yolo_forward_calls
            print(
                f"YOLOv7 FLOPs (입력 {self.opt.imgsz}x{self.opt.imgsz})\n"
                f"  • 1 forward 당: {self.flops_str} FLOPs\n"
                f"  • forward 호출 수: {self.yolo_forward_calls}\n"
                f"  • 총 FLOPs: {total_flops/1e12:.3f} TFLOPs"
            )

        # DRL FLOPs (Agent 쪽에 flops_per_forward, flops_str 있다고 가정)
        flops_per_forward = getattr(self.Agent, "flops_per_forward", None)
        if flops_per_forward is not None and self.drl_forward_calls > 0:
            total_drl_flops = flops_per_forward * self.drl_forward_calls
            flops_str = getattr(self.Agent, "flops_str", "N/A")
            print(
                f"DRL 제어 네트워크 FLOPs\n"
                f"  • 1 forward 당: {flops_str} FLOPs\n"
                f"  • forward 호출 수: {self.drl_forward_calls}\n"
                f"  • 총 FLOPs: {total_drl_flops/1e9:.3f} GFLOPs"
            )

        gc.collect()
        torch.cuda.empty_cache()
        print("모든 시퀀스 테스트 종료")

def main_yolo():
    dataset_path = Path("/home/hyhy/Datasets/FR_Dataset/MOT17/train")

    opt = Options(
        source=str(dataset_path),
        name='MOT17_YOLO_Test',
        img_size=640,
        imgsz=640,
        conf_thres=0.3,
        iou_thres=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_yolo=True,          # YOLO 모드
        mot_det_type="FRCNN",   # 어떤 시퀀스 쓸지 (폴더 이름에 FRCNN 포함)
    )

    device = torch.device(opt.device)
    # YOLO 모델은 이 모드에서만 로딩
    model = attempt_load('/home/hyhy/Desktop/yolov7.pt', map_location=device)

    if device.type == "cuda":
        model.half()
    else:
        model.float()

    agent = Agent(num_episodes=1, load=True, n_actions=4, device=device)

    tester = SOT_with_DRL_Test(
        agent=agent,
        dataset_path=dataset_path,
        yolo_model=model,
        opt=opt
    )

    print("평가 시작...(YOLO 모드)")
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

def main_det():
    dataset_path = Path("/home/hyhy/Datasets/FR_Dataset/MOT17/train")

    opt = Options(
        source=str(dataset_path),
        name='MOT17_PublicDet_Test',
        img_size=640,
        imgsz=640,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_yolo=False,         # det.txt 모드
        mot_det_type="FRCNN",   # "SDP", "DPM", "ALL" 로 바꿔가며 실험 가능
    )

    device = torch.device(opt.device)

    # YOLO를 쓰지 않으니 굳이 모델 로드할 필요 없음
    model = None

    agent = Agent(num_episodes=1, load=True, n_actions=4, device=device)

    tester = SOT_with_DRL_Test(
        agent=agent,
        dataset_path=dataset_path,
        yolo_model=model,   # 내부에서 use_yolo=False면 _infer_yolo 안 쓰니까 None이어도 됨
        opt=opt
    )

    print("평가 시작...(Public det 모드)")
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
    # 1) YOLO + FLOPs 모드
    #main_yolo()

    # 2) MOT17 public det 모드
    main_det()