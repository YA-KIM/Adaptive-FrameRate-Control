****************Adaptive Frame-Rate Control for MOT with DQN****************  

**핵심 아이디어**

연산·전력 예산을 아끼기 위해 프레임레이트(FPS)를 ‘행동(action)’으로 선택하는 DQN 기반 컨트롤러.
이미지 → Detector(YOLOv7) → Tracker(SORT) → State(바운딩박스+운동량 시계열) → DQN이 최적 FPS(예: 5/10/15/30) 선택 → 다음 프레임을 건너뛰며 샘플링.

  
**왜 필요한가? (연구의 의의)**

- 엣지 디바이스나 실시간 시스템에서, 고정 FPS는 낭비 혹은 정확도 저하를 유발.

- 추적 난이도가 낮은 구간(정적/완만 이동)에서는 FPS를 낮춰 연산·전력 절감, 난이도가 높은 구간(급가속/회전/군중)에서는 FPS를 높여 정확도 유지.

- 기존 MOT 파이프라인(Detector/Tracker)을 바꾸지 않고, 샘플링 정책만 최적화하는 경량·실용적 접근.
  

**1) 시스템 개요**

<img width="1968" height="912" alt="image" src="https://github.com/user-attachments/assets/b21507ad-e3c7-4bb6-85dc-63354bc06592" />

<img width="978" height="516" alt="image" src="https://github.com/user-attachments/assets/4b587fa9-fde9-4ba0-bd42-a3c154200af8" />



Detector: yolov7.pt(class 0: person)

Tracker: SORT Kalman (상태: (cx, cy, h, w, v, a, ω))

State: 최근 history_length 프레임의 시계열 스택(총 9차원/프레임)

Action: Frame_Rates = {5, 10, 15, 30} 중 하나 선택

  
**2) 저장소 구조**

Adaptive-FrameRate-Control/

├─ Tester_MOT7_ds.py           # ✅ TEST (MOT 다중 객체, DeepSORT 확장 인터페이스)

├─ Trainer_MOT7_ds.py          # ✅ TRAIN (MOT 다중 객체)

├─ tester_mot7_sort.py         # 🚧 TEST (단일 객체) — 확장/연구용

├─ trainer_mot7_sort.py        # 🚧 TRAIN (단일 객체) — 확장/연구용

├─ utility/

│  ├─ agent_MOT.py             # DQN Agent (policy/feature/target, 메모리, 보상 등)

│  ├─ model.py                 # FeatureExtractor(1D-CNN), DQN 정의

│  ├─ moment.py                # Moment/History_Supervisor (트랙 히스토리 관리)

│  ├─ tools.py, config.py, ... # 보조 유틸

└─ yolov7_object_tracking/

   ├─ utils/                   # YOLOv7 유틸(letterbox 등)
   
   └─ runs/                    # 결과물 저장(root)



  
**3) 데이터셋**

MOT17 (예시 경로)

/home/<user>/Datasets/FR_Dataset/MOT17/test/*FRCNN*/img1/*.jpg


Tester_MOT7_ds.py / Trainer_MOT7_ds.py의 dataset_path와 Options(source=...)를 환경에 맞게 수정.  


**4) 실행 방법**

✅ 테스트 (MOT, DeepSORT 확장 경로)
python Tester_MOT7_ds.py


주요 옵션: Options(source, weights, conf_thres, iou_thres, device)

사전학습된 DQN을 사용하려면: Agent(load=True) + Agent.Load_Ver 지정

출력

yolov7_object_tracking/runs/MOT/<seq>/<start_idx>/

 ├─ img1_<trk_idx>/*.jpg      # 시각화된 결과

 └─ <seq>.txt                 # 프레임별 로그(BBox, Vel/Acc, AngVel)


  

✅ 학습 (MOT)
python Trainer_MOT7_ds.py


주요 옵션:
Options(img_size, conf_thres, iou_thres, device, project),
Agent(Version, history_length, GAMMA, EPS, w_iou, w_theta, w_FR)

출력

models/<Version>_policy.pth
models/<Version>_feature.pth
yolov7_object_tracking/runs/MOT/...  # 시각화/로그

🚧 단일 객체 추적 확장 (연구용)
python tester_mot7_sort.py
python trainer_mot7_sort.py

SORT를 단일 객체 시나리오로 통제해 모듈 인터페이스를 확장/검증하는 실험용 스크립트(구현 진행 중)


  

**5) 네트워크 상세**

- 입력/상태

트랙별 시계열 윈도우 history_length = 8

시점 벡터(9차원):
[cx, cy, h, w, vx, vy, ax, ay, ω]

텐서 모양(배치 B):
BB ∈ R^{B×8×4}, M ∈ R^{B×8×5} → permute(0,2,1) → R^{B×4×8}, R^{B×5×8}


  
- FeatureExtractor (분기형 1D-CNN → concat)

BBox branch: Conv1d(4→32) → ReLU → Conv1d(32→64) → ReLU → Flatten → 64×8=512

Motion branch: Conv1d(5→16) → ReLU → Conv1d(16→32) → ReLU → Flatten → 32×8=256

가중 결합 & 연결: 학습 가능한 스칼라 BB_weights, M_weights로 분기 기여도 학습
최종 feature 크기 = 512 + 256 = 768

권장 전처리: cx,cy,h,w는 이미지 크기로 정규화([0,1]), v/a/ω는 z-score 표준화.

  

- DQN (MLP Head)

입력: 768

Linear(768→128) → ReLU → Dropout(0.2) → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→4)

출력: Q ∈ R^{B×4} (FPS 후보 4개에 대한 Q값)

파라미터 수(참고)

FeatureExtractor: ≈ 8.4K

DQN: ≈ 106.9K

총 ≈ 115K (경량, 실시간 적합)

  


**6) 보상(Reward) 설계**


1. 정확도(IoU): 현재 상태 moment에서 행동(FPS) expected_FR을 가정하여 다음 BBox를 외삽(BB_Predict) → 실제 다음 상태 post_moment와 IoU
R_iou = IoU(predicted, post)

2. 선형성(궤적 일관성): ω(각속도) 기반 예측 각도와 실측 각도 차이
R_theta = -|θ_pred - θ_post|

3. 에너지(낮은 FPS 선호):
R_fr = prev_FR - expected_FR

  
종합 보상:

R = w_iou*R_iou + w_theta*R_theta + w_FR*R_fr
(기본: w_iou=10, w_theta=0.25, w_FR=0.2)


실패 패널티: 대상 소실/불확실 프레임 누적 등 상황별 -10 등 패널티 부여 (코드 내 로직 참고)
