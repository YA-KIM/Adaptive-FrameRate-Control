from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utility.model import *
from utility.moment import *
from utility.tools import *  
from utility.config import (
    use_cuda, Tensor, LongTensor, criterion  # 기존 프로젝트 전역 타입/손실 사용
)

# =========================
# DQN Agent (MOT 전용)
# =========================
class Agent:
    """
    DQN 기반 프레임레이트 제어 에이전트.
    - state -> feature_extractor(1D CNN) -> policy_net(DQN) -> Q(s,a)
    - select_action: ε-greedy
    - optimize_model: 표준 DQN 타깃으로 학습
    """

    def __init__(
        self,
        alpha: float = 0.2,
        nu: float = 3.0,
        threshold: float = 0.5,
        num_episodes: Optional[int] = None,
        load: bool = False,
        n_actions: int = 4,
        device: Optional[torch.device] = None,
        version: str = "MOT_Ver7",
        load_version: str = "MOT_Ver5",
        save_dir: str = "/home/hyhy/Desktop/SYD_DtoS/DRL_FR/models",
        frame_rates: Optional[List[int]] = None,
    ):
        # -------- 기본 하이퍼파라미터 -------
        self.n_actions = n_actions
        self.history_length = 8

        self.Version = version
        self.Load_Ver = load_version

        self.GAMMA = 0.900
        self.EPS = 1.0
        self.EPS_min = 0.01

        # Reward 가중치 (음/양 모두 가능하도록 유지)
        self.w_iou = 2.0
        self.w_theta = 0.1
        self.w_FR = 0.2

        # Trigger 보상
        self.alpha = alpha
        self.nu = nu
        self.threshold = threshold

        # 행동공간 (FPS)
        self.Frame_Rates = frame_rates or [5, 10, 15, 30]

        # 디바이스
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # -------- 경로 --------
        self.save_path = save_dir
        self.save_version_path = os.path.join(self.save_path, self.Version)
        self.load_version_path = os.path.join(self.save_path, self.Load_Ver)

        os.makedirs(self.save_path, exist_ok=True)

        # -------- 네트워크 --------
        self.feature_extractor = FeatureExtractor()
        self.policy_net = DQN(self.history_length, self.n_actions)

        if load:
            self._safe_load(self.policy_net, self.load_version_path + "_policy.pth")
            self._safe_load(self.feature_extractor, self.load_version_path + "_feature.pth")
            self.feature_extractor.eval()

        self.target_net = DQN(self.history_length, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 디바이스 이동
        self.feature_extractor.to(self.device)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        # -------- 학습 설정 --------
        self.BATCH_SIZE = 32
        self.num_episodes = num_episodes
        self.memory = ReplayMemory(10000)  # 프로젝트 구현 유지
        self.TARGET_UPDATE = 1
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.feature_extractor.parameters()),
            lr=1e-6
        )

        self.steps_done = 0  # ε-greedy 스텝 카운터

    # =========================
    # 저장/로드
    # =========================
    def save_network(self) -> None:
        os.makedirs(os.path.dirname(self.save_version_path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), self.save_version_path + "_policy.pth")
        torch.save(self.feature_extractor.state_dict(), self.save_version_path + "_feature.pth")
        print(f"[Agent] Saved -> {self.save_version_path}_policy.pth / _feature.pth")

    def _safe_load(self, model: nn.Module, path: str) -> None:
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            print(f"[Agent] Loaded weights from: {path}")
        else:
            print(f"[Agent] Warning: weight file not found: {path} (skipped load)")

    # =========================
    # Geometry / Kinematics
    # =========================
    @staticmethod
    def _to_xyxy_from_cxcyhw(box_cxcyhw: np.ndarray | torch.Tensor) -> Tuple[float, float, float, float]:
        """[cx,cy,h,w] -> [x1,y1,x2,y2]"""
        if isinstance(box_cxcyhw, torch.Tensor):
            cx, cy, h, w = box_cxcyhw.tolist()
        else:
            cx, cy, h, w = box_cxcyhw
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
        return x1, y1, x2, y2

    @staticmethod
    def _iou_xyxy(a: Tuple[float, float, float, float],
                  b: Tuple[float, float, float, float]) -> float:
        """IoU for two boxes in [x1,y1,x2,y2]."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1) 
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return float(inter_area / denom)

    @staticmethod
    def _angle_of(moment: np.ndarray) -> float:
        """rad; moment shape (1, >=7), vx=[:,4], vy=[:,5]"""
        vx, vy = float(moment[0, 4]), float(moment[0, 5])
        return float(np.arctan2(vy, vx))

    def _predict_bbox_cxcyhw(self, prev: np.ndarray, cur: np.ndarray, fr: int) -> np.ndarray:
        """
        현재 moment(cur)와 prev 차이를 이용한 1-step ahead 예측.
        입력/출력: (1,9)에서 [:4]=[cx,cy,h,w].
        """
        t = {30: 1, 15: 2, 10: 3, 5: 6}.get(int(fr), 1)

        cx, cy, h, w = [float(x) for x in cur[0, 0:4]]
        vx, vy = float(cur[0, 4]), float(cur[0, 5])

        # prev 없으면 0 벡터 취급
        prev = prev if prev is not None else np.zeros((1, 9), dtype=np.float32)
        dh = float(cur[0, 2] - prev[0, 2])
        dw = float(cur[0, 3] - prev[0, 3])

        new_cx = cx + vx * t
        new_cy = cy + vy * t
        new_h = h + dh * t
        new_w = w + dw * t

        out = np.array([[new_cx, new_cy, new_h, new_w]], dtype=np.float32)
        return out

    def _predict_angle(self, cur: np.ndarray, fr: int) -> float:
        """현재 각도 + (각속도 * Δt)"""
        t = {30: 1, 15: 2, 10: 3, 5: 6}.get(int(fr), 1)
        ang_vel = float(cur[0, 8])
        return self._angle_of(cur) + ang_vel * t

    # =========================
    # Reward
    # =========================
    def compute_reward(
        self,
        moment: np.ndarray,        # 현재 스텝 직전 (FR 예측의 근거가 된 상태)
        prev_moment: np.ndarray,   # 바로 직전 상태
        post_moment: np.ndarray,   # FR 적용 후 다음 상태
        prev_Fr: int,
        expected_FR: int
    ) -> float:
        """
        총 보상:
          + w_iou * IoU(예측 박스, 실제 다음 프레임 박스)
          + w_theta * (-|θ_pred - θ_post|)
          + w_FR   * (prev_FR - expected_FR)
        => 항마다 음/양 가능: 전체 보상도 음수가 나올 수 있음.
        """
        # 1) IoU
        # 예측 bbox (cx,cy,h,w) -> xyxy 변환
        pred_cxcyhw = self._predict_bbox_cxcyhw(prev_moment, moment, expected_FR)[0]
        post_cxcyhw = post_moment[0, 0:4]

        pred_xyxy = self._to_xyxy_from_cxcyhw(pred_cxcyhw)
        post_xyxy = self._to_xyxy_from_cxcyhw(post_cxcyhw)
        R_iou = self._iou_xyxy(post_xyxy, pred_xyxy)
        if np.isnan(R_iou) or np.isinf(R_iou):
            R_iou = 0.0

        # 2) 각도 일관성 (작을수록 좋음 → 음수)
        theta_pred = self._predict_angle(moment, expected_FR)
        theta_post = self._angle_of(post_moment)
        R_theta = -abs(theta_pred - theta_post)

        # 3) 에너지 (낮은 FPS 선호)
        R_fr = float(prev_Fr - expected_FR)

        total = self.w_iou * R_iou + self.w_theta * R_theta + self.w_FR * R_fr
        return float(total)

    def compute_trigger_reward(
        self,
        actual_state_xyxy: Tuple[float, float, float, float],
        gt_xyxy: Tuple[float, float, float, float]
    ) -> float:
        """최종 IoU가 임계 이상이면 +nu, 아니면 -nu."""
        res = self._iou_xyxy(actual_state_xyxy, gt_xyxy)
        return float(self.nu if res >= self.threshold else -self.nu)

    # =========================
    # Action Selection
    # =========================
    def get_best_next_action4MOT_Test(self, state):
        with torch.no_grad():
            inpu = state.cuda() if use_cuda else state
            q_values = self.policy_net(inpu)               # shape: (B, n_actions)
            best_actions = q_values.argmax(dim=1)          # shape: (B,)

            # Frame_Rates도 batch로 매핑
            frame_rates = [self.Frame_Rates[a.item()] for a in best_actions]

            return best_actions.tolist(), frame_rates      # 리스트 형태로 반환


    def get_best_next_action(self, state):
        """
        Returns the action with the highest Q-value for a given state,
        along with the corresponding frame rate.
        """
        with torch.no_grad():  # 손실 계산 시에도 호출되므로 학습 비활성화
            if use_cuda:
                inpu = state.cuda()  # tensor를 CUDA로 전송
            else:
                inpu = state

            q_values = self.policy_net(inpu)
            # print(f"{q_values}")  # 디버깅용

            best_action = q_values.argmax(dim=1).item()
            return best_action, self.Frame_Rates[best_action]


    def select_action(self, state):
        sample = random.random()
        # epsilon value is assigned by self.EPS
        eps_threshold = self.EPS
        # self.steps_done is to count how many steps the agent used to get final bounding box
        self.steps_done += 1

        if state is None:
            print("Warning: Computed state is None.")
            # 기본값을 반환하도록 처리
            state = torch.zeros(1, 768)

        # Exploration 
        if sample < eps_threshold:
            Exploration = random.randrange(self.n_actions)
            return Exploration, self.Frame_Rates[Exploration]
        # Exploitation        
        else:
            return self.get_best_next_action(state)  # best_action, self.Frame_Rates[best_action]


    def select_action_model(self, state):
        """
        Select an action during the interaction with environment, using greedy policy
        This implementation should be used when testing
        ----------
        Argument:
        state - the state varible of current agent, consisting of (o,h), should conform to input shape of policy net
        ----------
        Return:
        An action index which is generated by policy net
        """
        return self.get_best_next_action(state)


    # =========================
    # Feature Extract
    # =========================
    def get_features_Test(self, track_id, state_histories: list[np.ndarray]):
        batch_bb = []
        batch_m = []

        for state_history in state_histories:
            if state_history is None or len(state_history) == 0:
                continue

            if state_history.ndim == 1:
                state_history = np.expand_dims(state_history, axis=0)

            # 최신 → 과거
            state_history = state_history[::-1]

            bb_numpy = state_history[:, :4]
            m_numpy = state_history[:, 4:]

            batch_bb.append(bb_numpy)
            batch_m.append(m_numpy)

        if len(batch_bb) == 0:
            return None

        bb_tensor = torch.tensor(np.array(batch_bb).astype(np.float32), device=self.device)
        m_tensor = torch.tensor(np.array(batch_m).astype(np.float32), device=self.device)

        self.feature_extractor.to(self.device)
        feature = self.feature_extractor(bb_tensor, m_tensor)
        return feature


    def get_features(self, current_obj_id, state_history):
        if state_history is None or len(state_history) == 0 or current_obj_id is None:
            return None

        if state_history.ndim == 1:
            state_history = np.expand_dims(state_history, axis=0)

        # 최신 -> 과거 순
        state_history = state_history[::-1]

        bb_numpy = state_history[:, :4]
        m_numpy = state_history[:, 4:]

        bb_tensor = torch.tensor(bb_numpy.astype(np.float32), device=self.device).unsqueeze(0)
        m_tensor = torch.tensor(m_numpy.astype(np.float32), device=self.device).unsqueeze(0)

        self.feature_extractor.to(self.device)
        feature = self.feature_extractor(bb_tensor, m_tensor)
        return feature

    # =========================
    # Optimization
    # =========================
    def optimize_model(self, verbose: bool = False) -> None:
        """표준 DQN 1스텝 학습 (배치 부족/NaN 방지 가드 포함)."""
        if len(self.memory) < self.BATCH_SIZE:
            return

        # 샘플링
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # next_state 텐서 만들기
        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), dtype=torch.bool, device=self.device)
        next_states_list = [s for s in batch.next_state if s is not None]
        if len(next_states_list) > 0:
            non_final_next_states = Variable(torch.cat(next_states_list)).to(self.device)
        else:
            non_final_next_states = None

        # state 텐서
        valid_states = [s for s in batch.state if s is not None]
        if len(valid_states) == 0:
            return
        state_batch = Variable(torch.cat(valid_states)).to(self.device)

        # 액션/보상
        action_batch = Variable(torch.LongTensor(batch.action).view(-1, 1)).to(self.device)
        reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1, 1)).to(self.device)

        # 배치 패딩 (부족 시 0으로 채움)
        if state_batch.size(0) < self.BATCH_SIZE:
            pad = torch.zeros(self.BATCH_SIZE - state_batch.size(0), state_batch.size(1), device=self.device)
            state_batch = torch.cat([state_batch, pad], dim=0)
            action_pad = torch.zeros(self.BATCH_SIZE - action_batch.size(0), 1, dtype=action_batch.dtype, device=self.device)
            reward_pad = torch.zeros(self.BATCH_SIZE - reward_batch.size(0), 1, dtype=reward_batch.dtype, device=self.device)
            action_batch = torch.cat([action_batch, action_pad], dim=0)
            reward_batch = torch.cat([reward_batch, reward_pad], dim=0)

        # Q(s,a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # target
        next_state_values = torch.zeros(self.BATCH_SIZE, 1, device=self.device)
        if non_final_next_states is not None:
            with torch.no_grad():
                q_next = self.target_net(non_final_next_states)
                next_state_values[non_final_mask] = q_next.max(1)[0].unsqueeze(1)

        expected = reward_batch + self.GAMMA * next_state_values

        # 안정성 체크
        if torch.isnan(state_action_values).any() or torch.isnan(expected).any():
            if verbose:
                print("[Agent] NaN detected; skip step.")
            return
        if torch.isinf(state_action_values).any() or torch.isinf(expected).any():
            if verbose:
                print("[Agent] Inf detected; skip step.")
            return

        loss = criterion(state_action_values, expected)

        if torch.isnan(loss) or torch.isinf(loss):
            if verbose:
                print("[Agent] Invalid loss; skip step.")
            return

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()