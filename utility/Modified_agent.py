import sys
import os
from utility.model import *
from utility.tools import *
from Tracking.detect_track import get_next_frame_index

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv

from tqdm.notebook import tqdm
from utility.config import *

import glob
from PIL import Image
from collections import deque
from sort import *
from utility.moment import *

class Agent2():
    def __init__(self, state_dim, n_actions, history_length, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.95, lr=1e-4, memory_capacity=10000, max_history=10):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.history_length = history_length
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # 네트워크 초기화
        self.policy_net = DQN(history_length, n_actions)
        self.target_net = DQN(history_length, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = 64
        self.steps_done = 0


        self.max_history = max_history  # Maximum trajectory history length
        self.history_queue = deque(maxlen=max_history)  # Queue to store trajectory data
        self.dynamic_history = True 

    def save_network(self):
        torch.save(self.policy_net, self.save_path + "_" + self.model_name + "_" +self.classe)
        print('Saved')

    def load_network(self):
        if not use_cuda:
            return torch.load(self.save_path + "_" + self.model_name + "_" + self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path + "_" + self.model_name + "_" + self.classe)
    
    def reset(self, image_files, ground_truths):
        print(f"Image files: {len(image_files)}")  # Debug: print the length of image_files
        print(f"Ground truths: {len(ground_truths)}")  # Debug: print the length of ground_truths

        if len(image_files) == 0 or len(ground_truths) == 0 :
            print("Error: One or more lists are empty.")
            return None  # Return None or an appropriate fallback

        initial_state = self.compose_state([image_files[0], ground_truths[0]])
        return initial_state

    
    #############################
    # 1. Functions to compute reward
    def intersection_over_union(self, box1, box2):
        """
        Compute IoU value over two bounding boxes
        Each box is represented by four elements vector: (left, right, top, bottom)
        Origin point of image system is on the top left
        """
        box1_left, box1_right, box1_top, box1_bottom = box1
        box2_left, box2_right, box2_top, box2_bottom = box2
        
        inter_top = max(box1_top, box2_top)
        inter_left = max(box1_left, box2_left)
        inter_bottom = min(box1_bottom, box2_bottom)
        inter_right = min(box1_right, box2_right)
        inter_area = max(((inter_right - inter_left) * (inter_bottom - inter_top)), 0)
        
        box1_area = (box1_right - box1_left) * (box1_bottom - box1_top)
        box2_area = (box2_right - box2_left) * (box2_bottom - box2_top)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    #Assume Angle with arctanm(vy/vx)
    def angle_assumer(self,moment):
        #assume moment = (1,8)
        vx,vy=moment[0,5:7] #slicing

        theta = np.arctan(vy/vx) #(rad)

        return theta
    
    def BB_Predict(self,moment,FrameRate):
        t=0
        BoundB=moment[0,0:4] #[[x1,y1,x2,y2]]
        Vel=moment[0,4:6]
        Vel=np.append(Vel,Vel)

        if FrameRate == 60:
            t=1
        elif FrameRate == 30:
            t=2
        elif FrameRate == 20:
            t=3  
        elif FrameRate == 10:
            t=6

        # H,W,S, ds 바탕으로 추론하는 BB predict로 바꿔야하긴 해...
        Predict_B=BoundB+ t*Vel

        return Predict_B
    
    def Angle_Predict(self, moment, FrameRate):
        ang_vel = moment[0:7]

        t=0
        if FrameRate == 60:
            t=1
        elif FrameRate == 30:
            t=2
        elif FrameRate == 20:
            t=3  
        elif FrameRate == 10:
            t=6
        
        current_theta=self.angle_assumer(moment)
        predict_theta=current_theta + ang_vel*t

        return predict_theta

    
    # Reward is calculated after state is made with FR expected.
    def compute_reward(self, moment, prev_moment, post_moment, prev_Fr, expected_FR):

        current_BB = moment[0,0:4] #FR 예측 직전, 즉 현재 action을 야기한 state
        post_BB = post_moment[0,0:4] # state를 보고 예측한 FR 기반으로 다시 찍은 state
        # Weight Options
        w_iou = 0.1
        w_theta = 1.0
        w_FR = 0.1

        # 1. Accuracy Reward
        predicted_BB = self.BB_Predict(moment,expected_FR)
        Rw_Iou = self.intersection_over_union(post_BB,predicted_BB)

        # 2. Reward with Linearity
        angl_diff=self.Angle_Predict(moment,expected_FR) - self.angle_assumer(moment)
        Rw_theta = -abs(angl_diff)

        # 3. Reward with Energy
        Rw_FR = prev_Fr - expected_FR

        # 4. total Reward
        total_reward = w_iou*Rw_Iou + w_theta*Rw_theta + w_FR*Rw_FR

        return total_reward
    
    
    ###########################
    # 2. Functions to get actions     
    def get_best_action(self, state):
        """
        Returns the action with the highest Q-value for a given state,
        along with the corresponding frame rate.
        """
        frame_rates = [10, 20, 30]  # Define frame rate options
        with torch.no_grad():
            q_values = self.policy_net(state)
            best_action = q_values.argmax(dim=1).item()  # Find the best action
            return best_action, frame_rates[best_action]  # Return action and corresponding frame rate

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy during training.
        Returns the selected action and the corresponding frame rate.
        """
        self.steps_done += 1
        frame_rates = [10, 20, 30]

        if random.random() < self.epsilon:  # Random action
            random_action = random.randrange(self.n_actions)
            return random_action, frame_rates[random_action]
        else:  # Best action based on Q-value
            return self.get_best_action(state)

    
    def select_action_model(self, state):
        """
        Select an action using greedy policy during evaluation.
        """
        return self.get_best_action(state)
    

    ########################
    # 3. Functions to form input tensor to policy network
    def compose_state(self, log_data, dtype=torch.float32):
        """
        로그 데이터를 기반으로 현재 상태(state)를 구성한다.
        """
        try:
            if not log_data:
                print("[WARNING] Log data is empty. Returning default state.")
                return torch.zeros(self.state_dim * self.max_history, dtype=dtype)

            state_features = []
            for entry in log_data[-self.max_history:]:  # 최신 `max_history`개의 로그만 사용
                parsed_data = self.parse_log_line(entry)
                if parsed_data:
                    bbox = parsed_data.get("BBox", [0, 0, 0, 0])
                    velocity = parsed_data.get("Vel", [0, 0])
                    acceleration = parsed_data.get("Acc", [0, 0])
                    angular_velocity = parsed_data.get("AngVel", 0)
                    state_vector = bbox + velocity + acceleration + [angular_velocity]
                    state_features.append(state_vector)

            while len(state_features) < self.max_history:  # 데이터 부족 시 0으로 패딩
                state_features.insert(0, [0] * len(state_features[0]))

            state_tensor = torch.tensor(state_features, dtype=dtype).unsqueeze(0).to(device)
            return state_tensor


        except Exception as e:
            print(f"[ERROR] Error composing state: {e}")
            return torch.zeros(self.state_dim * self.max_history, dtype=dtype)

    def parse_log_line(self, line):
        """
        로그 파일의 한 줄을 파싱하여 Bounding Box, 속도, 가속도, 각속도 정보를 추출
        """
        try:
            if "BBox:" in line:
                bbox_part = line.split("BBox:")[1].split(", Vel:")[0].strip()
                bbox = list(map(float, bbox_part.strip("[]").split(", ")))

                vel_part = line.split("Vel:")[1].split(", Acc:")[0].strip()
                velocity = list(map(float, vel_part.strip("[]").split(", ")))

                acc_part = line.split("Acc:")[1].split(", AngVel:")[0].strip()
                acceleration = list(map(float, acc_part.strip("[]").split(", ")))

                ang_vel_part = line.split("AngVel:")[1].strip()
                angular_velocity = float(ang_vel_part)

                return {"BBox": bbox, "Vel": velocity, "Acc": acceleration, "AngVel": angular_velocity}
        except Exception as e:
            print(f"[ERROR] Failed to parse log line: {line.strip()} - {e}")
        return None

    ########################
    # 4. Main training functions
    def optimize_model(self):
        """
        Optimize the DQN model using replay memory.
        """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = nn.SmoothL1Loss()(state_action_values.squeeze(), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy strategy."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def train(self, image_files, ground_truths, episodes=10):
        for episode in range(episodes):
            total_reward = 0
            state = self.reset(image_files, ground_truths)

            for t in range(len(image_files) - 1):
                action, selected_fps = self.select_action(state)

                next_state, reward, done = self.process_next_state(state, action, image_files, ground_truths, t)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model()

                total_reward += reward

                if done:
                    break

            self.decay_epsilon()

            print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")

            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())


        
    ########################
    # 5. Predict and evaluate functions
    def evaluate(self, image_files, ground_truths):

        ground_truth_boxes = []
        predicted_boxes = []

        print("Evaluating model...")

        for current_index in range(len(image_files) - 1):
            state = self.compose_state([image_files[current_index]])

            action, selected_fps = self.get_best_action(state)

            next_index = get_next_frame_index(current_index, selected_fps, len(image_files))
            if next_index is None:
                break  # 더 이상 이미지가 없으면 평가 종료

            next_image_files = [image_files[next_index]]

            # 객체 추적 수행 (예: YOLO로 객체 감지)
            predicted_bbox = self.track_objects(next_image_files)
            gt_bbox = ground_truths[next_index]  # 해당 프레임의 ground truth 바운딩 박스

            ground_truth_boxes.append(gt_bbox)
            predicted_boxes.append(predicted_bbox)

        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes)

        print("Final evaluation result:")
        print(stats)

        return stats
    
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

# Replay Memory를 위한 Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
'''
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """저장된 Transition 추가"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """랜덤 샘플링"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Feature Extractor for Bounding Box and Motion
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # Bounding Box Features (4개 입력)
        self.conv1_bb = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2_bb = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Motion Features (속도, 가속도, 각속도 5개)
        self.conv1_m = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.conv2_m = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

    def forward(self, BB, M):
        """Bounding Box와 Motion Feature를 추출"""
        # Bounding Box Feature
        BB = BB.permute(0, 2, 1)
        BB = torch.relu(self.conv1_bb(BB))
        BB = torch.relu(self.conv2_bb(BB))
        BB = BB.view(BB.size(0), -1) 

        # Motion Feature
        M = M.permute(0, 2, 1)
        M = torch.relu(self.conv1_m(M))
        M = torch.relu(self.conv2_m(M))
        M = M.view(M.size(0), -1)

        # Feature 결합
        features = torch.cat([BB, M], dim=-1)  
        return features

# DQN 네트워크 정의
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions, history_length=10):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim * history_length, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-Values 출력
'''
class Agent:
    def __init__(self, state_dim, n_actions, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = ReplayMemory(10000)
        self.feature_extractor = FeatureExtractor().to(device)

        self.policy_net = DQN(state_dim, n_actions).to(device)
        self.target_net = DQN(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """ε-탐욕 정책을 사용하여 행동 선택"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).argmax(dim=1).item()

    def optimize_model(self, batch_size=64):
        """DQN 업데이트"""
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros(batch_size, device=device)
        next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_q_values = (next_q_values * self.gamma) + reward_batch
        loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε-탐욕 감소
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def compose_state(self, tracked_dets, tracks, current_fps):
        bbox_xyxy = tracked_dets[:, :4]
        #print(f"[DEBUG] compose_state() 호출됨 - type(tracked_dets): {type(tracked_dets)}")
        #print(f"[DEBUG] compose_state() tracked_dets 내용: {tracked_dets}")

        velocities = [track.velocities[-1] if hasattr(track, 'velocities') and track.velocities else (0, 0) for track in tracks]
        accelerations = [track.accelerations[-1] if hasattr(track, 'accelerations') and track.accelerations else (0, 0) for track in tracks]
        angular_velocities = [track.ang_velocities[-1] if hasattr(track, 'ang_velocities') and track.ang_velocities else 0.0 for track in tracks]

        state = []
        for i in range(len(bbox_xyxy)):
            x1, y1, x2, y2 = bbox_xyxy[i]
            velocity = velocities[i]
            acceleration = accelerations[i]
            angular_velocity = angular_velocities[i]
            state.append([x1, y1, x2, y2, *velocity, *acceleration, angular_velocity, current_fps])

        return np.array(state, dtype=np.float32)



    def update_target_network(self):
        """Target Network 업데이트"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
