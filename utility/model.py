import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# 학습이 원활이 진행되지 않을 경우, 1D CNN의 layer를 깊게 해보는 시도 필요.
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        #bb=[cx,cy,w,h] : [중심x, 중심y, 넓이, 높이]: input 4 (위치 정보와 물체의 크기만,, BB를 다 담는것은 학습에 오히려 방해가 될 수도 있음)
        self.conv1_bb= nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2_bb=nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # CNN for Motion Data (5 features: vx, vy, ax, ay, ang_vel)
        self.conv1_m = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.conv2_m = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.BB_weights = nn.Parameter(torch.ones(1))
        self.M_weights = nn.Parameter(torch.ones(1))

    def forward(self, BB, M):
        '''
        BB: tensor는 (batch,History,input_vector_num)으로 구성되어있음 (batch,5,8) [BB1, BB2 ...]
        permute를 사용하면 (batch, input_num, History=sequence input) [x1(1),x1(2),...X(5)]

        BB: (batch, 5, 8) → 5개 프레임, 8개 Bounding Box 특징
        M:  (batch, 5, 5) → 5개 프레임, 5개 Motion 정보 (vx, vy, ax, ay, ang_vel)
        
        프레임은 10개로 조절해야하긴 함 : 5->10
        '''
        assert BB.device == M.device == self.BB_weights.device, \
            f"Device mismatch: BB ({BB.device}), M ({M.device}), BB_weights ({self.BB_weights.device})"

        #Bounding Box Feature Extract
        BB=BB.permute(0,2,1)
        BB=F.relu(self.conv1_bb(BB)) # BB=nn.ReLu()(self.conv1_bb(BB))와 동일; nn.ReLu를 모듈로 호출후에 대입하나 바로 대입되는 레이어로 호출하나의 차이
        BB=F.relu(self.conv2_bb(BB))
        BB=BB.view(BB.size(0),-1) #Faltten -> (batch, Output features*History)=(Batch,64*10); 1D conv는 공간적인 정보를 추출하므로 텐서를 1차원 vector로 변환

        #Motion Feature Extract by 1D conv
        M=M.permute(0,2,1)
        M=F.relu(self.conv1_m(M))
        M=F.relu(self.conv2_m(M))
        M=M.view(M.size(0),-1) #(Batch,16*10)
        #학습 가능한 가중치를 부여, DRL Reward를 loss로 학습함.

        features= torch.cat([self.BB_weights*BB,self.M_weights*M],dim=-1) # (batch, 96=64(bb)+32(m))
        # 가중치 업데이트가 궁금하면; print(f"weight_bb: {model.weight_bb.item()}, weight_m: {model.weight_m.item()}")
        #->만약 학습이 잘 안된다면, Batch정규화, layer정규화를 통해 Motion와 Bounding Box가 같은 범위 내에서 놀게 해야함.

        return features
    
    #학습을 진행해보면서 layer개수, output feature 줄어드는 것 등을 조절해야함
class DQN(nn.Module): 
    def __init__(self, history_length, n_actions):
        super(DQN, self).__init__()
        self.state_dim=96
        self.history_length = history_length
        self.classifier = nn.Sequential(
            nn.Linear(self.state_dim * history_length, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_actions))
    def forward(self, features):
      return self.classifier(features)


# Policy based on Actor-Critic, we don't have to use in Frame Rate control Project.  
''' 
class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return self.actor_linear(x), self.critic_linear(x)

    '''
    






'''
BB와 Motion을 분리해서 할당해야함. 1D CNN을 분리해서 적용한 값이 동시에 FCN에 들어가는 형태..


import torch
import torch.nn as nn

class MotionAwareCNN(nn.Module):
    def __init__(self):
        super(MotionAwareCNN, self).__init__()

        # CNN for Bounding Box (8 features: x1, y1, x2, y2, w, h, cx, cy)
        self.conv1_bb = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
        self.conv2_bb = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # CNN for Motion Data (5 features: vx, vy, ax, ay, ang_vel)
        self.conv1_m = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.conv2_m = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fully Connected Layer
        self.fc = nn.Linear(96, 3)  # 96 = CNN(BB 64) + CNN(Motion 32)

    def forward(self, BB, M):
        """
        BB: (batch, 5, 8) → 5개 프레임, 8개 Bounding Box 특징
        M:  (batch, 5, 5) → 5개 프레임, 5개 Motion 정보 (vx, vy, ax, ay, ang_vel)
        """
        # CNN 처리 (Bounding Box)
        BB = BB.permute(0, 2, 1)  # (batch, 8, 5) → CNN 입력 형태
        BB = nn.ReLU()(self.conv1_bb(BB))
        BB = nn.ReLU()(self.conv2_bb(BB))
        BB = BB.view(BB.size(0), -1)  # Flatten (batch, 64)

        # CNN 처리 (Motion)
        M = M.permute(0, 2, 1)  # (batch, 5, 5) → (batch, 5, 5)
        M = nn.ReLU()(self.conv1_m(M))
        M = nn.ReLU()(self.conv2_m(M))
        M = M.view(M.size(0), -1)  # Flatten (batch, 32)

        # CNN 특징 결합
        x = torch.cat([BB, M], dim=-1)  # (batch, 96)

        # 최종 예측
        action = self.fc(x)
        return action
        '''