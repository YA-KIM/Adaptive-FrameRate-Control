from utility.model import *
from utility.tools import *
import os
import imageio
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
from torch.autograd import Variable

from tqdm.notebook import tqdm
from utility.config import *
from utility.moment import *

import glob
from PIL import Image

class Agent():
    def __init__(self, alpha=0.2, nu=3.0, threshold=0.5, num_episodes = None, load=False, n_actions=3, device =None):
        # basic settings
        self.n_actions = n_actions               # total number of actions : len([20, 30, 60])=3
        self.history_length= 8                   # Numver of Windowing #Check Point
        self.Version='Ver9'                      # "Ver_#로 학습하고고
        self.Load_Ver='Ver7'                     # 학습시킨 Ver로 테스트, 따라서 load를 해야함.
        self.GAMMA = 0.900                       # decay weight
        self.EPS = 1                             # initial epsilon value, decayed every epoch
        self.EPS_min=0.1
        # === Reward Weight ===
        self.w_iou = 10
        self.w_theta = 0.25
        self.w_FR = 0.1

        self.alpha = alpha                       # €[0, 1]  Scaling factor
        self.nu = nu                             # Reward of Trigger
        self.threshold = threshold               # threshold of IoU to consider as True detection
        self.actions_history = None              # action history vector as record, later initialized in train/predict
        self.steps_done = 0                      # to count how many steps it used to compute the final bdbox
        self.Frame_Rates = [20,30,60]            # 행동 공간을 선언
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # networks
        self.save_path = '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/models'         # path to save network
        self.save_version_path = os.path.join(self.save_path,self.Version)
        self.load_version_path = os.path.join(self.save_path, self.Load_Ver)

        if not load:
            self.feature_extractor = FeatureExtractor()  # NetWork: 1D CNN
        else:
            self.feature_extractor = self.load_extractor()
            self.feature_extractor.eval()            # a pre-trained CNN model as feature extractor
        
        if not load:
            self.policy_net = DQN(self.history_length, self.n_actions)
        else:
            self.policy_net = self.load_network() # policy net - DQN, inputs state vector, outputs q value for each action
            self.feature_extractor = self.load_extractor()
            self.feature_extractor.eval() 
        
        self.target_net = DQN(self.history_length, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()                    # target net - same DQN as policy net, works as frozen net to compute loss
                                                  # initialize as the same as policy net, use eval to disable Dropout
            
        # training settings
        self.BATCH_SIZE = 32                    # batch size
        self.num_episodes = num_episodes         # number of total episodes
        self.memory = ReplayMemory(10000)        # experience memory object
        self.TARGET_UPDATE = 1                   # frequence of update target net
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)  # optimizer
        
        if use_cuda:
            self.feature_extractor = self.feature_extractor.cuda()
            self.target_net = self.target_net.cuda()
            self.policy_net = self.policy_net.cuda()
            self.feature_extractor = self.feature_extractor.cuda()

    def save_network(self):
        os.makedirs(os.path.dirname(self.save_version_path), exist_ok=True)
        print(f"Saving model to: {self.save_version_path + '_policy.pth'}")
        print(f"Saving extractor to: {self.save_version_path + '_feature.pth'}") 
        torch.save(self.policy_net.state_dict(), self.save_version_path + "_policy.pth")
        torch.save(self.feature_extractor.state_dict(), self.save_version_path + "_feature.pth")
        print('Saved')

    def load_network(self):
        model = DQN(self.history_length, self.n_actions)
        if not use_cuda:
            state_dict = torch.load(self.load_version_path + "_policy.pth", map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(self.load_version_path + "_policy.pth")
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def load_extractor(self):
        model = FeatureExtractor()  # 빈 모델 인스턴스 생성
        if not use_cuda:
            state_dict = torch.load(self.load_version_path + "_feature.pth", map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(self.load_version_path + "_feature.pth")
        model.load_state_dict(state_dict)  # state_dict 로드
        model.to(self.device)              # 디바이스 설정
        return model

    
    #############################
    # 1. Functions to compute reward
    def intersection_over_union(self, box1, box2):
        """
        Compute IoU value over two bounding boxes
        Each box is represented by four elements vector: (left, right, top, bottom)
        Origin point of image system is on the top left
        """
        box1_left, box1_bottom, box1_right, box1_top = box1
        box2_left, box2_bottom, box2_right, box2_top = box2
        
        inter_top = min(box1_top, box2_top)
        inter_left = max(box1_left, box2_left)
        inter_bottom = max(box1_bottom, box2_bottom)
        inter_right = min(box1_right, box2_right)
        inter_area = max(((inter_right - inter_left) * (inter_top - inter_bottom)), 0)
        
        box1_area = (box1_right - box1_left) * (box1_top - box1_bottom)
        box2_area = (box2_right - box2_left) * (box1_top - box1_bottom)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou
    
    #Assume Angle with arctanm(vy/vx)
    def angle_assumer(self,moment):
        #assume moment = (1,8)
        vx,vy=moment[0,5:7] #slicing

        theta = np.arctan2(vy, vx) #(rad)

        return theta
    
    def BB_Predict(self,prev_moment,moment,FrameRate):
        t=0
        BoundB=moment[0,0:4] #[[cx,cy,h,w]
        cx,cy,h,w = BoundB
        vx=moment[0,4]
        vy=moment[0,5]
        if prev_moment is None:
            prev_moment = np.zeros((1, 9), dtype=np.float32) 
        dh=(moment[0,2]-prev_moment[0,2])
        dw=(moment[0,3]-prev_moment[0,3])

        if FrameRate == 60:
            t=1
        elif FrameRate == 30:
            t=2
        elif FrameRate == 20:
            t=3  

        new_cx=cx+vx*t
        new_cy=cy+vy*t
        new_h=h+dh*t
        new_w=w+dw*t

        Predict_B = torch.tensor(BoundB).clone() 
        Predict_B = torch.tensor(BoundB, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32).unsqueeze(0)
        Predict_B[0, 0:4] = torch.tensor([new_cx, new_cy, new_h, new_w], device=Predict_B.device, dtype=Predict_B.dtype)

        return Predict_B
    
    def Angle_Predict(self, moment, FrameRate):
        ang_vel = moment[0,8]

        t=0
        if FrameRate == 60:
            t=1
        elif FrameRate == 30:
            t=2
        elif FrameRate == 20:
            t=3  
        
        current_theta=self.angle_assumer(moment)
        predict_theta=current_theta + ang_vel*t

        return predict_theta

    
    # Reward is calculated after state is made with FR expected.
    def compute_reward(self, moment, prev_moment, post_moment, prev_Fr, expected_FR):

        current_BB = moment[0,0:4] #FR 예측 직전, 즉 현재 action을 야기한 state
        post_BB = post_moment[0,0:4] # state를 보고 예측한 FR 기반으로 다시 찍은 state
        # Weight Options
        w_iou = self.w_iou
        w_theta = self.w_theta
        w_FR = self.w_FR

        # 1. Accuracy Reward
        predicted_BB = self.BB_Predict(prev_moment,moment,expected_FR)
        cx,cy,h,w = current_BB
        predicted_BB = cx-w/2, cy-h/2, cx+w/2, cy+h/2  
        cx,cy,h,w = post_BB
        post_BB = cx-w/2, cy-h/2, cx+w/2, cy+h/2  
        Rw_Iou = self.intersection_over_union(post_BB,predicted_BB)
        Rw_Iou = np.nan_to_num(Rw_Iou, nan=0.0) 

        # 2. Reward with Linearity
        angl_diff=self.Angle_Predict(moment,expected_FR) - self.angle_assumer(post_moment)
        Rw_theta = -abs(angl_diff)

        # 3. Reward with Energy
        Rw_FR = prev_Fr - expected_FR
        
        # 4. total Reward
        # print(f"reward = Rw_Iou: {Rw_Iou} + Rw_theta: {Rw_theta} + Rw_FR: {Rw_FR}")
        total_reward = w_iou*Rw_Iou + w_theta*Rw_theta + w_FR*Rw_FR

        return total_reward
    
    def compute_trigger_reward(self, actual_state, ground_truth):
        """
        Compute the reward based on final IoU before *trigger*
        The reward will be +nu if final IoU is larger than threshold, and -nu if not
        ----------
        Argument:
        actual_state - final bounding box before trigger
        ground_truth - ground truth bounding box of current object
        *all bounding boxes comes in four elements vector (left, right, top, bottom)
        ----------
        Return:
        reward       - +nu/-nu depends on final IoU
        """
        res = self.intersection_over_union(actual_state, ground_truth)
        if res>=self.threshold:
            return self.nu
        return -1*self.nu
    
    ###########################
    # 2. Functions to get actions 
        #E-greedy policy: Exploitation Part
    def get_best_next_action(self, state):
        """
        Returns the action with the highest Q-value for a given state,
        along with the corresponding frame rate.
        """  
    
        with torch.no_grad(): # 사실 loss function을 구할 때 한번 더 호출되니까,,, no_grad로 학습을 꺼둠.
            if use_cuda:
                inpu = state.cuda()  #tensor 가 미분 가능하게 requires_gard = TRUE로 #이건 그냥 cuda 전송임
            else:
                inpu = state

            q_values = self.policy_net(inpu)
            #print(f"{q_values}")  #만약 보고싶으면 여기기
            best_action = q_values.argmax(dim=1).item()

            return best_action, self.Frame_Rates[best_action]    #self.Frame_Rates로 선언해둠.

        # Function with the E-greedy policy applied
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
            return self.get_best_next_action(state) # best_action, self.Frame_Rates[best_action]
        

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
    
    ########################
    # 3. Functions to form input tensor to policy network
    # 얘는 loss 계산할 때 한 번 더 호출되지 않음...
    # 굳이 no grad로 잠그고, 후반부에 다시 feature를 통과시킬 필요가 없음.
    def get_features(self, current_obj_id, state_history):
        if state_history is None or len(state_history) == 0 or current_obj_id is None:
            return None

        if state_history.ndim == 1:
            state_history = np.expand_dims(state_history, axis=0)

        state_history = state_history[::-1]  # 최신 -> 과거 순

        bb_numpy = state_history[:, :4]
        m_numpy = state_history[:, 4:]

        bb_tensor = torch.tensor(bb_numpy.astype(np.float32), device=self.device).unsqueeze(0)
        m_tensor = torch.tensor(m_numpy.astype(np.float32), device=self.device).unsqueeze(0)

        self.feature_extractor.to(self.device)
        feature = self.feature_extractor(bb_tensor, m_tensor)

        return feature

    
    # Action history를 일단 사용하지 않아보자. 만약 네트워크 성능이 부족하면 그 때 추가.
    '''
    def update_history(self, action):
        """
        Update action history vector with a new action
        ---------
        Argument:
        action         - a new taken action that should be updated into action history
        ---------
        Return:
        actions_history - a tensor of (9x9), encoding action history information
        """
        action_vector = torch.zeros(self.n_actions)
        action_vector[action] = 1
        for i in range(0,8,1):
            self.actions_history[i][:] = self.actions_history[i+1][:]
        self.actions_history[8][:] = action_vector[:]
        return self.actions_history
    '''
    


    # 우선, 시계열 데이터 1D CNN -> FCN 형태를 고려해보고 학습이 잘 안될 시 : LG AI랑 일단 동일한 방식식
    # vector->FCN concat(feature)
    ''' 
    def compose_state(self, image, dtype=FloatTensor):
        """
        Compose image feature and action history to a state variable
        ---------
        Argument:
        image - raw image data
        ---------
        state - a state variable, which is concatenation of image feature vector and action history vector
        """
        image_feature = self.get_features(image, dtype)
        image_feature = image_feature.view(1,-1)
        history_flatten = self.actions_history.view(1,-1).type(dtype)
        state = torch.cat((image_feature, history_flatten), 1)
        return state
    '''
    
    
    
    ########################
    # 4. Main training functions
    def optimize_model(self, verbose):
        """
        Sample a batch from experience memory and use this batch to optimize the model (DQN)
        """
        # if there are not enough memory, just do not do optimize
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        if len(self.memory) > 10000:
            self.memory.pop(0)


        # Sample a batch of transitions from memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))  # Unpack the transitions into individual elements
        
        # Fetch next_state_batch, excluding final states
        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool()
        next_states = [s for s in batch.next_state if s is not None]
        # Convert next states to tensor
        if len(next_states) > 0:
            non_final_next_states = Variable(torch.cat(next_states)).type(Tensor)
        '''
        if len(next_states) == 0:
            if verbose:
                print("Warning: next_states 리스트가 비어있어 optimize_model을 건너뜁니다.")
            return
        '''
        # Fetch valid states (those that are not None)
        valid_states = [state for state in batch.state if state is not None]
        if len(valid_states) > 0:
            state_batch = Variable(torch.cat(valid_states)).type(Tensor)
            
        # Move to GPU if available
        if use_cuda:
            state_batch = state_batch.cuda()

        # Fetch action_batch and reward_batch
        action_batch = Variable(torch.LongTensor(batch.action).view(-1, 1)).type(LongTensor)
        reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1, 1)).type(Tensor)
        batch_size = self.BATCH_SIZE
        if state_batch.size(0) < batch_size:
            padding_size = batch_size - state_batch.size(0)
            padding = torch.zeros(padding_size, state_batch.size(1)).type(Tensor)  # 텐서의 크기 맞추기
            state_batch = torch.cat((state_batch, padding), dim=0)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # intialize state value for next states
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE, 1).type(Tensor)) 

        if use_cuda:
            non_final_next_states = non_final_next_states.cuda()
        
        # target_net is a frozen net that used to compute q-values, we do not update its weights
        with torch.no_grad():
            d = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = d.max(1)[0].view(-1, 1)

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        if torch.isnan(state_action_values).any():
            print("NaN detected in state_action_values!")

        if torch.isnan(expected_state_action_values).any():
            print("NaN detected in expected_state_action_values!")

        if torch.isinf(state_action_values).any():
            print("Inf detected in state_action_values!")

        if torch.isinf(expected_state_action_values).any():
            print("Inf detected in expected_state_action_values!")


        # 손실 계산
        loss = criterion(state_action_values, expected_state_action_values)
        
        # if verbose:
          #  print("Loss:{}".format(loss))
            
        if torch.isnan(loss):
            print("[Warning] Loss is NaN. Skipping optimization.")

            return
        
        # optimize
        self.optimizer.zero_grad()        
        loss.backward()
        self.optimizer.step()
    #History를 input으로 받아서 내부에서 update하는걸 목표로,, 
    #일단 kal man count랑 train count랑 동기화가 되어야하는데,,,
    # History는 class History_Supervisor 형태로 관리되는인스턴스
    # train 코드에서 History를 init하고 handle해야하는데? 
    # 우선 class 형태로 Sort와 train을 handle하거나
    # self. current vector 형태로 활용해야할지도,, 
    '''
    def train(self, History, verbose = False):
        """
        Use data in a train_loader to train an agent.
        This train_loader must contain images for only one class
        Each episode is done when this agent has interacted with all training images
        Each episode is performed as following:
        - Fetch a new training image
        - The agent take an action to interacte with this image using epsilon-greedy policy
          Each step will be pushed into experience replay
          After each step, update the weights of this network once
          The interaction finishes when triggered or up to 20 steps
        - Update the target net after the whole episode is done
        - Decrease epsilon
        - Save Network
        """

        for i_episode in range(self.num_episodes):
            # Start i_episode
            print("Episode "+str(i_episode))
            img_id = 0
            # Traverse every training image to do interaction
            for key, value in  train_loader.items():
                
                if verbose:
                    img_id += 1
                    print("Training on Img {}/{}".format(img_id, len(train_loader.items())))
                    
                # fetch one image and ground_truth from train_loader
                image, ground_truth_boxes = extract(key, train_loader)
                original_image = image.clone()
                ground_truth = ground_truth_boxes[0]
                
                # initialization setting
                self.actions_history = torch.zeros((9,self.n_actions))
                new_image = image
                state = self.compose_state(image)
                
                original_coordinates = [xmin, xmax, ymin, ymax]
                self.current_coord = original_coordinates
                new_equivalent_coord = original_coordinates
              
                done = False
                t = 0
                
                # interaction with environment (image)
                while not done:
                    # increase step count
                    t += 1
                    
                    # take action according to epsilon-greedy policy
                    action = self.select_action(state, self.current_coord, ground_truth)
                    
                    # if action ==0, trigger
                    if action == 0:
                        next_state = None
                        closest_gt = self.get_max_bdbox(ground_truth_boxes, self.current_coord)
                        reward = self.compute_trigger_reward(self.current_coord, closest_gt)
                        done = True
                    
                    # if not, compute next coordinate
                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(self.current_coord, action)
                        new_xmin = self.rewrap(int(new_equivalent_coord[2])-16)
                        new_xmax = self.rewrap(int(new_equivalent_coord[3])+16)
                        new_ymin = self.rewrap(int(new_equivalent_coord[0])-16)
                        new_ymax = self.rewrap(int(new_equivalent_coord[1])+16)
                        
                        # fetch new_image (a crop of whole image) according to new coordinate
                        new_image = original_image[:, new_xmin:new_xmax, new_ymin:new_ymax]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        
                        
                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox(ground_truth_boxes, new_equivalent_coord)
                        reward = self.compute_reward(new_equivalent_coord, self.current_coord, closest_gt)
                        self.current_coord = new_equivalent_coord
                    
                    # tolerate
                    if t == 20:
                        done = True
                        
                    self.memory.push(state, int(action), next_state, reward)

                    # Move to the next state
                    state = next_state
                    image = new_image
                    
                    # Perform one step of the optimization (on the target network)
                    self.optimize_model(verbose)
                    
            
            # update target net every TARGET_UPDATE episodes
            # Traget Network는 Episdoe(one Video)마다 변화시키는게 안정적임.
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # linearly decrease epsilon on first 5 episodes
            if i_episode < 5:
                self.EPS -= 0.18
                
            # Save network every episode
            self.save_network()

            print('Complete')
        
    def get_max_bdbox(self, ground_truth_boxes, actual_coordinates):
        """
        A simple function to hanlde more than 1 object in a picture
        It will compute IoU over every ground truth box and current coordinate and choose the largest one
        And return the corresponding ground truth box as actual ground truth
        """
        max_iou = False
        max_gt = []
        for gt in ground_truth_boxes:
            iou = self.intersection_over_union(actual_coordinates, gt)
            if max_iou == False or max_iou < iou:
                max_iou = iou
                max_gt = gt
        return max_gt
    
    
    
    
    ########################
    # 5. Predict and evaluate functions
    def predict_image(self, image, plot=False, verbose=False):
        """
        Run agent on a single image, taking actions until 40 steps or triggered
        The prediction process is following:
        - Initialization
        - Input state vector into policy net and get action
        - Take action and step into new state
        - Terminate if trigger or take up to 40 steps
        ----------
        Argument:
        image                - Input image, should be resized to (224,224) first
        plot                 - Bool variable, if True, plot all intermediate bounding box
        verbose              - Bool variable, if True, print out intermediate bouding box and taken action
        ---------
        Return:
        new_equivalent_coord - The final bounding box coordinates
        cross_flag           - If it should apply cross on the image, if done with trigger, True; if done with 40 steps, False
        steps                - how many steps it consumed
        """
        # set policy net to evaluation model, disable dropout
        self.policy_net.eval()
        
        # initialization
        original_image = image.clone()
        self.actions_history = torch.zeros((9,self.n_actions))
        state = self.compose_state(image)
        
        new_image = image
        self.current_coord = [0, 224, 0, 224]
        steps = 0
        done = False
        cross_flag = True
        
        # start interaction
        while not done:
            steps += 1
            # take action according to greedy policy
            action = self.select_action_model(state)
            
            if action == 0:
                next_state = None
                new_equivalent_coord = self.current_coord
                done = True
            else:
                self.actions_history = self.update_history(action)
                new_equivalent_coord = self.calculate_position_box(self.current_coord, action)
                
                new_xmin = self.rewrap(int(new_equivalent_coord[2])-16)
                new_xmax = self.rewrap(int(new_equivalent_coord[3])+16)
                new_ymin = self.rewrap(int(new_equivalent_coord[0])-16)
                new_ymax = self.rewrap(int(new_equivalent_coord[1])+16)
                
                new_image = original_image[:, new_xmin:new_xmax, new_ymin:new_ymax]
                try:
                    new_image = transform(new_image)
                except ValueError:
                    break            
                
                next_state = self.compose_state(new_image)
                self.current_coord = new_equivalent_coord
            
            if steps == 40:
                done = True
                cross_flag = False
            
            state = next_state
            image = new_image
            
            if verbose:
                print("Iteration:{} - Action:{} - Position:{}".format(steps, action, new_equivalent_coord))
            
            # if plot, print out current bounding box
            if plot:
                show_new_bdbox(original_image, new_equivalent_coord, color='b', count=steps)
                
            
        
        # if plot, save all changing in bounding boxes as a gif
        if plot:
#             images = []
#             tested = 0
#             while os.path.isfile('media/movie_'+str(tested)+'.gif'):
#                 tested += 1
#             # filepaths
#             fp_out = "media/movie_"+str(tested)+".gif"
#             images = []
#             for count in range(1, steps+1):
#                 images.append(imageio.imread(str(count)+".png"))
            
#             imageio.mimsave(fp_out, images)
            
#             for count in range(1, steps + 1):
#                 os.remove(str(count)+".png")
                
                
        return new_equivalent_coord, cross_flag, steps
    
    def predict_multiple_objects(self, image, plot=False, verbose=False):
        """
        Iteratively predict multiple objects, when one object is detected, draw a cross on it
        Perform up to 100 steps
        """
        
        new_image = image.clone()
        all_steps = 0
        bdboxes = []   
        
        while 1:
            bdbox, cross_flag, steps = self.predict_image(new_image, plot, verbose)
            bdboxes.append(bdbox)
            
            if cross_flag:
                mask = torch.ones((224,224))
                middle_x = round((bdbox[0] + bdbox[1])/2)
                middle_y = round((bdbox[2] + bdbox[3])/2)
                length_x = round((bdbox[1] - bdbox[0])/8)
                length_y = round((bdbox[3] - bdbox[2])/8)

                mask[middle_y-length_y:middle_y+length_y,int(bdbox[0]):int(bdbox[1])] = 0
                mask[int(bdbox[2]):int(bdbox[3]),middle_x-length_x:middle_x+length_x] = 0

                new_image *= mask
                
            all_steps += steps
                
            if all_steps > 100:
                break
                    
        return bdboxes
        
    
    def evaluate(self, dataset):
        """
        Conduct evaluation on a given dataset
        For each image in this dataset, using this agent to predict a bounding box on it
        Save predicted bdbox and ground truth bdbox to two lists
        Send these two lists to tool function eval_stats_at_threshold and get results
        *you can manually define threshold by setting threshold argument of this tool function*
        
        Return a dataframe that contains the result
        """
        ground_truth_boxes = []
        predicted_boxes = []
        print("Predicting boxes...")
        for key, value in dataset.items():
            image, gt_boxes = extract(key, dataset)
            bbox = self.predict_multiple_objects(image)
            ground_truth_boxes.append(gt_boxes)
            predicted_boxes.append(bbox)

        print("Computing recall and ap...")
        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes)
        print("Final result : \n"+str(stats))
        return stats
    '''
