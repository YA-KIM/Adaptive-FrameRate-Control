use_cuda = True
import torch
import torch.nn as nn
from collections import namedtuple
import torchvision.transforms as transforms
#Selecting Tensor shape depending on CUDA availability.
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


#Saving Path of Model trained by DRL Network.
SAVE_MODEL_PATH = '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/models/q_network'
if use_cuda:
    #Loss Function
    #|Traget Q-Valye-current Q value|^2
    #Bellman equation (너무 학습이 느리면 MSE로 ㄱㄱㄱ)
    
    criterion=nn.SmoothL1Loss().cuda()
    #criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()
    #Back Propagation

#Input transformation for applying to Network.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
])