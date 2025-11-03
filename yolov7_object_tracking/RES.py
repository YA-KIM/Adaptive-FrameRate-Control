import torch
import torchvision.models as models

# ResNet101 모델 다운로드 및 초기화 (사전 학습된 가중치 포함)
model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

# GPU 사용을 원할 경우 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 모델을 평가 모드로 설정 (추론 시 사용)
model.eval()

from torchvision.models import resnet101, ResNet101_Weights

# 커스텀 디렉토리 설정
torch.hub.set_dir('/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking')
# 모델 가중치를 저장
torch.save(model.state_dict(), '/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/resnet101.pt')

checkpoint = torch.load('/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/resnet101.pt', map_location=device)
print(checkpoint.keys())
