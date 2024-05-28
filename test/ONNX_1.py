import torch
import torch.onnx
import sys
import os

# YOLOv5 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

from models.yolo import Model
from utils.general import check_yaml, intersect_dicts
from utils.torch_utils import select_device

# 모델 경로와 출력 ONNX 파일 경로 설정
model_path = 'runs/train/exp3/weights/car_number.pt'
onnx_file_path = 'car_number.onnx'
cfg = 'yolov5/models/yolov5m.yaml'  # YOLOv5 모델 구성 파일

# 디바이스 선택 (CPU 사용)
device = select_device('cpu')

# YOLOv5 모델 로드
model = Model(cfg, ch=3, nc=46).to(device)  # 모델 초기화 (nc=38은 클래스 수에 따라 조정)

# 체크포인트 로드
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['model'].float().state_dict()

# 클래스 수가 다른 경우 맞춤형 state_dict 사용
model_state_dict = model.state_dict()
state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
model_state_dict.update(state_dict)
model.load_state_dict(model_state_dict)

model.eval()

# 더미 입력 생성 (배치 크기: 1, 채널 수: 3, 이미지 크기: 640x640)
dummy_input = torch.randn(5, 3, 640, 640).to(device)

# ONNX 파일로 내보내기
torch.onnx.export(model, dummy_input, onnx_file_path, verbose=True, opset_version=12, input_names=['images'], output_names=['output'])

print(f"ONNX 모델이 {onnx_file_path}로 저장되었습니다.")
