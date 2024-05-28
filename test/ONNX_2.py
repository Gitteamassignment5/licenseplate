# netron fireplug.onnx flowchart 

import onnx

import numpy as np

# Load the ONNX model
onnx_model = onnx.load("fireplug.onnx")
version = onnx_model.ir_version
print("ONNX 모델 ir 버전:", version)

# ONNX 모델의 버전 정보
version_info = onnx_model.opset_import[0]
# 메이저 버전과 마이너 버전 추출
major_version = version_info.version
minor_version = version_info.domain
# ONNX 버전 출력
print(f"ONNX opset 버전: {major_version}.{minor_version}")

import onnxruntime as ort
# ONNX Runtime 버전 확인
version = ort.__version__
print("ONNX Runtime 버전:", version)