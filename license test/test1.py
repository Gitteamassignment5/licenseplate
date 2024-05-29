import onnxruntime as ort
import cv2
import numpy as np

# ONNX 모델 로드
ort_session = ort.InferenceSession(r'D:\license\test\fireplug.onnx')

# 모델의 첫 번째 입력 이름 가져오기
input_name = ort_session.get_inputs()[0].name
print(f"Model input name: {input_name}")

# 이미지 전처리
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Ensure the array is of type float32
    return image

def predict(image, ort_session, input_name):
    image = preprocess(image)
    outputs = ort_session.run(None, {input_name: image})
    # 결과 후처리 (예: 소화전 클래스의 인덱스가 0이라고 가정)
    predicted = np.argmax(outputs[0], axis=1)
    return predicted[0]

# 이미지 로드 및 예측
image = cv2.imread(r'D:\license\test\test1.jpg')
is_fire_hydrant = predict(image, ort_session, input_name)
print(f"Is fire hydrant: {is_fire_hydrant}")
