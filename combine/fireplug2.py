import onnxruntime as ort
import numpy as np
import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_shape = (640, 640)
    image = cv2.resize(image, input_shape)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def load_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def run_inference(session, image):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image})
    return outputs

def postprocess_output(output, score_threshold=0.5):
    output = output[0]
    boxes = output[:, :4]
    scores = output[:, 4]
    classes = output[:, 5]
    
    # 디버깅 정보 출력
    print("Boxes:", boxes)
    print("Scores:", scores)
    print("Classes:", classes)
    
    # 소화전 클래스 ID가 0이라고 가정 (이것은 실제 데이터셋에 따라 다를 수 있습니다)
    fire_hydrant_class_id = 1
    
    # 클래스가 소화전이고 점수가 임계값보다 높은 경우를 확인
    fire_hydrant_detected = np.any((classes == fire_hydrant_class_id) & (scores > score_threshold))
    
    return fire_hydrant_detected

def detect_fire_hydrants(image_path, model_path, score_threshold=0.5):
    image = preprocess_image(image_path)
    session = load_model(model_path)
    outputs = run_inference(session, image)
    
    # 출력 결과 디버깅
    print("Model outputs:", outputs)
    
    is_fire_hydrant_detected = postprocess_output(outputs, score_threshold)
    return is_fire_hydrant_detected

# Example usage
model_path = r'D:\license\test\fireplug.onnx'
image_path = r'D:\license\test\test1.jpg'
result = detect_fire_hydrants(image_path, model_path)
print(f"Fire hydrant detected: {result}")
