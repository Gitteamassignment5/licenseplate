import cv2
import os
import numpy as np

# YOLOv5 모델 입력 이미지 크기
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# YOLOv5 모델 로드
net = cv2.dnn.readNetFromONNX(r'D:\license\test\car_number.onnx') 
net = cv2.dnn.readNetFromONNX(r'D:\license\test\fireplug.onnx') 
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def read_image(image_path):
    """이미지 파일을 읽어오는 함수"""
    if os.path.exists(image_path):
        try:
            img = cv2.imread(image_path)
            if img is not None:
                return img
            else:
                print('이미지가 비어있습니다.')
        except Exception as e:
            print(f'이미지를 읽는 중 오류가 발생했습니다: {image_path}')
    else:
        print(f'경로가 존재하지 않습니다: {image_path}')
    return None

def get_detections(img, net):
    """이미지에서 객체 감지를 수행하는 함수"""
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image, detections):
    """비최대 억제(NMS)를 사용하여 중복된 박스를 제거하는 함수"""
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
    
    return boxes_np, confidences_np, index

def yolo_predictions(img, net):
    """YOLO 모델을 사용하여 예측을 수행하는 함수"""
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    return boxes_np, index

def croptheROI(image, bbox, index):
    """감지된 객체의 ROI(관심 영역)를 자르는 함수"""
    rois = []
    if len(index) > 0:
        for i in index.flatten():
            x, y, w, h = bbox[i]
            rois.append(image[y:y+h, x:x+w])
    return rois

def detect_number_plate_yolo(image_path, net):
    """이미지에서 번호판을 감지하고 ROI를 추출하는 함수"""
    img = read_image(image_path)
    if img is None:
        return None, None, None, None

    boxes_np, nm_index = yolo_predictions(img, net)
    rois = croptheROI(img, boxes_np, nm_index)

    return boxes_np, nm_index, img, rois
