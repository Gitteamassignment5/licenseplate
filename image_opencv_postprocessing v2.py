import cv2
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# YOLOv5 모델 로드
net = cv2.dnn.readNetFromONNX('D:/license/test/best.onnx') 
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def read_image(image_path):
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

def image_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def bilateral_filter(gray_img):
    return cv2.bilateralFilter(gray_img, 11, 17, 17)

def canny_edge_detection(filtered_img):
    return cv2.Canny(filtered_img, 170, 200)

def invert_color(grayscale_img):
    return cv2.bitwise_not(grayscale_img)

def binarize_img(inverted_img):
    _, binary = cv2.threshold(inverted_img, 100, 255, cv2.THRESH_BINARY)
    return binary

def dilate_image(binary_image):
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel3)
    return thre_mor

def get_detections(img, net):
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
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    return boxes_np, index

def croptheROI(image, bbox, index):
    rois = []
    if len(index) > 0:
        for i in index.flatten():
            x, y, w, h = bbox[i]
            rois.append(image[y:y+h, x:x+w])
    return rois

def detect_number_plate_yolo(image_path, net):
    img = read_image(image_path)
    if img is None:
        return None, None, None, None

    boxes_np, nm_index = yolo_predictions(img, net)
    rois = croptheROI(img, boxes_np, nm_index)

    return boxes_np, nm_index, img, rois

def preprocessing(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(crop, crop, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    return cropped_image

def draw_number_plate(image, NumberPlateCnt):
    if NumberPlateCnt is not None:
        cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    return image

def find_contours(dilated_img, roi):
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi, contours, -1, (0, 255, 0), 3)
    return roi, contours

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# 메인 실행 코드
image_path = 'D:/license/img1/023170-5.jpg'  # 이미지 경로 설정
save_dir = 'D:/license/test/123'
ensure_dir(save_dir)  # 저장 경로 디렉토리 확인 및 생성

boxes_np, nm_index, image, rois = detect_number_plate_yolo(image_path, net)

if rois:
    for idx, roi in enumerate(rois):
        roi = imutils.resize(roi, width=500)
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # 이미지 전처리 및 시각화
        gray = image_to_grayscale(roi)

        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        ax[0, 0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('Original Image')

        ax[0, 1].imshow(gray, cmap='gray')
        ax[0, 1].set_title('Grayscale Conversion')

        filtered = bilateral_filter(gray)
        ax[0, 2].imshow(filtered, cmap='gray')
        ax[0, 2].set_title('Bilateral Filter')

        edged = canny_edge_detection(filtered)
        ax[1, 0].imshow(edged, cmap='gray')
        ax[1, 0].set_title('Canny Edges')

        inverted = invert_color(edged)
        binary = binarize_img(inverted)
        dilated = dilate_image(binary)
        ax[1, 1].imshow(dilated, cmap='gray')
        ax[1, 1].set_title('Dilated Image')

        processed_img, contours = find_contours(dilated, roi.copy())
        ax[1, 2].imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        ax[1, 2].set_title('Contours on Image')

        fig.tight_layout()
        plt.show()

        # 번호판 탐지
        cropped_image = preprocessing(roi)

        if cropped_image is not None:
            plt.figure(figsize=(10, 7))
            plt.imshow(cropped_image, cmap='gray')
            plt.title('Detected Number Plate Region')
            plt.savefig(f'{save_dir}/detected_number_plate_region_{idx}.png')  # 결과 저장
            plt.show()

        # 번호판 외곽선 그리기 및 시각화
        final_img = draw_number_plate(roi, contours)
        plt.figure(figsize=(10, 7))
        plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        plt.title('Final Image with Number Plate Contours')
        plt.savefig(f'{save_dir}/final_image_with_number_plate_contours_{idx}.png')
        plt.show()