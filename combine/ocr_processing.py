import cv2
import numpy as np
import pytesseract
import os
import matplotlib.pyplot as plt
from image_opencv_postprocessing_v2 import detect_number_plate_yolo, net

# Tesseract 경로 설정 (Windows에서만 필요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def preprocessing(crop):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # CLAHE 적용하여 이미지 대비를 향상시킴
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 가우시안 블러를 적용하여 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Otsu 이진화
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 모폴로지 변환을 통해 노이즈 제거 및 텍스트 강조
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # 확대하여 인식률 높이기
    morph = cv2.resize(morph, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 윤곽선을 강조하여 선명도 향상
    edges = cv2.Canny(morph, 100, 200)
    morph = cv2.bitwise_or(morph, edges)

    return morph

def extract_and_recognize_text(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found or unable to load.")
        return
    
    # YOLO를 사용하여 번호판 탐지
    plates_info = detect_number_plate_yolo(image_path)
    
    if plates_info is None or plates_info[0] is None:
        print("No plates detected.")
        return
    
    boxes_np, nm_index, img, rois = plates_info

    plate_infos = []
    plate_chars = []

    for i, (x, y, w, h) in enumerate(boxes_np):
        plate_img = img[y:y+h, x:x+w]
        preprocessed_img = preprocessing(plate_img)

        # Tesseract를 사용하여 문자 인식
        chars = pytesseract.image_to_string(preprocessed_img, lang='kor', config='--psm 7 --oem 0')
        result_chars = ''
        has_digit = False

        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c

        if has_digit and len(result_chars) > 0:
            plate_infos.append({'x': x, 'y': y, 'w': w, 'h': h, 'chars': result_chars})
            plate_chars.append(result_chars)

            # 결과 이미지 시각화
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, result_chars, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if plate_infos:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Detected License Plates')
        plt.show()
        
        # 인식된 번호판 문자 출력
        for info in plate_infos:
            print(f"Detected Plate: {info['chars']} at (x: {info['x']}, y: {info['y']}, w: {info['w']}, h: {info['h']})")
    else:
        print("No valid license plate characters detected.")

    # 마지막 배열값 가져오기
    if plate_chars:
        return plate_chars[-1]
    else:
        return None

