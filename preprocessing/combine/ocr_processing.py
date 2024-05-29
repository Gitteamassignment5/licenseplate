# ocr_processing.py
import cv2
import numpy as np
import pytesseract
import os
import csv
import imutils
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

    # 전처리된 이미지 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(morph, cmap='gray')
    plt.title('Preprocessed Image')
    plt.show()
    
    return morph

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)