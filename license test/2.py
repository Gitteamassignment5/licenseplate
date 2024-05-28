# ocr_module.py
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

# 메인 실행 코드
input_image_path = r'D:\license\img1'  # 입력 이미지가 저장된 경로
save_image_path = 'D:/license/results/images'  # 중간 결과 이미지를 저장할 경로
save_text_path = 'D:/license/results/text'  # 인식된 텍스트를 저장할 경로

# 저장 경로가 존재하지 않으면 생성
os.makedirs(save_image_path, exist_ok=True)
os.makedirs(save_text_path, exist_ok=True)

# CSV 파일 생성 및 헤더 작성
csv_file_path = os.path.join(save_text_path, 'recognized_text.csv')
with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image Number', 'Recognized Text'])

# 이미지 불러오기 및 처리
for images in range(1, 21):
    img_path = os.path.join(input_image_path, f"{images}.jpg")
    boxes_np, nm_index, image, rois = detect_number_plate_yolo(img_path, net)
    
    if rois:
        for idx, roi in enumerate(rois):
            roi = imutils.resize(roi, width=500)
            cropped_image = preprocessing(roi)

            if cropped_image is not None:
                # 번호판 영역 이미지 저장
                save_img_filename = os.path.join(save_image_path, f'plate_{images}_{idx}.png')
                cv2.imwrite(save_img_filename, cropped_image)

                # pytesseract를 사용하여 텍스트 인식
                custom_config = r'--psm 6 --oem 3'  # 다양한 설정을 시도해 볼 수 있습니다
                text = pytesseract.image_to_string(cropped_image, lang='kor', config=custom_config)
                print(f"{images}-{idx}: {text}")

                # 인식된 텍스트를 CSV 파일에 저장
                with open(csv_file_path, mode='a', newline='', encoding='utf-8-sig') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([f"{images}-{idx}", text])
    else:
        print(f"번호판을 찾을 수 없습니다: {img_path}")
