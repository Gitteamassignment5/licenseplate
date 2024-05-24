import os
import cv2
import pytesseract
import numpy as np
from PIL import Image
import csv

# Tesseract 경로 설정 (Tesseract가 설치된 경로를 설정)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image_for_ocr(image_path, output_dir, args):
    # 이미지 경로 확인
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    else:
        print(f"Image loaded successfully: {image_path}")
    img_for_crop = Image.open(image_path)
    height, width, channel = img.shape

    # Gray 이미지로 변환
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, 'gray_image.jpg'), img_gray)

    # 가우시안 Blur 필터 적용
    img_blur = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=0)
    cv2.imwrite(os.path.join(output_dir, 'blur_image.jpg'), img_blur)

    # 문턱값 설정
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(os.path.join(output_dir, 'threshold_image.jpg'), img_thresh)

    # 노이즈 제거를 위한 모폴로지 연산
    kernel = np.ones((3, 3), np.uint8)
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, 'morphology_image.jpg'), img_thresh)

    # 컨투어 찾기
    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours found: {len(contours)}")
    if not contours:
        print("No contours found")
        return

    # 컨투어 정보 저장
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, (x, y), (x + w, y + h), (255, 255, 255), 1)
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    # 중간 결과 저장
    cv2.imwrite(os.path.join(output_dir, 'contours.jpg'), temp_result)

    # 컨투어 필터링
    MAX_AREA, MIN_AREA = 2000, 50  # Adjusted areas for more flexibility
    MIN_RATIO, MAX_RATIO = 0.2, 1.2  # Adjusted ratio range
    first_course_contours = []
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        if MIN_AREA < area < MAX_AREA and MIN_RATIO < ratio < MAX_RATIO:
            first_course_contours.append(d)

    if len(first_course_contours) == 0:
        print("No suitable contours found after first filter")
        return
    print(f"Number of suitable contours: {len(first_course_contours)}")

    # 번호판 위치 추정 (첫 번째 큰 컨투어 사용)
    x, y, w, h = first_course_contours[0]['x'], first_course_contours[0]['y'], first_course_contours[0]['w'], first_course_contours[0]['h']
    cropped_plate = img_for_crop.crop((x, y, x + w, y + h))
    cropped_plate_path = os.path.join(output_dir, 'cropped_plate.jpg')
    cropped_plate.save(cropped_plate_path)
    print(f"Cropped plate image saved to: {cropped_plate_path}")

    # OCR을 위한 이미지 전처리
    cropped_plate_cv = np.array(cropped_plate)
    cropped_plate_gray = cv2.cvtColor(cropped_plate_cv, cv2.COLOR_BGR2GRAY)
    _, cropped_plate_thresh = cv2.threshold(cropped_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cropped_plate_enlarged = cv2.resize(cropped_plate_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cropped_plate_enlarged = cv2.GaussianBlur(cropped_plate_enlarged, (5, 5), 0)
    
    cv2.imwrite(os.path.join(output_dir, 'cropped_plate_processed.jpg'), cropped_plate_enlarged)

    # OCR 수행
    result = pytesseract.image_to_string(cropped_plate_enlarged, lang="kor")
    print(f"OCR Result: {result}")

    # OCR 결과가 비어 있지 않은지 확인
    if not result.strip():
        print("OCR result is empty")
        return

    # OCR 결과 저장
    car_number_csv_path = os.path.join(output_dir, 'car_number.csv')
    try:
        with open(car_number_csv_path, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([result])
        print(f"OCR result saved to: {car_number_csv_path}")
    except IOError as e:
        print(f"Failed to write to file: {e}")

# Example usage
args = type('', (), {})()
args.imgW = 100
args.imgH = 32

# Uploaded image path
uploaded_image_path = 'D:/license/test/123/unnamed.png'
output_directory = 'D:/license/test/123test/'

process_image_for_ocr(uploaded_image_path, output_directory, args)