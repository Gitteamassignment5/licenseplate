import cv2
import pytesseract
import datetime
import pandas as pd
import os
import imutils
from ocr_processing import preprocessing  # ocr_processing에서 preprocessing 함수 import
from image_opencv_postprocessing_v2 import detect_number_plate_yolo, net

# Tesseract OCR 경로 설정 (예: Windows의 경우)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 요일과 금지되는 차량 번호 마지막 숫자 매핑
restriction_map = {
    0: [1, 6],  # 월요일
    1: [2, 7],  # 화요일
    2: [3, 8],  # 수요일
    3: [4, 9],  # 목요일
    4: [5, 0],  # 금요일
}

def extract_license_plate_number(image_path):
    # 이미지 파일 경로 확인
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")
    
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 열 수 없습니다: {image_path}. 파일 경로를 확인하세요.")
    
    print(f"이미지 읽기 성공: {image_path}")
    
    # 번호판 영역 추출 (OCR 모듈의 함수 사용)
    boxes_np, nm_index, image, rois = detect_number_plate_yolo(image_path, net)
    
    if rois:
        for idx, roi in enumerate(rois):
            roi = imutils.resize(roi, width=500)
            cropped_image = preprocessing(roi)

            if cropped_image is not None:
                # OCR로 텍스트 추출
                custom_config = r'--psm 6 --oem 3'
                text = pytesseract.image_to_string(cropped_image, lang='kor', config=custom_config)
                print(f"OCR 결과: {text}")
                
                # 숫자만 추출
                number = ''.join(filter(str.isdigit, text))
                return number
    else:
        print(f"번호판을 찾을 수 없습니다: {image_path}")
        return None

def classify_vehicle(car_number):
    # 차량 번호 앞 두 자리 추출
    front_number = int(car_number[:2])
    
    # 차량 유형 분류
    if 1 <= front_number <= 69:
        return "승용차"
    elif 70 <= front_number <= 79:
        return "승합차"
    elif 80 <= front_number <= 97:
        return "화물차"
    elif front_number in [98, 99]:
        return "특수차"
    else:
        return "알 수 없음"

def can_enter_public_office(car_number):
    # 차량 번호 마지막 숫자 추출
    last_digit = int(car_number[-1])
    
    # 오늘의 요일 추출 (0: 월요일, 1: 화요일, ..., 6: 일요일)
    today = datetime.datetime.today().weekday()
    
    if today < 5:  # 월요일(0)부터 금요일(4)까지 확인
        if last_digit in restriction_map[today]:
            return "출입 불가"
        else:
            return "출입 가능"
    else:
        return "출입 가능 (주말)"

def main(image_paths, csv_output_path):
    results = []
    for image_path in image_paths:
        car_number = extract_license_plate_number(image_path)
        if car_number:
            vehicle_type = classify_vehicle(car_number)
            result = can_enter_public_office(car_number)
            results.append({
                "차량 번호": car_number,
                "차량 유형": vehicle_type,
                "출입 가능 여부": result,
                "날짜": datetime.datetime.today().strftime('%Y-%m-%d')  # 날짜 형식 지정
            })
        else:
            results.append({
                "차량 번호": "인식 실패",
                "차량 유형": "알 수 없음",
                "출입 가능 여부": "알 수 없음",
                "날짜": datetime.datetime.today().strftime('%Y-%m-%d')  # 날짜 형식 지정
            })
    
    # 결과 디렉토리가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')  # UTF-8 BOM 인코딩 사용
    print(f"결과가 {csv_output_path}에 저장되었습니다.")
