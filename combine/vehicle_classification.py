import cv2
import pytesseract
import datetime
import pandas as pd
import os
import imutils
from ocr_processing import preprocessing, extract_and_recognize_text  # ocr_processing에서 preprocessing 및 새로운 함수 import
from image_opencv_postprocessing_v2 import detect_number_plate_yolo  # net 제거

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

def extract_license_plate_last_char(image_path):
    # 이미지 파일 경로 확인
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")
    
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 열 수 없습니다: {image_path}. 파일 경로를 확인하세요.")
    
    print(f"이미지 읽기 성공: {image_path}")
    
    # 번호판 영역 및 문자를 추출하는 외부 모듈 함수 사용
    plate_chars = extract_and_recognize_text(image_path)
    return plate_chars

def classify_vehicle(plate_chars):
    # 숫자와 문자 분리
    digits = ''.join(filter(str.isdigit, plate_chars))
    
    if len(digits) < 2:
        return "알 수 없음"
    
    # 차량 번호 앞 두 자리 추출
    front_number = int(digits[:2])
    
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

def can_enter_public_office(plate_chars):
    # 숫자와 문자 분리
    digits = ''.join(filter(str.isdigit, plate_chars))
    
    if not digits:
        return "출입 가능 (숫자 아님)"
    
    # 차량 번호 마지막 숫자 추출
    last_digit = int(digits[-1])
    
    # 오늘의 요일 추출 (0: 월요일, 1: 화요일, ..., 6: 일요일)
    today = datetime.datetime.today().weekday()
    
    if today < 5:  # 월요일(0)부터 금요일(4)까지 확인
        if last_digit in restriction_map[today]:
            return "출입 불가능"
        else:
            return "출입 가능"
    else:
        return "출입 가능 (주말)"

def main(image_paths, csv_output_path):
    results = []
    for image_path in image_paths:
        plate_chars = extract_license_plate_last_char(image_path)
        if plate_chars:
            vehicle_type = classify_vehicle(plate_chars)
            result = can_enter_public_office(plate_chars)
            results.append({
                "차량 번호": plate_chars,
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

# 예시로 이미지 경로를 입력하여 함수 호출

