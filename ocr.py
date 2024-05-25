import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image, ImageEnhance
import math
import os

# Tesseract 경로 설정 (Windows에서만 필요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 이미지 파일 경로 설정
images_path = 'D:/license/img1'  # 이미지 파일이 저장된 경로
save_path = 'D:/license/test/123/'  # 결과 이미지를 저장할 경로

# 처리할 이미지 파일 목록 (원하는 파일명으로 수정)
image_files = [
    '02_3170-4.jpg',
    '02라3170-6.jpg',
    '02라4201.jpg',
    '02라8214.jpg',
    '02로6470.jpg',
    '02루8771.jpg'
]

# 저장 경로가 존재하지 않으면 생성
os.makedirs(save_path, exist_ok=True)

for file_name in image_files:
    img_path = os.path.join(images_path, file_name)
    if not os.path.isfile(img_path):
        print(f"파일을 찾을 수 없습니다: {img_path}")
        continue  # 파일이 없으면 다음 반복으로 넘어감

    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {img_path}")
        continue
    
    img_for_crop = Image.open(img_path)
    height, width, channel = img.shape

    # -------- Gray 이미지로 바꿔주기 -------- #
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------- 명암 대비 조절 -------- #
    img_enhanced = ImageEnhance.Contrast(Image.fromarray(img_gray)).enhance(2.0)
    img_gray = np.array(img_enhanced)

    plt.figure(figsize=(12, 10))
    plt.imshow(img_gray, cmap='gray')
    plt.title("Gray Image with Enhanced Contrast")
    plt.show()

    # -------- 가우시안 Blur 필터 적용 = 부드럽게 해서 잡다한 노이즈 제거 -------- #
    img_blur = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=0)
    plt.figure(figsize=(12, 10))
    plt.imshow(img_blur, cmap='gray')
    plt.title("Gaussian Blur Image")
    plt.show()

    # -------- 문턱값 설정 ------- #
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 11)
    plt.figure(figsize=(12, 10))
    plt.imshow(img_thresh, cmap='gray')
    plt.title("Threshold Image")
    plt.show()

    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    # -------- 생성된 Contours 정보 저장 ------- #
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=1)

        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),  # 사각형의 중심 좌표 x,y
            'cy': y + (h / 2)
        })

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')
    plt.title("Contours")
    plt.show()

    # -------- 생성된 Contours 편하게 추출 ------- #
    MAX_AREA, MIN_AREA = 1200, 150  # 작은 크기도 인식해야 하므로 크기를 줄여준다 + MAX 설정
    MIN_RATIO, MAX_RATIO = 0.2, 0.9  # 번호판 속 숫자의 컨투어 박스 비율에 맞도록 재설정
    MIN_WIDTH, MIN_HEIGHT = 3, 8
    first_course_contours = []
    middle_course_contours = []
    possible_contours = []

    cnt = 0
    average_w = 0
    average_h = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if MIN_AREA < area < MAX_AREA and MIN_RATIO < ratio < MAX_RATIO:  # 번호판 인식 조건설정
            d['idx'] = cnt
            cnt += 1
            area = (d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h'])  # 컨투어박스 좌표
            average_w += d['w']
            average_h += d['h']
            first_course_contours.append(d)

    if len(first_course_contours) == 0:
        print(f"유효한 컨투어를 찾을 수 없습니다: {img_path}")
        continue

    average_w = math.floor(average_w / len(first_course_contours))
    average_h = math.floor(average_h / len(first_course_contours))

    first_course_contours = sorted(first_course_contours, key=lambda x: x['x'])  # 내림차순

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    for d in first_course_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=1)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')
    plt.title("Filtered Contours")
    plt.show()

    for d in first_course_contours:
        if average_w - 8 < d['w'] < average_w + 8 and average_h - 3 < d['h'] < average_h + 5:
            middle_course_contours.append(d)

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    for d in middle_course_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=1)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')
    plt.title("Final Contours")
    plt.show()

    sorted_contours = sorted(middle_course_contours, key=lambda x: x['x'])  # 내림차순
    middle_course_contours.clear()
    f = []
    cnt = 0
    check = 0

    for d in sorted_contours:
        if cnt == 0:
            middle_course_contours.append(d)
            check = 0
        elif cnt == 2:
            if d['x'] - (f['x'] + f['w']) > f['w']:
                if f['x'] + f['w'] < d['x']:
                    middle_course_contours.append(d)
                    check = 0
        else:
            if f['x'] + f['w'] < d['x']:
                middle_course_contours.append(d)
                check = 0
        cnt += 1
        f = d.copy()
        check = 0

    cnt = 0
    img_back = Image.new("RGB", (13 * average_w, 2 * average_h), (256, 256, 256))  # 배경화면

    for d in middle_course_contours:
        if cnt != 1:
            area = (d['x'], d['y'] - 2, d['x'] + d['w'], d['y'] + d['h'] + 3)  # 컨투어박스 좌표
            cropped = img_for_crop.crop(area)  # 다시 불러온 이미지 crop
            img_back.paste(cropped, area)  # 붙이기
            possible_contours.append(d)
            cnt += 1
        else:  # 한글 직전 부분
            area = (d['x'], d['y'] - 2, d['x'] + d['w'], d['y'] + d['h'] + 3)  # 컨투어박스 좌표
            dx = d['x']
            dw = d['w']
            cropped = img_for_crop.crop(area)  # 다시 불러온 이미지 crop
            img_back.paste(cropped, area)  # 붙이기
            possible_contours.append(d)
            #  ----------- 한번더 넣기 --------------
            area = (d['x'] + d['w'], d['y'] - 2, d['x'] + d['w'] + d['w'] + 9, d['y'] + d['h'] + 3)
            cropped = img_for_crop.crop(area)  # 다시 불러온 이미지 crop
            img_back.paste(cropped, area)  # 붙이기
            possible_contours.append(d)
            cnt += 1

    img_back = img_back.resize((26 * average_w, 4 * average_h))  # 크기늘리기
    save_img_path = os.path.join(save_path, 'plate_%d.jpg' % image_files.index(file_name))
    img_back.save(save_img_path, format='PNG')  # 이미지 저장
    img_back.show()

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=1)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')
    plt.title("Final Possible Contours")
    plt.show()

    result = pytesseract.image_to_string(img_back, lang="kor")
    print(result)

    data = result + "\n"
    with open(os.path.join(save_path, 'car_number.txt'), mode='a') as f:
        f.write(data)

    plt.show()