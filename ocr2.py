import cv2
import numpy as np
import pytesseract
import os
import csv

# Tesseract 경로 설정 (Windows에서만 필요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# TESSDATA_PREFIX 설정
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# 이미지 경로 및 결과 저장 경로 설정
input_image_path = r'D:\license\img1'  # 입력 이미지가 저장된 경로
save_image_path = 'D:/license/results/images'  # 중간 결과 이미지를 저장할 경로
save_text_path = 'D:/license/results/text'  # 인식된 텍스트를 저장할 경로

# 저장 경로가 존재하지 않으면 생성
os.makedirs(save_image_path, exist_ok=True)
os.makedirs(save_text_path, exist_ok=True)

# CSV 파일 생성 및 헤더 작성
csv_file_path = os.path.join(save_text_path, 'recognized_text.csv')
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image Number', 'Recognized Text'])

# 이미지 불러오기 및 처리
for images in range(1, 21):
    img_path = os.path.join(input_image_path, f"{images}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {img_path}")
        continue

    height, width, channel = img.shape

    # RGB를 Gray로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 처리
    blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    # 이미지 구별을 쉽게 하기 위해 Thresholding (검정색 또는 흰색으로 이미지를 변경)
    img_thresh = cv2.adaptiveThreshold(
        blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    # findContours 함수를 사용하여 canny 이미지에 대해 contours들을 검출
    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_TREE,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1,
                     color=(255, 255, 255))

    # cv2.boundingRect()를 통해 윤곽선을 감싸는 사각형을 구하고, contours_dict에 append한다.
    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h),
                      color=(255, 255, 255), thickness=2)

        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    # Contours 추려내기
    MIN_AREA = 80  # 번호판 윤곽선 최소 범위
    MIN_WIDTH, MIN_HEIGHT = 2, 8  # 최소 너비 높이
    MIN_RATIO, MAX_RATIO = 0.25, 1.0  # 최소 비율 범위 지정

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                      color=(255, 255, 255), thickness=2)

    # Contours 추려내기 2
    MAX_DIAG_MULTIPLYER = 5  # 대각선 길이
    MAX_ANGLE_DIFF = 12.0    # 각도 차이
    MAX_AREA_DIFF = 0.5      # 면적 차이
    MAX_WIDTH_DIFF = 0.8     # 너비 차이
    MAX_HEIGHT_DIFF = 0.2    # 높이 차이
    MIN_N_MATCHED = 3        # 최소 일치 컨투어 수

    def find_chars(contour_list):
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                # 대각선 길이
                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                # np.linalg.norm(a - b) 벡터 a와 벡터 b 사이의 거리를 구한다.
                # 삼각함수 사용
                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])  # 면적의 비율
                width_diff = abs(d1['w'] - d2['w']) / d1['w']  # 너비의 비율
                height_diff = abs(d1['h'] - d2['h']) / d1['h']  # 높이의 비율

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            # 번호판 후보군의 윤곽선 개수가 3보다 작으면 번호판일 확률이 낮다. 이유는 한국 번호판은 총 7자리이기 때문이다.
            matched_contours_idx.append(d1['idx'])

            # 윤곽선 개수가 3보다 작을 때는 continue를 통해 제외
            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            # for문 속에서 조건을 충족하는 경우 최종 후보군 리스트에 append
            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']),
                          pt2=(d['x'] + d['w'], d['y'] + d['h']),
                          color=(255, 255, 255), thickness=2)

    # 이미지 회전 + 영역 추출
    PLATE_WIDTH_PADDING = 1.3
    PLATE_HEIGHT_PADDING = 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        # 센터 좌표 구하기
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        # 번호판의 기울어진 각도를 구하기 (삼각함수 이용)
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        # cv2.getRotationMatrix2D() 로테이션매트릭스를 구한다.
        # cv2.warpAffine() 이미지를 변형한다.
        # cv2.getRectSubPix() 회전된 이미지에서 원하는 부분만을 잘라낸다.
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    if len(plate_imgs) > 0:
        img_cropped = plate_imgs[0]  # 첫 번째 번호판 이미지 사용

        # 결과 이미지 저장
        save_img_filename = os.path.join(save_image_path, f'plate_{images}.png')
        cv2.imwrite(save_img_filename, img_cropped)

        # pytesseract를 사용하여 텍스트 인식
        text = pytesseract.image_to_string(img_cropped, lang='kor', config=r'--psm 6 --oem 3')
        print(images, text, end='')

        # 인식된 텍스트를 CSV 파일에 저장
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([images, text])
    else:
        print(f"번호판을 찾을 수 없습니다: {img_path}")