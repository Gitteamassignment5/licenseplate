import cv2
import numpy as np
import pytesseract
from PIL import Image
import math
import matplotlib.pyplot as plt

for m in range(0, 6):
    img = cv2.imread('car_%d.jpg' % m)
    img_for_crop = Image.open("car_%d.jpg" % m)
    height, width, channel = img.shape

    # 1. Grayscale 변환
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur 적용
    img_blur = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)

    # 3. Adaptive Thresholding
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 4. Contours 찾기
    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    # 5. 번호판으로 가능한 Contours 필터링
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    possible_contours = []
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            possible_contours.append(d)

    # 6. 번호판 후보 영역 찾기
    def find_possible_plate(contours, width, height):
        possible_plates = []
        for c in contours:
            for d in contours:
                if c is d:
                    continue
                if abs(c['cx'] - d['cx']) < width and abs(c['cy'] - d['cy']) < height:
                    possible_plates.append(c)
                    possible_plates.append(d)
        return possible_plates

    plates = find_possible_plate(possible_contours, width//7, height//7)
    if plates:
        plates = sorted(plates, key=lambda x: x['x'])

        # 7. 번호판 영역 crop
        plate_imgs = []
        for p in plates:
            x, y, w, h = p['x'], p['y'], p['w'], p['h']
            plate_img = img_for_crop.crop((x, y, x+w, y+h))
            plate_imgs.append(plate_img)

        # 8. 번호판 영역 OCR 수행
        for idx, plate_img in enumerate(plate_imgs):
            plate_img = plate_img.resize((plate_img.width * 2, plate_img.height * 2))
            plate_img = plate_img.convert('L')
            plate_img_np = np.array(plate_img)
            _, plate_img_thresh = cv2.threshold(plate_img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plate_img = Image.fromarray(plate_img_thresh)

            # OCR 수행
            result = pytesseract.image_to_string(plate_img, lang="kor+eng", config='--psm 7')
            result = result.strip()
            print(f"Plate {m}-{idx}: {result}")

            # 결과를 텍스트 파일에 저장
            data = f"Image {m} - Plate {idx}: {result}\n"
            with open('car_number.txt', mode='a') as f:
                f.write(data)

            # 번호판 이미지 저장
            plate_img.save(f'plate_{m}_{idx}.jpg')

    plt.show()