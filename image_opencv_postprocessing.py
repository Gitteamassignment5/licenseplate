import cv2
import os
import imutils
import matplotlib.pyplot as plt

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

def image_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur_img(gray_img):
    return cv2.GaussianBlur(gray_img, (7, 7), 0)

def invert_color(grayscale_img):
    return cv2.bitwise_not(grayscale_img)

def binarize_img(inverted_img):
    _, binary = cv2.threshold(inverted_img, 100, 255, cv2.THRESH_BINARY)
    return binary

def dilate_image(binary_image):
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel3)
    return thre_mor

def bilateral_filter(gray_img):
    return cv2.bilateralFilter(gray_img, 11, 17, 17)

def canny_edge_detection(filtered_img):
    return cv2.Canny(filtered_img, 170, 200)

def find_contours(edged_image, original_img):
    contours, hierarchy = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]  # 상위 30개의 외곽선만 선택

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return original_img, contours

def detect_number_plate(contours, img):
    NumberPlateCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # 꼭짓점이 4개인 외곽선 선택
            NumberPlateCnt = approx  # 번호판 외곽선 저장
            x, y, w, h = cv2.boundingRect(c)
            ROI = img[y:y + h, x:x + w]  # 번호판 영역 추출
            break
    return NumberPlateCnt, ROI if NumberPlateCnt is not None else None

def draw_number_plate(image, NumberPlateCnt):
    if NumberPlateCnt is not None:
        cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    return image

# 메인 실행 코드
image_path = 'path_to_your_image.jpg'  # 이미지 경로 설정
image = read_image(image_path)

if image is not None:
    image = imutils.resize(image, width=500)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지 전처리 및 시각화
    gray = image_to_grayscale(image)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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

    processed_img, contours = find_contours(dilated, image.copy())
    ax[1, 2].imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    ax[1, 2].set_title('Contours on Image')

    fig.tight_layout()
    plt.show()

    # 번호판 탐지
    NumberPlateCnt, ROI = detect_number_plate(contours, img_rgb)

    if ROI is not None:
        plt.figure(figsize=(10, 7))
        plt.imshow(ROI)
        plt.title('Detected Number Plate Region')
        plt.show()

    # 번호판 외곽선 그리기 및 시각화
    final_img = draw_number_plate(image, NumberPlateCnt)
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Number Plate')
    plt.show()