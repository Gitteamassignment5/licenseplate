import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import csv

# YOLO 모델 설정
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# CSV 파일 열기
with open('car_number.csv', mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # CSV 파일에 헤더 작성
    csvwriter.writerow(['Image', 'Plate', 'Result', 'Numbers', 'Letters'])

    for m in range(0, 6):
        img = cv2.imread(f'car_{m}.jpg')
        height, width, channels = img.shape

        # YOLO로 객체 검출
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # 번호판 검출 및 OCR 수행
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                plate_img = img[y:y+h, x:x+w]
                plate_img_pil = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
                plate_img_pil = plate_img_pil.resize((plate_img_pil.width * 2, plate_img_pil.height * 2))
                plate_img_pil = plate_img_pil.convert('L')
                plate_img_np = np.array(plate_img_pil)
                _, plate_img_thresh = cv2.threshold(plate_img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plate_img_pil = Image.fromarray(plate_img_thresh)

                # OCR 수행
                result = pytesseract.image_to_string(plate_img_pil, lang="kor+eng", config='--psm 7')
                result = result.strip()
                print(f"Plate {m}-{i}: {result}")

                # 결과를 CSV 파일에 저장
                numbers = ''.join(filter(str.isdigit, result))
                letters = ''.join(filter(str.isalpha, result))
                csvwriter.writerow([f"Image {m}", f"Plate {i}", result, numbers, letters])

                # 번호판 이미지 저장
                plate_img_pil.save(f'plate_{m}_{i}.jpg')

                # 원본 이미지에 번호판 위치 사각형 그리기
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                print(f"Numbers: {numbers}")
                print(f"Letters: {letters}")

            # 번호판 위치 표시된 원본 이미지 저장
            cv2.imwrite(f'car_with_plate_{m}.jpg', img)

        plt.show()