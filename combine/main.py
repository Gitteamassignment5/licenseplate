import hgtk
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import subprocess
import pandas as pd
import datetime
import os

# 각 모듈 임포트 (실제 모듈 이름으로 대체해야 함)
import image_opencv_postprocessing_v2 as A  # YOLO 전처리 모듈
import ocr_processing as C  # OCR 모듈
import vehicle_classification as VC  # vehicle_classification 모듈

def koreanCleaner(text):
    return hgtk.text.compose(hgtk.text.decompose(text))

def displayWaveImage(data, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def get_day():
    weeks = ['월', '화', '수', '목', '금', '토', '일']
    today = datetime.datetime.today().weekday()
    return weeks[today]

def check_new_entries(file_path):
    # CSV 파일을 읽고 새로운 항목을 확인
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    new_entries = df.iloc[-1]  # 마지막 행을 새로운 항목으로 가정
    return new_entries

def generate_message(car_number, day):
    # 메시지 생성
    return f"{car_number} 번호판 차량은 {day}요일에는 주차를 하시는 날이 아닙니다."

def test_imports():
    try:
        # Korean 모듈 테스트
        test_text = "안녕하세요"
        cleaned_text = koreanCleaner(test_text)
        print(f"Korean Cleaner Output: {cleaned_text}")
        
        # librosa와 matplotlib 테스트
        # 가상 오디오 데이터를 생성하여 테스트
        sr = 22050  # 샘플링 레이트
        t = 5.0  # 신호의 지속 시간 (초)
        x = librosa.tone(frequency=440.0, sr=sr, length=int(sr * t))
        displayWaveImage(x, sr)
        print("Librosa and Matplotlib: Wave image displayed.")
        
        # pandas 테스트
        test_csv_path = r"D:\license\results\text\results.csv"  # 절대 경로로 변경
        new_entry = check_new_entries(test_csv_path)
        print(f"Pandas: New entry - {new_entry}")
        
        # datetime 테스트
        current_day = get_day()
        print(f"Datetime: Current day is {current_day}")
        
        # 커스텀 모듈 테스트
        # YOLO 전처리 모듈 테스트
        # preprocessed_image = A.preprocess("D:/license/img1")  # 절대 경로로 변경
        # print(f"Preprocessing Module: Preprocessed image - {preprocessed_image}")
        
        # YOLO 모델 모듈 테스트
        # yolo_results = B.detect_objects(preprocessed_image)
        # print(f"YOLO Model Module: YOLO results - {yolo_results}")
        
        # OCR 모듈 테스트
        # ocr_results = C.preprocessing("D:license/img1")  # 절대 경로로 변경
        # print(f"OCR Module: OCR results - {ocr_results}")
        
        # TTS 모듈 테스트
        # 아래 주석을 해제하고 유효한 경로를 제공하여 TTS 테스트
        # ttsPath = "D:/weekpark_test/model/tts/glowtts/coqui_tts-December-08-2021_03+15PM-0000000"  # 절대 경로로 변경
        # subprocess.run([
        #     "tts",
        #     "--text", cleaned_text,
        #     "--model_path", f"{ttsPath}/checkpoint_190000.pth.tar",
        #     "--config_path", f"{ttsPath}/config.json",
        #     "--out_path", "D:/weekpark_test/output.wav"  # 절대 경로로 변경
        # ])
        # displayWaveImage("D:/weekpark_test/output.wav")  # 절대 경로로 변경
        # ipd.display(ipd.Audio("D:/weekpark_test/output.wav"))  # 절대 경로로 변경
        # print("TTS Module: TTS output generated and displayed.")
        
        print("모든 임포트와 함수 호출이 성공적으로 완료되었습니다.")
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # vehicle_classification_main 함수 실행
    image_paths = [r"D:\license\img1\1.jpg"]  # 업로드된 이미지 파일 경로
    csv_output_path = r"D:\license\results\text\results.csv"  # 결과 CSV 파일 경로 설정

    print(f"이미지 파일 경로: {image_paths}")
    print(f"CSV 파일 경로: {csv_output_path}")

    VC.main(image_paths, csv_output_path)
