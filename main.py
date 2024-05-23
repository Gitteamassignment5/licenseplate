import korean
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import subprocess
import pandas as pd
import datetime

# 각 모듈 임포트 (실제 모듈 이름으로 대체해야 함)
import A_preprocessing_module as A  # YOLO 전처리 모듈
import B_yolo_model_module as B     # YOLO 모델 모듈
import C_ocr_module as C            # OCR 모듈
import D_tts_module as D            # TTS 모듈

def koreanCleaner(text):
    return "".join(korean.tokenize(text))

def displayWaveImage(path):
    x, sr = librosa.load(path)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.show()

def get_day():
    weeks = ['월', '화', '수', '목', '금', '토', '일']
    today = datetime.datetime.today().weekday()
    return weeks[today]

def check_new_entries(file_path):
    # CSV 파일을 읽고 새로운 항목을 확인
    df = pd.read_csv(file_path)
    new_entries = df.iloc[-1]  # 마지막 행을 새로운 항목으로 가정
    return new_entries

def generate_message(car_number, day):
    # 메시지 생성
    return f"{car_number} 번호판 차량은 {day}요일에는 주차를 하시는 날이 아닙니다."

def main():
    file_path = "f.csv"
    
    # 새로운 항목 확인
    new_entry = check_new_entries(file_path)
    car_number = new_entry['차량번호']  # '차량번호' 열에 차량 번호가 있다고 가정
    
    # 현재 요일 가져오기
    day = get_day()
    
    # 메시지 생성
    input_text = generate_message(car_number, day)
    inputText = koreanCleaner(input_text)

    # YOLO 전처리 모듈 사용 예시
    preprocessed_image = A.preprocess("input_image.jpg")
    
    # YOLO 모델 모듈 사용 예시
    yolo_results = B.detect_objects(preprocessed_image)
    
    # OCR 모듈 사용 예시
    ocr_results = C.recognize_text("input_image_with_text.jpg")
    
    # TTS 모듈 사용 예시
    ttsPath = "model/tts/glowtts/coqui_tts-December-08-2021_03+15PM-0000000"
    subprocess.run([
        "tts",
        "--text", inputText,
        "--model_path", f"{ttsPath}/checkpoint_190000.pth.tar",
        "--config_path", f"{ttsPath}/config.json",
        "--out_path", "output.wav"
    ])

    # 음성 파일 시각화 및 재생 (TTS 실행 후 사용)
    displayWaveImage("output.wav")
    ipd.display(ipd.Audio("output.wav"))

if __name__ == "__main__":
    main()