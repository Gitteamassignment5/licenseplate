import hgtk
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import subprocess
import pandas as pd
import datetime

# 각 모듈 임포트 (실제 모듈 이름으로 대체해야 함)
import image_opencv_postprocessing_v2 as A  # YOLO 전처리 모듈
import B_yolo_model_module as B     # YOLO 모델 모듈
import ocr as C            # OCR 모듈
import TTS as D            # TTS 모듈

def koreanCleaner(text):
    return hgtk.text.compose(hgtk.text.decompose(text))

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

def test_imports():
    try:
        # Korean 모듈 테스트
        test_text = "안녕하세요"
        cleaned_text = koreanCleaner(test_text)
        print(f"Korean Cleaner Output: {cleaned_text}")
        
        # librosa와 matplotlib 테스트
        test_audio_path = "test_audio.wav"  # 실제 경로로 대체
        displayWaveImage(test_audio_path)
        print("Librosa and Matplotlib: Wave image displayed.")
        
        # pandas 테스트
        test_csv_path = "f.csv"  # 실제 경로로 대체
        new_entry = check_new_entries(test_csv_path)
        print(f"Pandas: New entry - {new_entry}")
        
        # datetime 테스트
        current_day = get_day()
        print(f"Datetime: Current day is {current_day}")
        
        # 커스텀 모듈 테스트
        # YOLO 전처리 모듈 테스트
        preprocessed_image = A.preprocess("input_image.jpg")
        print(f"Preprocessing Module: Preprocessed image - {preprocessed_image}")
        
        # YOLO 모델 모듈 테스트
        yolo_results = B.detect_objects(preprocessed_image)
        print(f"YOLO Model Module: YOLO results - {yolo_results}")
        
        # OCR 모듈 테스트
        ocr_results = C.recognize_text("input_image_with_text.jpg")
        print(f"OCR Module: OCR results - {ocr_results}")
        
        # TTS 모듈 테스트
        # 아래 주석을 해제하고 유효한 경로를 제공하여 TTS 테스트
        # ttsPath = "model/tts/glowtts/coqui_tts-December-08-2021_03+15PM-0000000"
        # subprocess.run([
        #     "tts",
        #     "--text", cleaned_text,
        #     "--model_path", f"{ttsPath}/checkpoint_190000.pth.tar",
        #     "--config_path", f"{ttsPath}/config.json",
        #     "--out_path", "output.wav"
        # ])
        # displayWaveImage("output.wav")
        # ipd.display(ipd.Audio("output.wav"))
        # print("TTS Module: TTS output generated and displayed.")
        
        print("모든 임포트와 함수 호출이 성공적으로 완료되었습니다.")
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# 테스트 함수 실행
test_imports()