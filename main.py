# 필요한 라이브러리 임포트
import korean
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import subprocess

# 각 모듈 임포트 (실제 모듈 이름으로 대체해야 함)
import A_preprocessing_module as A
import B_yolo_model_module as B
import C_ocr_module as C
import D_tts_module as D

def koreanCleaner(text):
    return "".join(korean.tokenize(text))

def displayWaveImage(path):
    x, sr = librosa.load(path)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.show()

def main():
    # 입력 텍스트와 경로 설정
    input_text = "신은 우리의 수학 문제에는 관심이 없다. 신은 다만 경험적으로 통합할 뿐이다."
    inputText = koreanCleaner(input_text)

    # 전처리 모듈 사용 예시
    preprocessed_text = A.preprocess(inputText)
    
    # YOLO 모델 모듈 사용 예시
    yolo_results = B.detect_objects("input_image.jpg")
    
    # OCR 모듈 사용 예시
    ocr_results = C.recognize_text("input_image_with_text.jpg")
    
    # TTS 모듈 사용 예시 (주석 처리)
    # ttsPath = "model/tts/glowtts/coqui_tts-December-08-2021_03+15PM-0000000"
    # subprocess.run([
    #     "tts",
    #     "--text", preprocessed_text,
    #     "--model_path", f"{ttsPath}/checkpoint_190000.pth.tar",
    #     "--config_path", f"{ttsPath}/config.json",
    #     "--out_path", "output.wav"
    # ])

    # 음성 파일 시각화 및 재생 (TTS 실행 후 사용)
    # displayWaveImage("output.wav")
    # ipd.display(ipd.Audio("output.wav"))

main()