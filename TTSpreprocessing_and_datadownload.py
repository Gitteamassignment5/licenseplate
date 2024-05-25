import os
from glob import glob
import torchaudio
from pathlib import Path
from tqdm import tqdm
import csv
import torch
import zipfile
import tarfile
import requests
import subprocess

# 설정 변수
target_rate = 22050
download_dir = "download_data/"
save_dir = "resample1/"
zeroth_dir = "zeroth_korean/"
tedxhr_dir = "pansori_tedxkr/"
kss_dir = "kss/"

data_dir_list = [zeroth_dir, tedxhr_dir, kss_dir]

for i in [download_dir, save_dir]:
    for j in data_dir_list:
        os.makedirs(i + j, exist_ok=True)

# 리샘플링 함수 정의
def saveResampleWav(wavList, save_dir, working_dir, target_rate):
    for itemPath in tqdm(wavList):
        savePath = os.path.join(save_dir, working_dir, "wavs", Path(itemPath).stem + ".wav")
        speech_array, sample_rate = torchaudio.load(itemPath)
        speech_array = torchaudio.functional.resample(speech_array, sample_rate, target_rate)
        speech_array = torch.unsqueeze(torch.mean(speech_array, axis=0), dim=0)
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        torchaudio.save(savePath, speech_array, target_rate)

# CSV 생성 함수 정의
def saveCsv(textList, save_dir, working_dir, csvReadLine):
    with open(os.path.join(save_dir, working_dir, "metadata.csv"), "w", encoding="utf-8") as file:
        wr = csv.writer(file, delimiter='|')
        for textPath in textList:
            with open(textPath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                file_name, text, speaker_name = csvReadLine(line, textPath)
                if not os.path.exists(os.path.join(save_dir, working_dir, file_name)):
                    continue
                wr.writerow([file_name, text, text, speaker_name])

# Kaggle 데이터셋 다운로드 및 압축 해제
os.environ['KAGGLE_USERNAME'] = "your_kaggle_username"
os.environ['KAGGLE_KEY'] = "your_kaggle_key"

subprocess.run(['kaggle', 'datasets', 'download', '-d', 'bryanpark/korean-single-speaker-speech-dataset'])

# 다운로드된 파일 압축 해제
with zipfile.ZipFile('korean-single-speaker-speech-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall(os.path.join(download_dir, kss_dir))

# requests를 사용하여 파일 다운로드
def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong")

# Zeroth와 TEDxKR 데이터셋 다운로드 및 압축 해제
zeroth_url = 'https://www.openslr.org/resources/40/zeroth_korean.tar.gz'
tedxhr_url = 'https://www.openslr.org/resources/58/pansori-tedxkr-corpus-1.0.tar.gz'

download_file(zeroth_url, 'zeroth_korean.tar.gz')
download_file(tedxhr_url, 'pansori-tedxkr-corpus-1.0.tar.gz')

with tarfile.open('zeroth_korean.tar.gz', 'r:gz') as tar_ref:
    tar_ref.extractall(os.path.join(download_dir, zeroth_dir))

with tarfile.open('pansori-tedxkr-corpus-1.0.tar.gz', 'r:gz') as tar_ref:
    tar_ref.extractall(os.path.join(download_dir, tedxhr_dir))

# TEDxKR 전처리
textList = glob(download_dir + tedxhr_dir + "**/*.txt", recursive=True)
wavList = glob(download_dir + tedxhr_dir + "**/*.flac", recursive=True)
working_dir = tedxhr_dir

def readLine_tedxhr(line, textPath):
    lineSplit = line.split(' ', 1)
    file_name = os.path.join("wavs", Path(lineSplit[0]).stem + ".wav")
    text = lineSplit[1].strip()
    speaker_name = "tedxhr_" + Path(textPath).parent.name
    return [file_name, text, speaker_name]

saveResampleWav(wavList, save_dir, working_dir, target_rate)
saveCsv(textList, save_dir, working_dir, readLine_tedxhr)

# Zeroth 전처리
textList = glob(download_dir + zeroth_dir + "**/*.txt", recursive=True)
wavList = glob(download_dir + zeroth_dir + "**/*.flac", recursive=True)
working_dir = zeroth_dir

def readLine_zeroth(line, textPath):
    lineSplit = line.split(' ', 1)
    file_name = os.path.join("wavs", Path(lineSplit[0]).stem + ".wav")
    text = lineSplit[1].strip()
    speaker_name = "zeroth_" + Path(textPath).parent.name
    return [file_name, text, speaker_name]

saveResampleWav(wavList, save_dir, working_dir, target_rate)
saveCsv(textList, save_dir, working_dir, readLine_zeroth)

# KSS 전처리
textList = glob(os.path.join(download_dir, kss_dir, "**/*.txt"), recursive=True)
wavList = glob(os.path.join(download_dir, kss_dir, "**/*.wav"), recursive=True)
working_dir = kss_dir

def readLine_kss(line, textPath):
    lineSplit = line.split('|')
    file_name = os.path.join("wavs", Path(lineSplit[0]).stem + ".wav")
    text = lineSplit[3].strip()
    speaker_name = "kss"
    return [file_name, text, speaker_name]

saveResampleWav(wavList, save_dir, working_dir, target_rate)
saveCsv(textList, save_dir, working_dir, readLine_kss)