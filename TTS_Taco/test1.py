import os
import argparse
import torch
import numpy as np
import pandas as pd
from jamo import hangul_to_jamo
from models.tacotron import Tacotron
from util.text import text_to_sequence, sequence_to_text, number_to_hangul
from util.plot_alignment import plot_alignment
from util.hparams import *
from datetime import datetime

# 폴더 생성
checkpoint_dir = './checkpoint/1'
save_dir = './output'
os.makedirs(save_dir, exist_ok=True)

# 요일을 한글로 변환하는 함수
def get_korean_day_of_week():
    days = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    today = datetime.today().weekday()  # 0 = 월요일, 6 = 일요일
    return days[today]

def inference(text, idx, model):
    seq = text_to_sequence(text)
    enc_input = torch.tensor(seq, dtype=torch.int64).unsqueeze(0)
    sequence_length = torch.tensor([len(seq)], dtype=torch.int32)
    dec_input = torch.from_numpy(np.zeros((1, mel_dim), dtype=np.float32))
    
    pred, alignment = model(enc_input, sequence_length, dec_input, is_training=False, mode='inference')
    pred = pred.squeeze().detach().numpy()
    alignment = np.squeeze(alignment.detach().numpy(), axis=0)

    np.save(os.path.join(save_dir, 'mel-{}'.format(idx)), pred, allow_pickle=False)

    input_seq = sequence_to_text(seq)
    alignment_dir = os.path.join(save_dir, 'align-{}.png'.format(idx))
    plot_alignment(alignment, alignment_dir, input_seq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True, help='Path to the checkpoint file')
    args = parser.parse_args()
    
    # 모델 로드
    model = Tacotron(K=16, conv_dim=[128, 128])
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model'])

    # CSV 파일에서 차량 번호 읽기
    file_path = '../results/text/results.csv'
    data = pd.read_csv(file_path)

    # '차량 번호' 컬럼 이름을 사용하여 데이터 읽기
    vehicle_numbers = data['차량 번호']  # CSV 파일의 차량 번호 컬럼 이름

    # 현재 요일 가져오기
    day_of_week = get_korean_day_of_week()

    # 문장 생성 및 음성 합성
    for i, number in enumerate(vehicle_numbers):
        hangul_number = number_to_hangul(str(number))
        sentence = f'{hangul_number}번님 {day_of_week}에는 주차하는 날이 아닙니다'
        jamo_sentence = ''.join(list(hangul_to_jamo(sentence)))
        inference(jamo_sentence, i, model)