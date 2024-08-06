import re
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm
import h5py


# 문장 약간 깔끔하게 전처리
def clean_text(text):
     # 1. 숫자와 글자 사이의 불필요한 문자 제거
    text = re.sub(r'(\d)[^\s\w]+(\w)', r'\1\2', text)    
    # 2. 특수문자 제거 (괄호, 큰따옴표, 작은따옴표, 마침표, 쉼표를 제외)
    text = re.sub(r'[^\w\s()"\'.,]', '', text)
    # 3. 여러 줄의 공백을 하나의 공백으로 대체하되, 줄바꿈 문자는 유지
    text = re.sub(r'(\n\s*\n)+', r'\n\n', text)
    return text

# 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.to("cuda:0")

# 파일 경로
file_path = '/data/shared/nlp/merged_file_noise.txt'
output_path = '/data/shared/nlp/noise_embeddings.h5'

# 배치 크기 설정
batch_size = 65536
print('batch_size: ', batch_size)

# 진행 상황을 표시할 tqdm progress bar 생성
total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))

with tqdm(total=total_lines, desc="Generating embeddings") as pbar:
    with open(file_path, 'r', encoding='utf-8') as file, h5py.File(output_path, 'w') as hf:
        embeddings_dataset = hf.create_dataset('noise_embeddings', (total_lines, 384), dtype=np.float32)
        batch = []
        batch_start_idx = 0
        for line in file:
            clean_line = clean_text(line.strip())
            batch.append(clean_line)
            if len(batch) == batch_size:
                embeddings = model.encode(batch, convert_to_numpy=True)
                embeddings_dataset[batch_start_idx:batch_start_idx + batch_size] = embeddings
                batch_start_idx += batch_size
                batch = []
                pbar.update(batch_size)
        if batch:
            embeddings = model.encode(batch, convert_to_numpy=True)
            embeddings_dataset[batch_start_idx:batch_start_idx + len(batch)] = embeddings
            pbar.update(len(batch))

        

