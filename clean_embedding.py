import re
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm
import h5py


# 모델 로드
print("load embedding model----------")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.to("cuda:0")

# 파일 경로
file_path = '/data/shared/nlp/merged_file_clean.txt'
output_path = '/data/shared/nlp/clean_embeddings.h5'

# 배치 크기 설정
batch_size = 131072
print('batch_size: ', batch_size)

# 진행 상황을 표시할 tqdm progress bar 생성
total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))

with tqdm(total=total_lines, desc="Generating embeddings") as pbar:
    with open(file_path, 'r', encoding='utf-8') as file, h5py.File(output_path, 'w') as hf:
        embeddings_dataset = hf.create_dataset('clean_embeddings', (total_lines, 384), dtype=np.float32)
        batch = []
        batch_start_idx = 0
        for line in file:
            batch.append(line)
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