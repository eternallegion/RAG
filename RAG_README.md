# 🔍 Wikipedia RAG QA System

Wikipedia 문서를 기반으로 FAISS 벡터 인덱스를 구축하고, Gemma-2b-it 모델을 이용해 질의응답을 수행하는 RAG(Retrieval-Augmented Generation) 파이프라인입니다.

---

## 📁 파일 구조

```
.
├── noise_embedding.py        # 노이즈 Wikipedia 문서 임베딩 생성
├── clean_embedding.py        # 클린 Wikipedia 문서 임베딩 생성
├── noise_indexing.ipynb      # 노이즈 임베딩 → FAISS IVF 인덱스 빌드
├── clean_indexing.ipynb      # 클린 임베딩 → FAISS IVF 인덱스 빌드
├── rag_v3.py                 # RAG 추론 (질문 → 검색 → 답변 생성)
├── post_processing.py        # 답변 후처리 및 제출 파일 생성
└── check_submission.ipynb    # 제출 파일 검증 및 최종 정리
```

---

## ⚙️ 전체 파이프라인

```
Wikipedia 문서 (노이즈 / 클린)
        ↓
[1단계] 임베딩 생성
  noise_embedding.py / clean_embedding.py
        ↓  
[2단계] FAISS 인덱스 빌드
  noise_indexing.ipynb / clean_indexing.ipynb
        ↓
[3단계] RAG 추론
  rag_v3.py
        ↓
[4단계] 후처리
  post_processing.py
        ↓
[5단계] 제출 파일 검증
  check_submission.ipynb
```

---

## 🗂️ 데이터 경로 (기본값)

코드 내 경로는 아래와 같이 설정되어 있습니다. 환경에 맞게 수정이 필요합니다.

| 파일 | 경로 |
|------|------|
| 노이즈 Wikipedia 텍스트 | `/data/shared/nlp/merged_file_noise.txt` |
| 클린 Wikipedia 텍스트 | `/data/shared/nlp/merged_file_clean.txt` |
| 노이즈 임베딩 (HDF5) | `/data/shared/nlp/noise_embeddings.h5` |
| 클린 임베딩 (HDF5) | `/data/shared/nlp/clean_embeddings.h5` |
| 노이즈 FAISS 인덱스 | `/data/shared/nlp/noise_index.ivf` |
| 클린 FAISS 인덱스 | `/data/shared/nlp/clean_index.ivf` |
| QA 질문 파일 | `./result/qa_test.json` |
| RAG 출력 결과 | `./output.csv` |
| 최종 제출 파일 | `./submission.csv` |

---

## 🔧 환경 요구사항

### 하드웨어
- CUDA 지원 GPU (VRAM 16GB 이상 권장, Gemma-2b-it 로딩 기준)
- 충분한 RAM (대용량 임베딩 파일 처리용, 32GB 이상 권장)

### 소프트웨어
- Python 3.10+
- CUDA Toolkit 11.x 이상

---

## 💿 설치 방법

```bash
pip install -r requirements.txt
```

> PyTorch는 CUDA 버전에 맞게 별도 설치를 권장합니다:
> ```bash
> # CUDA 11.8
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> # CUDA 12.1
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🚀 실행 방법

### 1단계: 임베딩 생성

노이즈 데이터와 클린 데이터 각각에 대해 `all-MiniLM-L6-v2` 모델로 임베딩을 생성하고 HDF5 파일로 저장합니다.

```bash
# 노이즈 Wikipedia 임베딩 생성
python noise_embedding.py

# 클린 Wikipedia 임베딩 생성
python clean_embedding.py
```

**주요 설정값:**

| 항목 | 노이즈 | 클린 |
|------|--------|------|
| 배치 크기 | 65,536 | 131,072 |
| 임베딩 모델 | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` |
| 임베딩 차원 | 384 | 384 |
| GPU | `cuda:0` | `cuda:0` |

> 노이즈 데이터는 전처리(`clean_text`) 후 임베딩됩니다.

---

### 2단계: FAISS IVF 인덱스 빌드

생성된 HDF5 임베딩 파일을 청크 단위로 로드하여 FAISS IVF(Inverted File) 인덱스를 구축합니다.

```bash
jupyter notebook noise_indexing.ipynb
jupyter notebook clean_indexing.ipynb
```

**인덱스 설정:**

| 항목 | 값 |
|------|------|
| 인덱스 타입 | `IndexIVFFlat` |
| 유사도 메트릭 | 코사인 유사도 (Inner Product) |
| 클러스터 수 (`nlist`) | 1,000 |
| 청크 크기 | 100,000 |

---

### 3단계: RAG 추론

질문 파일(`qa_test.json`)을 읽어 FAISS로 관련 문서를 검색하고, Gemma-2b-it 모델로 답변을 생성합니다.

```bash
python rag_v3.py
```

**결과:** `output.csv` (컬럼: `id`, `sentences`, `queries`)

**주요 설정값:**

| 항목 | 값 |
|------|------|
| 임베딩 모델 | `sentence-transformers/all-MiniLM-L6-v2` |
| 생성 모델 | `google/gemma-2b-it` |
| 검색 개수 | 노이즈 5개, 클린 3개 |
| 유사도 임계값 | 0.7 이상만 context로 사용 |
| 최대 토큰 길이 | 1,024 |
| 데이터 타입 | `bfloat16` |

**QA 질문 파일 형식 (`qa_test.json`):**

```json
[
  {"question": "Who is the president of France?"},
  {"question": "When was the Eiffel Tower built?"}
]
```

---

### 4단계: 후처리

생성된 답변을 정제하고 제출 포맷에 맞게 변환합니다.

```bash
python post_processing.py
```

**처리 내용:**
- `*`, `"`, `_` 특수문자 제거
- 개행 이후 텍스트 잘라내기
- "The answer is" 제거
- 문장 끝 마침표 제거
- `Yes/No/None` → 소문자 통일 (`yes/no`)
- 빈 셀 → `no` 처리
- 중복 ID, null 값, 다중 문장 여부 검증

**결과:** `submission.csv`

---

### 5단계: 제출 파일 최종 검증

```bash
jupyter notebook check_submission.ipynb
```

**처리 내용:**
- `(unknown)`, `Unknown`, `none` → `no` 치환
- 괄호 문자 제거
- "so I cannot answer this question" 이후 텍스트 제거
- null 값 최종 확인
- `submission2.csv` 및 `sentences.txt` 저장

---

## 📊 모델 정보

| 모델 | 용도 | 출처 |
|------|------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | 문서/질문 임베딩 | Hugging Face |
| `google/gemma-2b-it` | 답변 생성 | Hugging Face |

> `google/gemma-2b-it` 사용 시 Hugging Face 계정 로그인 및 모델 접근 권한 동의가 필요합니다.
> ```bash
> huggingface-cli login
> ```

---

## ⚠️ 주의사항

- **임베딩 모델 일치**: `noise_embedding.py`, `clean_embedding.py`, `rag_v3.py` 모두 동일한 임베딩 모델(`all-MiniLM-L6-v2`)을 사용해야 합니다.
- **파일 경로 수정**: 코드 내 하드코딩된 `/data/shared/nlp/` 경로를 실제 환경에 맞게 수정하세요.
- **GPU 메모리**: Gemma-2b-it 모델은 bfloat16 기준 약 5~6GB VRAM이 필요합니다.
- **인덱스 빌드 순서**: 반드시 임베딩 생성 후 인덱스 빌드를 실행해야 합니다.
