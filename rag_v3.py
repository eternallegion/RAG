import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import linecache


# set target hardware
device = 'cuda'

# mmodel loading
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)   # documents임베딩에 썼던 모델과 동일한 모델 사용

# load index
noise_index = faiss.read_index('/data/shared/nlp/noise_index.ivf', faiss.IO_FLAG_MMAP)
clean_index = faiss.read_index('/data/shared/nlp/clean_index.ivf', faiss.IO_FLAG_MMAP)
print("index file loaded")

# Embedding
def embed_question(question):
    return model.encode([question])[0]

def get_line(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()

# Load JSON file
with open('./result/qa_test.json', 'r') as f:
    qa_pairs = json.load(f)

# Extract question list
queries = [item['question'] for item in qa_pairs]

print("queries are ready!")

# Define Wikipedia data file path
# wikipedia_file_path = './processed_wikipedia.txt'
# file_path = '/data/shared/nlp/processed_wikipedia.txt/processed_wikipedia.txt'
noise_file_path = '/data/shared/nlp/merged_file_noise.txt'
clean_file_path = '/data/shared/nlp/merged_file_clean.txt'

# Simple search function using FAISS
def retrieve_passage(query, noise_k=5, clean_k=3):
    context_list = []
    # query_embedding = model.encode(query).astype('float32')
    # query_embedding = model.encode(query).astype('float16')

    query_embedding = embed_question(query).astype('float32')
    normalized_query = query_embedding / np.linalg.norm(query_embedding)

    noise_D, noise_I = noise_index.search(np.array([normalized_query]), noise_k)
    clean_D, clean_I = clean_index.search(np.array([normalized_query]), clean_k)

    # query와 연관성이 높은 검색결과의 경우만 context로 넣어줌
    if noise_D[0][0] > 0.7:
        for i in range(noise_k):
            context_list.append(get_line(noise_file_path, noise_I[0][0]+i-1).replace('_','(unknown)'))

    if clean_D[0][0] > 0.7:
        for i in range(clean_k):
            context_list.append(get_line(clean_file_path, clean_I[0][0]+i))

    return context_list

# Load QA model and tokenizer
qa_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
qa_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map=device,
    torch_dtype=torch.bfloat16
).to(device)

# Custom dataset
class QADataset(Dataset):
    def __init__(self, queries):
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        return idx, query
    
    # Create dataset and dataloader
dataset = QADataset(queries)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Generate answers
results = []

for batch in tqdm(dataloader):
    indices, batch_queries = batch

    # without RAG
    # batch_input_texts = [
    #     f"Generating an answer in an one word or a sentence without any an explanation\nQuery: {query} Answer:" for query in batch_queries
    # ]

    # Retrieve passages for each query in the batch
    batch_retrieved_passages = [retrieve_passage(query) for query in batch_queries]

    # Initialize an empty list to hold the input texts for the model
    batch_input_texts = []

    # Iterate over the batch queries and their corresponding retrieved passages
    for query, retrieved_passages in zip(batch_queries, batch_retrieved_passages):
        # Concatenate the retrieved passages with newline and index
        combined_passages = ""
        for i, passage in enumerate(retrieved_passages):
            combined_passages += f"{passage}\n"

            # 너무 긴 context는 제한
            if len(combined_passages) > 1023:
                combined_passages = combined_passages[:1023]

        # Define the system prompt
        # Design 1
        # system_prompt = """
        # You are an expert in Wikipedia and proficient in routing user questions to the appropriate contexts provided.
        # The contexts are numbered sentences that may contain noise.
        # Reference these contexts by their numbers to find and summarize the correct answer.
        # Please generate answers in short, A word answer would more prefer than the sentence answer.
        # If you cannot find an answer within the contexts, review them again to ensure accuracy.
        # Only generate an answer after thoroughly reviewing the given contexts.
        # Moreover, you must reference an answer that you generated.
        # You can not answer with this form:
        # For example, "The passage does not specify", "The context dose not provide",
        # "I cannot find this information in the context." or "The correct answer is".
        # Let's think step-by-step.
        # """

        # # Design 2
        # system_prompt = """
        # You are the Wikipedia expert, Answer in one word.
        # Reference given context and summarize a correct answer.
        # """

        # Design 3
        # system_prompt = """
        # You are the Wikipedia expert, Answer in one word.
        # Reference given context and summarize a correct answer.
        # If you can not answer just say no. An answer must be short."
        # """

        # Design 4
        # system_prompt = "If the context does not provide any information, just answer about following query./\n"

        # Design 5
        # system_prompt = 'Prompt: Provide a one-word answer to the query. Some contexts require predicting [MASK]. Provide your own one-word answer when the context does not provide any information. DO NOT say "context". Give a one-word Answer.\n'

        # Design 6
        # system_prompt = """
        # You are a expert of the Wikipedia, generates multiple sub-questions related to an input contexts.\n
        # The goal is to break down input query into a set of sub-problems / sub-questions that can be answers in isolation. \n
        # Generate multiple answers at first with search querie related to and answer in one word.:
        # """

        # Design 7
        system_prompt = """
        You are an expert of question and answering. Answer the question with one word.
        # Response Grunding
        - Warning: Context may not always be provided and some information is masked as (unknown).
        - Guess what is in (unknown).
        - You must answer in short without any explanation.\n"""
        # Create the input text for the current query
        answer_add = 'answer:'
        query_add = ''
        input_text = system_prompt + f'contexts: {combined_passages}\n{query_add}\nQuestion: {query}, {answer_add}'

        print(input_text)
        # Append the input text to the batch input texts list
        batch_input_texts.append(input_text)
   
    # Tokenize the batch input texts
    batch_input_ids = qa_tokenizer(batch_input_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    
    # Ensure input lengths do not exceed 1023 tokens
    for i, input_id in enumerate(batch_input_ids['input_ids']):
        if input_id.size(0) > 1024:
            batch_input_ids['input_ids'][i] = input_id[:1023]
            batch_input_ids['attention_mask'][i] = batch_input_ids['attention_mask'][i][:1023]

    # Generate responses using the model
    outputs = qa_model.generate(input_ids=batch_input_ids['input_ids'],
                                attention_mask=batch_input_ids['attention_mask'],
                                max_length=1024)

    # Decode the generated outputs
    batch_answers = [qa_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Process each query, answer, and index in the batch
    for idx, answer, query in zip(indices, batch_answers, batch_input_texts):
        # Extract the answer text
         # Extract the answer text
        answer = answer.split(answer_add)[1].strip()
        print("***answer:", answer)
        # Append the result to the results list
        results.append([idx.item(), answer, query])


# Save results to CSV
df = pd.DataFrame(results, columns=["id", "sentences", "queries"])
df.to_csv("output.csv", index=False)

print("CSV file has been created successfully.")
