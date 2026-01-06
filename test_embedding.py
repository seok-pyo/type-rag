import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

MODEL_NAME = "intfloat/e5-mistral-7b-instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # evaluation 모드
print(f"Model loaded on {device}")
print(f"Model config: {model.config.model_type}")

def get_embedding(text):
    formatted_text = f"passage: {text}"
    inputs = tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return embedding[0]

# 테스트
text1 = "Label: word. English: word"
text2 = "Label: typography. Meaning: 타이포그래피"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)

print(f"\nText 1: {text1}")
print(f"Embedding 1 (first 10): {emb1[:10]}")

print(f"\nText 2: {text2}")
print(f"Embedding 2 (first 10): {emb2[:10]}")

print(f"\nAre they equal? {np.array_equal(emb1, emb2)}")
print(f"Cosine similarity: {np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))}")
