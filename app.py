import ollama
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_NAME = "jhgan/ko-sroberta-multitask"
tokenizer = None
model = None
device = None

def load_model():
  global tokenizer, model, device
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = AutoModel.from_pretrained(MODEL_NAME)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)
  model.eval()
  print(f"Model loaded on {device}")

def get_query_embedding(query_text):
  formatted_text = query_text

  inputs = tokenizer(formatted_text, return_tensors="pt",
    padding=True, truncation=True, max_length=512)
  inputs = {k: v.to(device) for k, v in inputs.items()}

  with torch.no_grad():
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    embedding = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

  emb = embedding.cpu().numpy()[0]
  emb = emb/(np.linalg.norm(emb) + 1e-12)
  return emb.tolist()

@app.route('/embed', methods=['POST'])
def embed_query():
  data = request.get_json()
  query = data.get('query', '')

  if not query:
    return jsonify({'error': 'No query provided'}), 400

  try:
    embedding = get_query_embedding(query)
    return jsonify({'embedding': embedding})
  except Exception as e:
    return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_answer():
  data = request.get_json()
  query = data.get('query', '')
  context_items = data.get('context', [])

  if not query:
    return jsonify({'error': 'No query provided'}), 400

  if not context_items:
    return jsonify({'error': 'No context provided'}), 400

  try:
    context_text = "\n\n".join([
      f"[{i+1}] {item['label']}: {item['text']}"
      for i, item in enumerate(context_items)
    ])

    prompt = f"당신은 타이포그래피 전문가입니다. 아래의 참고 자료를 바탕으로 사용자의 질문에 답변해주세요.\n\n참고 자료:\n{context_text}\n\n질문: {query}\n\n답변 (참고 자료를 기반으로 명확하고 간결하게 답변해주세요):"

    response = ollama.chat(
      model='exaone3.5:7.8b',
      messages=[{'role': 'user', 'content': prompt}]
    )

    answer = response['message']['content']
    return jsonify({'answer': answer})

  except Exception as e:
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
  load_model()
  app.run(host='0.0.0.0', port=8000, debug=True)
