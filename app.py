from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_NAME = "intfloat/e5-mistral-7b-instruct"
tokenizer = None
model = None
device = None

def load_model():
  global tokenizer, model, device
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = AutoModel.from_pretrained(MODEL_NAME)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)
  print(f"Model loaded on {device}")

def get_query_embedding(query_text):
  formatted_text = f"query: {query_text}"

  inputs = tokenizer(formatted_text, return_tensors="pt",
    padding=True, truncation=True, max_length=512)
  inputs = {k: v.to(device) for k, v in inputs.items()}

  with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

  return embedding[0].tolist()

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

if __name__ == '__main__':
  load_model()
  app.run(host='0.0.0.0', port = 5000, debug=True)

