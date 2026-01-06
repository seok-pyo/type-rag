import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# 수정 1: 임베딩 전용 모델로 교체 (가볍고 안정적)
MODEL_NAME = "jhgan/ko-sroberta-multitask"

def load_model():
  print(f"Loading model: {MODEL_NAME}")
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = AutoModel.from_pretrained(MODEL_NAME)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)
  model.eval()  # evaluation 모드 (추론 전용)
  print(f"Model loaded on {device}")

  return tokenizer, model, device

def get_embedding(text, tokenizer, model, device):
  # ko-sroberta는 prefix 불필요
  formatted_text = text

  inputs = tokenizer(
    formatted_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
  )

  inputs = {k: v.to(device) for k, v in inputs.items()}

  with torch.no_grad():
    outputs = model(**inputs)

    # 기존 방식 (CLS pooling) 제거
    # embedding = outputs.last_hidden_state[:, 0, :]

    # 수정 3: mean pooling + attention mask (E5 권장 방식)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    embedding = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

  emb = embedding.cpu().numpy()[0]
  emb = emb / (np.linalg.norm(emb) + 1e-12)

  return emb

def load_graph_data(file_path):
  with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  return data

def create_node_text(node):
  parts = []

  if node.get('label'):
    parts.append(f"Label: {node['label']}는 다음 의미이다.")
  if node.get('subtitle_kr'):
    parts.append(f"한글: {node['subtitle_kr']}")
  if node.get('subtitle_hanja'):
    parts.append(f"한자: {node['subtitle_hanja']}")
  if node.get('subtitle_en'):
    parts.append(f"English: {node['subtitle_en']}")
  if node.get('meaning'):
    parts.append(f"Meaning: {node['meaning']}")
  if node.get('refer'):
    parts.append(f"Reference: {node['refer']}")
  if node.get('body'):
    parts.append(f"Body: {node['body']}")

  return '. '.join(parts)

def generate_all_embeddings(input_file, output_file):
  tokenizer, model, device = load_model()

  print(f"\nLoading graph data from {input_file}")
  graph_data = load_graph_data(input_file)
  nodes = graph_data['nodes']
  print(f"Found {len(nodes)} nodes")

  print("\nGenerating embeddings...")
  embeddings_data = []

  for i, node in enumerate(nodes):
    text = create_node_text(node)
    embedding = get_embedding(text, tokenizer, model, device)

    embeddings_data.append({
      'id': node['id'],
      'label': node['label'],
      'text': text,
      'embedding': embedding.tolist()
    })

    if (i + 1) % 10 == 0:
      print(f"Processed {i + 1} / {len(nodes)} nodes")

    # print(f"\nSaving embeddings to {output_file}")
  with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

  print("Done")

if __name__ == "__main__":
  generate_all_embeddings("graph_data.json", "embeddings.json")
