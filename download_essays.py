"""Download Paul Graham essays from HuggingFace for needle-in-haystack benchmark."""
from datasets import load_dataset
import os

# Download the dataset
print("Downloading Paul Graham essays dataset from HuggingFace...")
dataset = load_dataset("chromadb/paul_graham_essay", split="data")

# Concatenate all essays into one large corpus
print(f"Loaded {len(dataset)} entries")
corpus_parts = []

for item in dataset:
    # The dataset has 'document' field containing essay content
    if 'document' in item and item['document']:
        corpus_parts.append(item['document'])

corpus = "\n\n".join(corpus_parts)

# Save to file
output_path = "data/paul_graham/essays.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(corpus)

print(f"Saved corpus to {output_path}")
print(f"Total corpus length: {len(corpus)} characters")
print(f"Approximate tokens: {len(corpus.split())}")
