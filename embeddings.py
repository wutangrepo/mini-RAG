import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Load the model
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

def cosine_similarity(vec1, vec2):
    """1.0 = same direction (similar meaning), 0.0 = unrelated."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

demo = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Quantum computers process data with qubits.",
]

# Generate embeddings for the demo sentences
demo_embeddings = model.encode(demo)

# Calculate similarities
for i in range(len(demo_embeddings)):
    for j in range(i + 1, len(demo_embeddings)):
        sim = cosine_similarity(demo_embeddings[i], demo_embeddings[j])
        print(f"Similarity between '{demo[i]}' and '{demo[j]}': {sim:.2f}")

# Embed chunks from a JSON file
try:
    with open("chunks.json", "r") as f:
        chunks = json.load(f)
except FileNotFoundError:
    print("chunks.json file not found. Please ensure the file exists.")
    chunks = []

print(f"Embedding {len(chunks)} chunks...")

# Generate embeddings for the chunks
chunk_embeddings = model.encode(
    chunks,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True, # turning to dot product
)

print(f"\nEach chunk is now {chunk_embeddings.shape[1]} numbers.")
print(f"First 5 of them for chunk 0: {chunk_embeddings[0][:5]}")

# save for retrival
np.save("chunk_embeddings.npy", chunk_embeddings)