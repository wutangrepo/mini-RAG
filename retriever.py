import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # MUST be the same model

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
embeddings = np.load("embeddings.npy")   # shape: (n_chunks, 384)
print(f"Loaded {len(chunks)} chunks, embeddings shape {embeddings.shape}\n")

# ---------- The entire "search engine" ----------
def search(query, k=3):
    """
    Embed the query, compare it to every chunk, return the k best.
    All vectors are unit-length, so dot product == cosine similarity.
    """
    query_vec = model.encode([query], normalize_embeddings=True)[0]  # (384,)
    scores = embeddings @ query_vec       # (n_chunks,) — one similarity per chunk
    top = np.argsort(scores)[::-1][:k]    # indices of the k highest scores, best first
    return [(chunks[i], float(scores[i])) for i in top]

# ---------- Baseline for comparison: naive keyword matching ----------
def keyword_score(query, chunk):
    """Fraction of query words that appear *literally* in the chunk."""
    q = set(query.lower().split())
    c = set(chunk.lower().split())
    return len(q & c) / len(q) if q else 0.0

# ---------- Interactive loop ----------
while True:
    try:
        query = input("\nSearch (Enter to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not query:
        break

    print(f"\nQuery: {query!r}")
    for rank, (chunk, score) in enumerate(search(query, k=3), 1):
        kw = keyword_score(query, chunk)
        preview = chunk.replace("\n", " ")
        preview = preview[:200] + ("..." if len(preview) > 200 else "")
        print(f"  #{rank}  semantic={score:.3f}  keywords={kw:.2f}")
        print(f"      {preview}")

print("\nBye!")