import re
import json

# ---------- Approach 1: fixed-size chunks (simplest possible) ----------
def chunk_fixed_size(text, chunk_size=500, overlap=50):
    """Cut text every `chunk_size` characters, sharing `overlap` characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# ---------- Approach 2: sentence-aware chunks (better for RAG) ----------
def split_sentences(text):
    # Split after ".", "!", or "?" followed by whitespace
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def chunk_by_sentences(text, max_chars=500, overlap_sentences=1):
    """
    Group sentences into chunks up to `max_chars` characters.
    `overlap_sentences` sentences are shared between neighboring chunks.
    """
    sentences = split_sentences(text)
    chunks, current = [], []
    length = 0

    for sentence in sentences:
        # If adding this sentence would exceed the limit, close the chunk
        if current and length + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            # Keep the last N sentences as overlap for the next chunk
            current = current[-overlap_sentences:] if overlap_sentences else []
            length = sum(len(s) + 1 for s in current)
        current.append(sentence)
        length += len(sentence) + 1  # +1 for the space

    if current:
        chunks.append(" ".join(current))
    return chunks

# ---------- Run it ----------
if __name__ == "__main__":
    with open("sample.txt", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_by_sentences(text, max_chars=500, overlap_sentences=1)

    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"=== Chunk {i} ({len(chunk)} chars) ===")
        print(chunk, "\n")

    # Save for the next step of your RAG pipeline (embedding)
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Saved to chunks.json")