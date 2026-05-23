"""
Chunk MASA technical documents using a token-level sliding window.

Strategy (state of the art for technical docs):
  - Window: 256 tokens, Overlap: 64 tokens (25% overlap)
  - Preserves document boundaries; each chunk tagged with doc_id + chunk_id
  - Output: chunks.json  (list of {doc_id, chunk_id, text})
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_hf_token

from transformers import AutoTokenizer

ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 256
OVERLAP_TOKENS = 64
DOCS_DIR = "base_conocimiento"
OUTPUT = "chunks.json"


def load_documents(docs_dir: str) -> dict:
    docs = {}
    for doc_dir in sorted(Path(docs_dir).iterdir()):
        if not doc_dir.is_dir():
            continue
        md_file = doc_dir / "doc.md"
        if md_file.exists():
            docs[doc_dir.name] = md_file.read_text(encoding="utf-8")
    return docs


def chunk_text(text: str, tokenizer, chunk_size: int, overlap: int) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        chunks.append(chunk_text.strip())
        if end >= len(token_ids):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks(docs_dir: str = DOCS_DIR, output: str = OUTPUT) -> list:
    load_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    # Disable the "sequence longer than 512" warning: the limit applies to model
    # inference, not tokenization. We encode full docs only to split them into
    # ≤256-token chunks, so each chunk fed to the model is always within limit.
    tokenizer.model_max_length = int(1e9)
    docs = load_documents(docs_dir)
    print(f"Loaded {len(docs)} documents")

    all_chunks = []
    for doc_id, text in docs.items():
        text_chunks = chunk_text(text, tokenizer, CHUNK_TOKENS, OVERLAP_TOKENS)
        for chunk_id, chunk_text_str in enumerate(text_chunks):
            if chunk_text_str:
                all_chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text_str,
                })

    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {len(all_chunks)} chunks → {output}")
    return all_chunks


if __name__ == "__main__":
    build_chunks()
