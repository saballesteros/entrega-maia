"""
Fine-tune BAAI/bge-small-en-v1.5 with hard negatives.

Technique (state of the art for dense retrieval):
  - Loss: MultipleNegativesRankingLoss (InfoNCE-style)
    Ref: Karpukhin et al. DPR (2020), Wang et al. BGE (2023)
  - Hard negatives from consultas_centro_control.json (hard_negative_doc_id)
  - BGE instruction prefix for query encoding
  - Best-matching chunk per doc (token overlap) instead of always chunk[0]
  - InformationRetrievalEvaluator on 20% hold-out to drive save_best_model

Output: models/encoder/
"""
import json
import random
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_hf_token

ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
CHUNKS_FILE = "chunks.json"
CONSULTAS_FILE = "consultas_centro_control.json"
OUTPUT_DIR = "models/encoder"
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
EVAL_RATIO = 0.20
# BGE instruction prefix (required for bge-small-en-v1.5)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_doc_to_chunks(chunks_file: str) -> dict:
    with open(chunks_file) as f:
        chunks = json.load(f)
    doc_to_chunks = {}
    for c in chunks:
        doc_to_chunks.setdefault(c["doc_id"], []).append(c["text"])
    return doc_to_chunks


def best_chunk_for_query(query: str, chunks: list[str]) -> str:
    """Return the chunk with highest token overlap with the query."""
    query_tokens = set(query.lower().split())
    best_idx, best_score = 0, -1
    for i, chunk in enumerate(chunks):
        score = len(query_tokens & set(chunk.lower().split()))
        if score > best_score:
            best_score, best_idx = score, i
    return chunks[best_idx]


def build_examples(consultas_file: str, doc_to_chunks: dict, train_items: list) -> list:
    examples = []
    for item in train_items:
        query = QUERY_PREFIX + item["query"]
        pos_id = item["doc_id"]
        neg_id = item.get("hard_negative_doc_id")

        if pos_id not in doc_to_chunks:
            continue

        pos_text = best_chunk_for_query(item["query"], doc_to_chunks[pos_id])

        if neg_id and neg_id in doc_to_chunks:
            neg_text = best_chunk_for_query(item["query"], doc_to_chunks[neg_id])
            examples.append(InputExample(texts=[query, pos_text, neg_text]))
        else:
            examples.append(InputExample(texts=[query, pos_text]))

    return examples


def build_ir_evaluator(eval_items: list, doc_to_chunks: dict) -> InformationRetrievalEvaluator:
    """Build an IR evaluator from held-out consultas and all document chunks."""
    queries, corpus, relevant_docs = {}, {}, {}

    for i, item in enumerate(eval_items):
        qid = str(i)
        queries[qid] = QUERY_PREFIX + item["query"]
        relevant_docs[qid] = set()

        pos_id = item["doc_id"]
        if pos_id not in doc_to_chunks:
            continue
        for chunk_text in doc_to_chunks[pos_id]:
            cid = f"{pos_id}_{len(corpus)}"
            corpus[cid] = chunk_text
            relevant_docs[qid].add(cid)

    # Add all chunks from non-relevant docs as distractors
    eval_doc_ids = {item["doc_id"] for item in eval_items}
    for doc_id, chunks in doc_to_chunks.items():
        if doc_id in eval_doc_ids:
            continue
        for chunk_text in chunks:
            corpus[f"{doc_id}_{len(corpus)}"] = chunk_text

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="ir_eval",
        show_progress_bar=False,
    )


def finetune_encoder(
    chunks_file: str = CHUNKS_FILE,
    consultas_file: str = CONSULTAS_FILE,
    output_dir: str = OUTPUT_DIR,
):
    load_hf_token()
    print("Loading chunks...")
    doc_to_chunks = load_doc_to_chunks(chunks_file)

    with open(consultas_file) as f:
        all_consultas = json.load(f)

    random.seed(42)
    shuffled = all_consultas[:]
    random.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * EVAL_RATIO))
    eval_items  = shuffled[:n_eval]
    train_items = shuffled[n_eval:]
    print(f"Consultas — train: {len(train_items)} | eval: {len(eval_items)}")

    print("Building training examples...")
    examples = build_examples(consultas_file, doc_to_chunks, train_items)
    print(f"  {len(examples)} examples ({sum(1 for e in examples if len(e.texts)==3)} with hard negatives)")

    print("Building IR evaluator...")
    evaluator = build_ir_evaluator(eval_items, doc_to_chunks)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(ENCODER_MODEL)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(loader) * EPOCHS * 0.1)
    model.fit(
        train_objectives=[(loader, loss_fn)],
        evaluator=evaluator,
        epochs=EPOCHS,
        evaluation_steps=len(loader),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LR},
        output_path=output_dir,
        show_progress_bar=True,
        save_best_model=True,
    )
    print(f"Encoder saved → {output_dir}")
    return model


if __name__ == "__main__":
    finetune_encoder()
