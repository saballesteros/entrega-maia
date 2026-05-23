"""
Fine-tune meta-llama/Llama-3.2-1B-Instruct with QLoRA for tool calling.

Key design decisions (backed by recent literature):
  - System prompt includes ALL valid enum values so the 1B model can 'copy'
    them from context rather than guess. Critical for exact match on small LLMs.
    Ref: Gorilla (Patil 2023), ToolACE (Liu 2024), Small LLMs for FC (2024)
  - ResponseOnlyCollator does TEXT-LEVEL splitting at the assistant boundary
    (not token-level pattern search), which is guaranteed to find the boundary.
  - QLoRA 4-bit NF4 + LoRA on all projections. Ref: Dettmers et al. 2023
  - Loss computed ONLY on the tool call tokens (response-only SFT).

Output: models/decoder/  (LoRA adapter checkpoint)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_hf_token, build_tools_prompt, load_tools_def, enrich_tools_def

import faiss
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

DECODER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ENCODER_DIR    = "models/encoder"
RETRIEVAL_INDEX = "retrieval_index.json"
TRAIN_CSV       = "train.csv"
TOOLS_DEF_FILE  = "tools_definition.json"
OUTPUT_DIR      = "models/decoder"
TOP_K = 3

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 1024
BATCH_SIZE   = 4
GRAD_ACCUM   = 8
EPOCHS       = 5          # increased from 3
LR           = 2e-4

# The response starts immediately after this header in Llama-3.2 chat format
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class ResponseOnlyCollator:
    """
    Tokenize examples and mask everything BEFORE the assistant response with
    label=-100 so the loss is computed only on the tool call.

    Uses TEXT-LEVEL splitting (not token-level pattern search) so the boundary
    is always found correctly regardless of tokenizer behavior.
    Left-pads so that truncation never cuts off the response.
    """

    def __init__(self, tokenizer, response_template: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_template = response_template

    def _encode_example(self, text: str):
        idx = text.rfind(self.response_template)
        if idx == -1:
            # Template not found — mask everything (exclude from loss)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = ids[-self.max_length :]
            return ids, [-100] * len(ids)

        boundary = idx + len(self.response_template)
        prefix = text[:boundary]
        suffix = text[boundary:]

        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

        # If too long, truncate the prefix (keep the full response in loss)
        total = len(prefix_ids) + len(suffix_ids)
        if total > self.max_length:
            keep_prefix = self.max_length - len(suffix_ids)
            prefix_ids = prefix_ids[-keep_prefix:] if keep_prefix > 0 else []

        input_ids = prefix_ids + suffix_ids
        labels    = [-100] * len(prefix_ids) + list(suffix_ids)
        return input_ids, labels

    def __call__(self, examples):
        all_ids, all_labels = [], []
        for ex in examples:
            ids, labs = self._encode_example(ex["text"])
            all_ids.append(ids)
            all_labels.append(labs)

        # Left-pad to the longest sequence in the batch
        max_len = max(len(ids) for ids in all_ids)
        pad_id  = self.tokenizer.pad_token_id

        padded_ids, padded_labels, masks = [], [], []
        for ids, labs in zip(all_ids, all_labels):
            pad = max_len - len(ids)
            padded_ids.append([pad_id] * pad + ids)
            padded_labels.append([-100] * pad + labs)
            masks.append([0] * pad + [1] * len(ids))

        return {
            "input_ids":      torch.tensor(padded_ids,     dtype=torch.long),
            "attention_mask": torch.tensor(masks,          dtype=torch.long),
            "labels":         torch.tensor(padded_labels,  dtype=torch.long),
        }


def load_retrieval_index(index_file: str):
    with open(index_file) as f:
        data = json.load(f)
    vectors = np.array([item["vector"] for item in data], dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, data


def encode_queries(queries: list, encoder_dir: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(encoder_dir)
    prefixed = [QUERY_PREFIX + q for q in queries]
    embs = model.encode(prefixed, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    del model
    return embs.astype(np.float32)


def build_bm25_index(chunks: list) -> BM25Okapi:
    corpus = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(corpus)


def retrieve_hybrid(
    query: str,
    query_emb: np.ndarray,
    faiss_index,
    chunks: list,
    bm25: BM25Okapi,
    k: int = TOP_K,
    rrf_k: int = 60,
) -> list[str]:
    n_candidates = k * 3

    _, I_dense = faiss_index.search(query_emb.reshape(1, -1), n_candidates)
    dense_ranks = {int(I_dense[0][i]): i for i in range(len(I_dense[0]))}

    tokens = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokens))
    bm25_top = np.argsort(bm25_scores)[::-1][:n_candidates]
    bm25_ranks = {int(idx): rank for rank, idx in enumerate(bm25_top)}

    all_ids = set(dense_ranks) | set(bm25_ranks)
    rrf_scores = {}
    for idx in all_ids:
        score = 0.0
        if idx in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[idx] + 1)
        if idx in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[idx] + 1)
        rrf_scores[idx] = score

    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [chunks[i]["text"] for i in top_ids if i < len(chunks)]


def format_chat(query, context_chunks, tool_call, tokenizer, system_prompt):
    context = "\n\n---\n\n".join(context_chunks)
    user_content = f"Technical Documentation:\n{context}\n\nOperator Query: {query}"
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": tool_call},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def clean_train_data(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    before = len(df)
    df = df.drop_duplicates(subset=["query", "tool_call"])
    # For same query with conflicting labels, keep the most frequent label
    df = (
        df.groupby("query")["tool_call"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    df.columns = ["query", "tool_call"]
    # Drop rows where tool_call is NaN or empty
    df = df[df["tool_call"].notna() & (df["tool_call"].str.strip() != "")]
    print(f"Clean train: {len(df)} rows (from {before})")
    return df.reset_index(drop=True)


def finetune_decoder(
    train_csv: str = TRAIN_CSV,
    synthetic_csv: str = None,
    encoder_dir: str = ENCODER_DIR,
    retrieval_index: str = RETRIEVAL_INDEX,
    tools_def_file: str = TOOLS_DEF_FILE,
    output_dir: str = OUTPUT_DIR,
):
    load_hf_token()

    # Build system prompt dynamically with ALL valid enum values (critical for 1B model)
    # Enrich tools_def with all values observed in training data — the JSON file only
    # lists ~3 example values per parameter, but training has the full set.
    tools_def = load_tools_def(tools_def_file)
    enrich_tools_def(tools_def, train_csv)
    system_prompt = build_tools_prompt(tools_def)
    print("System prompt preview (first 300 chars):")
    print(system_prompt[:300])

    print("\nCleaning training data...")
    df = clean_train_data(train_csv)

    # Merge synthetic data if provided
    if synthetic_csv and Path(synthetic_csv).exists():
        df_synth = pd.read_csv(synthetic_csv)
        df_synth = df_synth[df_synth["tool_call"].notna() & (df_synth["tool_call"].str.strip() != "")]
        df = pd.concat([df, df_synth], ignore_index=True).drop_duplicates(subset=["query"])
        print(f"Merged {len(df_synth)} synthetic rows → total {len(df)} training examples")

    print("Loading retrieval index...")
    faiss_idx, chunks = load_retrieval_index(retrieval_index)

    print("Building BM25 index for hybrid retrieval...")
    bm25_idx = build_bm25_index(chunks)

    print("Encoding training queries...")
    query_embs = encode_queries(df["query"].tolist(), encoder_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Verify the response template appears in a formatted example (debug check)
    _sample = format_chat("test query", ["doc"], "no_action", tokenizer, system_prompt)
    assert RESPONSE_TEMPLATE in _sample, (
        f"RESPONSE_TEMPLATE not found in chat output!\nTemplate: {repr(RESPONSE_TEMPLATE)}\n"
        f"Sample (last 200): {_sample[-200:]}"
    )
    print(f"Response template found at char {_sample.rfind(RESPONSE_TEMPLATE)} ✓")

    print("Building training dataset with hybrid-retrieved context...")
    texts = []
    for idx, (_, row) in enumerate(df.iterrows()):
        ctx  = retrieve_hybrid(row["query"], query_embs[idx], faiss_idx, chunks, bm25_idx, TOP_K)
        text = format_chat(row["query"], ctx, row["tool_call"], tokenizer, system_prompt)
        texts.append(text)
    dataset = Dataset.from_dict({"text": texts})
    print(f"Dataset: {len(dataset)} examples")

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Loading decoder model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = ResponseOnlyCollator(tokenizer, RESPONSE_TEMPLATE, MAX_SEQ_LEN)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        args=training_args,
    )

    print("Fine-tuning decoder...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Decoder saved → {output_dir}")


if __name__ == "__main__":
    finetune_decoder()
