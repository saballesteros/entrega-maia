"""
Utilities: canonical formatting, tool call parsing, and validation.

Canonical format rules (exact match required):
  1. Parameter order as defined in tools_definition.json
  2. No spaces after commas or around =
  3. Single quotes for str values
  4. All lowercase EXCEPT protocol_id (UPPERCASE)
  5. Integers without quotes
"""
import json
import os
import re
from pathlib import Path


def _find_env_file(env_file: str) -> Path:
    """Walk upward from cwd until the .env file is found."""
    path = Path(env_file)
    if path.is_absolute():
        return path
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        candidate = parent / env_file
        if candidate.exists():
            return candidate
    return Path(env_file)  # fallback — let dotenv report the missing file


def load_hf_token(env_file: str = ".env") -> str:
    """Load HF_TOKEN from a .env file and authenticate with Hugging Face Hub."""
    from dotenv import load_dotenv
    from huggingface_hub import login

    path = _find_env_file(env_file)
    load_dotenv(path)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(f"HF_TOKEN not found in {path}")

    login(token=token, add_to_git_credential=False)
    return token


def load_kaggle_token(env_file: str = ".env") -> None:
    """Configure kagglehub auth from .env.

    Priority:
      1. KAGGLE_API_TOKEN — JSON string {"username":..,"key":..} or raw key.
      2. KAGGLE_USERNAME + KAGGLE_KEY — legacy credentials.
    Raises EnvironmentError if neither is available.
    """
    import json as _json
    from dotenv import load_dotenv

    path = _find_env_file(env_file)
    load_dotenv(path)

    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        try:
            data = _json.loads(token)
            os.environ["KAGGLE_USERNAME"] = data["username"]
            os.environ["KAGGLE_KEY"] = data["key"]
        except (_json.JSONDecodeError, KeyError):
            os.environ["KAGGLE_KEY"] = token
        return

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return

    raise EnvironmentError(
        f"No Kaggle credentials found in {path}. "
        "Set KAGGLE_API_TOKEN or KAGGLE_USERNAME + KAGGLE_KEY."
    )


VALID_TOOLS = [
    "get_telemetry", "get_crew_status", "get_module_status", "send_alert",
    "send_message", "schedule_maintenance", "activate_protocol",
    "control_system", "calculate_trajectory", "request_supply", "no_action",
]


def load_tools_def(path="tools_definition.json"):
    """Load tools_definition.json and normalize to {tool_name: {parameters: {...}}} format.

    The competition file uses {"tools": [...], "canonical_format": {...}}, where each
    tool is {"name": ..., "parameters": {...}, "parameter_order": [...], ...}.
    All parameter types are "enum"; we detect int vs str from the value types.
    """
    with open(path) as f:
        raw = json.load(f)

    tools_list = raw["tools"] if "tools" in raw and isinstance(raw["tools"], list) else None
    if tools_list is None:
        return raw  # already dict-keyed — legacy format

    result = {}
    for tool in tools_list:
        name = tool["name"]
        params = tool.get("parameters", {})
        order = tool.get("parameter_order", list(params.keys()))
        ordered_params = {}
        for pname in order:
            if pname not in params:
                continue
            pinfo = dict(params[pname])
            vals = pinfo.get("values", [])
            pinfo["type"] = "int" if vals and all(isinstance(v, int) for v in vals) else "str"
            ordered_params[pname] = pinfo
        result[name] = {"parameters": ordered_params}
    return result


def enrich_tools_def(tools_def: dict, train_csv: str) -> dict:
    """Add all parameter values observed in train.csv to tools_def.

    tools_definition.json only lists ~3 example values per parameter.
    Training data has the full set. The system prompt must include ALL of them
    so the model can copy the correct value from context.
    """
    import pandas as pd
    df = pd.read_csv(train_csv)
    for tc in df["tool_call"].dropna():
        m = re.match(r"^(\w+)\((.*)\)$", tc.strip(), re.DOTALL)
        if not m:
            continue
        tool_name = m.group(1)
        if tool_name not in tools_def:
            continue
        params_str = m.group(2)
        for pm in re.finditer(r"(\w+)='([^']*)'", params_str):
            pname, pval = pm.group(1), pm.group(2)
            if pname in tools_def[tool_name]["parameters"]:
                vals = tools_def[tool_name]["parameters"][pname].setdefault("values", [])
                if pval not in vals:
                    vals.append(pval)
        for pm in re.finditer(r"(\w+)=(\d+)", params_str):
            pname, pval = pm.group(1), int(pm.group(2))
            if pname in tools_def[tool_name]["parameters"]:
                vals = tools_def[tool_name]["parameters"][pname].setdefault("values", [])
                if pval not in vals:
                    vals.append(pval)
    return tools_def


def _parse_params(params_str):
    """Parse key=value pairs from a params string into a dict."""
    params = {}
    # Match key='value', key="value", or key=integer
    for m in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']|(\w+)\s*=\s*(\d+)', params_str):
        if m.group(1):
            params[m.group(1).lower()] = ("str", m.group(2))
        else:
            params[m.group(3).lower()] = ("int", int(m.group(4)))
    return params


def _extract_raw(text):
    """Extract the tool call string from potentially noisy model output."""
    text = text.strip()

    # Case-insensitive search for each known tool name
    for tool in VALID_TOOLS:
        if tool == "no_action":
            continue
        m = re.search(rf'\b{re.escape(tool)}\s*\([^)]*\)', text, re.DOTALL | re.IGNORECASE)
        if m:
            raw = m.group(0)
            # Normalize tool name to lowercase
            raw = tool + raw[len(tool):]
            return raw

    # Check for bare no_action
    if re.search(r'\bno_action\b', text, re.IGNORECASE):
        return "no_action"

    return None


def canonicalize(text: str, tools_def: dict) -> str:
    """Normalize model output to the canonical tool call format."""
    raw = _extract_raw(text)
    if raw is None or raw == "no_action":
        return "no_action"

    # Handle no_action with parens
    if re.match(r'^no_action\s*\(\s*\)$', raw.strip()):
        return "no_action"

    m = re.match(r'^(\w+)\s*\((.*)\)$', raw.strip(), re.DOTALL)
    if not m:
        return "no_action"

    tool_name = m.group(1).strip().lower()
    params_str = m.group(2)

    if tool_name not in tools_def:
        return "no_action"

    params = _parse_params(params_str)
    tool_params = tools_def[tool_name]["parameters"]

    ordered = []
    for param_name, param_info in tool_params.items():
        if param_name not in params:
            continue
        _, val = params[param_name]
        expected_type = param_info["type"]

        if expected_type == "int":
            try:
                ival = int(val)
            except (TypeError, ValueError):
                ival = 1
            ordered.append(f"{param_name}={ival}")
        else:
            sval = str(val).lower()
            if param_name == "protocol_id":
                sval = sval.upper()
            ordered.append(f"{param_name}='{sval}'")

    return f"{tool_name}({','.join(ordered)})"


def validate_tool_call(tc: str, tools_def: dict) -> bool:
    """Return True if the tool call string is already in canonical format."""
    return tc == canonicalize(tc, tools_def)


def build_tools_prompt(tools_def: dict) -> str:
    """Build a system prompt that includes ALL valid enum values for every parameter.

    This is the single most important factor for a 1B LLM to achieve high exact match:
    the model can 'copy' the correct enum values from context instead of guessing.
    Ref: Gorilla (Patil et al. 2023), ToolACE (Liu et al. 2024),
         Small LLMs for Function Calling (2024).
    """
    header = (
        "You are the AI controller for MASA's Kuntur Station space station (400 km above Earth).\n"
        "Given a control center operator query and retrieved technical documentation, "
        "output EXACTLY ONE tool call.\n"
        "Output ONLY the tool call string. No explanation. No other text.\n"
        "\n"
        "CANONICAL FORMAT RULES — any violation = wrong answer (0 points):\n"
        "  1. Parameter order must match the definition below exactly\n"
        "  2. No spaces after commas or around the = sign\n"
        "  3. Single quotes for string values:  module='jaguar'\n"
        "  4. ALL lowercase EXCEPT protocol_id (always UPPERCASE):  protocol_id='MASA-SEC-007'\n"
        "  5. Integer values WITHOUT quotes:  timeframe_hours=1  not  timeframe_hours='1'\n"
        "  6. Use no_action for purely informational or historical queries\n"
        "\n"
        "AVAILABLE TOOLS — pick the value that best matches the query:\n"
    )

    tool_lines = []
    for tool_name, tool_info in tools_def.items():
        if tool_name == "no_action":
            continue
        params = tool_info.get("parameters", {})
        if not params:
            tool_lines.append(f"  {tool_name}()")
            continue
        parts = []
        for pname, pinfo in params.items():
            vals  = pinfo.get("values", [])
            ptype = pinfo.get("type", "str")
            if vals:
                if ptype == "int":
                    opts = "|".join(str(v) for v in vals)
                else:
                    opts = "|".join(f"'{v}'" for v in vals)
                parts.append(f"{pname}={opts}")
            else:
                parts.append(pname)
        tool_lines.append(f"  {tool_name}({','.join(parts)})")

    tool_lines.append("  no_action   (no parameters — purely informational or historical query)")

    footer = (
        "\n"
        "OUTPUT FORMAT EXAMPLES:\n"
        "  get_telemetry(module='jaguar',metric='oxygen',timeframe_hours=12)\n"
        "  send_alert(module='quetzal',severity='critical',reason='radiation_spike')\n"
        "  activate_protocol(protocol_id='MASA-SEC-007',scope='station_wide')\n"
        "  no_action\n"
        "\n"
        "INVALID (wrong parameter order — evaluates as incorrect):\n"
        "  send_alert(severity='critical',module='quetzal',reason='radiation_spike')\n"
    )

    return header + "\n".join(tool_lines) + footer


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def evaluate_retrieval_recall(
    consultas_file: str,
    retrieval_index_file: str,
    encoder_path: str,
    ks: list = None,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
    batch_size: int = 64,
) -> dict:
    """Compute Recall@k for the encoder over consultas_centro_control.json.

    A query is counted as correct at @k if the ground-truth doc_id appears
    in the doc_ids of the top-k retrieved chunks.

    Returns {k: recall_float, ..., "per_query": [(query, gold_doc, hit@1, hit@k_max), ...]}
    """
    import json
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    if ks is None:
        ks = [1, 3, 5, 10]

    with open(consultas_file) as f:
        consultas = json.load(f)

    with open(retrieval_index_file) as f:
        index_data = json.load(f)

    vectors = np.array([item["vector"] for item in index_data], dtype=np.float32)
    faiss.normalize_L2(vectors)
    faiss_idx = faiss.IndexFlatIP(vectors.shape[1])
    faiss_idx.add(vectors)
    chunk_doc_ids = [item["doc_id"] for item in index_data]

    encoder = SentenceTransformer(encoder_path)
    raw_queries = [c["query"] for c in consultas]
    prefixed    = [query_prefix + q for q in raw_queries]
    embs = encoder.encode(prefixed, batch_size=batch_size,
                          normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    del encoder

    max_k = max(ks)
    _, I = faiss_idx.search(embs, max_k)

    hits = {k: 0 for k in ks}
    per_query = []
    for q_idx, item in enumerate(consultas):
        gold = item["doc_id"]
        retrieved = [chunk_doc_ids[i] for i in I[q_idx] if i < len(chunk_doc_ids)]
        row = {"query": raw_queries[q_idx], "gold_doc": gold}
        for k in ks:
            hit = gold in retrieved[:k]
            row[f"hit@{k}"] = hit
            if hit:
                hits[k] += 1
        per_query.append(row)

    n = len(consultas)
    result = {k: hits[k] / n for k in ks}
    result["per_query"] = per_query
    result["n"] = n
    return result


def plot_retrieval_recall(results_by_label: dict, figsize=(9, 4)):
    """Bar + line chart comparing Recall@k across encoder variants.

    results_by_label: {label_str: evaluate_retrieval_recall output, ...}
    e.g. {"Base BGE": base_results, "Fine-tuned BGE": ft_results}
    """
    import matplotlib.pyplot as plt

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=figsize)

    for (label, res), color in zip(results_by_label.items(), colors):
        ks = sorted(k for k in res if isinstance(k, int))
        vals = [res[k] * 100 for k in ks]
        ax.plot(ks, vals, marker="o", label=label, color=color, linewidth=2)
        for k, v in zip(ks, vals):
            ax.annotate(f"{v:.1f}%", (k, v),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=color)

    ks_all = sorted(k for k in next(iter(results_by_label.values())) if isinstance(k, int))
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k (%)")
    ax.set_title("Recall@k del Encoder sobre consultas_centro_control.json", fontweight="bold")
    ax.set_xticks(ks_all)
    ax.set_ylim(0, 108)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_tool_distribution(train_df, combined_df=None, figsize=(13, 6)):
    """Horizontal bar chart of tool-name frequencies.

    If combined_df is provided, shows original vs. combined side-by-side so the
    effect of data augmentation is immediately visible.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    def _counts(df):
        return df["tool_call"].str.extract(r"^(\w+)")[0].value_counts().sort_values()

    orig = _counts(train_df)

    if combined_df is not None:
        comb = _counts(combined_df).reindex(orig.index, fill_value=0)
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        axes[0].barh(orig.index, orig.values, color="#4C72B0")
        axes[0].set_title("Distribución original (train.csv)", fontweight="bold")
        axes[0].set_xlabel("Ejemplos")
        for v, name in zip(orig.values, orig.index):
            axes[0].text(v + 5, name, str(v), va="center", fontsize=8)

        axes[1].barh(comb.index, comb.values, color="#55A868")
        axes[1].set_title("Distribución con datos sintéticos", fontweight="bold")
        axes[1].set_xlabel("Ejemplos")
        for v, name in zip(comb.values, comb.index):
            axes[1].text(v + 5, name, str(v), va="center", fontsize=8)

        fig.suptitle("Distribución de herramientas antes y después de data augmentation",
                     fontsize=13, fontweight="bold", y=1.01)
    else:
        fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
        ax.barh(orig.index, orig.values, color="#4C72B0")
        ax.set_title("Distribución de herramientas — train.csv", fontweight="bold")
        ax.set_xlabel("Ejemplos")
        for v, name in zip(orig.values, orig.index):
            ax.text(v + 3, name, str(v), va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_encoder_metrics(encoder_dir="models/encoder", figsize=(9, 4)):
    """Plot IR evaluator recall@k curves saved by sentence-transformers."""
    import os, glob
    import matplotlib.pyplot as plt
    import pandas as pd

    pattern = os.path.join(encoder_dir, "eval", "*results.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"[plot_encoder_metrics] No eval CSV found in {encoder_dir}/eval/")
        return

    df = pd.read_csv(files[0])

    recall_cols = [c for c in df.columns if "Recall" in c or "recall" in c]
    if not recall_cols:
        print(f"[plot_encoder_metrics] Columns found: {list(df.columns)}")
        return

    x_col = "epoch" if "epoch" in df.columns else "steps"

    fig, ax = plt.subplots(figsize=figsize)
    for col in recall_cols:
        label = col.split("-")[-1] if "-" in col else col
        ax.plot(df[x_col], df[col], marker="o", label=label)

    ax.set_xlabel("Epoch" if x_col == "epoch" else "Paso")
    ax.set_ylabel("Recall")
    ax.set_title("Encoder — Recall@k en conjunto de evaluación IR", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decoder_loss(decoder_dir="models/decoder", figsize=(9, 4)):
    """Plot training loss from HuggingFace Trainer trainer_state.json."""
    import os, json
    import matplotlib.pyplot as plt

    state_path = os.path.join(decoder_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"[plot_decoder_loss] {state_path} not found. Run fine-tuning first.")
        return

    with open(state_path) as f:
        state = json.load(f)

    history = state.get("log_history", [])
    train_steps = [e["step"] for e in history if "loss" in e]
    train_loss  = [e["loss"] for e in history if "loss" in e]

    if not train_steps:
        print("[plot_decoder_loss] No training loss entries in trainer_state.json.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_steps, train_loss, color="#C44E52", linewidth=1.5, label="Train loss")
    ax.set_xlabel("Paso de entrenamiento")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title("Decoder — Curva de pérdida durante fine-tuning QLoRA", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_val_accuracy(val_preds, val_gold, figsize=(12, 5)):
    """Bar chart of per-tool accuracy on the validation set."""
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    for pred, gold in zip(val_preds, val_gold):
        tool = gold.split("(")[0]
        rows.append({"tool": tool, "correct": int(pred == gold)})

    df = pd.DataFrame(rows)
    per_tool = df.groupby("tool")["correct"].agg(["mean", "count"]).reset_index()
    per_tool = per_tool.sort_values("mean")

    overall = df["correct"].mean()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(per_tool["tool"], per_tool["mean"] * 100,
                   color=["#C44E52" if v < 0.7 else "#55A868" for v in per_tool["mean"]])

    for bar, (_, row) in zip(bars, per_tool.iterrows()):
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%  (n={int(row['count'])})", va="center", fontsize=8)

    ax.axvline(overall * 100, color="navy", linestyle="--",
               label=f"Overall: {overall*100:.1f}%")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Accuracy por herramienta — conjunto de validación (exact match)",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()
