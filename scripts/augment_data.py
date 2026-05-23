"""
Synthetic data augmentation for underrepresented tool classes.

Approach: Label-first generation (Wei et al. Self-Instruct 2022)
  1. Enumerate all valid canonical tool calls for each underrepresented class
  2. Randomly sample a target tool_call and 3 real few-shot seeds
  3. Ask the LLM to generate a natural operator query for that exact tool_call
  4. Validate length, deduplicate, save to synthetic_train.csv

Model: meta/llama-3.1-8b-instruct via NVIDIA Build free API
  Endpoint: https://integrate.api.nvidia.com/v1  (build.nvidia.com/models)
  License:  Meta Llama 3.1 Community License (research/academic use permitted)
  Justification: sufficient instruction-following at 8B scale, free tier,
  OpenAI-compatible API, no paid subscription required.

API key: NVIDIA_API_KEY in .env
"""
import os
import random
import re
import sys
import time
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import _find_env_file, enrich_tools_def, load_tools_def

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"

TARGET_PER_CLASS = 350
MIN_PER_CALL = 5      # minimum examples per unique tool_call combination (zero-coverage fill)
FEW_SHOT_K = 3
REQUEST_DELAY = 0.3


def load_nvidia_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(_find_env_file(".env"))
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY not found in .env")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def enumerate_calls(tool_name: str, tool_info: dict) -> list[str]:
    """Return all valid canonical tool_call strings for a tool."""
    params = tool_info["parameters"]
    if not params:
        return [f"{tool_name}()"]
    names = list(params.keys())
    values = [params[p]["values"] for p in names]
    calls = []
    for combo in product(*values):
        parts = []
        for pname, val in zip(names, combo):
            if params[pname]["type"] == "int":
                parts.append(f"{pname}={val}")
            else:
                sval = str(val).upper() if pname == "protocol_id" else str(val).lower()
                parts.append(f"{pname}='{sval}'")
        calls.append(f"{tool_name}({','.join(parts)})")
    return calls


def build_prompt(target_tc: str, few_shot: list[tuple]) -> str:
    examples = "\n".join(f'  Query: "{q}"\n  Tool call: {tc}' for q, tc in few_shot)
    return (
        "You generate realistic operator queries for MASA's Kuntur Station (400 km orbit).\n"
        "Modules: condor (command/control), quetzal (science lab), jaguar (life support), "
        "colibri (comms/navigation), vicuna (storage/docking), tucan (crew quarters).\n"
        "Crew: Commander Reyes, Pilot Valdivia, Specialists Nakamura & Al-Hassan, "
        "Engineer Kozlov, Medical Officer Mendoza.\n\n"
        "Examples:\n"
        f"{examples}\n\n"
        "Generate ONE new operator query (10–35 words, English) that requires exactly:\n"
        f"  {target_tc}\n\n"
        "Rules: write only the query text, no quotes, no tool call, no explanation."
    )


def generate_one(client, target_tc: str, few_shot: list) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": build_prompt(target_tc, few_shot)}],
            temperature=0.9,
            max_tokens=80,
        )
        text = resp.choices[0].message.content.strip()
        text = text.split("\n")[0].strip().strip('"').strip("'")
        text = re.sub(r"^Query:\s*", "", text, flags=re.IGNORECASE)
        if 8 <= len(text.split()) <= 50:
            return text
        return None
    except Exception as e:
        msg = str(e)
        print(f"  [API error] {type(e).__name__}: {msg}", file=sys.stderr)
        if "401" in msg or "Authentication" in msg or "Unauthorized" in msg:
            raise RuntimeError(f"NVIDIA API authentication failed — check NVIDIA_API_KEY in .env\n{e}") from e
        return None


def augment(
    train_csv: str,
    tools_def_file: str,
    output_csv: str = "data/synthetic/synthetic_train.csv",
    target_per_class: int = TARGET_PER_CLASS,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)

    tools_def = load_tools_def(tools_def_file)
    enrich_tools_def(tools_def, train_csv)

    df = pd.read_csv(train_csv).drop_duplicates(subset=["query", "tool_call"])
    tool_counts = df["tool_call"].str.extract(r"^(\w+)")[0].value_counts()

    print("Augmentation targets (TARGET_PER_CLASS={})".format(target_per_class))
    for t, info in tools_def.items():
        if t == "no_action":
            continue
        cur = tool_counts.get(t, 0)
        need = max(0, target_per_class - cur)
        print(f"  {t:<25} {cur:>5} → +{need}")

    client = load_nvidia_client()

    synthetic = []
    seen = set(df["query"].str.lower())

    for tool_name, tool_info in tools_def.items():
        if tool_name == "no_action":
            continue
        current = tool_counts.get(tool_name, 0)
        needed = max(0, target_per_class - current)
        if needed == 0:
            continue

        valid_calls = enumerate_calls(tool_name, tool_info)
        real = df[df["tool_call"].str.startswith(tool_name)][["query", "tool_call"]].values.tolist()
        if not real:
            real = [("check the system status", f"{tool_name}({','.join(list(tool_info['parameters'].keys())[:1])}='placeholder')")]

        print(f"\n[{tool_name}] generating {needed} examples...")
        generated = 0
        attempts = 0

        while generated < needed and attempts < needed * 5:
            attempts += 1
            target_tc = random.choice(valid_calls)
            few_shot = random.sample(real, min(FEW_SHOT_K, len(real)))
            query = generate_one(client, target_tc, few_shot)
            if query and query.lower() not in seen:
                seen.add(query.lower())
                synthetic.append({"query": query, "tool_call": target_tc})
                generated += 1
                if generated % 50 == 0:
                    print(f"  {generated}/{needed}")
            time.sleep(REQUEST_DELAY)

        print(f"  done: {generated}/{needed} generated")

    # Zero-coverage fill: ensure every valid tool_call combination has >= MIN_PER_CALL examples.
    # Targets combinations absent from train.csv (e.g. MASA-SEC-016, MASA-SEC-020) that the
    # per-tool balance loop skips because the tool itself has enough examples overall.
    # Capped at MAX_FILL_COMBOS per tool to avoid spending hours on high-cardinality tools.
    tc_counts_all = df["tool_call"].value_counts()
    synth_counts: dict[str, int] = {}
    for r in synthetic:
        synth_counts[r["tool_call"]] = synth_counts.get(r["tool_call"], 0) + 1

    print("\n[zero-coverage fill] scanning for combinations with < MIN_PER_CALL examples...")
    for tool_name, tool_info in tools_def.items():
        if tool_name == "no_action":
            continue
        valid_calls = enumerate_calls(tool_name, tool_info)
        zero_calls = [
            tc for tc in valid_calls
            if tc_counts_all.get(tc, 0) + synth_counts.get(tc, 0) < MIN_PER_CALL
        ]
        if not zero_calls:
            continue

        print(f"  [{tool_name}] {len(zero_calls)} under-covered combinations → generating {MIN_PER_CALL} each")

        real = df[df["tool_call"].str.startswith(tool_name)][["query", "tool_call"]].values.tolist()
        if not real:
            real = [("check the system status", f"{tool_name}()")]

        for combo_i, target_tc in enumerate(zero_calls, 1):
            generated = 0
            attempts = 0
            while generated < MIN_PER_CALL and attempts < MIN_PER_CALL * 8:
                attempts += 1
                few_shot = random.sample(real, min(FEW_SHOT_K, len(real)))
                query = generate_one(client, target_tc, few_shot)
                if query and query.lower() not in seen:
                    seen.add(query.lower())
                    synthetic.append({"query": query, "tool_call": target_tc})
                    synth_counts[target_tc] = synth_counts.get(target_tc, 0) + 1
                    generated += 1
                time.sleep(REQUEST_DELAY)
            print(f"    [{combo_i}/{len(zero_calls)}] {target_tc[:60]} → {generated}/{MIN_PER_CALL}")

    out = pd.DataFrame(synthetic)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"\nSaved {len(out)} synthetic rows → {output_csv}")
    return out


if __name__ == "__main__":
    augment(
        train_csv="train.csv",
        tools_def_file="tools_definition.json",
    )
