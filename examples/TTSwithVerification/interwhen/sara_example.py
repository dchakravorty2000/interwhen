import os
import re
import json
import asyncio

from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset
import logging

from interwhen.interject import stream_completion
from interwhen.utils.llm import init_llm_server
from interwhen.monitors.legal_reasoning_monitor import (
    LegalReasoningMonitor,
)

logging.basicConfig(level=logging.INFO)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


MODEL = "gemma4:12b"

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "interwhen" / "data" / "legal_data"

STATUTE_FOLDER = DATA_DIR / "source"
TEST_SET = DATA_DIR / "test_set3.txt"

OUT_REASONING = (
    Path(__file__).parent
    / "runtime_reasoning"
)

OUT_REASONING.mkdir(exist_ok=True)
OUT_REASONING.mkdir(exist_ok=True)



llm_server = {
    "url": "http://10.5.30.32:11434/v1/completions",
    "headers": {
        "Content-Type": "application/json",
        "Authorization": "Bearer ollama"
    },
    "payload": {
        "model": "gemma4:12b",
        "stream": True,
        "temperature": 0,
        "think": False,
        "context_length": 32000   
    },
}



print("Loading SARA...")

ds = load_dataset("jhu-clsp/SARA")


def is_entailment(example):

    return example["answer"] in (
        "Entailment",
        "Contradiction",
    )


train_examples = ds["train"].filter(
    is_entailment
)

test_examples = ds["test"].filter(
    is_entailment
)

examples = (
    list(train_examples)
    + list(test_examples)
)

print(f"Loaded {len(examples)} examples.")



statutes = {}

for f in sorted(os.listdir(STATUTE_FOLDER)):

    path = STATUTE_FOLDER / f

    with open(path, encoding="utf-8") as fh:

        txt = fh.read()

    m = re.search(r"(\d+)", f)

    if m:
        statutes[m.group(1)] = txt



def extract_subsection(statute_text, letter):

    pattern = rf"\({letter}\)(.*?)(?=\n\([a-z]\)|$)"

    m = re.search(
        pattern,
        statute_text,
        flags=re.S | re.I,
    )

    return m.group(0).strip() if m else ""


def retrieve_subsection(example_id):

    parts = example_id.split("_")

    sec = parts[0][1:]

    sub = parts[1] if len(parts) > 1 else None

    statute = statutes.get(sec, "")

    if not statute:
        return ""

    if sub:

        s = extract_subsection(
            statute,
            sub,
        )

        if s:
            return f"§{sec}\n{s}"

    return statute




def build_reasoning_prompt(
    example,
    statute,
):

    return f"""
You are an expert legal reasoning system.

Determine whether the HYPOTHESIS follows from the CASE FACTS under the STATUTE.

Reason naturally.

The LAST LINE of your response must contain exactly one word:

Entailment

or

Contradiction

--------------------------------------------------

STATUTE

{statute}

--------------------------------------------------

CASE FACTS

{example["text"]}

--------------------------------------------------

HYPOTHESIS

{example["question"]}
""".strip()


def extract_prediction(reasoning: str):
    lines = [l.strip() for l in reasoning.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        low = line.lower()
        if low in ("entailment", "contradiction"):
            return line
    return "UNKNOWN"  


async def process_example(example):

    statute = retrieve_subsection(
        example["id"]
    )

    prompt = build_reasoning_prompt(
        example,
        statute,
    )

    monitor = LegalReasoningMonitor(
        name="legal",
        example_id=example["id"],
        threshold=0.7,
    )

    reasoning = await stream_completion(
        prompt=prompt,
        llm_server=llm_server,
        monitors=[monitor],
        tokenizer=tokenizer
    )

    reasoning_path = (
        OUT_REASONING
        / f"{example['id']}.txt"
    )

    with open(
        reasoning_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(reasoning)

    prediction = extract_prediction(reasoning)

    prediction_path = (
        OUT_REASONING
        / f"{example['id']}.json"
    )

    with open(
        prediction_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "id": example["id"],
                "gold": example["answer"],
                "prediction": prediction,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"{example['id']} : "
        f"{prediction}"
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    with open(
        TEST_SET,
        "r",
        encoding="utf-8",
    ) as f:

        test_ids = {

            line.strip()

            for line in f

            if line.strip()

        }

    eval_examples = [

        ex

        for ex in examples

        if ex["id"] in test_ids

    ]

    print(
        f"Evaluating "
        f"{len(eval_examples)} "
        f"examples."
    )

    for example in tqdm(eval_examples):

        try:

            await process_example(
                example
            )

        except Exception as e:

            print(
                f"\n{example['id']} failed:"
            )

            print(e)

    print()

    print("Finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())