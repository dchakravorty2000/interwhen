# import logging

# logger = logging.getLogger(__name__)

# def verify_state(example_id, vu):
#     logger.info("=" * 80)
#     logger.info("verifier invoked.")
#     logger.info("Example ID: %s", example_id)
#     logger.info("VU: %s", vu)
#     logger.info("Returning PASS.")
#     logger.info("=" * 80)

#     return {
#         "status": "PASS"
#     }



import re
import json
import requests

from pathlib import Path

import logging

logger = logging.getLogger(__name__)

# ============================================================
# PATHS
# ============================================================

GRAPH_DIR = Path("interwhen/data/legal_data/case_graphs_final")
NODE_DIR = Path("interwhen/data/legal_data/grounded_nodes_final")

OLLAMA_URL = "http://10.5.30.32:11434/api/generate"
MODEL = "gemma4:12b"

# ============================================================
# CACHE
# ============================================================

GROUND_CACHE = {}
GRAPH_CACHE = {}
ALIGNMENT_CACHE = {}

# ============================================================
# NORMALIZATION
# ============================================================

def strip_prefix(text):

    text = re.sub(
        r"^(premise|conclusion)\s*\d*(?:\.\d+)*\s*:\s*",
        "",
        text,
        flags=re.I,
    )

    return text.strip()


def normalize_predicate(text):

    text = re.sub(r"\s+", "", text)

    text = text.replace("=<", "<=")

    return text

SECTION_REF_RE = re.compile(
    r"(?:section|§)\s*(\d+)\s*\(\s*([a-z])\s*\)\s*\(\s*([ivx]+)\s*\)",
    re.IGNORECASE,
)


def extract_section_hint(text):

    m = SECTION_REF_RE.search(text)

    if not m:
        return None

    sec, sub, clause = m.groups()

    return f"s{sec}_{sub}_{clause.lower()}"


# ============================================================
# LOADING
# ============================================================

def load_grounded_nodes(example_id):

    if example_id in GROUND_CACHE:
        return GROUND_CACHE[example_id]

    path = NODE_DIR / f"{example_id}.txt"

    if not path.exists():
        return []

    grounded = []

    with open(path, encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            grounded.append(
                normalize_predicate(
                    strip_prefix(line)
                )
            )

    GROUND_CACHE[example_id] = grounded

    return grounded


def load_graph(example_id):

    if example_id in GRAPH_CACHE:
        return GRAPH_CACHE[example_id]

    path = GRAPH_DIR / f"{example_id}_grounded.json"

    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        graph = json.load(f)

    GRAPH_CACHE[example_id] = graph

    return graph


def extract_children(node_entry):

    if isinstance(node_entry, dict):
        return node_entry.get("children", [])

    return []


# ============================================================
# LLM ALIGNMENT PROMPT
# ============================================================

def build_alignment_prompt(
    reasoning_step,
    grounded,
    top_k=1,
):

    grounded_block = "\n".join(
        f"{i+1}. {g}"
        for i, g in enumerate(grounded)
    )

    if top_k == 1:
        output_instruction = """
Choose EXACTLY ONE predicate.
Output ONLY the predicate.
"""
    else:
        output_instruction = f"""
Choose up to {top_k} predicates ranked from best to worst.

Output one predicate per line.

If fewer than {top_k} reasonable matches exist,
output only those.

If no reasonable match exists output exactly

NONE
"""

    return f"""
You are a legal trace alignment system.

Task

Given a natural language reasoning step and a list of grounded predicates,
identify the grounded predicates that best match the reasoning step.

This is a SYMBOL ALIGNMENT task.

Do NOT perform legal reasoning.

Do NOT determine whether the statement is true.

Simply identify which grounded predicate(s) the sentence was generated from.

Reasoning Step

"{reasoning_step}"

Grounded Predicates

{grounded_block}

Rules

- Never modify a predicate.
- Never invent a predicate.
- Copy predicates EXACTLY.
- Prefer exact numbers.
- Prefer exact operators.
- Prefer exact predicate names.
- Prefer exact section references.

{output_instruction}
""".strip()


# ============================================================
# GEMMA CALL
# ============================================================

def call_alignment_llm(prompt):

    payload = {

        "model": MODEL,

        "prompt": prompt,

        "stream": False,

        "think": False,

        "options": {
            "temperature": 0,
            "num_predict": 200 
        }

    }

    r = requests.post(

        OLLAMA_URL,

        json=payload,

        timeout=(10, 1800)

    )

    r.raise_for_status()

    return r.json()["response"].strip()


# ============================================================
# DISAMBIGUATION
# ============================================================

def disambiguate(
    example_id,
    reasoning_step,
    grounded,
    top_k=1,
):

    key = (
        example_id,
        reasoning_step,
        top_k,
    )

    if key in ALIGNMENT_CACHE:
        return ALIGNMENT_CACHE[key]

    #
    # Restrict conclusion search to statute predicates.
    #

    candidates = grounded

    if top_k == 1:

        hint = extract_section_hint(reasoning_step)

        statute_nodes = [
            g for g in grounded
            if g.startswith("s")
        ]

        if statute_nodes:
            candidates = statute_nodes

        if hint:

            filtered = [
                g for g in candidates
                if g.startswith(hint)
            ]

            if filtered:
                candidates = filtered

    prompt = build_alignment_prompt(
        reasoning_step,
        candidates,
        top_k=top_k,
    )

    try:
        response = call_alignment_llm(prompt)

    except Exception as e:

        logger.exception(e)

        return [] if top_k > 1 else None

    ranked = []

    candidate_set = set(candidates)

    for line in response.splitlines():

        line = line.strip()

        if not line:
            continue

        m = re.match(
            r'^(?:\d+[\.\)]\s*|[-*]\s*)?(.*)$',
            line
        )

        if not m:
            continue

        pred = normalize_predicate(
            m.group(1)
            .strip()
            .strip("`\"'.,:;")
        )

        if pred in candidate_set and pred not in ranked:
            ranked.append(pred)

    if top_k == 1:

        pred = ranked[0] if ranked else None

        ALIGNMENT_CACHE[key] = pred

        return pred

    ALIGNMENT_CACHE[key] = ranked[:top_k]

    return ranked[:top_k]


# ============================================================
# VERIFY SINGLE VU
# ============================================================

def verify_state(
    example_id,
    vu
):

    grounded = load_grounded_nodes(example_id)

    if len(grounded) == 0:

        return {
            "status": "FAIL"
        }

    graph = load_graph(example_id)

    if len(graph) == 0:

        return {
            "status": "FAIL"
        }

    # --------------------------------------------------------
    # DISAMBIGUATE PREMISES
    # --------------------------------------------------------

    grounded_premises = set()

    logger.info("Premise Disambiguation")

    for premise in vu["premises"]:

        preds = disambiguate(
            example_id,
            premise,
            grounded,
            top_k=10,
        )

        logger.info("Premise:")
        logger.info("  %s", premise)

        if preds:
            logger.info("Mapped Predicates:")
            for p in preds:
                logger.info("  -> %s", p)
        else:
            logger.info("  -> NONE")

        grounded_premises.update(preds)

    # --------------------------------------------------------
    # DISAMBIGUATE CONCLUSION
    # --------------------------------------------------------

    grounded_conclusion = disambiguate(
        example_id,
        vu["conclusion"],
        grounded
    )

    logger.info("Conclusion Disambiguation")

    logger.info("Conclusion:")
    logger.info("  %s", vu["conclusion"])

    if grounded_conclusion is not None:
        logger.info("Mapped Predicate:")
        logger.info("  -> %s", grounded_conclusion)
    else:
        logger.info("  -> NONE")

    # --------------------------------------------------------
    # CONCLUSION EXISTS
    # --------------------------------------------------------

    if grounded_conclusion not in graph:

        return {
            "status": "FAIL"
        }

    # --------------------------------------------------------
    # VERIFY GRAPH
    # --------------------------------------------------------

    node = graph[grounded_conclusion]

    required_children = set(
        extract_children(node)
    )

    matched_children = required_children.intersection(
        grounded_premises
    )

    if len(matched_children) > 0:

        return {
            "status": "PASS"
        }

    return {
        "status": "FAIL"
    }