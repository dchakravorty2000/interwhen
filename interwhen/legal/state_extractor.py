import re
import json
import requests

OLLAMA_URL = "http://10.5.30.32:11434/api/generate"
MODEL = "gemma4:12b"


def build_prompt(reasoning: str) -> str:

    return f"""
You are an expert legal reasoning analyzer.

A partial legal reasoning trace is provided.

Your task is to extract the FIRST COMPLETE Verifiable Unit (VU) that appears in the reasoning.

A Verifiable Unit (VU) consists of:

• One or more Premises.
• Exactly one Conclusion.

The conclusion MUST state whether a SPECIFIC statutory subsection
(for example §152(c)(3), §1(a)(i), etc.) is satisfied or not satisfied.

The conclusion MUST explicitly mention the subsection.

IMPORTANT

A VU should correspond to EXACTLY ONE statutory determination.

The premises should contain ONLY the information immediately necessary to support that statutory determination.

Do NOT include premises that were only used to derive intermediate facts earlier in the reasoning.

For example, if the reasoning is

Bob was born in 1984.
Alice was born in 1992.
Therefore Bob is older than Alice.
Section 152(c)(3) requires the child to be younger than the taxpayer.
Therefore Bob does not satisfy §152(c)(3).

The extracted VU should be

Premise 1.1: Bob is older than Alice.
Premise 1.2: Section 152(c)(3) requires the child to be younger than the taxpayer.
Conclusion 1: Bob does not satisfy §152(c)(3).

NOT

Premise: Bob born in 1984.
Premise: Alice born in 1992.
Premise: Bob is older than Alice.
Premise: ...
Conclusion: Bob does not satisfy §152(c)(3).

RULES

- Extract ONLY the FIRST complete VU.
- A VU should contain EXACTLY ONE statutory conclusion.
- The conclusion MUST mention the subsection.
- Include ONLY the premises directly supporting that conclusion.
- Do NOT include earlier reasoning that merely established intermediate facts.
- Copy the wording from the reasoning as closely as possible.
- Do NOT rewrite or paraphrase.
- Do NOT invent premises.
- Do NOT combine multiple statutory conclusions into one VU.
- If no complete statutory conclusion has been reached, output ONLY:

NONE

OUTPUT FORMAT

Premise 1.1: ...
Premise 1.2: ...
Conclusion 1: ...

Reasoning

--------------------------------------------------

{reasoning}

Extract the FIRST complete Verifiable Unit.
""".strip()


PREMISE_RE = re.compile(
    r"^Premise\s+\d+\.\d+:\s*(.*)",
    re.I
)

CONCLUSION_RE = re.compile(
    r"^Conclusion\s+\d+:\s*(.*)",
    re.I
)


def _parse_json(text):

    text = text.strip()

    text = re.sub(
        r"^```(?:text)?",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"```$",
        "",
        text
    ).strip()

    if text.upper() == "NONE":
        return None

    premises = []

    conclusion = None

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        m = PREMISE_RE.match(line)

        if m:

            premises.append(
                m.group(1).strip()
            )

            continue

        m = CONCLUSION_RE.match(line)

        if m:

            conclusion = (
                m.group(1).strip()
            )

    if len(premises) == 0:
        return None

    if conclusion is None:
        return None

    return {

        "premises": premises,

        "conclusion": conclusion

    }


def extract_state(reasoning):

    payload = {

        "model": MODEL,

        "prompt": build_prompt(reasoning),

        "stream": False,

        "think": False,

        "options": {

            "temperature": 0,

            "num_predict": 300
        }
    }

    r = requests.post(

        OLLAMA_URL,

        json=payload,

        timeout=(10, 1800)

    )

    r.raise_for_status()

    response = r.json()["response"]

    try:

        vu = _parse_json(response)

    except Exception as e:
        print(f"State extraction failed: {e}")
        return None

    return vu