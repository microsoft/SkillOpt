from __future__ import annotations

import re

from openai import AzureOpenAI, OpenAI

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either [\"CORRECT\", \"INCORRECT\", \"NOT_ATTEMPTED\"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- They do not contain any information that contradicts the gold target.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
- Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
```
These predicted answers are all INCORRECT because a factual statement in the answer contradicts the gold target.

The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
```
These predicted answers are all NOT_ATTEMPTED because the important information in the gold target is not included and there is no contradiction.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Just return the letters \"A\", \"B\", or \"C\", with no text around it.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED
""".strip()


def _build_grader_client() -> tuple[OpenAI | AzureOpenAI, str]:
    import os

    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', '').strip()
    api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '').strip() or '2025-04-01-preview'
    azure_key = os.environ.get('AZURE_OPENAI_API_KEY', '').strip()
    openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
    api_key = azure_key or openai_key
    if endpoint and api_version and api_key:
        model = os.environ.get('SEALQA_GRADER_AZURE_MODEL', '').strip() or os.environ.get('SEALQA_GRADER_MODEL', '').strip() or os.environ.get('AZURE_MODEL_NAME', '').strip() or os.environ.get('TEACHER_DEPLOYMENT', '').strip() or 'gpt-5.4'
        client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint.rstrip('/'))
        return client, model

    if openai_key:
        model = os.environ.get('SEALQA_GRADER_OPENAI_MODEL', '').strip() or os.environ.get('SEALQA_GRADER_MODEL', '').strip() or 'gpt-4.1-mini'
        return OpenAI(api_key=openai_key), model

    raise ValueError('Missing grader credentials for SealQA scoring.')


def _extract_text_content(content) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                parts.append(str(part.get('text', '')))
            else:
                text = getattr(part, 'text', None)
                if text:
                    parts.append(str(text))
        return '\n'.join(parts).strip()
    return str(content).strip()


def _normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r'\s+', ' ', lowered)
    lowered = re.sub(r'[^\w\s%.-]', '', lowered)
    return lowered.strip()


def _fallback_score(ground_truth: str, predicted: str) -> float:
    gold = _normalize_text(ground_truth)
    pred = _normalize_text(predicted)
    if not gold or not pred:
        return 0.0
    if gold == pred:
        return 1.0
    if gold in pred or pred in gold:
        return 1.0
    return 0.0


def score_sealqa(question: str, ground_truth: str, predicted: str) -> float:
    try:
        client, model = _build_grader_client()
    except ValueError:
        return _fallback_score(ground_truth, predicted)

    prompt = GRADER_TEMPLATE.format(question=question, target=ground_truth, predicted_answer=predicted)
    completion = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': prompt}])
    content = _extract_text_content(completion.choices[0].message.content).strip().upper()
    if content.startswith('A'):
        return 1.0
    return 0.0
