from skillopt_sleep.backend import _extract_json


def test_extract_json_object_after_chatty_braces():
    raw = 'Format: {score, reason}.\n{"score": 0.9, "reason": "ok"}'

    assert _extract_json(raw, "object") == {"score": 0.9, "reason": "ok"}


def test_extract_json_array_before_chatty_brackets():
    raw = '```json\n[{"op":"add","content":"Cite sources"}]\n```\nSee [1] above.'

    assert _extract_json(raw, "array") == [
        {"op": "add", "content": "Cite sources"}
    ]
