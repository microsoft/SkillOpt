# BabyVision Visual QA Heuristics

## Image Inspection
- First identify the main objects, their attributes, and their spatial relations before answering.
- If the question involves counting, compare all relevant instances carefully instead of stopping after the first match.
- If the question asks about color, size, position, or action, verify the specific visible evidence for that attribute.

## Multiple Choice
- Compare every option against the visible image evidence before deciding.
- Prefer the option that matches the image exactly; reject options that are only partially true or too vague.
- When two options are close, check the smallest discriminating visual detail.

## Open Answers
- Answer with the shortest phrase that is fully supported by the image.
- Match the expected level of specificity: not broader than the image evidence, not narrower than the question asks.

## Final Answer
- Output only the final answer inside <answer>...</answer>.
