# MMRB Multi-Image Reasoning Heuristics

## Cross-Image Alignment
- Track the role of each image by its index and compare evidence across all referenced images before deciding.
- When the question depends on sequence, correspondence, or retrieval, verify the relation between images instead of judging each image independently.

## Option Elimination
- For multiple-choice tasks, compare all options and reject choices that match only part of the visual evidence.
- If options differ by a small visual detail, use the most discriminative cue rather than a coarse scene impression.

## Open Answers
- For open-ended tasks, give the shortest answer that is fully supported by the combined images.
- Preserve exact entities, attributes, counts, and directions when the images support them directly.

## Final Answer
- Output only the final answer inside <answer>...</answer>.

