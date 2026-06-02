"""
Benchmark Environment Template
===============================
Copy this file and implement the TODO sections to add a new benchmark.

The EnvAdapter is responsible for:
1. Executing tasks using the target model + current skill document
2. Evaluating predictions against ground truth
3. Returning structured results for the training loop
"""
from skillopt.envs.base import EnvAdapter


class TemplateBenchmarkEnv(EnvAdapter):
    """
    Environment adapter for <Your Benchmark Name>.

    Rename this class and implement the abstract methods below.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # TODO: Initialize benchmark-specific state
        # Example: self.tools = load_tools(cfg)

    async def execute(self, item, skill: str, model):
        """
        Execute a single task with the target model.

        Args:
            item: DataItem with .id, .input, .ground_truth, .metadata
            skill: Current skill document content (Markdown string)
            model: Target model backend instance

        Returns:
            TaskResult with prediction, score, and trajectory
        """
        # Step 1: Build the prompt combining skill + task input
        prompt = self.build_prompt(item, skill)

        # Step 2: Call the target model
        # TODO: Customize the message format for your benchmark
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = await model.generate(messages)

        # Step 3: Parse the model response into a prediction
        prediction = self.parse_response(response.content)

        # Step 4: Score the prediction
        score = self.evaluate(prediction, item.ground_truth)

        # Step 5: Return structured result
        return {
            "item_id": item.id,
            "prediction": prediction,
            "score": score,
            "trajectory": messages + [{"role": "assistant", "content": response.content}],
        }

    def evaluate(self, prediction: str, ground_truth: str) -> float:
        """
        Score a prediction against the ground truth.

        Returns:
            Float between 0.0 (wrong) and 1.0 (correct)

        TODO: Implement your scoring metric. Common options:
        - Exact match: float(pred.strip().lower() == gt.strip().lower())
        - F1 score: compute token overlap
        - ANLS: for document QA tasks
        - Custom: any float in [0, 1]
        """
        # Placeholder — exact match
        return float(prediction.strip().lower() == ground_truth.strip().lower())

    def build_prompt(self, item, skill: str) -> str:
        """Combine skill document with task input."""
        return f"{skill}\n\n---\n\nQuestion: {item.input}"

    def parse_response(self, response: str) -> str:
        """
        Extract the answer from the model's raw response.

        TODO: Implement extraction logic. For example:
        - Extract text after "Answer:"
        - Parse JSON output
        - Extract from code blocks
        """
        return response.strip()
