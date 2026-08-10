# SkillOpt Design Spec

## __Problem Statement:__
Our current SA pipeline uses an LLM to perform 3 tasks, `boundary detection for annotations`, `verification for whether the annotations are correct`, and `extracting properties from annotations`.  These tasks all use prompts to guide the LLM into getting the correct answer, which often takes manual tuning to achieve acceptable results. 


__This raises 2 main problems:__

1. __Difficult to Tune__: When tuning LLM prompts, it is often not transparent as to what change will cause what outcome. This means that we are often left blindly making changes, without any concrete metric of whether the changes we make are actually beneficial or not. 
2. __Time Consuming__: Ties into the previous issue. The human review process takes time which significantly slows down engineering output and productivity.  


## __Solution__
To solve both of these issues, we plan to use SkillOpt. SkillOpt is a repo that allows us to tune prompts by following a similar loop as training a Deep Learning model. See [the deep learning analogy](../docs/guide/dl-analogy.md) for a clear comparison.
 

### __General Idea__

We use this general loop:

For each epoch, for each step:
1. Target executes tasks with current best prompt
2. From the task results, __trajectories__ are created containing data about task execution
3. __Optimizer__ (a LLM Model) analyzes __trajectories__ and __RolloutResults__, then creates a __patch__, an object containing suggested edits
4. Smaller __patches__ are merged into one patch 
5. The edits in a __patch__ will be ranked, and then clipped to be <= __learning rate__
6. Apply the top edits in the __patches__ to the prompt
7. Validate the new prompt with the __validation set__. Accept if score is higher than best score, else reject

This solves both of our problems: 
- By defining a clear metric, we can see how each prompt change affects the task score
- SkillOpt doesn't require the constant human review and edits manually editing prompts do, freeing up time


## __In Scope:__ 
1. Use SkillOpt to **tune the prompt** in the SA pipeline which handles `boundary detection`
2. Manually reviewing annotations to create a ground-truth dataset
3. Handling of partial credit (a field extracted but slightly malformed shouldn't score the same as a missing field)

## __Out of Scope:__
1. Use SkillOpt to **tune the prompt** in the SA pipeline which handles `validation` and `property extraction` 

2. **Streamline client onboarding with SkillOpt** so manual review and tuning of definitions is no longer needed.

   Current prompt-tuning process for a new client:
   1. Copy the most recent client definition matching the concept (minutes, agendas, etc.)
   2. Tailor the definition to the new client's files.
   3. Run a test batch of 8–10 files and export the output to an Excel sheet.
   4. Adam manually reviews and gives feedback on each row.
   5. Feedback goes back to Austin or Femina to tweak the definition.
   6. Repeat 3–5 until recall ≥ 92%.
   
## __File Structure:__
We are forking the SkilOpt repository. The only files we edit are under `configs/`, `data/`, and `skillopt/envs/_annotation_boundary` 
```
SkillOpt/
├── ...
├── configs/
│   ├── ...
│   └── annotation_boundary/        <-- Currently named as annotation_boundary. Need better name probably. 
│       └── default.yaml
├── data/
│   ├── ...
│   └── annotation_boundary_split/        <-- Currently named as annotation_boundary. Need better name probably. 
│       ├── test
|       ├── train
│       ├── val
|       └── split_manifest.json
├── skillopt/
│   ├── ...
│   └── envs/
│       ├── ...
|       ├── _common/
|       ├── _annotation_common/
│       └── _annotation_boundary/       <-- Currently named as annotation_boundary. Need better name probably. 
│           ├── prompts/
│           │   ├── analyst_error.md
│           │   ├── analyst_success.md
│           │   └── rollout_system.md
│           ├── skills/
│           │   └── initial.md
│           ├── __init__.py
│           ├── adapter.py
│           ├── dataloader.py
│           ├── evaluator.py
│           └── rollout.py
├── ...
├── OVERVIEW.md
└── SkillOpt_Design_Spec.md
```


## __Domain Types:__
```python
type UnitInterval = float
type SoftScore = float
type HardScore = int
```

```python
class BatchPhase(strEnum):
    TRAIN = "train"
    EVAL = "eval"
```

```python
class BatchSplit(strEnum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"
```

_Our code only writes, doesn't read, so this only validates on write_
```python
class ToolCallTrajectory(Basemodel):
    # Used when the LLM makes a tool call
    type: Literal["tool_call"]
    cmd: str
    obs: str
    turn: NotRequired[int]
```

```python
class StepTrajectory(Basemodel):
    # Used in cases of a stateful environment
    action: str
    env_feedback: str
    step: NotRequired[int]
    reasoning: NotRequired[str]
```

```python
class VerificationTrajectory(Basemodel):
    # Post-execution verification / enrichment info
    role: Literal["system"]
    content: str
```

```python
class MessageTrajectory(Basemodel):
    # Used for normal chats, like what the model was asked and what it said
    content: str
    role: NotRequired[str]
    type: NotRequired[str]
    turn: NotRequired[int]
```

```python
Trajectory = ToolCallTrajectory | StepTrajectory | VerificationTrajectory | MessageTrajectory
```

```python
class RolloutResult(TypedDict):
    id: str # required 
    soft: SoftScore # required 
    hard: HardScore # required 
    predicted_answer: str
    task_description: str # makes llm accurate
    question: str
    reference_text: str
    task_type: str # required for minibatch
    target_system_prompt: str
    target_user_prompt: str
    n_turns: int # cosmetic; for prints
```

## __Use Case Types:__
```python
AnnotationBoundryAdaptor(EnvAdapter)  #EnvAdapter is defined by SkillOpt
```

```python
AnnotationBoundryDataLoader(SplitDataLoader)
```


```python 
@dataclass 
AnnotatedChunk():
    # TODO: Check if Femina has a type for these
    chunk: str 
    ground_truth_span_indexs: list[tuple[int, int]]
    ground_truth_properties: list[Property]
```

```python 
# Returned from SplitDataLoader.load_split_items
@dataclass 
AnnotationBoundryTask():
    id: str
    annotated_chunk: AnnotatedChunk
```



```python
@dataclass
AnnotationBoundryBatchSpec(BatchSpec): 
    phase: BatchPhase
    split: BatchSplit
    seed: int
    batch_size: int
    payload: list[AnnotationBoundryTask]
    metadata: dict[str, Any] = field(default_factory=dict)

```


```python
# Enforce that AnnotationBoundryEnv should be = list[AnnotationBoundryBatchSpec.payload]
AnnotationBoundryEnv = NewType("AnnotationBoundryEnv", list[AnnotationBoundryTask]): 
```

## __Functions:__
`dataloader.py`
```python
AnnotationBoundryDataLoader.load_split_items(split_path: str) -> list[AnnotationBoundryTask]
```

`adaptor.py`
```python
# Use implementation from template_env
AnnotationBoundryAdaptor.__init__() -> None
```

```python
# Use implementation from template_env
AnnotationBoundryAdaptor.setup() -> None
```

```python;
AnnotationBoundryAdaptor.get_dataloader() -> AnnotationBoundryDataLoader
```

```python
AnnotationBoundryAdaptor.build_env_from_batch(BatchSpec: AnnotationBoundryBatchSpec) -> AnnotationBoundryEnv
```

```python
AnnotationBoundryAdaptor.build_train_env(batch_size: int, seed: int, **kwargs) -> AnnotationBoundryEnv
```

```python
# env_num is the number of eval cases to run. Feel like it should be called eval_num?
# Also env_num = 0 means run all cases, which I think is pretty stupid and unclear
AnnotationBoundryAdaptor.build_eval_env(env_num: int, split: BatchSplit, seed: int, **kwargs) -> AnnotationBoundryEnv
```

```python 
AnnotationBoundryAdaptor.get_task_types() -> list[literal["annotation_boundry"]]
```

```python
# Note that this function must also write a trajectory to disk at `<out_dir>/predictions/<id>/conversation.json` to be read by reflect
# Feel like it's not great that this happens in this function, but can't really change it?
AnnotationBoundryAdaptor.rollout() -> list[RolloutResult] # See if we need to make a unique rollout result later
```

`evaluator.py`
```python
from difflib import SequenceMatcher

# evalutator should ONLY be responsible for returning the metric
def intersection_over_union(prediction: str, ground_truth: str) -> UnitInterval:
    intersection = SequenceMatcher(None, prediction, ground_truth).find_longest_match()
    union = len(prediction) + len(ground_truth) - intersection.size
    return intersection.size / union if union else 0.0

def property_coverage_rate(prediction: str, ground_truth: str) -> UnitInterval:
    pass
```

`rollout.py`
```python
run_batch(env_manager, ...) -> list[RolloutResult]  # pretty sure we need a RolloutResult type
    items: list[dict] = env_manager
    return _rollout_one(item: dict, ...)
```

```python
from skillopt.model import chat_target

_rollout_one(item: dict, ...) -> RolloutResult
    prediction, _usage = chat_target(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
    )
    # 'hard' score routes trajectory to success vs. error analyst; `soft` is the actual score
    soft, hard = score(prediction, item.get("ground_truth", ""))
    
    # ... more code above and below but ignore the details
    return RolloutResult(...)
```


```python
#  score should handle the evaluator result threshold, not the evaluator 
_score(prediction: str, ground_truth: str) -> tuple[SoftScore, HardScore]:
    soft = intersection_over_union(prediction, ground_truth) # IoU score between 0-1
    hard = soft > 0.5 # arbitrarily set to 0.5 threshold
    return soft, hard
```


NOTES:
dataloader.load_split_data -> dataloader.build_train/eval_batch -> dataloader.build_env_from_batch -> adaptor.build_train/eval_env