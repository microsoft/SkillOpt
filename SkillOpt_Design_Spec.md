# SkillOpt Design Spec

## __Problem Statement:__
Our current SA pipeline uses an LLM to perform 3 tasks, `boundary detection for annotations & initial property assignment`, `verification for whether the annotations are correct`, and `extracting properties from annotations`.  These tasks all use prompts to guide the LLM into getting the correct answer, which often takes iterative manual tuning to achieve acceptable results. 


__This raises 2 main problems:__

1. __Difficult to Tune__: When tuning LLM prompts, it is often not transparent as to what change will cause what outcome. This means that we are often left blindly making changes, without any concrete metric of whether the changes we make are actually beneficial or not. 
2. __Time Consuming__: Ties into the previous issue. The human review process takes a long time which significantly slows down engineering output and productivity.  

This project looks only to fix these problems for `boundary detection for annotations & initial property assignment`, although the results should be generalizable such that it is easy to apply to the other 2 tasks


## __Solution__
To solve both of these issues, we plan to use SkillOpt. SkillOpt is a repo that allows us to tune prompts by following a similar loop as training a Deep Learning model. See [the deep learning analogy](../docs/guide/dl-analogy.md) for a clear comparison.

We also plan on following Alistair Cockburn's [Hexagonal Architecture](https://jmgarridopaz.github.io/content/hexagonalarchitecture.html) to ensure the pipeline's core depends only on the ports it owns, meaning future swaps with other interfaces can be done much easier using adapters. This also keeps the domain logic free of infrastructure concerns, so changes to how results are persisted don't ripple into how they're evaluated. A few other key factors to keep in mind when working with hexagonal architecture are:
1. Each port needs a test adapter. On the driven side that means an implementation and on the driving side, a caller.
2. Hexagonal Architecture reduces techical debt as more maintainability = less technical debt.
3. Technology evolves more frequently than business logic does. So, in applications where the business logic is tied to technology, you can't do technology changes without touching business logic which is annoying/bad. Hexagonal architecture avoids this by simply swapping the adapter.
4. When you start developing and coding, you can focus just on business logic, deferring decisions about which framework and technology you are going to use. You can choose a technology later, and code an adapter for it.

### __General Idea__

We use this general loop:

For each epoch, for each step:
1. Harness executes tasks with current best prompt
2. __trajectories__ and __RolloutResults__ are created containing data about task execution
3. __Optimizer__ (a LLM Model) analyzes __trajectories__ and __RolloutResults__, then creates a __patch__, an object containing suggested edits
4. Smaller __patches__ are merged into one patch 
5. The edits in a __patch__ will be ranked, and then clipped to be <= __learning rate__
6. Apply the top edits in the __patches__ to the prompt
7. Validate the new prompt with the __validation set__. Accept if score is higher than best score, else reject

This solves both of our problems: 
- By defining a clear metric, we can see how each prompt change affects the task score
- SkillOpt doesn't require the constant human review and edits manually editing prompts do, freeing up time


## __In Scope:__ 
1. Use SkillOpt to **tune the prompt** in the SA pipeline which handles `boundary detection & initial property assignment`
2. Manually reviewing annotations to create a ground-truth dataset
3. Handling of partial credit (a field extracted but slightly malformed shouldn't score the same as a missing field)

## __Out of Scope:__
1. Use SkillOpt to **tune the prompt** in the SA pipeline which handles `validation` and `property enrichment` 
2. **Streamline client onboarding with SkillOpt** so manual review and tuning of definitions is no longer needed.
   Current prompt-tuning process for a new client:
   1. Copy the most recent client definition matching the concept (minutes, agendas, etc.)
   2. Tailor the definition to the new client's files.
   3. Run a test batch of 8–10 files and export the output to an Excel sheet.
   4. Adam manually reviews and gives feedback on each row.
   5. Feedback goes back to Austin or Femina to tweak the definition.
   6. Repeat 3–5 until recall ≥ 92%.
   
## __File Structure:__
We are forking the SkilOpt repository. The only files we edit are under `configs/`, `data/`, and `skillopt/envs/` 
```
SkillOpt/
├── ...
├── configs/
│   ├── ...
│   └── annotation_detection/
│       └── default.yaml
├── data/
│   ├── ...
│   └── annotation_detection_split/
│       ├── test
|       ├── train
│       ├── val
|       └── split_manifest.json
├── skillopt/
│   ├── ...
│   └── envs/
│       ├── ...
|       ├── _common/
|       │   ├── adapters/
|       │   │   ├── json_file_result_writer.py
|       │   │   └── json_item_loader.py
|       │   └── ports/
|       │       ├── harness.py
|       │       ├── item_loader.py
|       │       └── result_writer.py
|       ├── _annotation_common/
|       │   └── adapters/
|       │       └── sa_pipeline_harness.py
│       └── _annotation_detection/
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
└── SkillOpt_Design_Spec.md
```


## __Domain Types:__
```python
type UnitInterval = float
type SoftScore = float
type HardScore = int

type Seed = int
type BatchSize = int
type FilePath = str
type Skill = str
type Chunk = str
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
    id: str  # required
    soft: SoftScore  # required
    hard: HardScore  # required
    predicted_answer: str
    task_description: str  # makes llm accurate
    question: str
    reference_text: str
    task_type: str  # required for minibatch
    target_system_prompt: str
    target_user_prompt: str
    n_turns: int  # cosmetic; for prints
```

```python
class DatasetType(StrEnum):
    MINUTE = "minute"
    AGENDA = "agenda"...
```

```python
class ApplicationType(StrEnum):
    CVPC_BC = "cvpc_bc"
    MF_CT = "mf_ct"...
```


## __Use Case Types:__
### Ports & Adaptors (all examples are driven)
```python
class Harness[InputT, LLMOutputT](ABC):
    @abstractmethod
    def forward(self, input: InputT, skill: Skill) -> LLMOutputT
```

```python
# Find the Chunk and SAOutput Type. 
class SAPipelineHarness(Harness[Chunk, SAOutput]):
    @override
    def forward(self: Harness, input: Chunk, skill: Skill) -> SAOutput
```
<br/><br/>

```python
class ItemLoader[ItemT](ABC):
    @abstractmethod
    def get_data(self) -> list[ItemT]
```

```python
class JSONItemLoader(ItemLoader):
    path: FilePath

    def __init__(self, path: FilePath):
        # Set Filepath here to use in get_data
        pass
    
    @override
    def get_data(self) -> list[dict[str, object]]
```

<br/><br/>

```python
class ResultWriter[ResultT](ABC):
    @abstractmethod
    def write(self, key: str, result: ResultT) -> None ... # return None since we are writing to files/db's
```

``` python
class JsonFileResultWriter[ResultT](ResultWriter[ResultT]):
    def __init__(self, out_dir: Path, filename: str) -> None:
        self._out_dir = out_dir
        self._filename = filename

    @override
    def write(self, key: str, result: ResultT) -> None:
        path = self._out_dir / "predictions" / key / self._filename
        ...
``` 
<br/><br/>

``` python
class CosmosFileResultWriter[ResultT](ResultWriter[ResultT]):
    def __init__(self, container: ContainerProxy, run_id: str) -> None:
        self._container = container
        self._run_id = run_id

    @override
    def write(self, key: str, result: ResultT) -> None:
        doc: CosmosWrittenResult = {"id": key, "runId": self._run_id, "payload": result}
        self._container.upsert_item(doc)
``` 
<br/><br/>

``` python
class MockResultWriter[ResultT: BaseModel](ResultWriter[ResultT]):
    def __init__(self) -> None:
        self.written: MockWrittenResult = {}

    @override
    def write(self, key: str, result: ResultT) -> None:
        self.written[key] = result
``` 
<br/><br/>

### Skill Opt 
```python
AnnotationDetectionAdaptor(EnvAdapter)  # EnvAdapter is defined by SkillOpt
```

```python
AnnotationDetectionDataLoader(SplitDataLoader)
```

```python 
@dataclass 
AnnotatedChunk():
    chunk: str 
    ground_truth_span_indexs: list[tuple[int, int]]
    ground_truth_properties: list[Property]
```

``` python
# This is how annotated chunks look like in the new domain types. Discuss with Doc to turn the list type into list[AnnotatedSpan]
class AgendaAnnotatedChunk(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    chunk_index: int
    chunk_text: Chunk
    annotated_spans: list[AgendaItemAnnotatedSpan]


class MotionAnnotatedChunk(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    chunk_index: int
    chunk_text: Chunk
    annotated_spans: list[MotionAnnotatedSpan]


# ... and more [concepts]AnnotatedChunks for all concepts
```

```python 
# Returned from SplitDataLoader.load_split_items
@dataclass 
AnnotationDetectionTask():
    id: str
    annotated_chunk: AnnotatedChunk
    # These properties will be used for task_type
    datasource: DatasourceType
    application: ApplicationType
    year: YearType

```


```python
@dataclass
AnnotationDetectionBatchSpec(BatchSpec): 
    phase: BatchPhase
    split: BatchSplit
    seed: Seed
    batch_size: BatchSize
    payload: list[AnnotationDetectionTask]
    metadata: dict[str, Any] = field(default_factory=dict)

```

```python
# Enforce that AnnotationDetectionEnv should be = list[AnnotationDetectionBatchSpec.payload]
AnnotationDetectionEnv = NewType("AnnotationDetectionEnv", list[AnnotationDetectionTask]): 
```

## __Files & Functions:__
`dataloader.py`
```python
AnnotationDetectionDataLoader.load_split_items(split_path: FilePath) -> list[AnnotationDetectionTask]
```

`adaptor.py`
```python
# Use implementation from template_env
AnnotationDetectionAdaptor.__init__() -> None
```

```python
# Use implementation from template_env
AnnotationDetectionAdaptor.setup() -> None
```

```python
AnnotationDetectionAdaptor.get_dataloader() -> AnnotationDetectionDataLoader
```

```python
AnnotationDetectionAdaptor.build_env_from_batch(BatchSpec: AnnotationDetectionBatchSpec) -> AnnotationDetectionEnv
```

```python
AnnotationDetectionAdaptor.build_train_env(batch_size: BatchSize, seed: Seed, **kwargs) -> AnnotationDetectionEnv
```

```python
# env_num is the number of eval cases to run. Feel like it should be called eval_num?
# Also env_num = 0 means run all cases, which I think is pretty stupid and unclear
AnnotationDetectionAdaptor.build_eval_env(env_num: Count, split: BatchSplit, seed: Seed, **kwargs) -> AnnotationDetectionEnv
```

```python 
AnnotationDetectionAdaptor.get_task_types() -> list[literal["annotation_detection"]]
```

```python
# Note that this function must also write a trajectory to disk at `<out_dir>/predictions/<id>/conversation.json` to be read by reflect
# Feel like it's not great that this happens in this function, but can't really change it?
AnnotationDetectionAdaptor.rollout(
    env_manager: AnnotationDetectionEnv,
    skill_content: Skill,
    out_dir: FilePath,) -> list[RolloutResult] 
```

`evaluator.py`
```python
# evalutator should ONLY be responsible for returning the metric

from difflib import SequenceMatcher


class SpanNotFoundError(ValueError):
    """A predicted span could not be located in its chunk."""


def find_boundary(chunk_text: str, span: str) -> tuple[int, int]:
    if (start := chunk_text.find(span)) == -1:
        raise SpanNotFoundError()
    return (start, start + len(span))


# for boundary overlap metrics
def intersection_over_union(prediction: list[int], ground_truth: list[int]) -> UnitInterval:
    pass


# for property correctness metric
def property_coverage_rate(prediction: str, ground_truth: str) -> UnitInterval:
    pass  # copy code from performance test of SA pipeline
```

`rollout.py`
```python
run_batch(env_manager, ...) -> list[RolloutResult]  # pretty sure we need a RolloutResult type
    items: list[dict] = env_manager
    results = [_rollout_one(item: dict, ...) for item in items]
    write_results_to_json(results) # need to write result per batch
    return results
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
    hard, soft = _score(prediction, item.get("ground_truth", ""))
    
    # ... more code above and below but ignore the details
    result = RolloutResult(...)
    write_results_to_json(result) # need to write result per item
    return result
```


```python
#  score should handle the evaluator result threshold, not the evaluator 
_score(prediction: str, ground_truth: str) -> tuple[HardScore, SoftScore]:
    property_metric = property_coverage_rate()
    iou_metric = intersection_over_union(prediction, ground_truth)

    soft = min(property_metric, iou_metric)
    hard = soft > 0.5 # arbitrarily set to 0.5 threshold
    
    return hard, soft
```








Notes about Hexagonal Architecture:


```
What exactly a port is and isn't is largely a matter of taste. At the one extreme, every use case could be given its own port, producing hundreds of ports for many applications. Alternatively, one could imagine merging all primary ports and all secondary ports so there are only two ports, a left side and a right side.

Neither extreme appears optimal.

The weather system described in the Known Uses has four natural ports: the weather feed, the administrator, the notified subscribers, the subscriber database. A coffee machine controller has four natural ports: the user, the database containing the recipes and prices, the dispensers, and the coin box. A hospital medication system might have three: one for the nurse, one for the prescription database, and one for the computer-controller medication dispensers.

It doesn't appear that there is any particular damage in choosing the "wrong" number of ports, so that remains a matter of intuition. My selection tends to favor a small number, two, three or four ports, as described above and in the Known Uses.
```



