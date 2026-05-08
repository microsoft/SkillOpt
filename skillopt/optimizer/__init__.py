"""ReflACT Optimizer -- skill update operations.

Analogous to the optimizer in neural network training: applies the computed
"gradient" (patches) to the current skill document to produce an updated
candidate skill.

Modules
-------
- skill: edit application (optimizer.step() / parameter update)
- clip: edit ranking and selection (gradient clipping)
- meta_reflect: epoch-level macro refinement (momentum)
- slow_update: longitudinal comparison and guidance (EMA / regularization)
"""
from skillopt.optimizer.skill import apply_edit, apply_patch  # noqa: F401
from skillopt.optimizer.clip import rank_and_select  # noqa: F401
