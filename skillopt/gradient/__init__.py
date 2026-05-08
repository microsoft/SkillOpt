"""ReflACT Gradient -- trajectory analysis and patch generation.

Analogous to gradient computation in neural network training: analyzes
minibatch rollout trajectories to produce skill-edit patches (the "gradient"
that drives skill updates).

Modules
-------
- reflect: minibatch trajectory analysis (gradient computation)
- aggregate: hierarchical patch merging (gradient aggregation)
- deep_probe: diagnostic probe generation (gradient probing)
"""
from skillopt.gradient.reflect import (  # noqa: F401
    run_minibatch_reflect,
)
from skillopt.gradient.aggregate import merge_patches  # noqa: F401
from skillopt.gradient.deep_probe import generate_deep_probe_instruction  # noqa: F401
