"""ReflACT Engine -- the training runner.

Analogous to the Runner in mmengine: orchestrates the full training pipeline
including rollout, gradient computation, aggregation, optimization, and
evaluation.
"""
from skillopt.engine.trainer import ReflACTTrainer  # noqa: F401

__all__ = ["ReflACTTrainer"]
