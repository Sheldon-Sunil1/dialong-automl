"""Training sub-package for DiaLong-AutoML — Phase 5.

Provides the canonical end-to-end training pipeline.

Usage::

    from dialong_automl.training import TrainingPipeline
    from dialong_automl.config import load_config

    cfg = load_config("configs/demo_fast.yaml")
    pipeline = TrainingPipeline(cfg)
    result = pipeline.run()
"""

from dialong_automl.training.pipeline import TrainingPipeline, PipelineResult

__all__ = ["TrainingPipeline", "PipelineResult"]
