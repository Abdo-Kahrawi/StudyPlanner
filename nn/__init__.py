"""
nn package for P1 Assessment-Aware Study Planner.
Allows `from nn.model import ...`, `from nn.predictor import ...`, etc.
"""
from .effort import EffortTranslator, build_feature_vector, apply_effort, features_for_plan

# model.py and predictor.py need torch. Import them lazily so that `import nn.effort`,
# the planner and the CI job (which does not install torch) work without it.
_LAZY = {"StudyScoreNet": "model", "get_model": "model", "eval_model": "model",
         "StudyPredictor": "predictor", "MarkPredictor": "predictor",
         "AverageMarkPredictor": "predictor"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "StudyScoreNet",
    "get_model",
    "eval_model",
    "EffortTranslator",
    "build_feature_vector",
    "apply_effort",
    "StudyPredictor",
    "MarkPredictor",
    "AverageMarkPredictor",
    "features_for_plan",
]
