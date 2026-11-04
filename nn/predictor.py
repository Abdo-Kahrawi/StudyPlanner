"""Predictor implementations used by the planner. Owner: Person A.

Both classes follow common.types.Predictor.
"""
import numpy as np

from common.types import Plan, StudentProfile


class MarkPredictor:
    name = "nn"

    def __init__(self, weights_path=None):
        self.weights_path = weights_path

    def predict(self, profile: StudentProfile, plan: Plan) -> np.ndarray:
        raise NotImplementedError("Person A, Week 3-5")


class AverageMarkPredictor:
    """Baseline: student's previous average mark, ignores effort."""

    name = "avg"

    def predict(self, profile: StudentProfile, plan: Plan) -> np.ndarray:
        raise NotImplementedError("Person A, Week 3")
