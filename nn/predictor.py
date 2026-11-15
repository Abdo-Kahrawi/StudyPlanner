"""
predictor.py - Clean OOP wrapper for Search engine.
Spec: predictor.predict_score(student_profile, proposed_plan)
  - loads saved weights
  - constructs vector using effort.py
  - passes to eval function in model.py
  - search engine does NOT interact beyond this class
"""
import os
import numpy as np
from typing import Union, Dict
import pandas as pd

from common import config
from common.types import Plan, StudentProfile
from .effort import build_feature_vector, features_for_plan, FEATURE_ORDER

# Where nn/train.py saves by default.
DEFAULT_WEIGHTS = str(config.WEIGHTS_DIR / "nn_weights.pt")
DEFAULT_SCALER = str(config.WEIGHTS_DIR / "scaler.npz")


class StudyPredictor:
    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, scaler_path: str = DEFAULT_SCALER, input_dim: int = len(FEATURE_ORDER)):
        from .model import get_model  # needs torch; imported here so the module loads without it
        self.weights_path = weights_path
        self.scaler_path = scaler_path
        self.model = get_model(input_dim=input_dim)
        self.scaler_mean = None
        self.scaler_scale = None
        self.feature_order = FEATURE_ORDER
        self._load()

    def _load(self):
        # load scaler if exists
        if os.path.exists(self.scaler_path):
            data = np.load(self.scaler_path, allow_pickle=True)
            self.scaler_mean = data["mean"]
            self.scaler_scale = data["scale"]
            # feature_order check
            if "feature_order" in data:
                self.feature_order = list(data["feature_order"])
        else:
            print(f"[Predictor] WARNING: scaler {self.scaler_path} not found, using no scaling. Train first.")

        # load weights
        if os.path.exists(self.weights_path):
            import torch
            state = torch.load(self.weights_path, map_location="cpu")
            try:
                self.model.load_state_dict(state)
            except RuntimeError as e:
                print(f"[Predictor] Weight load mismatch: {e}")
        else:
            print(f"[Predictor] WARNING: weights {self.weights_path} not found, using random init. Train first.")

        self.model.eval()

    def _scale(self, vec: np.ndarray) -> np.ndarray:
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (vec - self.scaler_mean) / (self.scaler_scale + 1e-8)
        return vec

    def predict_score(self, student_profile: Union[Dict, pd.Series, np.ndarray], proposed_plan: Union[int, Dict] = 0) -> float:
        """
        Main API for search.
        student_profile: dict with keys from FEATURE_ORDER + optionally id_assessment
                         or np array [D] or pd.Series
        proposed_plan: int = extra study sessions for this assessment
                       OR dict {assessment_id: sessions} for multi-assignment search (will extract if possible)
        returns: float score 0-100
        """
        # 1. Build vector using effort.py
        eff_vector = build_feature_vector(student_profile, proposed_plan)  # [D]

        # 2. Scale using training scaler
        scaled = self._scale(eff_vector)

        # 3. Pass to eval function in model.py
        from .model import eval_model
        score_arr = eval_model(self.model, scaled)  # returns np array [1]
        score = float(score_arr[0])

        # Monotonicity safety: ensure not negative etc. Already clipped in eval_model
        return score

    def predict_batch(self, profiles, plans):
        """Helper for evaluating 30 profiles requirement."""
        return [self.predict_score(p, pl) for p, pl in zip(profiles, plans)]

# convenience singleton for search module: from nn.predictor import predictor
# Search can do: from nn.predictor import StudyPredictor; predictor = StudyPredictor()


# ---------------------------------------------------------------------------
# common.types.Predictor implementations used by integration/ and eval/.
# ---------------------------------------------------------------------------
class MarkPredictor:
    """Trained NN behind the shared contract: predict(profile, plan) -> marks per assignment.

    Raises FileNotFoundError when no trained weights exist (run `python -m nn.train` first);
    eval.harness.load_predictor then falls back to the stub predictor.
    """

    name = "nn"

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, scaler_path: str = DEFAULT_SCALER):
        for path in (weights_path, scaler_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found. Train first: python -m nn.train")
        self.inner = StudyPredictor(weights_path, scaler_path)

    def predict(self, profile: StudentProfile, plan: Plan) -> np.ndarray:
        from .model import eval_model
        X = self.inner._scale(features_for_plan(profile, plan))
        return eval_model(self.inner.model, X).astype(float)


class AverageMarkPredictor:
    """Baseline: the student's previous average mark, ignores effort (same rule as nn/train.py:
    cumulative_avg_score if > 0, else prev_score if > 0, else `default`)."""

    name = "avg"

    def __init__(self, default: float = 55.0):
        self.default = default

    def predict(self, profile: StudentProfile, plan: Plan) -> np.ndarray:
        h = profile.history
        out = []
        for a in profile.assignments:
            mark = self.default
            for key in (f"{a.id}/cumulative_avg_score", f"{a.id}/prev_score",
                        "cumulative_avg_score", "prev_score", "prev_score_mean"):
                if float(h.get(key, 0.0)) > 0:
                    mark = float(h[key])
                    break
            out.append(mark)
        return np.clip(np.array(out, dtype=float), 0, 100)
