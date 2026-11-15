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
import torch
from typing import Union, Dict
import pandas as pd

from .model import get_model, eval_model, INPUT_DIM
from .effort import build_feature_vector, FEATURE_ORDER

class StudyPredictor:
    def __init__(self, weights_path: str = "nn_weights.pt", scaler_path: str = "scaler.npz", input_dim: int = INPUT_DIM):
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
        score_arr = eval_model(self.model, scaled)  # returns np array [1]
        score = float(score_arr[0])

        # Monotonicity safety: ensure not negative etc. Already clipped in eval_model
        return score

    def predict_batch(self, profiles, plans):
        """Helper for evaluating 30 profiles requirement."""
        return [self.predict_score(p, pl) for p, pl in zip(profiles, plans)]

# convenience singleton for search module: from nn.predictor import predictor
# Search can do: from nn.predictor import StudyPredictor; predictor = StudyPredictor()
