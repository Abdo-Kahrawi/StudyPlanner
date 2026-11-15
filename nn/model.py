"""
model.py - Neural architecture for OULAD score prediction.
P1: Assessment-Aware Study Planner

Input: ~18 features (prev_score, cumulative_avg, weight, weeks_since_start,
       4 weeks x 3 click groups =12, assessment_type 2 one-hot)
Target: assessment score 0-100

Architecture: small MLP with dropout + LayerNorm for stability.
Includes eval() wrapper used by predictor.py
"""
import torch
import torch.nn as nn
from typing import Union
import numpy as np

INPUT_DIM = 18  # matches your train_data.csv (excluding id_* and target)
# If your final feature set changes, update this or pass input_dim to get_model

class StudyScoreNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )
        # output 0-100 via scaling; we keep linear and clip in eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B] -> score

def get_model(input_dim: int = INPUT_DIM, **kwargs) -> StudyScoreNet:
    return StudyScoreNet(input_dim=input_dim, **kwargs)

def eval_model(model: nn.Module, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Eval wrapper required by spec.
    model: trained StudyScoreNet
    x: np.ndarray [N, D] or [D] or torch.Tensor
    returns: np.ndarray scores clipped to [0,100]
    """
    model.eval()
    with torch.no_grad():
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x.astype(np.float32))
        else:
            x_t = x.float()
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        preds = model(x_t).cpu().numpy()
        preds = np.clip(preds, 0.0, 100.0)
        return preds

def count_params(model):
    return sum(p.numel() for p in model.parameters())
