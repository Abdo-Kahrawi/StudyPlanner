"""NN tests incl. the monotonicity check (marks must not go DOWN with more effort). Owner: Person A."""
import os

import numpy as np
import pytest

from common.examples import example_profile
from nn.effort import FEATURE_ORDER, features_for_plan


def test_effort_adds_clicks_only_to_planned_assignment():
    p = example_profile()
    base, more = features_for_plan(p, (0, 0, 0)), features_for_plan(p, (0, 2, 0))
    click = [i for i, f in enumerate(FEATURE_ORDER) if f.startswith("group_")]
    assert base.shape == (3, len(FEATURE_ORDER))
    assert np.allclose(np.delete(more, click, axis=1), np.delete(base, click, axis=1))
    assert more[1, click].sum() - base[1, click].sum() == pytest.approx(2 * p.clicks_per_session)
    assert np.allclose(more[[0, 2]], base[[0, 2]])


def test_average_predictor_ignores_effort():
    from nn.predictor import AverageMarkPredictor
    p, avg = example_profile(), AverageMarkPredictor()
    assert np.allclose(avg.predict(p, (0, 0, 0)), 60.0)
    assert np.allclose(avg.predict(p, (5, 5, 5)), avg.predict(p, (0, 0, 0)))


def test_monotonicity():
    pytest.importorskip("torch")
    from nn.predictor import DEFAULT_SCALER, DEFAULT_WEIGHTS, MarkPredictor
    if not (os.path.exists(DEFAULT_WEIGHTS) and os.path.exists(DEFAULT_SCALER)):
        pytest.skip("no trained weights; run python -m nn.train")
    p, nn = example_profile(), MarkPredictor()
    marks = np.array([nn.predict(p, (n, n, n)) for n in range(11)])
    assert np.all(np.diff(marks, axis=0) >= -0.5), "NN marks drop when sessions are added"


def test_monotone_model_never_drops_with_more_clicks():
    torch = pytest.importorskip("torch")
    from nn.effort import CLICK_COLS
    from nn.model import MonotoneScoreNet
    torch.manual_seed(0)
    click = [FEATURE_ORDER.index(c) for c in CLICK_COLS]
    model = MonotoneScoreNet(click, input_dim=len(FEATURE_ORDER)).eval()
    x = torch.randn(64, len(FEATURE_ORDER))
    more = x.clone()
    more[:, click] += torch.rand(64, len(click)) * 3
    with torch.no_grad():
        assert torch.all(model(more) >= model(x) - 1e-5)
