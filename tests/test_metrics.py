import numpy as np
import pandas as pd

from rsna_knee.metrics import LABEL_COLUMNS, macro_auc, per_label_auc


def _make_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame(values)


def test_perfect_predictions_score_one():
    n = 20
    rng = np.random.default_rng(0)
    truth = {col: rng.integers(0, 2, size=n).astype(float) for col in LABEL_COLUMNS}
    y_true = _make_frame(truth)
    # perfect separation: predicted score equals truth exactly
    y_pred = _make_frame({col: truth[col].copy() for col in LABEL_COLUMNS})
    assert macro_auc(y_true, y_pred) == 1.0


def test_random_predictions_score_near_half():
    n = 4000
    rng = np.random.default_rng(1)
    truth = {col: rng.integers(0, 2, size=n).astype(float) for col in LABEL_COLUMNS}
    y_true = _make_frame(truth)
    y_pred = _make_frame({col: rng.random(n) for col in LABEL_COLUMNS})
    score = macro_auc(y_true, y_pred)
    assert 0.45 < score < 0.55


def test_nan_ground_truth_excluded_from_that_label_only():
    y_true = _make_frame(
        {
            "ACL": [1.0, 0.0, 1.0, 0.0],
            **{col: [1.0, 0.0, np.nan, np.nan] for col in LABEL_COLUMNS if col != "ACL"},
        }
    )
    y_pred = _make_frame({col: [0.9, 0.1, 0.8, 0.2] for col in LABEL_COLUMNS})
    scores = per_label_auc(y_true, y_pred)
    # ACL has 4 fully-labeled, well-separated rows -> perfect AUC
    assert scores["ACL"] == 1.0
    # every other label has exactly 2 non-NaN rows (one of each class), well separated
    for col in LABEL_COLUMNS:
        if col != "ACL":
            assert scores[col] == 1.0


def test_single_class_label_returns_nan_not_error():
    y_true = _make_frame({col: [1.0, 1.0, 1.0] for col in LABEL_COLUMNS})
    y_pred = _make_frame({col: [0.9, 0.8, 0.7] for col in LABEL_COLUMNS})
    scores = per_label_auc(y_true, y_pred)
    assert all(np.isnan(v) for v in scores.values())
    assert np.isnan(macro_auc(y_true, y_pred))


def test_matches_manual_macro_formula():
    n = 200
    rng = np.random.default_rng(2)
    truth = {col: rng.integers(0, 2, size=n).astype(float) for col in LABEL_COLUMNS}
    y_true = _make_frame(truth)
    y_pred = _make_frame({col: rng.random(n) for col in LABEL_COLUMNS})
    per_label = per_label_auc(y_true, y_pred)
    manual_mean = sum(per_label.values()) / len(LABEL_COLUMNS)
    assert abs(macro_auc(y_true, y_pred) - manual_mean) < 1e-12
