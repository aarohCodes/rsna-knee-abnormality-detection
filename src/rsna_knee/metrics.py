"""Competition metric: macro-averaged ROC AUC across the 12 label columns.

Final Score = (1/12) * sum_i AUC_i

Matches the competition's stated formula exactly (confirmed by the competition's own
Evaluation page): the unweighted mean of the per-label ROC AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def per_label_auc(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, label_columns: list[str] = LABEL_COLUMNS
) -> dict[str, float]:
    """Compute ROC AUC for each label column independently.

    Rows where a label's ground truth is NaN are excluded from that label's AUC only
    (a study missing one label can still contribute to the other eleven). A label with
    fewer than two ground-truth classes among its non-NaN rows has no defined AUC and is
    returned as NaN rather than raising, so a small/degenerate CV fold doesn't crash the
    whole evaluation.
    """
    scores: dict[str, float] = {}
    for col in label_columns:
        truth = y_true[col].to_numpy(dtype=float)
        mask = ~np.isnan(truth)
        truth = truth[mask]
        if len(np.unique(truth)) < 2:
            scores[col] = float("nan")
            continue
        pred = y_pred[col].to_numpy(dtype=float)[mask]
        scores[col] = float(roc_auc_score(truth, pred))
    return scores


def macro_auc(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, label_columns: list[str] = LABEL_COLUMNS
) -> float:
    """The competition's Final Score: mean of the 12 per-label AUCs.

    Labels with an undefined (NaN) AUC are excluded from the mean rather than counted as
    zero, since the competition's own test set is fully annotated and this situation only
    arises in small/imbalanced local CV folds.
    """
    scores = per_label_auc(y_true, y_pred, label_columns)
    values = [v for v in scores.values() if not np.isnan(v)]
    if not values:
        return float("nan")
    return float(np.mean(values))
