"""Merge gold (directly annotated) and silver (report-derived) labels into unified
training targets with per-label sample weights.

Only 58 of 4,407 training studies (1.3%) carry direct per-condition labels -- see
PLAN.md. For everything else, targets come from the LLM-based report-reading dataset
(pilkwang/rsna-knee-llm-labels), which we validated against the 58 gold studies at
0.87 macro AUC. Gold studies keep their hard 0/1 labels at elevated weight, since
they're the only labels read from images rather than text; silver studies use the
LLM's graded score as a soft BCE target, weighted by its per-label confidence.
"""
from __future__ import annotations

import pandas as pd

from rsna_knee.metrics import LABEL_COLUMNS

GOLD_WEIGHT = 3.0


def build_training_targets(
    train_labels: pd.DataFrame,
    silver_labels: pd.DataFrame,
    label_columns: list[str] = LABEL_COLUMNS,
    gold_weight: float = GOLD_WEIGHT,
) -> pd.DataFrame:
    """Return one row per training study with `StudyInstanceUID`, a soft target and a
    sample weight per label column, and `is_gold` indicating direct annotation.

    Studies with no row in `silver_labels` and no gold label fall back to target=0.5,
    weight=0.0 for every label (contributes nothing to the loss) rather than being
    dropped, so the caller doesn't need to special-case missing joins.
    """
    gold_mask = train_labels[label_columns].notna().any(axis=1)
    merged = train_labels[["StudyInstanceUID"]].merge(
        silver_labels, on="StudyInstanceUID", how="left", suffixes=("", "_silver")
    )

    out = {"StudyInstanceUID": train_labels["StudyInstanceUID"], "is_gold": gold_mask}
    for col in label_columns:
        conf_col = f"{col}__conf"
        target = merged[col].astype(float)
        weight = merged[conf_col].astype(float) if conf_col in merged else pd.Series(0.0, index=merged.index)

        # unmatched studies (no silver row at all): neutral target, zero weight
        no_silver = merged[col].isna() if col in merged else pd.Series(True, index=merged.index)
        target = target.where(~no_silver, 0.5)
        weight = weight.where(~no_silver, 0.0)

        gold_target = train_labels[col].astype(float)
        target = target.where(~gold_mask.to_numpy(), gold_target)
        weight = weight.where(~gold_mask.to_numpy(), gold_weight)

        out[col] = target
        out[f"{col}__weight"] = weight

    return pd.DataFrame(out)
