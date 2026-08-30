"""Merge gold (directly annotated) and silver (report-derived) labels into unified
training targets with per-label sample weights.

Only 58 of 4,407 training studies (1.3%) carry direct per-condition labels -- see
PLAN.md. For everything else, targets come from a public, report-derived label
dataset. Several such datasets exist publicly on Kaggle for this competition; each
was validated against the 58 gold studies before use (see PLAN.md's label-source
comparison). Two candidates (yunusgmsoy's "4-source-merged" and tasmeemreza's
"refined" datasets) turned out to bitwise-match the gold labels exactly on all 58
studies -- i.e. they'd baked the real answers back in rather than predicting them --
so despite scoring a suspicious 1.0 AUC, they were excluded as uninformative about
real quality on the ~4,349 studies that actually matter. stevenleehans's
llm_labels_v4_blend.csv passed that leakage check (no exact match, no degenerate/
binary-only values) and scored higher than the original baseline (0.893 vs 0.87
macro AUC against gold) -- it's the primary silver source now. Gold studies keep
their hard 0/1 labels at elevated weight, since they're the only labels read from
images rather than text; silver studies use the graded score as a soft BCE target,
weighted by per-label confidence.
"""
from __future__ import annotations

import pandas as pd

from rsna_knee.metrics import LABEL_COLUMNS

GOLD_WEIGHT = 3.0


def add_derived_confidence(
    silver_labels: pd.DataFrame, label_columns: list[str] = LABEL_COLUMNS
) -> pd.DataFrame:
    """Add `{col}__conf` columns for a silver-label source that provides graded
    scores but no native confidence (e.g. stevenleehans's llm_labels_v4_blend.csv,
    unlike pilkwang's dataset which ships its own confidence column).

    Confidence is derived from how far the graded score sits from the uninformative
    midpoint (0.5): confidence = |score - 0.5| * 2, so a score of exactly 0.5
    (maximally ambiguous) gets zero weight and a saturated 0 or 1 score gets full
    weight -- same spirit as the sources that do ship real confidence.
    """
    out = silver_labels.copy()
    for col in label_columns:
        out[f"{col}__conf"] = (out[col] - 0.5).abs() * 2
    return out


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
