"""Grouped, multilabel-stratified cross-validation folds.

Two leaks matter here (see PLAN.md section on the baseline notebook's CV design):

1. Grouping must be by report text, not just StudyInstanceUID: some reports are
   byte-identical templates shared across studies (e.g. unremarkable-knee boilerplate),
   and every study sharing a report gets the same derived silver label vector, so
   splitting such a group across train/holdout scores the model on a target whose
   source it was trained on.
2. `iterstrat`'s MultilabelStratifiedKFold has no notion of groups, so groups are
   collapsed to one representative row each (label = elementwise max across the
   group, treating NaN as 0 for this stratification-only purpose), stratified at the
   group level, and the resulting fold assignment is broadcast back to every member.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from rsna_knee._vendor.iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from rsna_knee.metrics import LABEL_COLUMNS


def report_hash(report: str) -> str:
    """Stable hash of normalized report text, used as the CV group key alongside
    StudyInstanceUID so duplicate/template reports never split across folds."""
    normalized = " ".join(str(report).split()).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def make_grouped_multilabel_folds(
    df: pd.DataFrame,
    group_col: str,
    label_columns: list[str] = LABEL_COLUMNS,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.Series:
    """Return a fold index (0..n_splits-1) per row of `df`, aligned to `df.index`.

    Every row sharing the same `group_col` value is guaranteed to land in the same
    fold. Stratification targets are built at the group level from the elementwise max
    of each group's label columns (NaN treated as 0 for this purpose only -- this
    never touches the actual training targets, just which fold a group lands in).
    """
    group_labels = (
        df[[group_col] + label_columns]
        .fillna(0.0)
        .groupby(group_col, sort=False)[label_columns]
        .max()
    )
    # binarize at 0.5 -- stratification only needs a rough positive/negative signal,
    # not full precision, and MultilabelStratifiedKFold expects discrete-ish labels.
    y = (group_labels.to_numpy() >= 0.5).astype(int)
    groups = group_labels.index.to_numpy()

    skf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_fold = {}
    # MultilabelStratifiedKFold.split needs an X; groups themselves serve fine as a
    # placeholder since only y drives the stratification.
    for fold, (_, val_idx) in enumerate(skf.split(groups.reshape(-1, 1), y)):
        for g in groups[val_idx]:
            group_fold[g] = fold

    return df[group_col].map(group_fold)


def select_holdout_ids(
    df: pd.DataFrame,
    group_col: str,
    study_col: str = "StudyInstanceUID",
    label_columns: list[str] = LABEL_COLUMNS,
    n_splits: int = 5,
    seed: int = 42,
    holdout_fold: int = 0,
) -> tuple[list[str], list[str]]:
    """Split `df`'s studies into (train_ids, holdout_ids) using one fold of
    `make_grouped_multilabel_folds` as the holdout. Intended for the baseline
    notebook's two-tier validation design (see PLAN.md): a ~20% holdout drives model
    selection using derived/silver labels (plenty of samples per class), while a much
    smaller "annotation check" -- the gold-labeled studies that land in that holdout
    -- approximates the actual competition metric closely enough to report, without
    enough samples to arbitrate alone.
    """
    folds = make_grouped_multilabel_folds(df, group_col, label_columns, n_splits, seed)
    is_holdout = folds == holdout_fold
    holdout_ids = df.loc[is_holdout, study_col].tolist()
    train_ids = df.loc[~is_holdout, study_col].tolist()
    return train_ids, holdout_ids
