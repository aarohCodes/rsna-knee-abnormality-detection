"""Loading and cleaning train.csv.

Per the competition's own Dataset Description: only a small subset of training studies
carry direct per-condition labels. The remaining rows have blank (NaN) label columns by
design -- NaN means "no direct label", not "confirmed negative". Treat that distinction
carefully; collapsing NaN to 0 silently turns "unknown" into "negative" and corrupts both
the label-expansion pipeline and any BCE loss computed directly off this frame.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna_knee.metrics import LABEL_COLUMNS

REQUIRED_COLUMNS = ["StudyInstanceUID", "Report", *LABEL_COLUMNS]


def load_train_labels(train_csv: str | Path) -> pd.DataFrame:
    """Load train.csv, validating schema and label dtypes.

    Label columns are coerced to float so that NaN (no label) and 0.0/1.0 (negative/
    positive) are distinguishable via `pandas.isna`, never collapsed into an int dtype
    that would force NaN to become 0.
    """
    df = pd.read_csv(train_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv is missing expected columns: {missing}")
    for col in LABEL_COLUMNS:
        df[col] = df[col].astype(float)
    return df


def label_coverage(df: pd.DataFrame, label_columns: list[str] = LABEL_COLUMNS) -> pd.Series:
    """Fraction of rows with a non-NaN value for each label column."""
    return df[label_columns].notna().mean()


def has_any_label(df: pd.DataFrame, label_columns: list[str] = LABEL_COLUMNS) -> pd.Series:
    """Boolean mask: True where a study has at least one directly annotated label."""
    return df[label_columns].notna().any(axis=1)
