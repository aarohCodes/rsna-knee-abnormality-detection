import numpy as np
import pandas as pd
import pytest

from rsna_knee.metrics import LABEL_COLUMNS
from rsna_knee.validate import make_grouped_multilabel_folds, report_hash


def test_report_hash_stable_and_normalizes_whitespace_and_case():
    a = report_hash("Meniscal tear.  Effusion present.")
    b = report_hash("meniscal tear. effusion present.")
    c = report_hash("Meniscal   tear.\nEffusion present.")
    assert a == b == c


def test_report_hash_differs_for_different_text():
    assert report_hash("normal knee") != report_hash("torn ACL")


def _synthetic_df(n=200, n_duplicate_groups=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    duplicate_reports = [f"template report {i}" for i in range(n_duplicate_groups)]
    for i in range(n):
        if i < n_duplicate_groups * 3:
            report = duplicate_reports[i % n_duplicate_groups]
        else:
            report = f"unique report {i}"
        row = {"StudyInstanceUID": f"study_{i}", "Report": report}
        for col in LABEL_COLUMNS:
            row[col] = float(rng.integers(0, 2))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["report_hash"] = df["Report"].map(report_hash)
    return df


def test_duplicate_report_groups_never_split_across_folds():
    df = _synthetic_df()
    folds = make_grouped_multilabel_folds(df, group_col="report_hash", n_splits=5, seed=0)
    df = df.assign(fold=folds)
    for group_hash, group_df in df.groupby("report_hash"):
        assert group_df["fold"].nunique() == 1, (
            f"group {group_hash} split across folds: {group_df['fold'].unique()}"
        )


def test_every_row_assigned_a_valid_fold():
    df = _synthetic_df()
    folds = make_grouped_multilabel_folds(df, group_col="report_hash", n_splits=5, seed=0)
    assert folds.isna().sum() == 0
    assert set(folds.unique()) <= set(range(5))
    assert len(folds) == len(df)


def test_folds_are_roughly_balanced_in_size():
    df = _synthetic_df(n=500, n_duplicate_groups=5)
    folds = make_grouped_multilabel_folds(df, group_col="report_hash", n_splits=5, seed=0)
    counts = folds.value_counts()
    # with mostly-unique groups and 5 folds, no fold should be wildly off from 500/5=100
    assert counts.min() > 60
    assert counts.max() < 140
