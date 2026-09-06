import numpy as np
import pandas as pd
import pytest

from rsna_knee.metrics import LABEL_COLUMNS
from rsna_knee.validate import make_grouped_multilabel_folds, report_hash, select_holdout_ids


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


def test_select_holdout_ids_partitions_without_overlap_or_gaps():
    df = _synthetic_df(n=300)
    train_ids, holdout_ids = select_holdout_ids(df, group_col="report_hash", n_splits=5, seed=0)

    assert set(train_ids).isdisjoint(set(holdout_ids))
    assert set(train_ids) | set(holdout_ids) == set(df["StudyInstanceUID"])
    assert len(train_ids) + len(holdout_ids) == len(df)


def test_select_holdout_ids_roughly_one_fifth_for_five_splits():
    df = _synthetic_df(n=500, n_duplicate_groups=5)
    train_ids, holdout_ids = select_holdout_ids(df, group_col="report_hash", n_splits=5, seed=0)
    # ~20% holdout, generous tolerance given group-based (not row-based) splitting
    assert 60 < len(holdout_ids) < 140


def test_select_holdout_ids_different_fold_gives_different_holdout():
    df = _synthetic_df(n=300)
    _, holdout_0 = select_holdout_ids(df, group_col="report_hash", n_splits=5, seed=0, holdout_fold=0)
    _, holdout_1 = select_holdout_ids(df, group_col="report_hash", n_splits=5, seed=0, holdout_fold=1)
    assert set(holdout_0) != set(holdout_1)


def test_select_holdout_ids_keeps_duplicate_report_groups_together():
    df = _synthetic_df(n=300)
    _, holdout_ids = select_holdout_ids(df, group_col="report_hash", n_splits=5, seed=0)
    holdout_set = set(holdout_ids)
    for _, group_df in df.groupby("report_hash"):
        ids_in_group = set(group_df["StudyInstanceUID"])
        # a duplicate-report group must be entirely in or entirely out of the holdout
        assert ids_in_group.issubset(holdout_set) or ids_in_group.isdisjoint(holdout_set)


def test_community_fold_matches_reference_implementation():
    """Must reproduce the public ensemble's split EXACTLY -- it is the only way to
    know which studies their published fold weights were trained on. Reference,
    verbatim from their notebook:
        int(hashlib.md5(rep.encode()).hexdigest()[:8], 16) % 5
    """
    import hashlib as _h

    from rsna_knee.validate import community_fold

    for text in ["normal knee mri", "", "Rodilla derecha: rotura de menisco", "x" * 500]:
        expected = int(_h.md5(text.encode()).hexdigest()[:8], 16) % 5
        assert community_fold(text) == expected


def test_community_fold_is_deterministic_and_unnormalized():
    """Deterministic across calls, and deliberately does NOT normalize whitespace/case
    the way report_hash does -- matching their bytes exactly is the whole point."""
    from rsna_knee.validate import community_fold

    assert community_fold("Some Report") == community_fold("Some Report")
    # differing only by case/whitespace should generally land differently; at minimum
    # the function must not be silently normalizing them together by construction
    assert community_fold("some report") == int(
        __import__("hashlib").md5(b"some report").hexdigest()[:8], 16
    ) % 5


def test_community_holdout_ids_partitions_completely():
    from rsna_knee.validate import community_holdout_ids

    df = pd.DataFrame(
        {
            "StudyInstanceUID": [f"s{i}" for i in range(200)],
            "Report": [f"report text number {i}" for i in range(200)],
        }
    )
    train, holdout = community_holdout_ids(df)

    assert set(train) | set(holdout) == set(df["StudyInstanceUID"])
    assert not (set(train) & set(holdout))
    # roughly a fifth should land in the holdout fold
    assert 0.1 < len(holdout) / len(df) < 0.3
