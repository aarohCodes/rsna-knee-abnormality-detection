import numpy as np
import pandas as pd
import pytest

from rsna_knee.data.label_expansion import GOLD_WEIGHT, add_derived_confidence, build_training_targets
from rsna_knee.metrics import LABEL_COLUMNS


def _train_labels(rows):
    base = {"StudyInstanceUID": [], "Report": []}
    for col in LABEL_COLUMNS:
        base[col] = []
    for r in rows:
        base["StudyInstanceUID"].append(r["id"])
        base["Report"].append(r.get("report", "x"))
        for col in LABEL_COLUMNS:
            base[col].append(r.get(col, np.nan))
    return pd.DataFrame(base)


def _silver_labels(rows):
    base = {"StudyInstanceUID": []}
    for col in LABEL_COLUMNS:
        base[col] = []
        base[f"{col}__conf"] = []
    for r in rows:
        base["StudyInstanceUID"].append(r["id"])
        for col in LABEL_COLUMNS:
            base[col].append(r.get(col, 0.28))
            base[f"{col}__conf"].append(r.get(f"{col}__conf", 0.05))
    return pd.DataFrame(base)


def test_gold_study_uses_hard_label_and_elevated_weight():
    train = _train_labels([{"id": "s1", "ACL": 1.0, "MCL": 0.0}])
    # fill remaining label columns with 0 so gold_mask is True (any label present)
    for col in LABEL_COLUMNS:
        if col not in ("ACL", "MCL"):
            train.loc[0, col] = 0.0
    silver = _silver_labels([{"id": "s1", "ACL": 0.6, "ACL__conf": 0.4}])  # silver disagrees

    out = build_training_targets(train, silver)
    row = out[out["StudyInstanceUID"] == "s1"].iloc[0]
    assert row["is_gold"] == True  # noqa: E712
    assert row["ACL"] == 1.0  # gold overrides silver's 0.6
    assert row["ACL__weight"] == GOLD_WEIGHT


def test_silver_only_study_uses_report_derived_score_and_confidence():
    train = _train_labels([{"id": "s2"}])  # no gold labels at all
    silver = _silver_labels([{"id": "s2", "ACL": 0.82, "ACL__conf": 0.95}])

    out = build_training_targets(train, silver)
    row = out[out["StudyInstanceUID"] == "s2"].iloc[0]
    assert row["is_gold"] == False  # noqa: E712
    assert row["ACL"] == 0.82
    assert row["ACL__weight"] == 0.95


def test_study_with_no_silver_row_gets_neutral_zero_weight_target():
    train = _train_labels([{"id": "s3"}])
    silver = _silver_labels([])  # empty -- no match for s3

    out = build_training_targets(train, silver)
    row = out[out["StudyInstanceUID"] == "s3"].iloc[0]
    assert row["ACL"] == 0.5
    assert row["ACL__weight"] == 0.0


def test_output_covers_every_input_study_exactly_once():
    train = _train_labels([{"id": "a"}, {"id": "b"}, {"id": "c", "ACL": 1.0}])
    for col in LABEL_COLUMNS:
        if col != "ACL":
            train.loc[2, col] = 0.0
    silver = _silver_labels([{"id": "a"}, {"id": "b"}])

    out = build_training_targets(train, silver)
    assert sorted(out["StudyInstanceUID"]) == ["a", "b", "c"]
    assert len(out) == 3


def test_add_derived_confidence_saturated_scores_get_full_weight():
    df = pd.DataFrame({"StudyInstanceUID": ["s1"], **{col: [1.0] for col in LABEL_COLUMNS}})
    out = add_derived_confidence(df)
    for col in LABEL_COLUMNS:
        assert out.loc[0, f"{col}__conf"] == pytest.approx(1.0)

    df0 = pd.DataFrame({"StudyInstanceUID": ["s1"], **{col: [0.0] for col in LABEL_COLUMNS}})
    out0 = add_derived_confidence(df0)
    for col in LABEL_COLUMNS:
        assert out0.loc[0, f"{col}__conf"] == pytest.approx(1.0)


def test_add_derived_confidence_midpoint_score_gets_zero_weight():
    df = pd.DataFrame({"StudyInstanceUID": ["s1"], **{col: [0.5] for col in LABEL_COLUMNS}})
    out = add_derived_confidence(df)
    for col in LABEL_COLUMNS:
        assert out.loc[0, f"{col}__conf"] == pytest.approx(0.0)


def test_add_derived_confidence_scales_linearly_from_midpoint():
    df = pd.DataFrame({"StudyInstanceUID": ["s1", "s2"], "ACL": [0.75, 0.25]})
    out = add_derived_confidence(df, label_columns=["ACL"])
    assert out.loc[0, "ACL__conf"] == pytest.approx(0.5)  # |0.75-0.5|*2
    assert out.loc[1, "ACL__conf"] == pytest.approx(0.5)  # |0.25-0.5|*2, symmetric


def test_add_derived_confidence_preserves_original_columns():
    df = pd.DataFrame({"StudyInstanceUID": ["s1"], "ACL": [0.7]})
    out = add_derived_confidence(df, label_columns=["ACL"])
    assert out.loc[0, "ACL"] == 0.7  # original score column untouched
    assert "StudyInstanceUID" in out.columns


def test_derived_confidence_feeds_build_training_targets_correctly():
    """End-to-end: a source with no native confidence, run through
    add_derived_confidence, should behave identically to a native-confidence source
    with the same values through build_training_targets."""
    train = _train_labels([{"id": "s1"}])
    silver_no_conf = pd.DataFrame({"StudyInstanceUID": ["s1"], "ACL": [0.9]})
    for col in LABEL_COLUMNS:
        if col != "ACL":
            silver_no_conf[col] = 0.5
    silver_with_conf = add_derived_confidence(silver_no_conf)

    out = build_training_targets(train, silver_with_conf)
    row = out[out["StudyInstanceUID"] == "s1"].iloc[0]
    assert row["ACL"] == pytest.approx(0.9)
    assert row["ACL__weight"] == pytest.approx(0.8)  # |0.9-0.5|*2
