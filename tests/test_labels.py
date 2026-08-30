import numpy as np
import pandas as pd
import pytest

from rsna_knee.data.labels import has_any_label, label_coverage, load_train_labels
from rsna_knee.metrics import LABEL_COLUMNS


def _write_csv(tmp_path, rows):
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_train_labels_preserves_nan_as_missing(tmp_path):
    rows = {
        "StudyInstanceUID": ["a", "b"],
        "Report": ["some report", "another report"],
        **{col: [1.0, np.nan] for col in LABEL_COLUMNS},
    }
    path = _write_csv(tmp_path, rows)
    df = load_train_labels(path)
    assert df.loc[0, "ACL"] == 1.0
    assert pd.isna(df.loc[1, "ACL"])


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "train.csv"
    pd.DataFrame({"StudyInstanceUID": ["a"], "Report": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_train_labels(path)


def test_label_coverage_and_has_any_label(tmp_path):
    rows = {
        "StudyInstanceUID": ["fully_labeled", "unlabeled", "partial"],
        "Report": ["r1", "r2", "r3"],
    }
    for i, col in enumerate(LABEL_COLUMNS):
        rows[col] = [1.0, np.nan, 1.0 if i == 0 else np.nan]
    path = _write_csv(tmp_path, rows)
    df = load_train_labels(path)

    coverage = label_coverage(df)
    assert coverage["ACL"] == pytest.approx(2 / 3)
    assert coverage["Fracture"] == pytest.approx(1 / 3)

    mask = has_any_label(df)
    assert mask.tolist() == [True, False, True]
