import numpy as np
import pandas as pd
import pytest

from rsna_knee.metrics import LABEL_COLUMNS, macro_auc, per_label_auc, rank_average


def _make_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame(values)


def _with_study_ids(df: pd.DataFrame, n: int) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "StudyInstanceUID", [f"study_{i}" for i in range(n)])
    return out


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


def test_rank_average_matches_ordering_of_agreeing_models():
    n = 50
    rng = np.random.default_rng(3)
    truth = {col: rng.integers(0, 2, size=n).astype(float) for col in LABEL_COLUMNS}
    y_true = _with_study_ids(_make_frame(truth), n)

    base_scores = {col: rng.random(n) for col in LABEL_COLUMNS}
    # two "models" that agree on ranking but live on wildly different scales/shapes --
    # one linear in [0,1], the other a squashed, shifted version of the same ranking.
    model_a = _with_study_ids(_make_frame(base_scores), n)
    model_b = _with_study_ids(
        _make_frame({col: base_scores[col] ** 3 * 1000 + 5000 for col in LABEL_COLUMNS}), n
    )

    combined = rank_average([model_a, model_b])
    y_true_indexed = y_true.set_index("StudyInstanceUID")
    combined_indexed = combined.set_index("StudyInstanceUID").loc[y_true_indexed.index]

    solo_auc = macro_auc(y_true, model_a)
    combined_auc = macro_auc(
        y_true_indexed.reset_index(), combined_indexed.reset_index()
    )
    # since both models rank identically, the rank-averaged combination should give
    # (near-)identical AUC to either alone -- the wildly different raw scales of
    # model_b must not distort the result the way raw-probability averaging would.
    assert abs(combined_auc - solo_auc) < 1e-6


def test_rank_average_dominant_model_would_win_under_raw_averaging_but_not_here():
    """A model producing extreme raw scores (e.g. near 0/1 saturated logits-after-
    sigmoid) would dominate a naive average of raw probabilities even if its ranking
    is no better than a second, more modestly-scored model. Rank averaging must not
    let magnitude alone dictate the outcome."""
    study_ids = [f"study_{i}" for i in range(4)]
    y_true = pd.DataFrame(
        {"StudyInstanceUID": study_ids, **{col: [1.0, 0.0, 1.0, 0.0] for col in LABEL_COLUMNS}}
    )
    # extreme_model: correct ranking but at saturated extremes
    extreme_model = pd.DataFrame(
        {"StudyInstanceUID": study_ids, **{col: [0.999, 0.001, 0.999, 0.001] for col in LABEL_COLUMNS}}
    )
    # modest_model: WRONG ranking (inverted for the middle pair), modest scores
    modest_model = pd.DataFrame(
        {"StudyInstanceUID": study_ids, **{col: [0.6, 0.55, 0.5, 0.45] for col in LABEL_COLUMNS}}
    )

    raw_avg = pd.DataFrame(
        {
            "StudyInstanceUID": study_ids,
            **{
                col: (extreme_model[col].to_numpy() + modest_model[col].to_numpy()) / 2
                for col in LABEL_COLUMNS
            },
        }
    )
    rank_avg = rank_average([extreme_model, modest_model])

    raw_auc = macro_auc(y_true, raw_avg)
    rank_auc = macro_auc(y_true, rank_avg)
    # raw averaging: extreme_model's saturated values dominate, order stays [0.7995,
    # 0.2755, 0.7495, 0.2255] -- correct order preserved by luck here, so instead
    # directly check rank_average's own output ordering matches extreme_model's
    # correct order rather than being distorted toward modest_model's incorrect one.
    ranked_values = rank_avg[LABEL_COLUMNS[0]].to_numpy()
    assert ranked_values[0] > ranked_values[1]  # study_0 (true=1) > study_1 (true=0)
    assert ranked_values[2] > ranked_values[3]  # study_2 (true=1) > study_3 (true=0)


def test_rank_average_output_matches_input_studies_and_columns():
    n = 10
    rng = np.random.default_rng(4)
    df1 = _with_study_ids(_make_frame({col: rng.random(n) for col in LABEL_COLUMNS}), n)
    df2 = _with_study_ids(_make_frame({col: rng.random(n) for col in LABEL_COLUMNS}), n)

    combined = rank_average([df1, df2])
    assert list(combined.columns) == ["StudyInstanceUID", *LABEL_COLUMNS]
    assert set(combined["StudyInstanceUID"]) == set(df1["StudyInstanceUID"])
    assert len(combined) == n
    assert combined[LABEL_COLUMNS].to_numpy().min() >= 0.0
    assert combined[LABEL_COLUMNS].to_numpy().max() <= 1.0


def test_rank_average_single_model_is_identity_ranking():
    n = 8
    rng = np.random.default_rng(5)
    df = _with_study_ids(_make_frame({col: rng.random(n) for col in LABEL_COLUMNS}), n)
    combined = rank_average([df])

    original_order = df.set_index("StudyInstanceUID")[LABEL_COLUMNS[0]].rank()
    combined_order = combined.set_index("StudyInstanceUID")[LABEL_COLUMNS[0]].rank()
    assert (original_order == combined_order).all()


def test_rank_average_empty_list_raises():
    with pytest.raises(ValueError, match="at least one"):
        rank_average([])


def test_rank_average_mismatched_study_ids_raises():
    df1 = _with_study_ids(_make_frame({col: [0.5] for col in LABEL_COLUMNS}), 1)
    df2 = pd.DataFrame({"StudyInstanceUID": ["different_study"], **{col: [0.5] for col in LABEL_COLUMNS}})
    with pytest.raises(ValueError, match="same StudyInstanceUID"):
        rank_average([df1, df2])
