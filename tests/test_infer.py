from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from rsna_knee.infer import run_inference, run_inference_tta
from rsna_knee.metrics import LABEL_COLUMNS


def _series_with_slots(study_ids):
    return pd.DataFrame(
        {
            "StudyInstanceUID": study_ids,
            "SeriesInstanceUID": [f"series_{i}" for i in range(len(study_ids))],
            "slot": ["SAG_T1"] * len(study_ids),
        }
    )


class _MeanPoolingModel(nn.Module):
    """A tiny real model whose output depends deterministically on the input images'
    mean pixel value, so tests can control predictions via what
    load_series_representative is mocked to return."""

    def __init__(self, num_labels=len(LABEL_COLUMNS)):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, images, presence_mask):
        b = images.shape[0]
        mean_val = images.mean(dim=(1, 2, 3, 4))  # (B,)
        # map [0,1] image mean to a wide logit range so sigmoid ranks are stable
        logits = (mean_val.unsqueeze(1) - 0.5) * 20
        return logits.expand(b, self.num_labels)


def test_run_inference_output_shape_and_format():
    study_ids = ["a", "b", "c"]
    series_with_slots = _series_with_slots(study_ids)
    model = _MeanPoolingModel()

    fake_image = np.full((336, 336), 0.5, dtype=np.float32)
    with patch("rsna_knee.datasets.load_series_representative", return_value=fake_image):
        preds = run_inference(model, study_ids, series_with_slots, series_root="fake_root", device="cpu")

    assert list(preds.columns) == ["StudyInstanceUID", *LABEL_COLUMNS]
    assert sorted(preds["StudyInstanceUID"]) == sorted(study_ids)
    assert preds[LABEL_COLUMNS].to_numpy().min() >= 0.0
    assert preds[LABEL_COLUMNS].to_numpy().max() <= 1.0


def test_run_inference_center_fraction_reaches_load_series_representative():
    study_ids = ["a"]
    series_with_slots = _series_with_slots(study_ids)
    model = _MeanPoolingModel()

    fake_image = np.full((336, 336), 0.5, dtype=np.float32)
    with patch(
        "rsna_knee.datasets.load_series_representative", return_value=fake_image
    ) as mock_load:
        run_inference(model, study_ids, series_with_slots, series_root="fake_root", device="cpu", center_fraction=0.35)

    assert mock_load.call_args.kwargs["center_fraction"] == 0.35


def test_run_inference_tta_runs_one_pass_per_center_fraction():
    study_ids = ["a", "b"]
    series_with_slots = _series_with_slots(study_ids)
    model = _MeanPoolingModel()

    fake_image = np.full((336, 336), 0.5, dtype=np.float32)
    with patch(
        "rsna_knee.datasets.load_series_representative", return_value=fake_image
    ) as mock_load:
        run_inference_tta(
            model, study_ids, series_with_slots, series_root="fake_root", device="cpu",
            center_fractions=(0.3, 0.5, 0.7),
        )

    seen_fractions = {call.kwargs["center_fraction"] for call in mock_load.call_args_list}
    assert seen_fractions == {0.3, 0.5, 0.7}


def test_run_inference_tta_output_matches_expected_format():
    study_ids = ["a", "b", "c"]
    series_with_slots = _series_with_slots(study_ids)
    model = _MeanPoolingModel()

    fake_image = np.full((336, 336), 0.5, dtype=np.float32)
    with patch("rsna_knee.datasets.load_series_representative", return_value=fake_image):
        preds = run_inference_tta(
            model, study_ids, series_with_slots, series_root="fake_root", device="cpu"
        )

    assert list(preds.columns) == ["StudyInstanceUID", *LABEL_COLUMNS]
    assert sorted(preds["StudyInstanceUID"]) == sorted(study_ids)
    assert preds[LABEL_COLUMNS].to_numpy().min() >= 0.0
    assert preds[LABEL_COLUMNS].to_numpy().max() <= 1.0


def test_run_inference_tta_combines_varying_predictions_by_rank():
    """Different center_fractions producing genuinely different image content (and
    thus different per-study predictions) should still yield a valid, consistently
    ordered rank-averaged combination -- not crash, not collapse to ties."""
    study_ids = ["a", "b", "c"]
    series_with_slots = _series_with_slots(study_ids)
    model = _MeanPoolingModel()

    # different mean pixel value per study, consistent ordering across "passes" so
    # the combined ranking should recover a > b > c on every label
    means_by_study = {"a": 0.9, "b": 0.5, "c": 0.1}

    def fake_representative(series_dir, physical_crop_mm=None, resize_px=336, center_fraction=0.5, **kwargs):
        # series_dir path looks like fake_root/<study_id>/<series_id>
        study_id = series_dir.parts[-2]
        return np.full((resize_px, resize_px), means_by_study[study_id], dtype=np.float32)

    with patch("rsna_knee.datasets.load_series_representative", side_effect=fake_representative):
        preds = run_inference_tta(
            model, study_ids, series_with_slots, series_root="fake_root", device="cpu"
        )

    preds = preds.set_index("StudyInstanceUID")
    for col in LABEL_COLUMNS:
        assert preds.loc["a", col] > preds.loc["b", col] > preds.loc["c", col]
