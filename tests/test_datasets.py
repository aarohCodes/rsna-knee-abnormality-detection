from unittest.mock import patch

import numpy as np
import pandas as pd

from rsna_knee.data.dicom_io import SERIES_SLOTS
from rsna_knee.datasets import StudyDataset
from rsna_knee.metrics import LABEL_COLUMNS


def _series_with_slots(study_id: str, slot: str = "SAG_T1"):
    return pd.DataFrame(
        {
            "StudyInstanceUID": [study_id],
            "SeriesInstanceUID": ["series_1"],
            "slot": [slot],
        }
    )


def _targets(study_id: str):
    row = {"StudyInstanceUID": study_id}
    for col in LABEL_COLUMNS:
        row[col] = 1.0
        row[f"{col}__weight"] = 1.0
    return pd.DataFrame([row])


def test_cache_avoids_redundant_decode_across_getitem_calls():
    study_id = "study_1"
    series_df = _series_with_slots(study_id)
    targets = _targets(study_id)
    dataset = StudyDataset([study_id], series_df, series_root="fake_root", targets=targets, cache_images=True)

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.5, dtype=np.float32)
    with patch(
        "rsna_knee.datasets.load_series_representative", return_value=fake_image
    ) as mock_load:
        dataset[0]
        dataset[0]
        dataset[0]

    assert mock_load.call_count == 1


def test_cache_disabled_decodes_every_call():
    study_id = "study_1"
    series_df = _series_with_slots(study_id)
    targets = _targets(study_id)
    dataset = StudyDataset(
        [study_id], series_df, series_root="fake_root", targets=targets, cache_images=False
    )

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.5, dtype=np.float32)
    with patch(
        "rsna_knee.datasets.load_series_representative", return_value=fake_image
    ) as mock_load:
        dataset[0]
        dataset[0]

    assert mock_load.call_count == 2


def test_cache_is_per_study_not_global():
    series_df = pd.concat([_series_with_slots("a"), _series_with_slots("b")], ignore_index=True)
    targets = pd.concat([_targets("a"), _targets("b")], ignore_index=True)
    dataset = StudyDataset(["a", "b"], series_df, series_root="fake_root", targets=targets, cache_images=True)

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.5, dtype=np.float32)
    with patch(
        "rsna_knee.datasets.load_series_representative", return_value=fake_image
    ) as mock_load:
        dataset[0]  # study a
        dataset[1]  # study b -- different study, must decode
        dataset[0]  # study a again -- must hit cache

    assert mock_load.call_count == 2


def test_cached_output_matches_uncached_output():
    study_id = "study_1"
    series_df = _series_with_slots(study_id)
    targets = _targets(study_id)
    dataset = StudyDataset([study_id], series_df, series_root="fake_root", targets=targets, cache_images=True)

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.5, dtype=np.float32)
    with patch("rsna_knee.datasets.load_series_representative", return_value=fake_image):
        first = dataset[0]
        second = dataset[0]

    assert (first["images"] == second["images"]).all()
    assert (first["presence_mask"] == second["presence_mask"]).all()
