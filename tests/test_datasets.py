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


def test_cache_evicts_least_recently_used_beyond_max_size():
    study_ids = ["a", "b", "c"]
    series_df = pd.concat([_series_with_slots(s) for s in study_ids], ignore_index=True)
    targets = pd.concat([_targets(s) for s in study_ids], ignore_index=True)
    dataset = StudyDataset(
        study_ids, series_df, series_root="fake_root", targets=targets,
        cache_images=True, max_cache_studies=2,
    )

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.5, dtype=np.float32)
    with patch(
        "rsna_knee.datasets.load_series_representative", return_value=fake_image
    ) as mock_load:
        dataset[0]  # cache: [a]
        dataset[1]  # cache: [a, b]
        dataset[2]  # over max_cache_studies=2 -> evicts a (least recently used); cache: [b, c]
        assert mock_load.call_count == 3

        dataset[2]  # c is cached -- no decode
        assert mock_load.call_count == 3

        dataset[0]  # a was evicted -- must re-decode
        assert mock_load.call_count == 4

    assert len(dataset._cache) == 2


def test_cache_access_refreshes_lru_order():
    study_ids = ["a", "b", "c"]
    series_df = pd.concat([_series_with_slots(s) for s in study_ids], ignore_index=True)
    targets = pd.concat([_targets(s) for s in study_ids], ignore_index=True)
    dataset = StudyDataset(
        study_ids, series_df, series_root="fake_root", targets=targets,
        cache_images=True, max_cache_studies=2,
    )

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.5, dtype=np.float32)
    with patch(
        "rsna_knee.datasets.load_series_representative", return_value=fake_image
    ) as mock_load:
        dataset[0]  # cache: [a]
        dataset[1]  # cache: [a, b]
        dataset[0]  # touch a -- LRU order becomes [b, a]
        dataset[2]  # over max -- evicts b (now least recently used), not a
        assert mock_load.call_count == 3

        dataset[0]  # a should still be cached
        assert mock_load.call_count == 3

        dataset[1]  # b was evicted -- must re-decode
        assert mock_load.call_count == 4


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


def test_center_fraction_defaults_to_middle():
    study_id = "study_1"
    series_df = _series_with_slots(study_id)
    targets = _targets(study_id)
    dataset = StudyDataset(
        [study_id], series_df, series_root="fake_root", targets=targets, cache_images=False
    )
    assert dataset.center_fraction == 0.5


def test_center_fraction_passed_through_to_load_series_representative():
    study_id = "study_1"
    series_df = _series_with_slots(study_id)
    targets = _targets(study_id)
    dataset = StudyDataset(
        [study_id], series_df, series_root="fake_root", targets=targets,
        cache_images=False, center_fraction=0.35,
    )

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.5, dtype=np.float32)
    with patch(
        "rsna_knee.datasets.load_series_representative", return_value=fake_image
    ) as mock_load:
        dataset[0]

    assert mock_load.call_args.kwargs["center_fraction"] == 0.35
