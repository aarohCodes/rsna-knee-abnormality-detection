import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rsna_knee.data.dicom_io import SERIES_SLOTS
from rsna_knee.datasets import StudyDataset, compute_safe_max_cache_studies
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


def _with_fake_psutil(available_gb: float):
    """Context manager that makes `import psutil` inside compute_safe_max_cache_studies
    succeed and report a specific available-memory value, without needing psutil
    actually installed (it isn't, in this dev environment)."""
    fake_psutil = MagicMock()
    fake_psutil.virtual_memory.return_value = MagicMock(available=available_gb * 1024**3)
    return patch.dict(sys.modules, {"psutil": fake_psutil})


def test_compute_safe_cache_falls_back_when_psutil_unavailable():
    with patch.dict(sys.modules, {"psutil": None}):
        result = compute_safe_max_cache_studies(num_workers=2, fallback_available_gb=13.0)
    # should use the fallback, not raise -- exact value covered by the psutil-present
    # test below (same formula), this just checks the no-psutil path doesn't crash
    assert isinstance(result, int)
    assert result > 0


def test_compute_safe_cache_uses_real_available_memory_when_psutil_present():
    with _with_fake_psutil(available_gb=20.0):
        result = compute_safe_max_cache_studies(num_workers=2, target_fraction=0.4, per_study_mb=2.0)
    # 20GB * 1024 * 0.4 / 2.0MB / 2 workers = 2048 studies
    assert result == 2048


def test_compute_safe_cache_more_workers_gives_smaller_per_worker_budget():
    with _with_fake_psutil(available_gb=20.0):
        cache_1_worker = compute_safe_max_cache_studies(num_workers=1, per_study_mb=2.0)
        cache_4_workers = compute_safe_max_cache_studies(num_workers=4, per_study_mb=2.0)
    # more workers -> each worker's independent cache must shrink to keep the same
    # total memory budget, per the docstring's whole rationale
    assert cache_4_workers < cache_1_worker
    assert cache_1_worker == pytest.approx(cache_4_workers * 4, rel=0.01)


def test_compute_safe_cache_clamps_to_min_and_max():
    with _with_fake_psutil(available_gb=0.001):  # essentially no memory available
        result = compute_safe_max_cache_studies(num_workers=2, min_studies=200, max_studies=5000)
    assert result == 200

    with _with_fake_psutil(available_gb=10000.0):  # implausibly huge
        result = compute_safe_max_cache_studies(num_workers=1, min_studies=200, max_studies=5000)
    assert result == 5000


def _multi_slot_series(study_id: str, slots: list[str]):
    return pd.DataFrame(
        {
            "StudyInstanceUID": [study_id] * len(slots),
            "SeriesInstanceUID": [f"series_{i}" for i in range(len(slots))],
            "slot": slots,
        }
    )


def test_slot_fallback_disabled_by_default():
    study_id = "study_1"
    # A controlled test found the fallback made results worse, not better (see
    # datasets.py's use_slot_fallbacks docstring) -- must stay off unless explicitly
    # requested, so the known-good default pipeline isn't silently affected.
    series_df = _multi_slot_series(study_id, ["COR_FLUID_FS"])
    targets = _targets(study_id)
    dataset = StudyDataset([study_id], series_df, series_root="fake_root", targets=targets, cache_images=False)

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.7, dtype=np.float32)
    with patch("rsna_knee.datasets.load_series_representative", return_value=fake_image):
        item = dataset[0]

    cor_t1_idx = SERIES_SLOTS.index("COR_T1")
    assert item["presence_mask"][cor_t1_idx] == 0.0
    assert item["images"][cor_t1_idx].max() == 0


def test_missing_cor_t1_falls_back_to_cor_fluid_fs():
    study_id = "study_1"
    # COR_T1 absent, COR_FLUID_FS present -- per SLOT_FALLBACKS, COR_T1's slot
    # position should still get filled (from COR_FLUID_FS) rather than left masked.
    series_df = _multi_slot_series(study_id, ["COR_FLUID_FS"])
    targets = _targets(study_id)
    dataset = StudyDataset(
        [study_id], series_df, series_root="fake_root", targets=targets, cache_images=False,
        use_slot_fallbacks=True,
    )

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.7, dtype=np.float32)
    with patch("rsna_knee.datasets.load_series_representative", return_value=fake_image):
        item = dataset[0]

    cor_t1_idx = SERIES_SLOTS.index("COR_T1")
    cor_fluid_fs_idx = SERIES_SLOTS.index("COR_FLUID_FS")
    assert item["presence_mask"][cor_t1_idx] == 1.0
    assert item["presence_mask"][cor_fluid_fs_idx] == 1.0
    # both slots carry real image content, not a zeroed-out placeholder
    assert item["images"][cor_t1_idx].max() > 0
    assert item["images"][cor_fluid_fs_idx].max() > 0


def test_missing_cor_t1_with_no_fallback_available_stays_masked():
    study_id = "study_1"
    # Neither COR_T1 nor its fallback (COR_FLUID_FS) present -- should stay masked,
    # same as any other genuinely-absent slot.
    series_df = _multi_slot_series(study_id, ["SAG_T1"])
    targets = _targets(study_id)
    dataset = StudyDataset(
        [study_id], series_df, series_root="fake_root", targets=targets, cache_images=False,
        use_slot_fallbacks=True,
    )

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.7, dtype=np.float32)
    with patch("rsna_knee.datasets.load_series_representative", return_value=fake_image):
        item = dataset[0]

    cor_t1_idx = SERIES_SLOTS.index("COR_T1")
    assert item["presence_mask"][cor_t1_idx] == 0.0
    assert item["images"][cor_t1_idx].max() == 0


def test_present_cor_t1_is_not_overridden_by_fallback():
    study_id = "study_1"
    # Both COR_T1 and COR_FLUID_FS present -- COR_T1's own series must be used, not
    # silently replaced by the fallback.
    series_df = _multi_slot_series(study_id, ["COR_T1", "COR_FLUID_FS"])
    targets = _targets(study_id)
    dataset = StudyDataset(
        [study_id], series_df, series_root="fake_root", targets=targets, cache_images=False,
        use_slot_fallbacks=True,
    )

    calls = []

    def fake_load(series_dir, *args, **kwargs):
        calls.append(str(series_dir))
        return np.full((dataset.resize_px, dataset.resize_px), 0.7, dtype=np.float32)

    with patch("rsna_knee.datasets.load_series_representative", side_effect=fake_load):
        dataset[0]

    # the COR_T1 row's own series (series_0) must be the one used for that slot, not
    # COR_FLUID_FS's series_1
    assert any("series_0" in c for c in calls)


def test_slices_per_slot_default_uses_single_crop_loader():
    study_id = "study_1"
    series_df = _series_with_slots(study_id)
    targets = _targets(study_id)
    dataset = StudyDataset([study_id], series_df, series_root="fake_root", targets=targets, cache_images=False)

    fake_image = np.full((dataset.resize_px, dataset.resize_px), 0.5, dtype=np.float32)
    with patch("rsna_knee.datasets.load_series_representative", return_value=fake_image) as mock_single, \
         patch("rsna_knee.datasets.load_series_multi_slice") as mock_multi:
        item = dataset[0]

    assert mock_single.call_count == 1
    assert mock_multi.call_count == 0
    assert item["images"].shape == (len(SERIES_SLOTS), 3, dataset.resize_px, dataset.resize_px)
    assert item["presence_mask"].shape == (len(SERIES_SLOTS),)


def test_slices_per_slot_greater_than_one_uses_multi_slice_loader():
    study_id = "study_1"
    series_df = _series_with_slots(study_id)
    targets = _targets(study_id)
    K = 4
    dataset = StudyDataset(
        [study_id], series_df, series_root="fake_root", targets=targets, cache_images=False,
        slices_per_slot=K,
    )

    fake_stack = np.full((K, dataset.resize_px, dataset.resize_px), 0.5, dtype=np.float32)
    with patch("rsna_knee.datasets.load_series_representative") as mock_single, \
         patch("rsna_knee.datasets.load_series_multi_slice", return_value=fake_stack) as mock_multi:
        item = dataset[0]

    assert mock_single.call_count == 0
    assert mock_multi.call_count == 1
    assert mock_multi.call_args.kwargs["k"] == K
    # (S, K, 3, H, W) -- one extra dim vs the slices_per_slot=1 case
    assert item["images"].shape == (len(SERIES_SLOTS), K, 3, dataset.resize_px, dataset.resize_px)
    assert item["presence_mask"].shape == (len(SERIES_SLOTS),)  # unchanged -- still one flag per slot, not per slice


def test_slices_per_slot_absent_slot_stays_zero_and_masked():
    study_id = "study_1"
    # Only SAG_T1 present -- every other slot (including its K sub-images) should
    # stay all-zero and masked, same semantics as the K=1 case.
    series_df = _series_with_slots(study_id, slot="SAG_T1")
    targets = _targets(study_id)
    K = 3
    dataset = StudyDataset(
        [study_id], series_df, series_root="fake_root", targets=targets, cache_images=False,
        slices_per_slot=K,
    )

    fake_stack = np.full((K, dataset.resize_px, dataset.resize_px), 0.7, dtype=np.float32)
    with patch("rsna_knee.datasets.load_series_multi_slice", return_value=fake_stack):
        item = dataset[0]

    sag_t1_idx = SERIES_SLOTS.index("SAG_T1")
    other_idx = SERIES_SLOTS.index("COR_FLUID_FS")
    assert item["presence_mask"][sag_t1_idx] == 1.0
    assert item["presence_mask"][other_idx] == 0.0
    assert item["images"][sag_t1_idx].max() > 0
    assert item["images"][other_idx].max() == 0
