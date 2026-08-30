import numpy as np
import pytest

from rsna_knee.data.preprocess import (
    center_crop_to_physical_extent,
    effective_pixel_pitch_mm,
    min_resolvable_feature_mm,
    normalize_intensity,
    resize_to,
)


def test_center_crop_produces_correct_physical_size():
    image = np.zeros((512, 512), dtype=np.float32)
    cropped = center_crop_to_physical_extent(image, pixel_spacing=(0.5, 0.5), physical_extent_mm=130.0)
    assert cropped is not None
    assert cropped.shape == (260, 260)  # 130mm / 0.5mm-per-pixel


def test_center_crop_is_actually_centered():
    image = np.zeros((100, 100), dtype=np.float32)
    image[45:55, 45:55] = 1.0  # a marker at the true center
    cropped = center_crop_to_physical_extent(image, pixel_spacing=(1.0, 1.0), physical_extent_mm=20.0)
    assert cropped is not None
    assert cropped.shape == (20, 20)
    assert cropped.sum() == 100  # the whole marker should be captured, centered


def test_center_crop_returns_none_when_fov_too_small():
    image = np.zeros((50, 50), dtype=np.float32)
    # requesting 130mm crop at 1mm/px needs a 130x130 image -- larger than we have
    assert center_crop_to_physical_extent(image, pixel_spacing=(1.0, 1.0), physical_extent_mm=130.0) is None


def test_resize_to_produces_requested_shape():
    image = np.random.rand(260, 260).astype(np.float32)
    resized = resize_to(image, size_px=336)
    assert resized.shape == (336, 336)


def test_normalize_intensity_maps_to_unit_range_and_clips_outliers():
    stack = np.zeros((5, 10, 10), dtype=np.float32)
    stack[:] = np.linspace(0, 100, 500).reshape(5, 10, 10)
    stack[0, 0, 0] = 100000.0  # one extreme outlier voxel

    normed = normalize_intensity(stack)
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0
    # the outlier should be clipped to the top of the range, not stretch everything else
    assert normed.max() == pytest.approx(1.0)
    # a typical mid-range value should not be crushed near zero by the outlier
    assert normed[1, 5, 5] > 0.1


def test_normalize_intensity_constant_stack_returns_zeros_not_nan():
    stack = np.full((3, 4, 4), 7.0, dtype=np.float32)
    normed = normalize_intensity(stack)
    assert np.all(normed == 0.0)
    assert not np.isnan(normed).any()


def test_nyquist_bound_relationship():
    pitch = effective_pixel_pitch_mm(physical_extent_mm=130.0, resize_px=336)
    assert pitch == pytest.approx(0.387, abs=0.001)
    assert min_resolvable_feature_mm(pitch) == pytest.approx(0.774, abs=0.002)
