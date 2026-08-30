"""Physical-scale cropping/resizing and intensity normalization.

Per PLAN.md section 4: a fixed-pixel resize with no regard for physical spacing hands
the encoder images whose real-world scale differs by a factor of several across this
corpus, and can destroy sub-2-pixel features (a meniscal tear is 1-3mm) before the
first convolution. Crop to a fixed physical field of view, then resize to a pixel
count chosen so the effective pitch clears the Nyquist bound for the smallest feature
worth resolving.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_PHYSICAL_CROP_MM = 130.0
DEFAULT_RESIZE_PX = 336


def center_crop_to_physical_extent(
    image: np.ndarray,
    pixel_spacing: tuple[float, float],
    physical_extent_mm: float = DEFAULT_PHYSICAL_CROP_MM,
) -> np.ndarray | None:
    """Center-crop `image` (H, W) to a square of `physical_extent_mm` millimetres.

    Returns None if the crop would exceed the image's actual extent -- silently
    cropping to less than the requested physical size would defeat the point of a
    physical-scale crop (every study would end up at a different effective FOV).
    """
    row_spacing, col_spacing = pixel_spacing
    crop_h = int(round(physical_extent_mm / row_spacing))
    crop_w = int(round(physical_extent_mm / col_spacing))
    h, w = image.shape
    if crop_h > h or crop_w > w:
        return None
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return image[top : top + crop_h, left : left + crop_w]


def resize_to(image: np.ndarray, size_px: int = DEFAULT_RESIZE_PX) -> np.ndarray:
    return cv2.resize(image, (size_px, size_px), interpolation=cv2.INTER_LINEAR)


def normalize_intensity(stack: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile-normalize a stack of slices (N, H, W) to [0, 1] using the 1st/99th
    percentile over the whole sampled stack (not per-slice), so slices keep their
    relative contrast and a single bright outlier doesn't compress everything else.
    """
    lo, hi = np.percentile(stack, [low, high])
    if hi <= lo:
        return np.zeros_like(stack, dtype=np.float32)
    clipped = np.clip(stack, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def effective_pixel_pitch_mm(physical_extent_mm: float, resize_px: int) -> float:
    return physical_extent_mm / resize_px


def min_resolvable_feature_mm(effective_pitch_mm: float) -> float:
    """Nyquist bound: the smallest feature width the given pitch can resolve."""
    return 2.0 * effective_pitch_mm
