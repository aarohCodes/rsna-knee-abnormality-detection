"""PyTorch Dataset producing per-study (images, presence_mask, target, weight) tuples.

Ties together series-slot assignment (data/series.py), representative-image loading
(data/dicom_io.py), and confidence-weighted targets (data/label_expansion.py).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna_knee.data.dicom_io import (
    SERIES_SLOTS,
    SLOT_FALLBACKS,
    load_series_multi_slice,
    load_series_representative,
)
from rsna_knee.data.preprocess import DEFAULT_PHYSICAL_CROP_MM, DEFAULT_RESIZE_PX
from rsna_knee.metrics import LABEL_COLUMNS

# Each cached study is len(SERIES_SLOTS) * resize_px^2 float32 -- at the default 336px,
# ~2.7MB/study. With num_workers>0 each worker process holds its own independent
# cache, and shuffling can scatter a given study across different workers on
# different epochs, so per-worker memory isn't naturally bounded by "one copy of the
# dataset" -- it can approach (num_workers * full dataset) in the worst case. Capping
# per-worker cache size keeps memory bounded regardless of worker count or dataset size.
DEFAULT_MAX_CACHE_STUDIES = 1500

# float32 single-channel image per slot, before the 3-channel repeat done at
# __getitem__ time (the cache stores pre-repeat images -- see _load_images_and_mask).
_BYTES_PER_STUDY_DEFAULT_MB = len(SERIES_SLOTS) * DEFAULT_RESIZE_PX * DEFAULT_RESIZE_PX * 4 / 1024**2


def compute_safe_max_cache_studies(
    num_workers: int,
    target_fraction: float = 0.4,
    per_study_mb: float = _BYTES_PER_STUDY_DEFAULT_MB,
    fallback_available_gb: float = 13.0,
    min_studies: int = 200,
    max_studies: int = 5000,
) -> int:
    """Pick a per-worker cache size from *actually available* memory instead of a
    hardcoded guess. Increasing num_workers alone already multiplies worst-case cache
    memory (each worker's cache is independent -- see DEFAULT_MAX_CACHE_STUDIES's
    docstring), so a bigger cache budget must divide by num_workers, not just scale up.

    Only spends `target_fraction` of available memory on the cache (default 40%),
    leaving headroom for the model, activations, and everything else sharing the same
    process. Falls back to `fallback_available_gb` if the actual available memory
    can't be read (e.g. `psutil` not installed) rather than guessing something
    larger and risking OOM.
    """
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / 1024**3
    except Exception:
        available_gb = fallback_available_gb

    budget_mb = available_gb * 1024 * target_fraction
    per_worker_studies = int(budget_mb / per_study_mb / max(num_workers, 1))
    return max(min_studies, min(max_studies, per_worker_studies))


class StudyDataset(Dataset):
    def __init__(
        self,
        study_ids: list[str],
        series_with_slots: pd.DataFrame,
        series_root: Path,
        targets: pd.DataFrame,
        physical_crop_mm: float = DEFAULT_PHYSICAL_CROP_MM,
        resize_px: int = DEFAULT_RESIZE_PX,
        cache_images: bool = True,
        max_cache_studies: int = DEFAULT_MAX_CACHE_STUDIES,
        center_fraction: float = 0.5,
        use_slot_fallbacks: bool = False,
        slices_per_slot: int = 1,
    ):
        self.study_ids = list(study_ids)
        self.series_by_study = {
            study: group for study, group in series_with_slots.groupby("StudyInstanceUID")
        }
        self.series_root = Path(series_root)
        self.targets = targets.set_index("StudyInstanceUID")
        self.physical_crop_mm = physical_crop_mm
        self.resize_px = resize_px
        self.cache_images = cache_images
        self.max_cache_studies = max_cache_studies
        # Off by default: a controlled test (rsna-knee-base-fallback, see STATUS.md
        # roadmap item 5) found this made results WORSE, not better -- tier1 macro
        # 0.7931->0.7791 and MCL itself 0.663->0.641, despite the data-coverage
        # rationale seeming sound. Leading hypothesis: marking a substituted slot
        # mask=1.0 identically to a genuine one gives the model no way to distinguish
        # real ligament-optimized T1 contrast from substituted fluid-sensitive
        # contrast at the same slot position, adding noise rather than removing a
        # gap. Kept available (not deleted) for a future, better-controlled retry
        # (e.g. a distinct fallback indicator) but must not silently affect the
        # known-good default pipeline used by our current best checkpoint (base v2).
        self.use_slot_fallbacks = use_slot_fallbacks
        # 1 (default) = existing single-crop-per-slot behavior, images shaped
        # (S, resize_px, resize_px) -- unchanged for every existing caller. >1 loads
        # `slices_per_slot` evenly-spaced crops per slot instead of one (roadmap item
        # 6: only ~18% of each study's files were being used), images shaped
        # (S, slices_per_slot, resize_px, resize_px); KneeMultiLabelModel.forward
        # accepts either shape (mean-pools over the extra dimension when present, a
        # no-op for slices_per_slot=1, so existing checkpoints stay loadable).
        self.slices_per_slot = slices_per_slot
        # Which slice-window to sample per series (0.5 = middle, the default). TTA
        # varies this across several StudyDataset instances at inference time and
        # combines their predictions -- see infer.run_inference_tta.
        self.center_fraction = center_fraction
        # DICOM decode dominates cost -- caching per study avoids re-decoding on every
        # epoch. Only helps beyond one epoch if the DataLoader keeps worker processes
        # alive across epochs (persistent_workers=True when num_workers>0); PyTorch
        # respawns workers every epoch by default, which would silently make an
        # in-process cache useless. Bounded LRU (see DEFAULT_MAX_CACHE_STUDIES above)
        # rather than an unbounded dict, to keep per-worker memory predictable.
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.study_ids)

    def _load_one_series(self, series_dir: Path) -> np.ndarray | None:
        """Load either one crop (resize_px, resize_px) or `slices_per_slot` evenly-
        spaced crops (slices_per_slot, resize_px, resize_px) from a series, depending
        on `self.slices_per_slot` -- the single call site every slot-filling branch
        below goes through, so they don't need to duplicate this branch themselves.
        """
        if self.slices_per_slot <= 1:
            return load_series_representative(
                series_dir, self.physical_crop_mm, self.resize_px, center_fraction=self.center_fraction
            )
        return load_series_multi_slice(
            series_dir, self.physical_crop_mm, self.resize_px, k=self.slices_per_slot
        )

    def _load_images_and_mask(self, study_id: str) -> tuple[np.ndarray, np.ndarray]:
        if self.cache_images and study_id in self._cache:
            self._cache.move_to_end(study_id)
            return self._cache[study_id]

        series_rows = self.series_by_study.get(study_id, pd.DataFrame(columns=["slot"]))

        per_slot_shape = (
            (len(SERIES_SLOTS), self.resize_px, self.resize_px)
            if self.slices_per_slot <= 1
            else (len(SERIES_SLOTS), self.slices_per_slot, self.resize_px, self.resize_px)
        )
        images = np.zeros(per_slot_shape, dtype=np.float32)
        mask = np.zeros(len(SERIES_SLOTS), dtype=np.float32)

        for slot_idx, slot in enumerate(SERIES_SLOTS):
            matches = series_rows[series_rows["slot"] == slot]
            if matches.empty and self.use_slot_fallbacks and slot in SLOT_FALLBACKS:
                # See SLOT_FALLBACKS' docstring: COR_T1 is missing in ~52% of studies,
                # non-selectively, so substituting the next-best coronal view beats
                # leaving this slot position all-zero/masked-out. If the fallback slot
                # (e.g. COR_FLUID_FS) is itself present as its own slot too, its image
                # is intentionally reused here -- duplicated coronal signal is still
                # strictly more informative than none.
                matches = series_rows[series_rows["slot"] == SLOT_FALLBACKS[slot]]
            if matches.empty:
                continue
            series_dir = self.series_root / study_id / matches.iloc[0]["SeriesInstanceUID"]
            img = self._load_one_series(series_dir)
            if img is not None:
                images[slot_idx] = img
                mask[slot_idx] = 1.0

        if mask.sum() == 0 and not series_rows.empty:
            # No series matched a canonical slot (or all failed to load) but the study
            # has series available -- fall back to any one of them in slot 0 rather
            # than feeding the model an all-masked study (undefined softmax).
            for _, row in series_rows.iterrows():
                series_dir = self.series_root / study_id / row["SeriesInstanceUID"]
                img = self._load_one_series(series_dir)
                if img is not None:
                    images[0] = img
                    mask[0] = 1.0
                    break

        if self.cache_images:
            self._cache[study_id] = (images, mask)
            if len(self._cache) > self.max_cache_studies:
                self._cache.popitem(last=False)  # evict least-recently-used
        return images, mask

    def __getitem__(self, idx: int) -> dict:
        study_id = self.study_ids[idx]
        images, mask = self._load_images_and_mask(study_id)
        # (S, H, W) -> (S, 3, H, W), or (S, K, H, W) -> (S, K, 3, H, W) when
        # slices_per_slot > 1 -- 3-channel repeat always happens on the last two dims.
        images_3ch = np.repeat(images[..., np.newaxis, :, :], 3, axis=-3)

        if study_id in self.targets.index:
            target_row = self.targets.loc[study_id]
            target = np.array([target_row[c] for c in LABEL_COLUMNS], dtype=np.float32)
            weight = np.array([target_row[f"{c}__weight"] for c in LABEL_COLUMNS], dtype=np.float32)
        else:
            target = np.full(len(LABEL_COLUMNS), 0.5, dtype=np.float32)
            weight = np.zeros(len(LABEL_COLUMNS), dtype=np.float32)

        return {
            "study_id": study_id,
            "images": torch.from_numpy(images_3ch),
            "presence_mask": torch.from_numpy(mask),
            "target": torch.from_numpy(target),
            "weight": torch.from_numpy(weight),
        }
