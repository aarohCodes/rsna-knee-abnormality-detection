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

from rsna_knee.data.dicom_io import SERIES_SLOTS, load_series_representative
from rsna_knee.data.preprocess import DEFAULT_PHYSICAL_CROP_MM, DEFAULT_RESIZE_PX
from rsna_knee.metrics import LABEL_COLUMNS

# Each cached study is len(SERIES_SLOTS) * resize_px^2 float32 -- at the default 336px,
# ~2.7MB/study. With num_workers>0 each worker process holds its own independent
# cache, and shuffling can scatter a given study across different workers on
# different epochs, so per-worker memory isn't naturally bounded by "one copy of the
# dataset" -- it can approach (num_workers * full dataset) in the worst case. Capping
# per-worker cache size keeps memory bounded regardless of worker count or dataset size.
DEFAULT_MAX_CACHE_STUDIES = 1500


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
        # DICOM decode dominates cost -- caching per study avoids re-decoding on every
        # epoch. Only helps beyond one epoch if the DataLoader keeps worker processes
        # alive across epochs (persistent_workers=True when num_workers>0); PyTorch
        # respawns workers every epoch by default, which would silently make an
        # in-process cache useless. Bounded LRU (see DEFAULT_MAX_CACHE_STUDIES above)
        # rather than an unbounded dict, to keep per-worker memory predictable.
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.study_ids)

    def _load_images_and_mask(self, study_id: str) -> tuple[np.ndarray, np.ndarray]:
        if self.cache_images and study_id in self._cache:
            self._cache.move_to_end(study_id)
            return self._cache[study_id]

        series_rows = self.series_by_study.get(study_id, pd.DataFrame(columns=["slot"]))

        images = np.zeros((len(SERIES_SLOTS), self.resize_px, self.resize_px), dtype=np.float32)
        mask = np.zeros(len(SERIES_SLOTS), dtype=np.float32)

        for slot_idx, slot in enumerate(SERIES_SLOTS):
            matches = series_rows[series_rows["slot"] == slot]
            if matches.empty:
                continue
            series_dir = self.series_root / study_id / matches.iloc[0]["SeriesInstanceUID"]
            img = load_series_representative(series_dir, self.physical_crop_mm, self.resize_px)
            if img is not None:
                images[slot_idx] = img
                mask[slot_idx] = 1.0

        if mask.sum() == 0 and not series_rows.empty:
            # No series matched a canonical slot (or all failed to load) but the study
            # has series available -- fall back to any one of them in slot 0 rather
            # than feeding the model an all-masked study (undefined softmax).
            for _, row in series_rows.iterrows():
                series_dir = self.series_root / study_id / row["SeriesInstanceUID"]
                img = load_series_representative(series_dir, self.physical_crop_mm, self.resize_px)
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
        images_3ch = np.repeat(images[:, np.newaxis, :, :], 3, axis=1)  # (S, 3, H, W)

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
