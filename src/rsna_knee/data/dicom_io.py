"""DICOM reading, slice ordering, series-slot classification, and laterality.

Techniques here follow the public baseline notebook (pilkwang/rsna-knee-baseline-v1),
whose reasoning is documented in PLAN.md:

- Slice order must come from geometry (ImageOrientationPatient x ImagePositionPatient),
  not from the SOP Instance UID filename, which is unordered.
- `Fluid_Sensitive`/`Fat_Suppression` in train_series.csv are degenerate (they always
  agree), so the true weighting (T1/T2/PD) and fat-sat are recovered from DICOM header
  fields and combined with `Anatomical_Plane` into one of six canonical slots.
- Knee laterality (left/right) is recovered from the sign of the image center's
  patient-coordinate x, since the `Laterality` tag is frequently absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

from rsna_knee.data.preprocess import (
    DEFAULT_PHYSICAL_CROP_MM,
    DEFAULT_RESIZE_PX,
    center_crop_to_physical_extent,
    normalize_intensity,
    resize_to,
)

SERIES_SLOTS = [
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "SAG_FLUID_NOFS",
    "COR_T1",
    "SAG_T1",
]

_MIDLINE_UNRESOLVED_BAND_MM = 15.0


@dataclass(frozen=True)
class SliceGeometry:
    """The minimal geometry fields needed for ordering and laterality, extracted from
    a DICOM dataset so the rest of this module doesn't depend on pydicom directly."""

    position: np.ndarray  # ImagePositionPatient, shape (3,)
    row_cosine: np.ndarray  # first 3 values of ImageOrientationPatient
    col_cosine: np.ndarray  # last 3 values of ImageOrientationPatient
    pixel_spacing: tuple[float, float]  # (row spacing, col spacing), mm
    rows: int
    columns: int
    instance_number: int | None = None


def geometry_from_dataset(ds) -> SliceGeometry | None:
    """Extract SliceGeometry from a pydicom Dataset. Returns None if the geometry tags
    required for ordering/laterality (ImageOrientationPatient, ImagePositionPatient) are
    absent, so callers can fall back to InstanceNumber-based ordering.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None:
        return None
    spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    return SliceGeometry(
        position=np.array([float(x) for x in ipp], dtype=float),
        row_cosine=np.array([float(x) for x in iop[:3]], dtype=float),
        col_cosine=np.array([float(x) for x in iop[3:6]], dtype=float),
        pixel_spacing=(float(spacing[0]), float(spacing[1])),
        rows=int(getattr(ds, "Rows", 0)),
        columns=int(getattr(ds, "Columns", 0)),
        instance_number=(
            int(ds.InstanceNumber) if getattr(ds, "InstanceNumber", None) is not None else None
        ),
    )


def slice_normal(geom: SliceGeometry) -> np.ndarray:
    """The through-plane normal n = row_cosine x col_cosine."""
    return np.cross(geom.row_cosine, geom.col_cosine)


def slice_scalar_position(geom: SliceGeometry) -> float:
    """k = position . normal -- increases monotonically along the true slice stack."""
    return float(np.dot(geom.position, slice_normal(geom)))


def order_slices(paths_and_geometry: list[tuple[Path, SliceGeometry | None]]) -> list[Path]:
    """Order a series' slice files by true spatial position.

    Uses geometry (position projected onto the slice normal) when every slice has it;
    falls back to InstanceNumber only when geometry is missing for any slice, since a
    partial geometric ordering mixed with a partial InstanceNumber ordering would be
    worse than either alone.
    """
    if all(geom is not None for _, geom in paths_and_geometry):
        return [
            p
            for p, _ in sorted(
                paths_and_geometry, key=lambda pg: slice_scalar_position(pg[1])
            )
        ]
    if all(geom is not None and geom.instance_number is not None for _, geom in paths_and_geometry):
        return [
            p
            for p, _ in sorted(paths_and_geometry, key=lambda pg: pg[1].instance_number)
        ]
    # Nothing usable to sort by -- return input order unchanged rather than guessing.
    return [p for p, _ in paths_and_geometry]


def knee_side(geometries: list[SliceGeometry]) -> str | None:
    """Determine 'left' or 'right' from the median sign of each slice's image-center
    x-coordinate in patient space. +x points to the patient's left (DICOM LPS
    convention). Returns None (unresolved) if the median center is within
    _MIDLINE_UNRESOLVED_BAND_MM of the midline, where the sign is not reliably better
    than chance.
    """
    centers_x = []
    for geom in geometries:
        row_extent = geom.row_cosine * geom.pixel_spacing[1] * geom.columns / 2.0
        col_extent = geom.col_cosine * geom.pixel_spacing[0] * geom.rows / 2.0
        center = geom.position + row_extent + col_extent
        centers_x.append(center[0])
    median_x = float(np.median(centers_x))
    if abs(median_x) < _MIDLINE_UNRESOLVED_BAND_MM:
        return None
    return "right" if median_x < 0 else "left"


def classify_weighting(
    scanning_sequence: str | None,
    repetition_time: float | None,
    echo_time: float | None,
) -> str:
    """Recover T1 / T2 / PD contrast weighting from acquisition parameters.

    Gradient-echo sequences break the TR/TE heuristic (their TR is short by design), so
    those are checked via ScanningSequence first.
    """
    seq = (scanning_sequence or "").upper()
    if "GR" in seq:
        return "GR"
    if repetition_time is None or echo_time is None:
        return "UNKNOWN"
    if repetition_time < 800:
        return "T1"
    if echo_time >= 60:
        return "T2"
    return "PD"


def classify_slot(
    anatomical_plane: str,
    weighting: str,
    fat_suppressed: bool,
) -> str | None:
    """Map (plane, weighting, fat-sat) to one of the six canonical series slots.

    Returns None for combinations that don't correspond to a defined slot (e.g. axial
    T1), so callers can decide whether to drop or fold such series in with the closest
    slot rather than silently mis-slotting them.
    """
    fluid = weighting in ("T2", "PD")
    plane = anatomical_plane.upper()[:3]  # SAG / COR / AXI
    if plane == "SAG" and fluid and fat_suppressed:
        return "SAG_FLUID_FS"
    if plane == "COR" and fluid and fat_suppressed:
        return "COR_FLUID_FS"
    if plane == "AXI" and fluid and fat_suppressed:
        return "AX_FLUID_FS"
    if plane == "SAG" and fluid and not fat_suppressed:
        return "SAG_FLUID_NOFS"
    if plane == "COR" and weighting == "T1" and not fat_suppressed:
        return "COR_T1"
    if plane == "SAG" and weighting == "T1" and not fat_suppressed:
        return "SAG_T1"
    return None


def load_series_representative(
    series_dir: Path,
    physical_crop_mm: float = DEFAULT_PHYSICAL_CROP_MM,
    resize_px: int = DEFAULT_RESIZE_PX,
    n_percentile_slices: int = 5,
    center_fraction: float = 0.5,
) -> np.ndarray | None:
    """Load one representative, preprocessed image for a series: order its slices by
    true spatial position, take the slice at `center_fraction` through the ordered
    stack (0.5 = middle, the default), crop to a fixed physical FOV, resize to a
    Nyquist-adequate pixel pitch, and percentile-normalize using a small sample of
    slices around that center (not just the one slice) for a more stable intensity
    range. `center_fraction` != 0.5 is for multi-crop TTA at inference time -- e.g.
    0.35/0.5/0.65 sample three different, overlapping-but-distinct views of the same
    series to average over, rather than relying on a single slice choice.

    Returns None if the series has no decodable slices or none of them survive the
    physical-FOV crop (their acquired field of view is smaller than the crop target).
    """
    paths = sorted(series_dir.glob("*.dcm"))
    if not paths:
        return None

    datasets = []
    geometries = []
    for p in paths:
        try:
            ds = pydicom.dcmread(p)
        except Exception:
            continue
        geom = geometry_from_dataset(ds)
        datasets.append((p, ds))
        geometries.append(geom)

    if not datasets:
        return None

    ordered_paths = order_slices(list(zip((p for p, _ in datasets), geometries)))
    path_to_ds = {p: ds for p, ds in datasets}
    ordered_ds = [path_to_ds[p] for p in ordered_paths]

    center = int(round(center_fraction * (len(ordered_ds) - 1)))
    center = max(0, min(len(ordered_ds) - 1, center))
    half_window = n_percentile_slices // 2
    sample_idx = range(max(0, center - half_window), min(len(ordered_ds), center + half_window + 1))

    pixel_spacing = getattr(ordered_ds[center], "PixelSpacing", None)
    if pixel_spacing is None:
        return None
    pixel_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))

    cropped_slices = []
    for i in sample_idx:
        try:
            arr = ordered_ds[i].pixel_array.astype(np.float32)
        except Exception:
            continue
        cropped = center_crop_to_physical_extent(arr, pixel_spacing, physical_crop_mm)
        if cropped is None:
            return None
        cropped_slices.append(cropped)

    if not cropped_slices:
        return None

    stack = np.stack(cropped_slices, axis=0)
    normalized = normalize_intensity(stack)
    representative = normalized[len(normalized) // 2]
    return resize_to(representative, resize_px)
