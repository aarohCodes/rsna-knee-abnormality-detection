"""Join train_series.csv/test_series.csv with DICOM-header-derived slot classification.

`Fluid_Sensitive`/`Fat_Suppression` in the CSVs are degenerate (always agree in this
corpus -- see dicom_io module docstring), so the true weighting is read from one
representative DICOM per series and combined with the CSV's `Anatomical_Plane` to
assign one of the six canonical slots.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydicom

from rsna_knee.data.dicom_io import SERIES_SLOTS, classify_slot, classify_weighting


def _first_dicom_file(series_dir: Path) -> Path | None:
    files = sorted(series_dir.glob("*.dcm"))
    return files[0] if files else None


def read_series_weighting(series_dir: Path) -> tuple[str, bool]:
    """Read one representative slice from a series directory and return
    (weighting, fat_suppressed). Any single slice in a series carries the same
    acquisition parameters, so reading one is sufficient and far cheaper than reading
    every slice just to classify the series.
    """
    path = _first_dicom_file(series_dir)
    if path is None:
        return "UNKNOWN", False
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    weighting = classify_weighting(
        scanning_sequence=str(getattr(ds, "ScanningSequence", "")),
        repetition_time=(
            float(ds.RepetitionTime) if getattr(ds, "RepetitionTime", None) is not None else None
        ),
        echo_time=float(ds.EchoTime) if getattr(ds, "EchoTime", None) is not None else None,
    )
    scan_options = str(getattr(ds, "ScanOptions", "") or "")
    series_desc = str(getattr(ds, "SeriesDescription", "") or "")
    sequence_name = str(getattr(ds, "SequenceName", "") or "")
    text = f"{scan_options} {series_desc} {sequence_name}".upper()
    fat_suppressed = any(tag in text for tag in ("FS", "FAT SAT", "SPAIR", "STIR", "SPIR"))
    return weighting, fat_suppressed


def assign_slots(series_df: pd.DataFrame, series_root: Path) -> pd.DataFrame:
    """Add `weighting`, `fat_suppressed`, and `slot` columns to a train_series.csv /
    test_series.csv dataframe by reading one DICOM header per series.

    Rows whose series directory isn't present locally (e.g. outside a downloaded
    sample) get `weighting="UNKNOWN"` and `slot=None` rather than raising, so this can
    run over a partial local sample without crashing on the missing majority.
    """
    weightings = []
    fat_sats = []
    slots = []
    for _, row in series_df.iterrows():
        series_dir = series_root / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
        if not series_dir.is_dir():
            weightings.append("UNKNOWN")
            fat_sats.append(False)
            slots.append(None)
            continue
        weighting, fat_suppressed = read_series_weighting(series_dir)
        weightings.append(weighting)
        fat_sats.append(fat_suppressed)
        slots.append(classify_slot(row["Anatomical_Plane"], weighting, fat_suppressed))

    out = series_df.copy()
    out["weighting"] = weightings
    out["fat_suppressed"] = fat_sats
    out["slot"] = slots
    return out


def slot_presence_mask(study_slots: list[str | None]) -> dict[str, bool]:
    """A study's per-slot presence mask over the six canonical slots."""
    present = set(s for s in study_slots if s is not None)
    return {slot: (slot in present) for slot in SERIES_SLOTS}
