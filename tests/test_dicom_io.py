from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rsna_knee.data.dicom_io import (
    SliceGeometry,
    classify_slot,
    classify_weighting,
    knee_side,
    load_series_multi_slice,
    order_slices,
    slice_scalar_position,
)


def _geom(z_mm: float, instance_number: int | None = None, x_center_offset: float = 0.0) -> SliceGeometry:
    """A simple axial-ish stack: row/col cosines are the standard identity in-plane
    axes, so the slice normal is +z, and position varies only in z. x_center_offset is
    the desired image-CENTER x for laterality tests; ImagePositionPatient is the
    slice's corner (per DICOM convention), so it must be back-computed from the
    center by subtracting the same row/col half-extents `knee_side` adds back on.
    """
    pixel_spacing = (1.0, 1.0)
    rows, columns = 100, 100
    row_extent_x = pixel_spacing[1] * columns / 2.0  # row_cosine=[1,0,0] -> all in x
    return SliceGeometry(
        position=np.array([x_center_offset - row_extent_x, 0.0, z_mm]),
        row_cosine=np.array([1.0, 0.0, 0.0]),
        col_cosine=np.array([0.0, 1.0, 0.0]),
        pixel_spacing=pixel_spacing,
        rows=rows,
        columns=columns,
        instance_number=instance_number,
    )


def test_slice_scalar_position_increases_along_normal():
    g1 = _geom(z_mm=0.0)
    g2 = _geom(z_mm=5.0)
    assert slice_scalar_position(g2) > slice_scalar_position(g1)


def test_order_slices_by_geometry_ignores_filename_order():
    # Deliberately shuffled: file "c" is physically first, "a" is physically last.
    items = [
        (Path("a.dcm"), _geom(z_mm=10.0)),
        (Path("b.dcm"), _geom(z_mm=5.0)),
        (Path("c.dcm"), _geom(z_mm=0.0)),
    ]
    ordered = order_slices(items)
    assert ordered == [Path("c.dcm"), Path("b.dcm"), Path("a.dcm")]


def test_order_slices_falls_back_to_instance_number_when_geometry_missing():
    items = [
        (Path("a.dcm"), None),
        (Path("b.dcm"), None),
    ]
    # geometry missing entirely -> can't sort by geometry; instance_number also None
    # on these entries, so falls back to input order.
    assert order_slices(items) == [Path("a.dcm"), Path("b.dcm")]

    items2 = [
        (Path("x.dcm"), _geom(z_mm=0.0, instance_number=3)),
        (Path("y.dcm"), None),
        (Path("z.dcm"), _geom(z_mm=0.0, instance_number=1)),
    ]
    # one entry has no geometry at all -> can't use geometry path (requires all present)
    # and can't use instance_number path either since y.dcm's is None -> input order.
    assert order_slices(items2) == [Path("x.dcm"), Path("y.dcm"), Path("z.dcm")]


def test_knee_side_right_and_left():
    # Negative x center -> right knee (DICOM LPS: +x points to patient's left)
    right_geoms = [_geom(z_mm=z, x_center_offset=-40.0) for z in range(5)]
    assert knee_side(right_geoms) == "right"

    left_geoms = [_geom(z_mm=z, x_center_offset=40.0) for z in range(5)]
    assert knee_side(left_geoms) == "left"


def test_knee_side_unresolved_near_midline():
    near_midline = [_geom(z_mm=z, x_center_offset=2.0) for z in range(5)]
    assert knee_side(near_midline) is None


@pytest.mark.parametrize(
    "seq,tr,te,expected",
    [
        ("GR", 500, 20, "GR"),
        (None, 500, 15, "T1"),
        (None, 2000, 80, "T2"),
        (None, 2000, 20, "PD"),
        (None, None, None, "UNKNOWN"),
    ],
)
def test_classify_weighting(seq, tr, te, expected):
    assert classify_weighting(seq, tr, te) == expected


@pytest.mark.parametrize(
    "plane,weighting,fat_sat,expected",
    [
        ("Sagittal", "T2", True, "SAG_FLUID_FS"),
        ("Coronal", "PD", True, "COR_FLUID_FS"),
        ("Axial", "T2", True, "AX_FLUID_FS"),
        ("Sagittal", "PD", False, "SAG_FLUID_NOFS"),
        ("Coronal", "T1", False, "COR_T1"),
        ("Sagittal", "T1", False, "SAG_T1"),
        ("Axial", "T1", False, None),
    ],
)
def test_classify_slot(plane, weighting, fat_sat, expected):
    assert classify_slot(plane, weighting, fat_sat) == expected


def test_load_series_multi_slice_decodes_series_only_once():
    # The whole point of load_series_multi_slice vs calling load_series_representative
    # k times: decode (file I/O + DICOM parsing) happens once regardless of k, only
    # the cheap crop/normalize/resize step repeats. Mock the decode step and assert
    # it's called exactly once even for k=5.
    fake_ordered_ds = [object()] * 30  # stand-in for an ordered DICOM stack
    fake_crop = np.zeros((16, 16), dtype=np.float32)
    with patch("rsna_knee.data.dicom_io._load_ordered_dicom_stack", return_value=fake_ordered_ds) as mock_decode, \
         patch("rsna_knee.data.dicom_io._extract_representative_crop", return_value=fake_crop) as mock_extract:
        result = load_series_multi_slice(Path("fake_series"), k=5)

    assert mock_decode.call_count == 1
    assert mock_extract.call_count == 5
    assert result.shape == (5, 16, 16)


def test_load_series_multi_slice_uses_evenly_spaced_fractions():
    fake_ordered_ds = [object()] * 30
    fake_crop = np.zeros((16, 16), dtype=np.float32)
    with patch("rsna_knee.data.dicom_io._load_ordered_dicom_stack", return_value=fake_ordered_ds), \
         patch("rsna_knee.data.dicom_io._extract_representative_crop", return_value=fake_crop) as mock_extract:
        load_series_multi_slice(Path("fake_series"), k=3)

    used_fractions = [call.args[-1] for call in mock_extract.call_args_list]
    assert used_fractions == pytest.approx([0.0, 0.5, 1.0])


def test_load_series_multi_slice_k1_uses_middle_fraction():
    fake_ordered_ds = [object()] * 30
    fake_crop = np.zeros((16, 16), dtype=np.float32)
    with patch("rsna_knee.data.dicom_io._load_ordered_dicom_stack", return_value=fake_ordered_ds), \
         patch("rsna_knee.data.dicom_io._extract_representative_crop", return_value=fake_crop) as mock_extract:
        result = load_series_multi_slice(Path("fake_series"), k=1)

    used_fractions = [call.args[-1] for call in mock_extract.call_args_list]
    assert used_fractions == pytest.approx([0.5])
    assert result.shape == (1, 16, 16)


def test_load_series_multi_slice_returns_none_when_series_undecodable():
    with patch("rsna_knee.data.dicom_io._load_ordered_dicom_stack", return_value=None):
        result = load_series_multi_slice(Path("fake_series"), k=3)
    assert result is None


def test_load_series_multi_slice_returns_none_if_any_crop_fails():
    fake_ordered_ds = [object()] * 30
    fake_crop = np.zeros((16, 16), dtype=np.float32)
    # second crop fails (e.g. that slice's FOV doesn't survive the physical crop)
    with patch("rsna_knee.data.dicom_io._load_ordered_dicom_stack", return_value=fake_ordered_ds), \
         patch("rsna_knee.data.dicom_io._extract_representative_crop", side_effect=[fake_crop, None, fake_crop]):
        result = load_series_multi_slice(Path("fake_series"), k=3)
    assert result is None
