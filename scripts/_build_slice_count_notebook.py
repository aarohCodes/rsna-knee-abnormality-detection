"""Generate diagnostics_slice_count/diagnostic_kernel.ipynb.

Not a competition submission -- a CPU-only diagnostic kernel (no GPU, no model, no
DICOM header parsing, just counting files) to scope roadmap item 6 (multi-slice/
volumetric input) before committing to building it. PLAN.md's original EDA plan
estimated "20-45 slices per series, median 30, tail to a few hundred" but that was
an initial estimate, not a corpus-wide measurement -- this gets exact numbers so any
volumetric design (how many slices per series to actually process, expected
decode/inference cost, memory footprint) is based on real data, not a guess.

Counts *.dcm files per series directory across the full training corpus (24,371
series) and reports the distribution overall and per canonical slot (since compute
budget for a volumetric redesign scales with slices actually used in the slots we
care about, not raw file count everywhere). This directly informs: how much more
compute/runtime a volumetric approach would cost versus the current one-crop-per-
series approach (relevant given the 9-hour submission runtime cap), and what a
reasonable per-series slice cap would be if we don't want to process every single
slice.
"""
import json
from pathlib import Path

CODE_CELLS = [
    # Cell 1: path discovery
    """
import sys
from pathlib import Path

INPUT_ROOT = Path('/kaggle/input')
OUT_DIR = Path('/kaggle/working')

print('Contents of /kaggle/input:', sorted(p.name for p in INPUT_ROOT.iterdir()))


_SKIP_DIR_NAMES = {'train_series', 'test_series'}


def find_dir_containing(marker_relpath, root=INPUT_ROOT, max_depth=8):
    \"\"\"Breadth-first search under /kaggle/input for a directory containing
    `marker_relpath`, pruning train_series/test_series before descending into them
    (they hold hundreds of thousands of per-study/per-series directories and none of
    our markers live inside them -- a naive search enumerates the whole tree instead,
    confirmed costing ~8 minutes per run without this).\"\"\"
    frontier = [root]
    for _ in range(max_depth + 1):
        next_frontier = []
        for candidate in frontier:
            if (candidate / marker_relpath).exists():
                return candidate
            try:
                children = [c for c in candidate.iterdir() if c.is_dir() and c.name not in _SKIP_DIR_NAMES]
            except OSError:
                continue
            next_frontier.extend(children)
        frontier = next_frontier
    raise FileNotFoundError(
        f'Could not find a directory containing {marker_relpath!r} under {root} '
        f'(searched up to depth {max_depth}). Top-level contents: '
        f'{sorted(p.name for p in root.iterdir())}'
    )


SRC_DATASET = find_dir_containing('rsna_knee/__init__.py')
print('Resolved SRC_DATASET:', SRC_DATASET)
sys.path.insert(0, str(SRC_DATASET))

COMPETITION_ROOT = find_dir_containing('sample_submission.csv')
print('Resolved COMPETITION_ROOT:', COMPETITION_ROOT)
""",
    # Cell 2: count *.dcm files per series directory -- pure filesystem listing, no
    # DICOM parsing at all, so this is fast even over the full corpus.
    """
import pandas as pd
import numpy as np

train_series_csv = pd.read_csv(COMPETITION_ROOT / 'train_series.csv')
train_series_dir = COMPETITION_ROOT / 'train_series'

print(f'Counting slices in {len(train_series_csv)} series directories...')
counts = []
for _, row in train_series_csv.iterrows():
    series_dir = train_series_dir / row['StudyInstanceUID'] / row['SeriesInstanceUID']
    n = len(list(series_dir.glob('*.dcm'))) if series_dir.is_dir() else 0
    counts.append(n)
train_series_csv['n_slices'] = counts
train_series_csv.to_csv(OUT_DIR / 'train_series_slice_counts.csv', index=False)

print('Slice-count distribution across all series:')
print(train_series_csv['n_slices'].describe())
print()
print('Percentiles:')
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f'  p{p}: {np.percentile(train_series_csv["n_slices"], p):.1f}')
""",
    # Cell 3: per-slot breakdown (needs the slot classification -- one DICOM header
    # per series, same cost as the earlier slot-coverage diagnostic)
    """
from rsna_knee.data.series import assign_slots
from rsna_knee.data.dicom_io import SERIES_SLOTS

print('Classifying slots (one header per series)...')
series_with_slots = assign_slots(train_series_csv.drop(columns=['n_slices']), train_series_dir)
series_with_slots['n_slices'] = train_series_csv['n_slices'].values

print('Slice-count distribution by canonical slot (only rows that classified into one):')
for slot in SERIES_SLOTS:
    subset = series_with_slots[series_with_slots['slot'] == slot]['n_slices']
    if len(subset) == 0:
        print(f'  {slot:16s} n/a (no series)')
        continue
    print(f'  {slot:16s} n={len(subset):5d}  mean={subset.mean():.1f}  median={subset.median():.0f}  p90={np.percentile(subset, 90):.0f}  max={subset.max()}')

# Total decode volume implied by different strategies -- directly informs runtime
# budget for a volumetric redesign against the 9-hour submission cap.
current_approach_decodes = len(series_with_slots)  # 1 representative slice per series today (n_percentile_slices=3 actually, but ~1 conceptually)
all_slices_total = int(series_with_slots['n_slices'].sum())
in_slot_series = series_with_slots[series_with_slots['slot'].notna()]
all_slices_in_slots = int(in_slot_series['n_slices'].sum())
print(f'\\nTotal series: {len(series_with_slots)}')
print(f'Total slices across ALL series (every plane/weighting, not just our 6 slots): {all_slices_total:,}')
print(f'Total slices within just our 6 canonical slots: {all_slices_in_slots:,}')
print(f'Ratio (all-slices-in-slots / current one-crop-per-series-slot): {all_slices_in_slots / len(in_slot_series):.1f}x more decodes per study if using every slice')
""",
]


def make_notebook(cells: list[str]) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": cell.strip("\n"),
            }
            for cell in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parents[1] / "diagnostics_slice_count" / "diagnostic_kernel.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_notebook(CODE_CELLS), indent=1))
    print(f"Wrote {out_path}")
