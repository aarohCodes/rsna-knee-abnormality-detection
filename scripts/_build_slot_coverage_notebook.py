"""Generate diagnostics_slot_coverage/diagnostic_kernel.ipynb.

Not a competition submission -- a CPU-only diagnostic kernel (no GPU, no model,
no training) to investigate roadmap item 5 (weak-label investigation). Across
all 6 real training runs so far (small v13/v3/v4, base v1/v2/v3), MCL (mean
holdout AUC 0.666) and Lateral Meniscus (0.730) are consistently our two
weakest labels by a wide margin -- and it isn't simply a class-imbalance
artifact, since Fracture and Synovitis are similarly or more rare in both gold
(9,11,12 positives/58 for MCL/Lateral OA/Baker's vs 18 for Fracture, 27 for
Synovitis) and silver (~15% prevalence for MCL/Lateral Meniscus vs ~7% for
Fracture, ~13% for Synovitis) yet score much higher (0.832, 0.876).

MCL is normally read off coronal sequences. Every training study has at least
one raw Coronal-plane series (confirmed locally from train_series.csv, 100%
coverage), so the gap -- if any -- must be in whether that coronal series gets
correctly classified into the COR_T1 slot specifically (needs real DICOM
header fields: ScanningSequence/RepetitionTime/EchoTime), which requires
reading the actual DICOM headers across the full corpus -- not available in
the small local sample, hence this kernel. This reads one header per series
(assign_slots, stop_before_pixels=True -- no pixel decode, cheap) across all
24,371 training series and tabulates per-slot coverage, so we can tell whether
weak MCL/Lateral Meniscus performance is a data-coverage bottleneck (fixable
by improving slot classification or crop) or a genuinely-harder-signal
bottleneck (would need a different lever, e.g. item 6's volumetric input).
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
    # Cell 2: scan every training series' DICOM header and classify slots (no GPU, no pixel decode)
    """
import pandas as pd

from rsna_knee.data.series import assign_slots
from rsna_knee.data.dicom_io import SERIES_SLOTS

train_series_csv = pd.read_csv(COMPETITION_ROOT / 'train_series.csv')
train_series_dir = COMPETITION_ROOT / 'train_series'

print(f'Scanning {len(train_series_csv)} series headers (stop_before_pixels, no pixel decode)...')
series_with_slots = assign_slots(train_series_csv, train_series_dir)
series_with_slots.to_csv(OUT_DIR / 'train_series_with_slots.csv', index=False)
print('Done. Weighting value counts:')
print(series_with_slots['weighting'].value_counts())
print()
print('Slot value counts (None = did not classify into any of the 6 canonical slots):')
print(series_with_slots['slot'].value_counts(dropna=False))
""",
    # Cell 3: per-study slot presence coverage -- the number that actually matters
    """
from rsna_knee.data.series import slot_presence_mask

coverage = {slot: 0 for slot in SERIES_SLOTS}
n_studies = 0
for study_id, group in series_with_slots.groupby('StudyInstanceUID'):
    mask = slot_presence_mask(group['slot'].tolist())
    for slot, present in mask.items():
        if present:
            coverage[slot] += 1
    n_studies += 1

print(f'Per-slot presence coverage across {n_studies} training studies:')
for slot in SERIES_SLOTS:
    pct = 100 * coverage[slot] / n_studies
    print(f'  {slot:16s} {coverage[slot]:5d} / {n_studies}  ({pct:.1f}%)')
""",
    # Cell 4: cross-check against the gold-labeled subset specifically -- do the studies
    # with a real MCL=1 gold label actually have a COR_T1 slot to learn from?
    """
train_labels = pd.read_csv(COMPETITION_ROOT / 'train.csv')
label_cols = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
              'PF OA','Effusion','Synovitis',\"Baker's\",'Contusion','Fracture']
gold_mask = train_labels[label_cols].notna().any(axis=1)
gold_ids = set(train_labels.loc[gold_mask, 'StudyInstanceUID'])
print(f'{len(gold_ids)} gold studies')

gold_slots = series_with_slots[series_with_slots['StudyInstanceUID'].isin(gold_ids)]
gold_coverage = {slot: 0 for slot in SERIES_SLOTS}
for study_id, group in gold_slots.groupby('StudyInstanceUID'):
    mask = slot_presence_mask(group['slot'].tolist())
    for slot, present in mask.items():
        if present:
            gold_coverage[slot] += 1

print(f'Per-slot presence coverage across the {len(gold_ids)} gold studies specifically:')
for slot in SERIES_SLOTS:
    pct = 100 * gold_coverage[slot] / len(gold_ids)
    print(f'  {slot:16s} {gold_coverage[slot]:5d} / {len(gold_ids)}  ({pct:.1f}%)')

# MCL is read off COR_T1; Lateral/Medial Meniscus off SAG_FLUID_FS or SAG_FLUID_NOFS.
# If MCL-positive gold studies are missing COR_T1 at a meaningfully higher rate than
# MCL-negative ones, that's a real, fixable data-coverage bottleneck rather than a
# purely-harder-signal one.
mcl_positive_ids = set(train_labels.loc[gold_mask & (train_labels['MCL'] == 1), 'StudyInstanceUID'])
mcl_negative_ids = gold_ids - mcl_positive_ids
print(f'\\nMCL-positive gold studies: {len(mcl_positive_ids)}, MCL-negative: {len(mcl_negative_ids)}')

for name, ids in [('MCL-positive', mcl_positive_ids), ('MCL-negative', mcl_negative_ids)]:
    subset = series_with_slots[series_with_slots['StudyInstanceUID'].isin(ids)]
    has_cor_t1 = 0
    for study_id, group in subset.groupby('StudyInstanceUID'):
        if slot_presence_mask(group['slot'].tolist())['COR_T1']:
            has_cor_t1 += 1
    pct = 100 * has_cor_t1 / len(ids) if ids else float('nan')
    print(f'  {name}: COR_T1 present in {has_cor_t1}/{len(ids)} ({pct:.1f}%)')
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
    out_path = Path(__file__).resolve().parents[1] / "diagnostics_slot_coverage" / "diagnostic_kernel.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_notebook(CODE_CELLS), indent=1))
    print(f"Wrote {out_path}")
