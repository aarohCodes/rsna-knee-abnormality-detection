"""Generate submission_base_multislice_fresh/submission_kernel.ipynb.

The decisive test of roadmap item 6 (multi-slice/volumetric input). The first test
(rsna-knee-base-multislice: resumed base v2, +2 epochs, K=3) scored tier1=0.7793,
below v2's 0.7931 -- but that result is confounded: it's the FIFTH attempt to beat
v2 via a short resumed extension (after slot-fallback, v3, lrdecay, unfreeze4), all
five landing in the same tight 0.769-0.779 tier1 band regardless of what was
changed. Can't tell "multi-slice doesn't help" apart from "2 epochs isn't enough
exposure to a suddenly-richer input for a model already converged on K=1."

This trains multi-slice FRESH from pretrained DINOv2-base for a full 6 epochs,
matching base v2's own epoch budget exactly (same pattern as the base-fallback
experiment) instead of a short resumed extension -- the clean, decisive comparison.
K=3 (start/middle/end of each series), BATCH_SIZE=3 (cut from 8 to compensate for
~3x more images/study), N_UNFROZEN_BLOCKS=2 and standard LR (matching v2's own
recipe exactly -- isolating K as the only variable that differs from v2's own
6-epoch fresh-from-pretrained run). Estimated ~5h45m based on the first
multi-slice run's per-epoch timing (~3040s/epoch), safely under the 9h cap.
"""
import json
from pathlib import Path

CODE_CELLS = [
    # Cell 1: path discovery, trivial fallback submission first
    """
import sys, shutil
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

DINOV2_PATH = find_dir_containing('preprocessor_config.json')
print('Resolved DINOV2_PATH:', DINOV2_PATH)

shutil.copy(COMPETITION_ROOT / 'sample_submission.csv', OUT_DIR / 'submission.csv')
print('Trivial fallback submission written.')
""",
    # Cell 2: imports
    """
import time
import numpy as np
import pandas as pd
import torch

from rsna_knee.data.dicom_io import SERIES_SLOTS
from rsna_knee.data.series import assign_slots
from rsna_knee.data.label_expansion import add_derived_confidence, build_training_targets
from rsna_knee.data.labels import load_train_labels
from rsna_knee.data.preprocess import DEFAULT_RESIZE_PX
from rsna_knee.datasets import StudyDataset, compute_safe_max_cache_studies
from rsna_knee.metrics import LABEL_COLUMNS, per_label_auc
from rsna_knee.models.vision import KneeMultiLabelModel
from rsna_knee.train import train_one_epoch
from rsna_knee.infer import run_inference
from rsna_knee.validate import report_hash, select_holdout_ids

def select_device():
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        (torch.zeros(1, device='cuda') + 1).cpu()
        return 'cuda'
    except Exception as e:
        print(f'CUDA reported available but failed a sanity op ({e!r}); falling back to CPU.')
        return 'cpu'


device = select_device()
print('device:', device)
""",
    # Cell 3: load data, build silver-expanded targets, SAME train/holdout split as v13
    """
SLICES_PER_SLOT = 3
BATCH_SIZE = 3  # cut from 8: each study now carries ~3x more images through the encoder
EPOCHS = 6  # matches base v2's own total epoch count exactly -- the whole point of this run
N_UNFROZEN_BLOCKS = 2
ENCODER_LR = 1e-5
HEAD_LR = 1e-3

train_labels = load_train_labels(COMPETITION_ROOT / 'train.csv')
train_series_csv = pd.read_csv(COMPETITION_ROOT / 'train_series.csv')

SILVER_ROOT = find_dir_containing('llm_labels_v4_blend.csv')
silver = pd.read_csv(SILVER_ROOT / 'llm_labels_v4_blend.csv')
silver = add_derived_confidence(silver)
targets = build_training_targets(train_labels, silver)

train_labels['report_hash'] = train_labels['Report'].map(report_hash)
train_ids, holdout_ids = select_holdout_ids(
    train_labels, group_col='report_hash', n_splits=5, seed=42, holdout_fold=0
)
print(f'train: {len(train_ids)} studies, holdout: {len(holdout_ids)} studies')

train_series_dir = COMPETITION_ROOT / 'train_series'
train_series_subset = train_series_csv[train_series_csv['StudyInstanceUID'].isin(train_ids)]
train_series_with_slots = assign_slots(train_series_subset, train_series_dir)

NUM_WORKERS = 4
per_study_mb = len(SERIES_SLOTS) * SLICES_PER_SLOT * DEFAULT_RESIZE_PX * DEFAULT_RESIZE_PX * 4 / 1024**2
max_cache_studies = compute_safe_max_cache_studies(NUM_WORKERS, per_study_mb=per_study_mb)
print(f'max_cache_studies per worker: {max_cache_studies} (per_study_mb={per_study_mb:.2f})')
train_dataset = StudyDataset(
    train_ids, train_series_with_slots, train_series_dir, targets,
    max_cache_studies=max_cache_studies, slices_per_slot=SLICES_PER_SLOT,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True
)
print('train dataset size:', len(train_dataset))
""",
    # Cell 4: train -- pretrained DINOv2-base backbone, FRESH (not resumed), multi-slice input
    """
model = KneeMultiLabelModel(
    num_slots=6, num_labels=12, hf_pretrained_path=str(DINOV2_PATH)
).to(device)
model.encoder.freeze_except_last_n_blocks(N_UNFROZEN_BLOCKS)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f'trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)')

optimizer = torch.optim.AdamW(model.discriminative_param_groups(ENCODER_LR, HEAD_LR))

for epoch in range(EPOCHS):
    t0 = time.time()
    loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f'epoch {epoch+1}/{EPOCHS}: loss={loss:.4f} ({time.time()-t0:.0f}s)')

torch.save(model.state_dict(), OUT_DIR / 'model_base_multislice_fresh_v1.pt')
print('Checkpoint saved.')
""",
    # Cell 5: two-tier holdout evaluation -- directly comparable to base v2 (same
    # epoch count, same everything except K=3 vs K=1)
    """
holdout_series_subset = train_series_csv[train_series_csv['StudyInstanceUID'].isin(holdout_ids)]
holdout_series_with_slots = assign_slots(holdout_series_subset, train_series_dir)

t0 = time.time()
holdout_predictions = run_inference(
    model, holdout_ids, holdout_series_with_slots, train_series_dir, device,
    batch_size=BATCH_SIZE, slices_per_slot=SLICES_PER_SLOT,
)
print(f'Holdout inference took {time.time()-t0:.0f}s')
holdout_predictions = holdout_predictions.set_index('StudyInstanceUID')

holdout_silver_targets = targets[targets['StudyInstanceUID'].isin(holdout_ids)].set_index('StudyInstanceUID')
y_true_silver = (holdout_silver_targets.loc[holdout_predictions.index, LABEL_COLUMNS] >= 0.5).astype(float)
silver_scores = per_label_auc(y_true_silver.reset_index(drop=True), holdout_predictions.reset_index(drop=True))
print('Holdout AUC vs SILVER labels (tier 1, primary signal) -- base v2 (K=1, 6ep, FRESH) was 0.7931, our best confirmed real score (0.812):')
for col, v in silver_scores.items():
    print(f'  {col:20s} {v:.3f}' if not np.isnan(v) else f'  {col:20s} n/a')
silver_vals = [v for v in silver_scores.values() if not np.isnan(v)]
print(f'  MACRO: {np.mean(silver_vals):.4f}' if silver_vals else '  MACRO: n/a')

holdout_gold_mask = train_labels.set_index('StudyInstanceUID').loc[holdout_ids, LABEL_COLUMNS].notna().any(axis=1)
holdout_gold_ids = [sid for sid in holdout_ids if holdout_gold_mask.get(sid, False)]
print(f'\\nHoldout AUC vs GOLD labels (tier 2, annotation check, n={len(holdout_gold_ids)}) -- base v2 was 0.7774:')
if len(holdout_gold_ids) >= 5:
    gold_true = train_labels.set_index('StudyInstanceUID').loc[holdout_gold_ids, LABEL_COLUMNS]
    gold_pred = holdout_predictions.loc[holdout_gold_ids, LABEL_COLUMNS]
    gold_scores = per_label_auc(gold_true.reset_index(drop=True), gold_pred.reset_index(drop=True))
    for col, v in gold_scores.items():
        print(f'  {col:20s} {v:.3f}' if not np.isnan(v) else f'  {col:20s} n/a')
    gold_vals = [v for v in gold_scores.values() if not np.isnan(v)]
    print(f'  MACRO: {np.mean(gold_vals):.4f}' if gold_vals else '  MACRO: n/a')
else:
    print('  too few gold studies in this holdout to report per-label AUC')
""",
    # Cell 6: single-pass inference on the real test set (no TTA -- see infer.py's
    # run_inference_tta docstring for why center_fraction-based TTA is a no-op here)
    """
test_csv = pd.read_csv(COMPETITION_ROOT / 'test.csv')
test_series_csv = pd.read_csv(COMPETITION_ROOT / 'test_series.csv')
test_series_dir = COMPETITION_ROOT / 'test_series'

test_study_ids = test_csv['StudyInstanceUID'].tolist()
test_series_with_slots = assign_slots(test_series_csv, test_series_dir)

predictions = run_inference(
    model, test_study_ids, test_series_with_slots, test_series_dir, device,
    batch_size=BATCH_SIZE, slices_per_slot=SLICES_PER_SLOT,
)
predictions.to_csv(OUT_DIR / 'submission.csv', index=False)
print(f'Wrote {len(predictions)} predictions to submission.csv')
predictions.head()
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
    out_path = Path(__file__).resolve().parents[1] / "submission_base_multislice_fresh" / "submission_kernel.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_notebook(CODE_CELLS), indent=1))
    print(f"Wrote {out_path}")
