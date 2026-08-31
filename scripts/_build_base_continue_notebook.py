"""Generate submission_base_continue/submission_kernel.ipynb.

Continues training DINOv2-base from its own first checkpoint (rsna-knee-weights-
base-v1, 3 epochs: tier1=0.7820, already beating small-backbone's 3-epoch tier1 of
0.7752, loss still dropping 0.560->0.517->0.493). Same successful pattern as
small's v13->v3->v4 progression. Same holdout split (seed=42, holdout_fold=0)
throughout for a fair, uncontaminated comparison.
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

WEIGHTS_ROOT = find_dir_containing('model_base_v1.pt')
print('Resolved WEIGHTS_ROOT:', WEIGHTS_ROOT)

# Write a trivial valid submission first -- if anything below fails partway, this
# is what gets scored instead of nothing.
shutil.copy(COMPETITION_ROOT / 'sample_submission.csv', OUT_DIR / 'submission.csv')
print('Trivial fallback submission written.')
""",
    # Cell 2: imports
    """
import numpy as np
import pandas as pd
import torch

from rsna_knee.data.series import assign_slots
from rsna_knee.data.label_expansion import add_derived_confidence, build_training_targets
from rsna_knee.data.labels import load_train_labels
from rsna_knee.datasets import StudyDataset, compute_safe_max_cache_studies
from rsna_knee.metrics import LABEL_COLUMNS, per_label_auc
from rsna_knee.models.vision import KneeMultiLabelModel
from rsna_knee.train import train_one_epoch
from rsna_knee.infer import run_inference, run_inference_tta
from rsna_knee.validate import report_hash, select_holdout_ids

def select_device():
    \"\"\"torch.cuda.is_available() only checks driver/device visibility, not that the
    installed PyTorch build supports the assigned GPU's compute capability -- Kaggle
    can assign an older P100 (sm_60) a newer PyTorch wheel has dropped kernel support
    for. Do a tiny real op and fall back to CPU if it fails.\"\"\"
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
BATCH_SIZE = 8  # DINOv2-base has ~4x DINOv2-small's params; smaller batch for GPU memory safety
ADDITIONAL_EPOCHS = 3  # base already did 3 (loss 0.560->0.517->0.493, still trending down)
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
# MUST match v13 exactly (same seed, same holdout_fold) so the holdout stays
# uncontaminated -- these extra epochs only ever see studies v13 already trained on.
train_ids, holdout_ids = select_holdout_ids(
    train_labels, group_col='report_hash', n_splits=5, seed=42, holdout_fold=0
)
print(f'train: {len(train_ids)} studies, holdout: {len(holdout_ids)} studies')

train_series_dir = COMPETITION_ROOT / 'train_series'
train_series_subset = train_series_csv[train_series_csv['StudyInstanceUID'].isin(train_ids)]
train_series_with_slots = assign_slots(train_series_subset, train_series_dir)

NUM_WORKERS = 4
max_cache_studies = compute_safe_max_cache_studies(NUM_WORKERS)
print(f'max_cache_studies per worker: {max_cache_studies}')
train_dataset = StudyDataset(
    train_ids, train_series_with_slots, train_series_dir, targets, max_cache_studies=max_cache_studies
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True
)
print('train dataset size:', len(train_dataset))
""",
    # Cell 4: resume from v13's checkpoint, continue training
    """
model = KneeMultiLabelModel(
    num_slots=6, num_labels=12, hf_pretrained_path=str(DINOV2_PATH)
).to(device)
model.load_state_dict(torch.load(WEIGHTS_ROOT / 'model_base_v1.pt', map_location=device))
print('Resumed from', WEIGHTS_ROOT / 'model_base_v1.pt')

model.encoder.freeze_except_last_n_blocks(N_UNFROZEN_BLOCKS)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f'trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)')

optimizer = torch.optim.AdamW(model.discriminative_param_groups(ENCODER_LR, HEAD_LR))

for epoch in range(ADDITIONAL_EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f'additional epoch {epoch+1}/{ADDITIONAL_EPOCHS}: loss={loss:.4f}')

torch.save(model.state_dict(), OUT_DIR / 'model_base_v2.pt')
print('Checkpoint saved.')
""",
    # Cell 5: two-tier holdout evaluation, same methodology as v13 for a fair comparison
    """
holdout_series_subset = train_series_csv[train_series_csv['StudyInstanceUID'].isin(holdout_ids)]
holdout_series_with_slots = assign_slots(holdout_series_subset, train_series_dir)

holdout_predictions = run_inference(
    model, holdout_ids, holdout_series_with_slots, train_series_dir, device, batch_size=BATCH_SIZE
)
holdout_predictions = holdout_predictions.set_index('StudyInstanceUID')

holdout_silver_targets = targets[targets['StudyInstanceUID'].isin(holdout_ids)].set_index('StudyInstanceUID')
y_true_silver = (holdout_silver_targets.loc[holdout_predictions.index, LABEL_COLUMNS] >= 0.5).astype(float)
silver_scores = per_label_auc(y_true_silver.reset_index(drop=True), holdout_predictions.reset_index(drop=True))
print('Holdout AUC vs SILVER labels (tier 1, primary signal) -- base(3ep) was 0.7820, small best was 0.7911:')
for col, v in silver_scores.items():
    print(f'  {col:20s} {v:.3f}' if not np.isnan(v) else f'  {col:20s} n/a')
silver_vals = [v for v in silver_scores.values() if not np.isnan(v)]
print(f'  MACRO: {np.mean(silver_vals):.4f}' if silver_vals else '  MACRO: n/a')

holdout_gold_mask = train_labels.set_index('StudyInstanceUID').loc[holdout_ids, LABEL_COLUMNS].notna().any(axis=1)
holdout_gold_ids = [sid for sid in holdout_ids if holdout_gold_mask.get(sid, False)]
print(f'\\nHoldout AUC vs GOLD labels (tier 2, annotation check, n={len(holdout_gold_ids)}) -- base(3ep) was 0.7891:')
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
    # Cell 6: multi-crop TTA inference on the real test set
    """
test_csv = pd.read_csv(COMPETITION_ROOT / 'test.csv')
test_series_csv = pd.read_csv(COMPETITION_ROOT / 'test_series.csv')
test_series_dir = COMPETITION_ROOT / 'test_series'

test_study_ids = test_csv['StudyInstanceUID'].tolist()
test_series_with_slots = assign_slots(test_series_csv, test_series_dir)

predictions = run_inference_tta(
    model, test_study_ids, test_series_with_slots, test_series_dir, device, batch_size=BATCH_SIZE
)
predictions.to_csv(OUT_DIR / 'submission.csv', index=False)
print(f'Wrote {len(predictions)} TTA predictions to submission.csv')
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
    out_path = Path(__file__).resolve().parents[1] / "submission_base_continue" / "submission_kernel.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_notebook(CODE_CELLS), indent=1))
    print(f"Wrote {out_path}")
