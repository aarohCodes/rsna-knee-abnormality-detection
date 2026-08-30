"""Generate submission/submission_kernel.ipynb.

v0 (validated on Kaggle: random-init ResNet18, 800 then full studies, no held-out
eval) proved the pipeline works end-to-end -- public scores 0.596/0.617. v2 swaps in
the real architecture: pretrained DINOv2 backbone (via a Kaggle Model source, no
internet needed), discriminative fine-tuning (only the last 2 of 12 blocks unfrozen,
separate encoder/head learning rates), the better-validated silver-label source, and
a real held-out split so we can see per-label AUC instead of only training loss.
TTA is deliberately left for a follow-up push -- too many new pieces to validate in
one untested-on-Kaggle run already.
"""
import json
from pathlib import Path

CODE_CELLS = [
    # Cell 1: make our package importable, write the trivial fallback submission first
    """
import sys, shutil
from pathlib import Path

INPUT_ROOT = Path('/kaggle/input')
OUT_DIR = Path('/kaggle/working')

print('Contents of /kaggle/input:', sorted(p.name for p in INPUT_ROOT.iterdir()))


def find_dir_containing(marker_relpath, root=INPUT_ROOT, max_depth=4):
    \"\"\"Search /kaggle/input at increasing depth for a directory containing
    `marker_relpath`, instead of assuming a fixed mount layout -- Kaggle's actual
    layout here is /kaggle/input/{competitions,datasets}/..., not a flat
    /kaggle/input/<slug>/ as commonly assumed.\"\"\"
    for depth in range(max_depth + 1):
        candidates = [root] if depth == 0 else root.glob('/'.join(['*'] * depth))
        for candidate in candidates:
            if candidate.is_dir() and (candidate / marker_relpath).exists():
                return candidate
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
print('Its contents:', sorted(p.name for p in COMPETITION_ROOT.iterdir()))

# preprocessor_config.json is specific to HuggingFace vision-model checkpoints (e.g.
# the mounted DINOv2 Model source) -- unambiguous among our attached inputs.
DINOV2_PATH = find_dir_containing('preprocessor_config.json')
print('Resolved DINOV2_PATH:', DINOV2_PATH)

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
from rsna_knee.datasets import StudyDataset
from rsna_knee.metrics import LABEL_COLUMNS, per_label_auc
from rsna_knee.models.vision import KneeMultiLabelModel
from rsna_knee.train import train_one_epoch
from rsna_knee.infer import run_inference
from rsna_knee.validate import report_hash, select_holdout_ids

def select_device():
    \"\"\"torch.cuda.is_available() only checks that a CUDA driver/device is visible,
    not that the installed PyTorch build actually supports its compute capability --
    Kaggle sometimes assigns an older GPU (e.g. a P100, sm_60) that a newer PyTorch
    wheel has dropped kernel support for, which fails on the first real op rather
    than at is_available() time. Do a tiny real op and fall back to CPU if it fails.
    \"\"\"
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
    # Cell 3: load data, build silver-expanded targets, split train/holdout
    """
BATCH_SIZE = 16
EPOCHS = 3
N_UNFROZEN_BLOCKS = 2
ENCODER_LR = 1e-5
HEAD_LR = 1e-3

train_labels = load_train_labels(COMPETITION_ROOT / 'train.csv')
train_series_csv = pd.read_csv(COMPETITION_ROOT / 'train_series.csv')

# Silver labels: stevenleehans/rsna-knee-llm-report-labels (llm_labels_v4_blend.csv),
# validated against the 58 gold studies at 0.893 macro AUC -- better than, and not
# leaked unlike some other public candidates (see label_expansion.py docstring for
# the leakage check). No native confidence column, so derive one.
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

train_dataset = StudyDataset(train_ids, train_series_with_slots, train_series_dir, targets)
# persistent_workers=True is required for the dataset's per-study image cache to help
# beyond the first epoch -- PyTorch respawns worker processes every epoch by default,
# which would otherwise silently discard each worker's in-process cache every time.
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True
)
print('train dataset size:', len(train_dataset))
""",
    # Cell 4: train -- pretrained DINOv2 backbone, discriminative fine-tuning
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
    loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f'epoch {epoch+1}/{EPOCHS}: loss={loss:.4f}')

torch.save(model.state_dict(), OUT_DIR / 'model_v2.pt')
print('Checkpoint saved.')
""",
    # Cell 5: two-tier holdout evaluation (per PLAN.md's baseline-notebook design)
    """
holdout_series_subset = train_series_csv[train_series_csv['StudyInstanceUID'].isin(holdout_ids)]
holdout_series_with_slots = assign_slots(holdout_series_subset, train_series_dir)

holdout_predictions = run_inference(
    model, holdout_ids, holdout_series_with_slots, train_series_dir, device, batch_size=BATCH_SIZE
)
holdout_predictions = holdout_predictions.set_index('StudyInstanceUID')

# Tier 1: silver-label holdout -- noisy but plenty of samples per class, drives
# model/epoch selection.
holdout_silver_targets = targets[targets['StudyInstanceUID'].isin(holdout_ids)].set_index('StudyInstanceUID')
y_true_silver = holdout_silver_targets.loc[holdout_predictions.index, LABEL_COLUMNS]
silver_scores = per_label_auc(y_true_silver.reset_index(drop=True), holdout_predictions.reset_index(drop=True))
print('Holdout AUC vs SILVER labels (tier 1, primary signal):')
for col, v in silver_scores.items():
    print(f'  {col:20s} {v:.3f}' if not np.isnan(v) else f'  {col:20s} n/a')
silver_vals = [v for v in silver_scores.values() if not np.isnan(v)]
print(f'  MACRO: {np.mean(silver_vals):.4f}' if silver_vals else '  MACRO: n/a')

# Tier 2: the gold-labeled studies that landed in this holdout -- approximates the
# real competition metric but too few samples to arbitrate alone; report only.
holdout_gold_mask = train_labels.set_index('StudyInstanceUID').loc[holdout_ids, LABEL_COLUMNS].notna().any(axis=1)
holdout_gold_ids = [sid for sid in holdout_ids if holdout_gold_mask.get(sid, False)]
print(f'\\nHoldout AUC vs GOLD labels (tier 2, annotation check, n={len(holdout_gold_ids)}):')
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
    # Cell 6: inference on the real test set + overwrite the trivial submission
    """
test_csv = pd.read_csv(COMPETITION_ROOT / 'test.csv')
test_series_csv = pd.read_csv(COMPETITION_ROOT / 'test_series.csv')
test_series_dir = COMPETITION_ROOT / 'test_series'

test_study_ids = test_csv['StudyInstanceUID'].tolist()
test_series_with_slots = assign_slots(test_series_csv, test_series_dir)

predictions = run_inference(
    model, test_study_ids, test_series_with_slots, test_series_dir, device, batch_size=BATCH_SIZE
)
predictions.to_csv(OUT_DIR / 'submission.csv', index=False)
print(f'Wrote {len(predictions)} real predictions to submission.csv')
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
    out_path = Path(__file__).resolve().parents[1] / "submission" / "submission_kernel.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_notebook(CODE_CELLS), indent=1))
    print(f"Wrote {out_path}")
