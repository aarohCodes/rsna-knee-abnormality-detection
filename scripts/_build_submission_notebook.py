"""Generate submission/submission_kernel.ipynb.

v0 goal (per PLAN.md step 7): a deliberately simple, fast, internet-off, GPU kernel
that trains briefly on a subset of the full mounted training data and produces a
valid submission.csv -- validating the whole Kaggle pipeline (package import, data
mount paths, kernel constraints, submission format) before investing in a longer,
higher-quality training run. pretrained=False throughout so no model-weight download
is needed (internet is disabled during the scored run).
"""
import json
from pathlib import Path

CODE_CELLS = [
    # Cell 1: make our package importable, write the trivial fallback submission first
    """
import sys, shutil
from pathlib import Path

SRC_DATASET = Path('/kaggle/input/rsna-knee-src')
sys.path.insert(0, str(SRC_DATASET))

COMPETITION_ROOT = Path('/kaggle/input/rsna-knee-abnormality-detection')
OUT_DIR = Path('/kaggle/working')

# Write a trivial valid submission first -- if anything below fails partway, this
# is what gets scored instead of nothing.
shutil.copy(COMPETITION_ROOT / 'sample_submission.csv', OUT_DIR / 'submission.csv')
print('Trivial fallback submission written.')
""",
    # Cell 2: imports
    """
import pandas as pd
import torch

from rsna_knee.data.series import assign_slots
from rsna_knee.data.label_expansion import build_training_targets
from rsna_knee.data.labels import load_train_labels
from rsna_knee.datasets import StudyDataset
from rsna_knee.models.vision import KneeMultiLabelModel
from rsna_knee.train import train_one_epoch
from rsna_knee.infer import run_inference

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
""",
    # Cell 3: load + subset training data (v0: keep it fast, not exhaustive)
    """
N_TRAIN_STUDIES_V0 = 800  # v0: fast correctness-check run, not full-scale training
BATCH_SIZE = 8
EPOCHS = 2

train_labels = load_train_labels(COMPETITION_ROOT / 'train.csv')
train_series_csv = pd.read_csv(COMPETITION_ROOT / 'train_series.csv')

# Need a public silver-labels source too: LLM-derived report labels, attached as a
# second input dataset (pilkwang/rsna-knee-llm-labels, CC0).
silver = pd.read_csv('/kaggle/input/rsna-knee-llm-labels/report_labels_v2.csv')
targets = build_training_targets(train_labels, silver)

# v0 subset: prioritize the 58 gold-labeled studies (small but high-value), fill the
# rest randomly up to the budget.
gold_ids = train_labels.loc[
    train_labels[[c for c in train_labels.columns if c not in ('StudyInstanceUID', 'Report')]].notna().any(axis=1),
    'StudyInstanceUID',
].tolist()
rest = train_labels.loc[~train_labels['StudyInstanceUID'].isin(gold_ids), 'StudyInstanceUID']
rest_sample = rest.sample(n=min(N_TRAIN_STUDIES_V0 - len(gold_ids), len(rest)), random_state=42).tolist()
study_ids = gold_ids + rest_sample
print(f'v0 training subset: {len(study_ids)} studies ({len(gold_ids)} gold)')

train_series_dir = COMPETITION_ROOT / 'train_series'
series_subset = train_series_csv[train_series_csv['StudyInstanceUID'].isin(study_ids)]
series_with_slots = assign_slots(series_subset, train_series_dir)

train_dataset = StudyDataset(study_ids, series_with_slots, train_series_dir, targets)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
print('train dataset size:', len(train_dataset))
""",
    # Cell 4: train
    """
model = KneeMultiLabelModel(
    num_slots=6, num_labels=12, backbone_name='resnet18', pretrained=False
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f'epoch {epoch+1}/{EPOCHS}: loss={loss:.4f}')

torch.save(model.state_dict(), OUT_DIR / 'model_v0.pt')
print('Checkpoint saved.')
""",
    # Cell 5: inference on the real test set + overwrite the trivial submission
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
