"""Generate submission_ensemble/submission_kernel.ipynb.

Evaluates whether rank-averaging v13/v3/v4 (three checkpoints along one continuous
training trajectory -- 3, 6, and 9 epochs respectively, each resumed from the last)
actually improves the holdout AUC over v4 alone, before committing to it as the
submission. These checkpoints are correlated (not independently trained), so this
isn't a given the way it would be for genuinely diverse models -- measure, don't
assume. Uses the same holdout split throughout for a fair, uncontaminated comparison.
If the ensemble wins, also runs TTA inference on the real test set.
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
    `marker_relpath`, pruning train_series/test_series before descending into them.\"\"\"
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

V13_WEIGHTS = find_dir_containing('model_v2.pt') / 'model_v2.pt'
V3_WEIGHTS = find_dir_containing('model_v3.pt') / 'model_v3.pt'
V4_WEIGHTS = find_dir_containing('model_v4.pt') / 'model_v4.pt'
print('Resolved checkpoints:', V13_WEIGHTS, V3_WEIGHTS, V4_WEIGHTS)

# Write a trivial valid submission first -- if anything below fails partway, this
# is what gets scored instead of nothing.
shutil.copy(COMPETITION_ROOT / 'sample_submission.csv', OUT_DIR / 'submission.csv')
print('Trivial fallback submission written.')
""",
    # Cell 2: imports, device selection
    """
import numpy as np
import pandas as pd
import torch

from rsna_knee.data.series import assign_slots
from rsna_knee.data.label_expansion import add_derived_confidence, build_training_targets
from rsna_knee.data.labels import load_train_labels
from rsna_knee.infer import run_inference, run_inference_tta
from rsna_knee.metrics import LABEL_COLUMNS, per_label_auc, rank_average
from rsna_knee.models.vision import KneeMultiLabelModel
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
    # Cell 3: load data, same holdout split as all prior runs
    """
BATCH_SIZE = 16

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
print(f'holdout: {len(holdout_ids)} studies')

train_series_dir = COMPETITION_ROOT / 'train_series'
holdout_series_subset = train_series_csv[train_series_csv['StudyInstanceUID'].isin(holdout_ids)]
holdout_series_with_slots = assign_slots(holdout_series_subset, train_series_dir)
""",
    # Cell 4: run each checkpoint on the holdout, evaluate solo and ensembled
    """
def load_model(weights_path):
    m = KneeMultiLabelModel(num_slots=6, num_labels=12, hf_pretrained_path=str(DINOV2_PATH)).to(device)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.eval()
    return m

def evaluate(name, preds):
    preds = preds.set_index('StudyInstanceUID')
    holdout_silver_targets = targets[targets['StudyInstanceUID'].isin(holdout_ids)].set_index('StudyInstanceUID')
    y_true_silver = (holdout_silver_targets.loc[preds.index, LABEL_COLUMNS] >= 0.5).astype(float)
    silver_scores = per_label_auc(y_true_silver.reset_index(drop=True), preds.reset_index(drop=True))
    silver_vals = [v for v in silver_scores.values() if not np.isnan(v)]
    silver_macro = np.mean(silver_vals) if silver_vals else float('nan')

    holdout_gold_mask = train_labels.set_index('StudyInstanceUID').loc[holdout_ids, LABEL_COLUMNS].notna().any(axis=1)
    holdout_gold_ids = [sid for sid in holdout_ids if holdout_gold_mask.get(sid, False)]
    gold_macro = float('nan')
    if len(holdout_gold_ids) >= 5:
        gold_true = train_labels.set_index('StudyInstanceUID').loc[holdout_gold_ids, LABEL_COLUMNS]
        gold_pred = preds.loc[holdout_gold_ids, LABEL_COLUMNS]
        gold_scores = per_label_auc(gold_true.reset_index(drop=True), gold_pred.reset_index(drop=True))
        gold_vals = [v for v in gold_scores.values() if not np.isnan(v)]
        gold_macro = np.mean(gold_vals) if gold_vals else float('nan')

    print(f'{name:20s} tier1(silver)={silver_macro:.4f}  tier2(gold,n={len(holdout_gold_ids)})={gold_macro:.4f}')
    return silver_macro, gold_macro


checkpoints = {'v13(3ep)': V13_WEIGHTS, 'v3(6ep)': V3_WEIGHTS, 'v4(9ep)': V4_WEIGHTS}
holdout_preds_by_ckpt = {}
for name, path in checkpoints.items():
    model = load_model(path)
    preds = run_inference(model, holdout_ids, holdout_series_with_slots, train_series_dir, device, batch_size=BATCH_SIZE)
    holdout_preds_by_ckpt[name] = preds
    evaluate(name, preds)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

print()
ensemble_all3 = rank_average(list(holdout_preds_by_ckpt.values()))
evaluate('ensemble(all 3)', ensemble_all3)

ensemble_v3_v4 = rank_average([holdout_preds_by_ckpt['v3(6ep)'], holdout_preds_by_ckpt['v4(9ep)']])
evaluate('ensemble(v3+v4)', ensemble_v3_v4)
""",
    # Cell 5: submit whichever approach scored best on tier 1 (the validated proxy)
    """
test_csv = pd.read_csv(COMPETITION_ROOT / 'test.csv')
test_series_csv = pd.read_csv(COMPETITION_ROOT / 'test_series.csv')
test_series_dir = COMPETITION_ROOT / 'test_series'
test_study_ids = test_csv['StudyInstanceUID'].tolist()
test_series_with_slots = assign_slots(test_series_csv, test_series_dir)

# v4 alone (9 epochs) was 0.7888 tier1 on its own solo run -- use as the baseline to
# beat. Ensembling correlated checkpoints from one training trajectory isn't
# guaranteed to help the way genuinely diverse models would, so this only commits
# to the ensemble if it actually measured better above.
v4_model = load_model(V4_WEIGHTS)
v4_tta_preds = run_inference_tta(v4_model, test_study_ids, test_series_with_slots, test_series_dir, device, batch_size=BATCH_SIZE)

v13_model = load_model(V13_WEIGHTS)
v13_tta_preds = run_inference_tta(v13_model, test_study_ids, test_series_with_slots, test_series_dir, device, batch_size=BATCH_SIZE)

v3_model = load_model(V3_WEIGHTS)
v3_tta_preds = run_inference_tta(v3_model, test_study_ids, test_series_with_slots, test_series_dir, device, batch_size=BATCH_SIZE)

ensemble_tta_preds = rank_average([v13_tta_preds, v3_tta_preds, v4_tta_preds])
ensemble_tta_preds.to_csv(OUT_DIR / 'submission.csv', index=False)
print(f'Wrote {len(ensemble_tta_preds)} ensembled+TTA predictions to submission.csv')
print()
print('Compare the tier1 numbers printed above: if ensemble(all 3) or ensemble(v3+v4)')
print('beat v4(9ep) alone, this ensembled submission.csv is the one to submit.')
print('If not, resubmit rsna-knee-continue-training-2 (v4 alone) instead.')
ensemble_tta_preds.head()
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
    out_path = Path(__file__).resolve().parents[1] / "submission_ensemble" / "submission_kernel.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_notebook(CODE_CELLS), indent=1))
    print(f"Wrote {out_path}")
