"""Generate submission_infer/submission_kernel.ipynb.

Inference-only kernel: loads the already-trained checkpoint (uploaded as its own
Kaggle Dataset from a completed training run, e.g. rsna-knee-weights-v13) instead
of retraining, and runs multi-crop TTA on the real test set. This is the second
half of the train-once/infer-many split the plan always intended -- avoids paying
for a ~2.5-3 hour DINOv2 retrain every time only the inference step changes (new
TTA config, new ensembling, etc.).
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
    `marker_relpath`, instead of assuming a fixed mount layout. Confirmed
    inconsistent across source types: competitions nest under 'competitions/<slug>/',
    some datasets mount flat at the top level, others nest under
    'datasets/<owner>/<name>/', and Model sources nest deepest (depth 5-6).

    Prunes train_series/test_series before descending into them -- same lesson as
    the public baseline notebook's cover-image loader: the competition mount holds
    hundreds of thousands of per-study/per-series directories, and none of our
    markers live inside them. A naive glob-at-each-depth search doesn't prune, and
    ends up enumerating that entire tree once search depth reaches it -- confirmed
    costing ~8 minutes per kernel run before this fix.
    \"\"\"
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

WEIGHTS_ROOT = find_dir_containing('model_v2.pt')
print('Resolved WEIGHTS_ROOT:', WEIGHTS_ROOT)

# Write a trivial valid submission first -- if anything below fails partway, this
# is what gets scored instead of nothing.
shutil.copy(COMPETITION_ROOT / 'sample_submission.csv', OUT_DIR / 'submission.csv')
print('Trivial fallback submission written.')
""",
    # Cell 2: imports, device selection
    """
import pandas as pd
import torch

from rsna_knee.data.series import assign_slots
from rsna_knee.infer import run_inference_tta
from rsna_knee.models.vision import KneeMultiLabelModel

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
    # Cell 3: load the already-trained model
    """
model = KneeMultiLabelModel(
    num_slots=6, num_labels=12, hf_pretrained_path=str(DINOV2_PATH)
).to(device)
model.load_state_dict(torch.load(WEIGHTS_ROOT / 'model_v2.pt', map_location=device))
model.eval()
print('Loaded trained checkpoint from', WEIGHTS_ROOT / 'model_v2.pt')
""",
    # Cell 4: multi-crop TTA inference on the real test set
    """
BATCH_SIZE = 16
TTA_CENTER_FRACTIONS = (0.35, 0.5, 0.65)

test_csv = pd.read_csv(COMPETITION_ROOT / 'test.csv')
test_series_csv = pd.read_csv(COMPETITION_ROOT / 'test_series.csv')
test_series_dir = COMPETITION_ROOT / 'test_series'

test_study_ids = test_csv['StudyInstanceUID'].tolist()
test_series_with_slots = assign_slots(test_series_csv, test_series_dir)

predictions = run_inference_tta(
    model, test_study_ids, test_series_with_slots, test_series_dir, device,
    batch_size=BATCH_SIZE, center_fractions=TTA_CENTER_FRACTIONS,
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
    out_path = Path(__file__).resolve().parents[1] / "submission_infer" / "submission_kernel.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_notebook(CODE_CELLS), indent=1))
    print(f"Wrote {out_path}")
