# RSNA Knee Abnormality Detection

Working repo for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition (code competition, Kernels-only submissions).

Full plan: see `.claude/plans` in the parent session, or `PLAN.md` in this repo (copied for
reference).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .
```

Kaggle credentials expected at `~/.kaggle/kaggle.json`.

## Data

Full dataset (500+ GB) lives on Kaggle and is mounted automatically inside every Kaggle
Notebook at `/kaggle/input/rsna-knee-abnormality-detection/` — it is **not** downloaded in
full locally. `scripts/download_data.py` pulls only the small CSVs plus a disk-budgeted
stratified sample of DICOMs for local development/EDA.

## Workflow

- Local: EDA, label-expansion pipeline, CPU-feasible baselines, unit tests.
- Kaggle Kernels: full-data training (GPU) and the mandatory scored submission kernel
  (9-hour runtime cap, internet disabled during scoring).

See `experiments/runs.csv` for the experiment log.
