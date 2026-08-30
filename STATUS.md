# Status (last updated 2026-08-30, ~21:40 CDT)

Read this file first in any new session/after context compaction to get oriented.
Full strategy/reasoning is in PLAN.md; this file is the living "where are we now."

## Competition
RSNA Knee Abnormality Detection (Kaggle, slug `rsna-knee-abnormality-detection`).
Code competition: submissions run via Kaggle Notebooks, GPU (T4 only -- **P100
submissions are rejected by this competition**, always set
`"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json), internet disabled at
scoring, 9-hour runtime cap **per submission** (not cumulative). Final entry
deadline 2026-10-15, final submission deadline 2026-10-22.

## Rules compliance notes (established with the user, keep following these)
- External data/labels from other public Kaggle datasets ARE allowed under the
  competition rules ("publicly available and equally accessible... at no cost") --
  user explicitly confirmed this after discussion. We use `stevenleehans`'s public
  silver-label dataset. Two other candidate label datasets
  (`yunusgmsoy/rsna-knee-llm-labels-4-source-merged`,
  `tasmeemreza/rsna-knee-refined-llm-labels`) were REJECTED after discovering they
  bitwise-match the gold labels exactly (data leakage) -- don't reuse those.
- Vendored code (`iterative-stratification`, BSD-3, in `src/rsna_knee/_vendor/`) is
  fine -- competition allows OSI-licensed open source.
- User said "use anything you want, but make sure it's allowed in the rules" -- so
  external public Kaggle datasets/models are fair game; just verify license +
  public accessibility before adding a new one, same way we checked the two above.

## Score trajectory (public leaderboard)
0.596 -> 0.617 -> 0.646 (random-init ResNet18 baselines, v0) -> **0.782** (DINOv2-
small, 6 epochs, TTA). Current visible competition leader: ~0.952. A submission from
the 9-epoch/v4 or ensemble run is still PENDING scoring as of this writing.

**Locally-validated holdout AUC has matched real public LB scores almost exactly**
(0.7833 predicted vs 0.782 actual) -- trust the tier-1 holdout number below as a
reliable proxy before spending a submission on something.

## What's built (all in `src/rsna_knee/`, tested, committed)
- Full DICOM pipeline: geometry-based slice ordering, series-slot classification,
  laterality normalization, Nyquist-aware crop/resize (`data/dicom_io.py`,
  `data/series.py`, `data/preprocess.py`)
- Label expansion: gold (58 studies) + silver (steven's dataset) with confidence
  weighting (`data/label_expansion.py`)
- Grouped multilabel-stratified CV, holdout selection, dedup-report-hash grouping
  (`validate.py`)
- Model: `KneeMultiLabelModel` supports both timm backbones and HuggingFace
  `transformers` backbones (DINOv2) via `hf_pretrained_path`; per-diagnosis
  attention pooling over series slots (`models/vision.py`); discriminative
  fine-tuning (`freeze_except_last_n_blocks`, `discriminative_param_groups`)
- Multi-crop TTA (`run_inference_tta`) + rank-based ensembling (`rank_average`)
- All notebook generators use a **pruned BFS path-discovery** helper
  (`find_dir_containing` in each `scripts/_build_*_notebook.py`) -- Kaggle's
  `/kaggle/input` mount layout is inconsistent (competitions nest under
  `competitions/<slug>/`, some datasets flat, some under `datasets/<owner>/<name>/`,
  Models nest deepest) and naive glob search wastes ~8min/run enumerating the huge
  DICOM tree if it doesn't prune `train_series`/`test_series` by name.

## Kaggle assets (all under account `aarohnanoti`)
- **Datasets**: `rsna-knee-src` (our package, re-upload via
  `kaggle_dataset_src/` + `datasets version` after any src change --
  **easy to forget, causes ImportError on next kernel push**),
  `rsna-knee-weights-v13` (3ep), `rsna-knee-weights-v3` (6ep, resumed from v13),
  `rsna-knee-weights-v4` (9ep, resumed from v3)
- **Kernels** (each has its own folder + `scripts/_build_*_notebook.py` generator --
  edit the generator, run it, then `kaggle kernels push -p <folder>`):
  - `submission/` -> `rsna-knee-submission-v0`: original v0/v13 combined train+infer
  - `submission_infer/` -> `rsna-knee-inference-tta`: loads v13, TTA infer only
  - `submission_continue/` -> `rsna-knee-continue-training`: v13->v3 (+3ep)
  - `submission_continue2/` -> `rsna-knee-continue-training-2`: v3->v4 (+3ep)
  - `submission_ensemble/` -> `rsna-knee-ensemble`: loads v13+v3+v4, evaluates solo
    + ensembled on holdout, submits whichever combo wins (currently: all-3 ensemble,
    tier1=0.7911, beat v4 alone's 0.7888)
  - `submission_base/` -> `rsna-knee-dinov2-base`: **currently RUNNING** (fresh
    training, DINOv2-base instead of -small, same recipe otherwise)

## Roadmap (user-approved sequence, execute in this order)
1. ~~See how 9-epoch submission scores~~ -- done, tier1 confirmed reliable proxy
2. ~~Ensemble v13+v3+v4 via rank averaging~~ -- done, small but real gain confirmed
   (measure, don't assume -- these are correlated checkpoints from one training
   trajectory, not independently trained, so this had to be verified empirically)
3. **DINOv2-base instead of -small -- IN PROGRESS**, kernel running now
   (`rsna-knee-dinov2-base`). When it finishes: compare its tier-1 holdout AUC
   against the small-backbone numbers above (v4 alone 0.7888, ensemble 0.7911).
   If better, upload its checkpoint as a new weights dataset and consider
   continuing training it the same way v13->v3->v4 worked, and/or ensembling it
   with the small-backbone checkpoints.
4. Combine multiple independent silver-label sources (rank-average `steven`'s
   labels with another *verified-not-leaked* public source) to push past the
   ~0.89 AUC ceiling any single silver source has against gold
5. Investigate weak labels specifically -- Lateral OA, MCL, Baker's have shown up
   weakest across holdout runs (check `data/interim/kernel_output_*/`.log files
   for exact per-label numbers); figure out if it's class imbalance or genuinely
   harder signal, and address accordingly
6. Multi-slice/volumetric input instead of one representative crop per series

## How to check on the running DINOv2-base kernel
```
.venv/Scripts/kaggle.exe kernels status aarohnanoti/rsna-knee-dinov2-base
# once COMPLETE:
.venv/Scripts/kaggle.exe kernels output aarohnanoti/rsna-knee-dinov2-base -p data/interim/kernel_output_base_v1 --force
# then read data/interim/kernel_output_base_v1/rsna-knee-dinov2-base.log for tier1/tier2 AUC
```

## Submission mechanics reminder
I can push/run kernels via the Kaggle API, but **clicking "Submit to Competition"
on kaggle.com is the user's action** -- no API for that on code competitions. After
any kernel completes with a good result, tell the user the kernel URL and ask them
to submit it.
