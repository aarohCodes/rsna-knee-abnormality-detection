# Status (last updated 2026-08-30, ~22:30 CDT)

Read this file first in any new session/after context compaction to get oriented.
Full strategy/reasoning is in PLAN.md; this file is the living "where are we now."

## Competition
RSNA Knee Abnormality Detection (Kaggle, slug `rsna-knee-abnormality-detection`).
Code competition: submissions run via Kaggle Notebooks, GPU (T4 only -- **P100
submissions are rejected by this competition**, always set
`"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json), internet disabled at
scoring, 9-hour runtime cap **per submission** (not cumulative). Final entry
deadline 2026-10-15, final submission deadline 2026-10-22. Public leaderboard is
only ~30% of test data; private (final) is the other 70%, revealed at the end --
current public scores could shift either direction at that point.

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
- User explicitly said NOT to pursue external MRI datasets (MRNet/OAI) or a
  RadImageNet-style domain-pretrained backbone "for now" -- don't start on either
  without asking again first.

## Score trajectory (public leaderboard)
0.596 -> 0.617 -> 0.646 (random-init ResNet18 baselines, v0) -> **0.782** (DINOv2-
small, 6 epochs, TTA, our best real score so far) -> **0.775** (v13+v3+v4 rank
ensemble -- REGRESSION, confirmed, see below). Current visible competition leader:
~0.952 (public LB, snapshot from early in the competition -- may have moved since).

All 5 submissions on 2026-08-30 (UTC) are used; the daily cap resets per UTC day,
so a fresh 5 became available starting 2026-08-31 00:00 UTC.

**Locally-validated holdout AUC has matched real public LB scores closely once**
(0.7833 predicted vs 0.782 actual) **but missed badly the second time**: the
ensemble's tier1 (0.7911, predicted best-yet) scored *worse* in reality (0.775)
than the plain 6-epoch single model (0.782). Two data points now, not one --
treat this as a real signal that tier1 (silver-label holdout) can diverge from
true generalization, not just as one-off public-LB sampling noise. Don't fully
trust tier1 deltas between closely-matched checkpoints/ensembles without a real
submission to confirm; still fine as a first-pass filter for clearly-better vs
clearly-worse.

## Holdout AUC leaderboard (tier1 = vs silver labels, n=871, reliable; tier2 = vs
## gold labels, n=11, noisy -- report only, don't tune on it)
| checkpoint | epochs | tier1 | tier2 |
|---|---|---|---|
| small v13 | 3 | 0.7752 | 0.7368 |
| small v3 | 6 | 0.7833 | 0.7318 |
| small v4 | 9 | 0.7888 | 0.7587 |
| small ensemble (v13+v3+v4) | -- | **0.7911** | 0.7516 |
| base v1 | 3 | 0.7820 | 0.7891 |
| base v2 | 6 (continuing now) | pending | pending |

Base already beats small at matched epoch count (3ep: 0.7820 vs 0.7752) --
continuing to train it is likely to become the new best.

## What's built (all in `src/rsna_knee/`, tested, committed)
- Full DICOM pipeline: geometry-based slice ordering, series-slot classification,
  laterality normalization, Nyquist-aware crop/resize (`data/dicom_io.py`,
  `data/series.py`, `data/preprocess.py`)
- Label expansion: gold (58 studies) + silver (steven's dataset) with confidence
  weighting (`data/label_expansion.py`)
- Grouped multilabel-stratified CV, holdout selection, dedup-report-hash grouping
  (`validate.py`)
- Model: `KneeMultiLabelModel` supports both timm backbones and HuggingFace
  `transformers` backbones (DINOv2 small AND base) via `hf_pretrained_path`;
  per-diagnosis attention pooling over series slots (`models/vision.py`);
  discriminative fine-tuning (`freeze_except_last_n_blocks`,
  `discriminative_param_groups`)
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
  `rsna-knee-weights-v13` (small, 3ep), `rsna-knee-weights-v3` (small, 6ep),
  `rsna-knee-weights-v4` (small, 9ep), `rsna-knee-weights-base-v1` (base, 3ep)
- **Kernels** (each has its own folder + `scripts/_build_*_notebook.py` generator --
  edit the generator, run it, then `kaggle kernels push -p <folder>`):
  - `submission/` -> `rsna-knee-submission-v0`: original v0/v13 combined train+infer
  - `submission_infer/` -> `rsna-knee-inference-tta`: loads v13, TTA infer only
  - `submission_continue/` -> `rsna-knee-continue-training`: v13->v3 (+3ep)
  - `submission_continue2/` -> `rsna-knee-continue-training-2`: v3->v4 (+3ep)
  - `submission_ensemble/` -> `rsna-knee-ensemble`: loads v13+v3+v4, evaluates solo
    + ensembled on holdout, submits whichever combo wins (currently: all-3
    ensemble, tier1=0.7911)
  - `submission_base/` -> `rsna-knee-dinov2-base`: base v1, 3 fresh epochs, DONE
    (tier1=0.7820, tier2=0.7891)
  - `submission_base_continue/` -> `rsna-knee-base-continue`: **currently RUNNING**,
    base v1->v2 (+3ep, resumed from rsna-knee-weights-base-v1)

## Roadmap (user-approved sequence, execute in this order)
1. ~~See how 9-epoch submission scores~~ -- done, tier1 confirmed reliable proxy
2. ~~Ensemble v13+v3+v4 via rank averaging~~ -- done, small but real gain confirmed
   (measure, don't assume -- these are correlated checkpoints from one training
   trajectory, not independently trained, so this had to be verified empirically)
3. **DINOv2-base instead of -small -- IN PROGRESS.** v1 (3ep) done and already
   beats small at matched epoch count. Continue-training kernel running now
   (`rsna-knee-base-continue`, v1->v2, +3ep). When it finishes: compare tier1
   against the table above; if still improving, keep extending the same way
   v13->v3->v4 worked for small, then consider ensembling base checkpoints
   together and/or with the small-backbone checkpoints.
4. Combine multiple independent silver-label sources (rank-average `steven`'s
   labels with another *verified-not-leaked* public source) to push past the
   ~0.89 AUC ceiling any single silver source has against gold
5. Investigate weak labels specifically -- Lateral OA, MCL, Baker's have shown up
   weakest across holdout runs (check `data/interim/kernel_output_*/`.log files
   for exact per-label numbers); figure out if it's class imbalance or genuinely
   harder signal, and address accordingly
6. Multi-slice/volumetric input instead of one representative crop per series

Explicitly deferred / do not start without asking again: external MRI datasets
(MRNet/OAI) and RadImageNet-style domain-specific pretrained backbones.

## How to check on the running base-continue kernel
```
.venv/Scripts/kaggle.exe kernels status aarohnanoti/rsna-knee-base-continue
# once COMPLETE:
.venv/Scripts/kaggle.exe kernels output aarohnanoti/rsna-knee-base-continue -p data/interim/kernel_output_base_continue_v1 --force
# then read data/interim/kernel_output_base_continue_v1/rsna-knee-base-continue.log for tier1/tier2 AUC
```

## Submission mechanics reminder
I can push/run kernels via the Kaggle API, but **clicking "Submit to Competition"
on kaggle.com is the user's action** -- no API for that on code competitions. After
any kernel completes with a good result, tell the user the kernel URL and ask them
to submit it.
