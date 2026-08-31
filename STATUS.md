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
| small ensemble (v13+v3+v4) | -- | 0.7911 (regressed to 0.775 real) | 0.7516 |
| base v1 | 3 | 0.7820 | 0.7891 |
| **base v2** | 6 | **0.7931** | 0.7774 |

Base v2 is the best single (non-ensembled) checkpoint tier1 to date, and its
epoch-scaling shape (single model, +3 epochs) matches the case that held up on
real LB before (small v13->v3), not the ensemble case that didn't. Not yet
submitted to competition -- ready to. `model_base_v2.pt` downloaded to
`data/interim/kernel_output_base_continue_v1/`, not yet re-uploaded as a
weights dataset (do that before any further continue-training off of it).

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
  `rsna-knee-weights-v4` (small, 9ep), `rsna-knee-weights-base-v1` (base, 3ep),
  `rsna-knee-weights-base-v2` (base, 6ep, tier1=0.7931, uploaded)
- **Kernels** (each has its own folder + `scripts/_build_*_notebook.py` generator --
  edit the generator, run it, then `kaggle kernels push -p <folder>`):
  - `submission/` -> `rsna-knee-submission-v0`: original v0/v13 combined train+infer
  - `submission_infer/` -> `rsna-knee-inference-tta`: loads v13, TTA infer only
  - `submission_continue/` -> `rsna-knee-continue-training`: v13->v3 (+3ep)
  - `submission_continue2/` -> `rsna-knee-continue-training-2`: v3->v4 (+3ep)
  - `submission_ensemble/` -> `rsna-knee-ensemble`: loads v13+v3+v4, evaluates solo
    + ensembled on holdout, submits whichever combo wins (currently: all-3
    ensemble, tier1=0.7911) -- **submitted, scored 0.775 real, a REGRESSION
    below the plain 6-epoch model's 0.782. Don't resubmit this kernel as-is.**
  - `submission_base/` -> `rsna-knee-dinov2-base`: base v1, 3 fresh epochs, DONE
    (tier1=0.7820, tier2=0.7891)
  - `submission_base_continue/` -> `rsna-knee-base-continue`: **COMPLETE**,
    base v1->v2 (+3ep), tier1=0.7931 (best single-checkpoint yet), tier2=0.7774.
    Not yet submitted to competition -- ready to submit.

## Roadmap (user-approved sequence, execute in this order)
1. ~~See how 9-epoch submission scores~~ -- done, tier1 confirmed reliable proxy
   for single-model epoch scaling (not for ensembles, see item 2)
2. ~~Ensemble v13+v3+v4 via rank averaging~~ -- done, but the real submission
   (0.775) came back WORSE than the plain 6-epoch model (0.782) despite a
   higher predicted tier1 (0.7911 vs 0.7833). Correcting an earlier wrong
   note here that called this "a real gain confirmed" -- it wasn't; that was
   written before the real submission result came back. Lesson: tier1 deltas
   between correlated/ensembled checkpoints are not reliable without a real
   submission to confirm, even though tier1 deltas from single-model epoch
   scaling have held up well (see item 1, item 3).
3. **DINOv2-base instead of -small -- v2 DONE, best single checkpoint yet
   (tier1=0.7931).** Ready to submit to competition (not yet submitted).
   Next decision: submit base v2 first to get a real-LB confirmation (given
   the item-2 lesson that tier1 alone isn't fully trustworthy), then decide
   whether to keep extending (v2->v3, +3 more epochs) based on how the real
   score compares to tier1's prediction -- do NOT jump straight to ensembling
   base checkpoints together without a real-submission check first, per the
   item-2 lesson.
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

## Next action (pending user)
`rsna-knee-base-continue` (base v2, tier1=0.7931) is COMPLETE and ready to
submit -- waiting on the user to open
https://www.kaggle.com/code/aarohnanoti/rsna-knee-base-continue and click
"Submit to Competition" (I have no API access to do this). 5/5 daily
submissions available (cap reset at 2026-08-31 00:00 UTC, all prior 5 used
on 2026-08-30 UTC). `model_base_v2.pt` has been uploaded as the `rsna-knee-weights-base-v2`
Kaggle dataset, ready to use as a resume point for further continue-training.

## Submission mechanics reminder
I can push/run kernels via the Kaggle API, but **clicking "Submit to Competition"
on kaggle.com is the user's action** -- no API for that on code competitions. After
any kernel completes with a good result, tell the user the kernel URL and ask them
to submit it.
