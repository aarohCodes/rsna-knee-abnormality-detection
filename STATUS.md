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
| base v3 | 9 | 0.7715 (regression) | 0.7496 (regression) |

**Base v2 (6 epochs) is the best checkpoint we have, full stop.** Base v3
(9 epochs) overfit: training loss kept falling nicely (0.4206->0.3806->
0.3394->0.2987) but BOTH holdout tiers dropped together, broadly across
almost every label -- not one noisy outlier, a real generalization reversal.
Lesson: for base with these settings (LR, 2 unfrozen blocks, batch 8), 6
epochs is the sweet spot and 9 overfits -- unlike small, where 9 epochs
(v4) was still improving. Don't assume the same epoch count generalizes
across backbones. Do not submit base v3. Do not continue-train past v2
along this same trajectory without changing something first (lower LR,
more regularization, or fewer unfrozen blocks) -- more epochs alone made
it worse here.

Base v2 is the best single (non-ensembled) checkpoint tier1 to date, and its
epoch-scaling shape (single model, +3 epochs from v1) matches the case that
held up on real LB before (small v13->v3), not the ensemble case that
didn't. Not yet submitted to competition -- ready to.

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
  - `submission_base_continue2/` -> `rsna-knee-base-continue-2`: **COMPLETE**,
    base v2->v3 (+3ep). tier1=0.7715, tier2=0.7496 -- a REGRESSION vs v2
    (0.7931/0.7774) despite lower training loss. Overfitting. **Do not
    submit this kernel.** model_base_v3.pt downloaded locally but not
    uploaded as a dataset (no reason to build further on a worse checkpoint).

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
3. **DINOv2-base instead of -small -- DONE. v2 (6ep, tier1=0.7931) is the
   winner; v3 (9ep) overfit and regressed (tier1=0.7715).** Stop extending
   epochs on this trajectory. Ready to submit base v2 to competition (not
   yet submitted). Moving on to item 4 next rather than more epoch-scaling
   or ensembling base checkpoints (v2+v3 ensembling isn't obviously safe
   either, since v3 is a worse, overfit checkpoint -- would need a real
   submission to check, not just assumed from tier1, per the item-2 lesson).
4. **Combine multiple independent silver-label sources -- INVESTIGATED, no
   viable improvement found.** Checked two candidates against our 58 gold
   studies (same leakage-check methodology as before):
   - `pilkwang/rsna-knee-llm-labels` (CC0): legitimately independent (LLM
     reads Report text only, confirmed via its `api_labeler.py` source --
     no gold access), 79.5% cellwise agreement with gold (not suspicious).
     But weaker alone (0.87 vs steven's 0.8932), and every combination
     strategy tried (confidence-weighted average, max-confidence-wins,
     steven-primary/pilkwang-backfill at 3 thresholds) scored BELOW steven
     alone (best blend: 0.8893). Diluting steven with a weaker source hurts,
     it doesn't help.
   - `rayanbabur/rsna-knee-2026-calibrated-soft-targets` (CC0): **REJECTED**.
     Its shipped `train_targets_soft.csv` exactly copies real gold labels
     for all 58 gold studies (696/696 cells, 100% match) -- same
     disqualifying pattern as the two datasets rejected earlier. Its
     self-reported "0.8991 gold AUC" is not reproducible against the file
     actually shipped (100% match would trivially score ~1.0, not 0.8991,
     so the report describes something other than what's in the zip) and
     can't be independently trusted.
   Conclusion: steven's `llm_labels_v4_blend.csv` remains the sole silver
   source. Don't revisit this item without a genuinely new, independently-
   verified-stronger candidate -- weaker sources measurably hurt when blended.
5. **Investigate weak labels -- IN PROGRESS.** Aggregated per-label tier1 AUC
   across all 6 real training runs (small v13/v3/v4, base v1/v2/v3) -- mean
   per label, sorted worst to best:
   MCL 0.666, Lateral Meniscus 0.730, Baker's 0.752, Medial Meniscus 0.768,
   PF OA 0.770, Lateral OA 0.782, ACL 0.788, Effusion 0.800, Contusion 0.802,
   Medial OA 0.819, Fracture 0.832, Synovitis 0.876.
   **MCL and Lateral Meniscus are consistently weakest by a wide margin, in
   every single run** -- correcting the earlier guess (Lateral OA, MCL,
   Baker's) which wasn't rigorous. Ruled out simple class imbalance as the
   explanation: Fracture (7.2% silver prevalence, 18/58 gold) and Synovitis
   (12.6% prevalence, 27/58 gold) are similarly-or-more rare yet score much
   higher (0.832, 0.876) -- rarity alone doesn't predict weakness.
   Every training study has >=1 raw Coronal-plane series (100% coverage,
   confirmed from train_series.csv), so if there's a data-coverage gap it
   must be in whether that series classifies into the COR_T1 slot
   specifically (MCL is normally read off coronal T1) -- needs real DICOM
   headers to check, not available in the small local sample. Pushed
   `rsna-knee-slot-coverage-diagnostic` (CPU-only, no GPU, no training --
   just reads one DICOM header per series across all 24,371 training series
   and tabulates slot coverage, cross-tabulated against MCL-positive vs
   MCL-negative gold studies). Waiting on results.
6. Multi-slice/volumetric input instead of one representative crop per series

Explicitly deferred / do not start without asking again: external MRI datasets
(MRNet/OAI) and RadImageNet-style domain-specific pretrained backbones.

## Next action (pending user)
`rsna-knee-base-continue` (base v2, tier1=0.7931, our best checkpoint) is
COMPLETE and ready to submit -- waiting on the user to open
https://www.kaggle.com/code/aarohnanoti/rsna-knee-base-continue and click
"Submit to Competition" (I have no API access to do this). Do NOT submit
`rsna-knee-base-continue-2` (base v3) -- confirmed regression, see above.
5/5 daily submissions available (cap reset at 2026-08-31 00:00 UTC, all
prior 5 used on 2026-08-30 UTC). Once base v2 is submitted and a real score
comes back, move on to roadmap item 4 (multi-source silver labels).

## Submission mechanics reminder
I can push/run kernels via the Kaggle API, but **clicking "Submit to Competition"
on kaggle.com is the user's action** -- no API for that on code competitions. After
any kernel completes with a good result, tell the user the kernel URL and ask them
to submit it.
