# Status (last updated 2026-08-31)

Read this file first in any new session/after context compaction to get oriented.
Full strategy/reasoning is in PLAN.md; this file is the living "where are we now."

## ACTIVE BLOCKER: Kaggle weekly GPU quota exhausted (30h/week)
As of 2026-08-31, `kernels push` for a GPU kernel fails with "Maximum weekly GPU
quota of 30.00 hours reached." No GPU kernels can run until this resets.
**Confirmed exact reset time via `kaggle quota` (a real CLI command, discovered
2026-08-31 -- use this directly instead of guessing or asking the user to check
manually next time): GPU refreshAt = 2026-09-05T00:00:00.** TPU quota is
completely unused (20h available) if ever worth targeting. **CPU-only kernels
(`enable_gpu: false`) are unaffected** -- confirmed working today (slot-coverage
and slice-count diagnostics both ran fine under this same quota exhaustion).
Base v2 (real score 0.812) is our
best checkpoint and is already submitted -- nothing is at risk, GPU-dependent work
is just paused until quota resets. See roadmap item 6 for the queued next GPU run.

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
0.596 -> 0.617 -> 0.646 (random-init ResNet18 baselines, v0) -> 0.782 (DINOv2-small,
6 epochs, TTA) -> 0.775 (v13+v3+v4 rank ensemble -- REGRESSION vs the 0.782 single
model) -> 0.790 (base v3, 9ep, submitted 2026-08-31 04:09 UTC) -> **0.812** (base v2,
6ep, submitted 2026-08-31 02:40 UTC -- our new best real score). Current visible
competition leader: ~0.952 (public LB, snapshot from early in the competition -- may
have moved since).

All 5 submissions on 2026-08-30 (UTC) were used; 2 of the fresh 5 for 2026-08-31 are
now used (base v2, base v3), 3 remain today.

**Tier1 (silver-label holdout) tracking vs real score, all data points so far:**
- small v13->v3 (single model, +3ep): predicted 0.7833, actual 0.782 -- close match.
- small ensemble (v13+v3+v4): predicted 0.7911 (best-yet), actual 0.775 -- WORSE
  than the plain 6-epoch model's 0.782, despite a higher predicted tier1. Ensembling
  correlated checkpoints is the one case where tier1 has actively misled us.
- base v2->v3 (single model, +3ep): predicted DROP (0.7931->0.7715), actual DROP
  confirmed (0.812->0.790) -- direction matched, though both absolute real scores
  came in ~0.02 HIGHER than tier1 predicted (opposite bias from the ensemble case).
  **Refined lesson**: for single-model epoch-scaling comparisons specifically, tier1's
  RELATIVE ranking has now held up correctly twice (small v13->v3, base v2->v3),
  even though absolute values drift either direction. The one confirmed failure mode
  remains specifically ensembling correlated checkpoints together -- keep trusting
  tier1 deltas for single-model epoch decisions, keep verifying ensembles with a
  real submission before trusting them.

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
   MCL-negative gold studies).

   **RESULTS: MCL's weakness is a real, systemic data-coverage gap.**
   COR_T1 (the slot MCL is read off) is present in only 48.1% of ALL
   training studies (2121/4407) and 51.7% of gold studies (30/58) -- roughly
   half the corpus has no usable coronal T1 series at all. Critically, this
   is NOT selective: MCL-positive gold studies have COR_T1 at 55.6% (5/9),
   MCL-negative at 51.0% (25/49) -- essentially identical, so it's not that
   positive cases are disproportionately missing the view. It's that the
   whole dataset is close to a coin flip on whether the diagnostic sequence
   for MCL exists at all, which drags down average performance across the
   board. By contrast, Lateral Meniscus's slot (SAG_FLUID_FS) has strong
   88.7% coverage -- so its weakness is more likely genuinely harder signal
   (plausibly the well-known posterolateral popliteus-tendon-hiatus
   mimicker that makes lateral meniscus reads inherently harder even for
   human radiologists), not a data problem, and probably isn't fixable the
   same way.

   Full slot coverage (all 4407 studies): SAG_FLUID_FS 88.7%, COR_FLUID_FS
   91.0%, AX_FLUID_FS 86.7%, SAG_FLUID_NOFS 46.2%, COR_T1 48.1%, SAG_T1
   31.9%. The three FLUID_FS slots are consistently well-covered; the two
   T1 slots and SAG_FLUID_NOFS are all in the 30-48% range -- this may
   partly explain other mid-weak labels too (PF OA, Baker's) if they lean
   on T1 views.

   **RESULT: the fallback made things WORSE, not better. Disabled by default.**
   `rsna-knee-base-fallback` (fresh 6ep, fallback enabled) scored tier1=0.7791
   (vs base v2's 0.7931 -- no fallback) and, critically, MCL itself got worse
   too (0.641 vs 0.663), the opposite of the hypothesis. 9 of 12 labels
   declined. Real caveat: this run trained 6 continuous epochs in one pass,
   while base v2's 6 epochs happened as two 3-epoch stages with AdamW's
   optimizer state reset in between (we only save/restore model weights, not
   optimizer state) -- so the comparison isn't perfectly clean, some of the
   gap could be training-trajectory variance rather than the fallback itself.
   Leading hypothesis for a real causal effect: marking a substituted slot
   `mask=1.0`, identical to a genuine one, gives the model no way to tell
   real ligament-optimized T1 contrast from substituted fluid-sensitive
   contrast at the same slot position -- adding noise rather than removing
   a gap. **Made `use_slot_fallbacks` an opt-in StudyDataset parameter,
   default `False`**, so this negative result can't silently regress the
   known-good default pipeline (base v2, real score 0.812) if we retrain
   further. `SLOT_FALLBACKS` and the substitution logic are kept in the
   codebase (not deleted) in case a better-controlled retry is worth it
   later (e.g. a distinct fallback indicator instead of mask=1.0), but
   don't re-enable without a cleaner ablation. 4 tests now cover this
   (3 fallback-behavior tests pass `use_slot_fallbacks=True` explicitly,
   1 new test confirms the default stays off). Full suite: 91/91 passing.
   Synced to `rsna-knee-src` (remember `--dir-mode zip`).

   Item 5 conclusion: MCL's weakness is real and explained (data coverage),
   but the fix attempted didn't pan out. Don't revisit without either a
   properly controlled ablation (same continuous-training methodology on
   both sides) or a redesigned fallback that distinguishes substituted from
   genuine content.
   Raw slot-coverage data: `train_series_with_slots.csv` in
   `data/interim/kernel_output_slot_coverage_v1/`.

   **Side experiment, RESULT: also negative, don't submit or pursue.**
   `rsna-knee-base-continue-lrdecay` (base v2 +3ep, ENCODER_LR/HEAD_LR halved
   to 5e-6/5e-4) scored tier1=0.7688, tier2=0.7093 -- WORSE than v2 (0.7931/
   0.7774) and even slightly worse than v3's full-LR overfit (0.7715/0.7496).
   Notably, final training loss (0.2970) landed almost identical to v3's
   (0.2987) despite the 2x smaller LR -- the learning rate barely changed the
   training trajectory at all. This suggests the overfitting isn't primarily
   a step-size problem; it's tied to total epoch exposure to the noisy
   silver labels, largely independent of LR in the range tested. **Base v2
   (6 epochs, tier1=0.7931, real=0.812) looks like a genuine, LR-insensitive
   ceiling for this training recipe.** Don't keep trying small LR/epoch
   variations around v2 -- two independent attempts (fallback, LR-decay)
   both failed to beat it. A different lever (item 6, or a genuinely
   different architecture/data change) is needed for further gains, not
   more tweaking of this recipe.

   **Synthesis (why this session stopped at 2 attempts, not 3+)**: the
   coronal fallback and the LR-decay experiment changed two *completely
   different* knobs (what data fills a slot vs. how fast weights update),
   yet both landed in almost the same degraded zone (tier1 ~0.769-0.779,
   both clearly worse than v2's 0.7931). That convergence, plus the LR
   experiment barely changing the final training loss at all despite a 2x
   LR change, is more consistent with a **silver-label noise ceiling**
   (past ~6 epochs of exposure, the model is fitting steven's report-
   derived label noise, not real signal, regardless of which training knob
   is turned) than with either fix being individually wrong. This makes
   item 6 (which adds genuinely new visual evidence per study, orthogonal
   to label quality) the more promising remaining lever over further
   recipe tweaking, and also weakly re-affirms item 4's conclusion (no
   better label source was found, so this ceiling isn't easily fixable
   from the label side either).

   **Third experiment, CONFIRMS the ceiling.** During a long idle stretch
   waiting on user input for item 6, ran one more cheap test:
   `rsna-knee-base-continue-unfreeze4` (base v2 +3ep, last 4 of 12 blocks
   unfrozen instead of 2, standard LR -- isolates capacity as the only
   changed variable). Result: tier1=0.7774, tier2=0.7568 -- still below v2
   (0.7931/0.7774), though the least-bad of the three extension attempts
   (v3=0.7715/0.7496, lrdecay=0.7688/0.7093, unfreeze4=0.7774/0.7568).
   Even meaningfully more capacity (32.8% of params trainable vs 16.5%)
   couldn't beat just stopping at v2. **Three independent knobs (slot
   content, LR, capacity) all land in the same degraded neighborhood below
   v2 -- this is now a solidly 3-way-confirmed silver-label noise ceiling
   around 6 epochs for this recipe, not a fixable bug.** Not submitted to
   the competition (clearly not better than v2 locally, no reason to spend
   a submission). **Stop trying to extend base v2's training via any
   nearby recipe variation -- exhausted.** The only remaining levers are
   item 6 (new visual evidence) or a genuinely different label source
   (already investigated in item 4, none found).
6. **Multi-slice/volumetric input -- APPROVED by user (via AskUserQuestion) and
   IMPLEMENTED.** Scoped first with a CPU-only diagnostic
   (`rsna-knee-slice-count-diagnostic`): median 30 slices/series (p10=20,
   p90=39, p95=45) but a heavy tail (p99=160, max=320); using every slice
   would be a 32.4x decode-volume increase, ruled out as blowing the 9-hour
   cap. Settled on `slices_per_slot` = K evenly-spaced crops per slot
   instead of one, K chosen small (3) for the first test.

   **Implementation** (`src/rsna_knee/`, 102/102 tests pass, verified
   against real local DICOM data, all committed):
   - `data/dicom_io.py`: refactored `load_series_representative` into
     `_load_ordered_dicom_stack` (decode+order, the expensive part) +
     `_extract_representative_crop` (cheap numpy ops), so the new
     `load_series_multi_slice(k=...)` pays the decode cost ONCE per series
     regardless of k, not k times -- decode/IO dominates runtime, so this
     matters a lot. New function is purely additive; existing function's
     behavior is byte-for-byte unchanged (confirmed: k=1 output equals the
     old function's output exactly).
   - `datasets.py`: `StudyDataset(..., slices_per_slot=K)`; images shape
     becomes `(S, K, resize_px, resize_px)` instead of `(S, resize_px,
     resize_px)` when K>1. Default K=1 reproduces prior behavior exactly.
   - `models/vision.py`: `KneeMultiLabelModel.forward` accepts both 5-dim
     `(B,S,C,H,W)` and 6-dim `(B,S,K,C,H,W)` images; the K dim is
     mean-pooled (NO new learnable params) before the existing
     `SlotAttentionHead` runs unchanged. This is why base v2's checkpoint
     loads and runs under the new code with zero shape/state_dict changes.
   - `infer.py`: `run_inference`/`run_inference_tta` take `slices_per_slot`.
     **Known limitation, documented in the code**: `load_series_multi_slice`
     always spans the full ordered stack (fractions 0..1) and ignores
     `center_fraction` entirely, so TTA's center_fraction variation is
     currently a no-op when `slices_per_slot>1` -- use a single-pass config
     until/unless TTA is redesigned around multi-slice.

   **First validation run, RESULT: didn't beat v2, but confounded --
   second, cleaner test approved and running.** `rsna-knee-base-multislice`
   (resumed base v2, `slices_per_slot=3`, 2 additional epochs) scored
   tier1=0.7793, tier2=0.7729 -- below v2 (0.7931/0.7774). Runtime was
   2h24m total (epochs ~3040s each, holdout inference 2026s), comfortably
   under the 9h cap -- that risk is resolved regardless of the accuracy
   result.

   **Important pattern**: this is now the FIFTH attempt to beat v2 (slot
   fallback, v3, lrdecay, unfreeze4, multislice) and all five land in a
   tight tier1 band of 0.769-0.779, despite changing completely different
   things (data content, LR, capacity, and now the input representation
   itself). Only v2 stands out at 0.7931. Honest alternative reading worth
   holding onto: v2's 0.7931 may be a favorable outlier in the noise rather
   than a "true" target -- five independent methods clustering together is
   data too, not just five failures.

   **Confound specific to this run**: multi-slice was resumed from v2 for
   only 2 epochs, the same short-resume pattern as the failed v3/lrdecay/
   unfreeze4 attempts -- can't cleanly separate "multi-slice doesn't help"
   from "2 epochs isn't enough exposure to a suddenly-richer input for a
   model already converged on K=1." Not submitted (not better locally).

   **Second, cleaner test APPROVED but BLOCKED**: `rsna-knee-base-
   multislice-fresh` was scoped (K=3 FRESH from pretrained DINOv2-base,
   6 epochs matching v2's own budget) but `kernels push` failed with
   "Maximum weekly GPU quota of 30.00 hours reached" -- **no more GPU
   kernels can run until Kaggle's weekly quota resets at
   2026-09-05T00:00:00** (confirmed via `kaggle quota` -- see the
   top-of-file ACTIVE BLOCKER note; use that command directly in future,
   don't guess or ask the user to check manually). CPU-only kernels are
   UNAFFECTED (confirmed: both diagnostic kernels today, slot-coverage and
   slice-count, ran fine and didn't count against this cap) -- any further
   data-analysis-only work remains possible in the meantime. The notebook/
   kernel-metadata for `submission_base_multislice_fresh` are built and
   committed, ready to push the moment quota resets -- no rework needed,
   just re-run `kaggle kernels push -p submission_base_multislice_fresh`.

   This was estimated ~5h45m based on the first multi-slice run's
   per-epoch timing (6 x ~3040s + setup + eval), safely under the 9h cap.
   This is the decisive test of whether multi-slice input helps at all,
   isolated from the "any further training past v2 degrades" confound.
   Base v2 (real=0.812) remains our best and is already submitted --
   nothing is at risk from this block, just further testing is paused.
   Reference to beat: v2's tier1=0.7931/real=0.812.

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
