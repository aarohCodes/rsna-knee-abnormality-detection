# RSNA Knee Abnormality Detection — Competition Plan

## Context

The user's goal is to win the **RSNA Knee Abnormality Detection** Kaggle competition
(`rsna-knee-abnormality-detection`), sponsored by the Radiological Society of North America,
$77,000 prize pool. The user has already entered. Final ranking is decided by the
**private leaderboard**, scored on the metric below — there is no existing project for this
competition yet, so this plan starts from a clean slate.

**Verified competition facts** (via Kaggle API + the user's own screenshots/description — not
scraped, so trustworthy):

- **Task**: predict, per knee-MRI study, the probability of 12 abnormalities: `ACL, MCL,
  Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis,
  Baker's, Contusion, Fracture`.
- **Metric** (confirmed by user screenshot): macro-averaged ROC AUC —
  `Final Score = (1/12) * Σ AUC_i` across the 12 targets.
- **Code competition** (confirmed via the competition's own Code Requirements, pasted by the
  user): `isKernelsSubmissionsOnly = true` — final submission must be produced by a Kaggle
  Notebook (Kernel). Max 5 submissions/day, max team size 5. **CPU and GPU notebooks both capped
  at ≤9 hours run-time. Internet access is disabled during the scored run.** Freely/publicly
  available external data and pretrained models are allowed, but since internet is off at
  submission time they must be attached to the kernel in advance (e.g. as a Kaggle Dataset),
  not fetched live via `transformers`/`torch.hub`/etc. The output file must be named exactly
  `submission.csv`.
- **Deadlines**: entry/team-merger deadline **2026-10-15**, final submission deadline
  **2026-10-22**.
- **Data layout** (confirmed via direct API file listing/content fetch):
  - `train.csv`: `StudyInstanceUID, Report, ACL, MCL, Medial Meniscus, Lateral Meniscus,
    Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture` — labels
    are binary 0/1. **Per the official Dataset Description: only a small subset of training
    studies carry per-condition labels; the rest are blank by design, and the competition
    explicitly suggests deriving labels for the unlabeled remainder from the `Report` text.**
    This is confirmed, not a guess — blanks mean "no direct label," not "confirmed negative."
    `Report` is free-text, multilingual (Spanish/Dutch/French/English seen).
  - `train_series.csv` / `test_series.csv`: `StudyInstanceUID, SeriesInstanceUID,
    Fluid_Sensitive, Fat_Suppression, Anatomical_Plane` (Sagittal/Coronal/Axial — not always
    correlated with each other despite often co-occurring). Each study has multiple series.
  - `train_series/{StudyInstanceUID}/{SeriesInstanceUID}/{SOPInstanceUID}.dcm`: DICOM slices,
    nested per-study/per-series. Series typically hold 20–45 slices (median 30), tail to a few
    hundred. Mixed transfer syntaxes (uncompressed Explicit VR LE, JPEG Lossless, JPEG 2000,
    Implicit VR LE) — `pydicom` alone won't decode all of them, need `pylibjpeg`/`gdcm` plugins.
    Every DICOM stripped to an allowlisted set of 86 metadata tags (some usual tags may be
    absent). Studies span diverse international sites, scanners, protocols, populations.
  - **`test.csv` has ONLY `StudyInstanceUID` — no `Report` column, confirmed both by direct
    inspection and the official docs ("The Report field will not be provided at the testing
    stage").** ~1,300 studies in the real test set (the visible `test.csv` is just a 3-row
    placeholder, replaced with real data during scoring). Radiology report text exists for
    training only — at inference time the model sees images alone. This makes sense clinically:
    the point is to help interpret a scan *before* a report is written.
  - `sample_submission.csv`: `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,
    Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`, all values
    default `0.5` — this is the exact required output format.
  - **Distribution notice** (official): abnormality prevalence is *not* guaranteed consistent
    across train / public leaderboard / private leaderboard — reinforces the need for robust,
    grouped/stratified CV rather than tuning to train-set class balance.
- **Rules highlights**: external/pretrained data allowed if "reasonably accessible to all at
  minimal cost" (open checkpoints fine, no paid/gated data); no private code sharing outside
  your team; a winner must hand over full training+inference code, weights (as a public Kaggle
  Dataset), and a reproducible writeup under CC-BY-NC 4.0 — so the pipeline should be clean and
  reproducible from the start, not notebook spaghetti.
- Public reference notebook for DICOM-loading patterns:
  `https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1`. Its actual Kaggle metadata
  (pulled via API) confirms it runs GPU-enabled, internet-disabled — matching the rules — and
  uses: vision backbone `metaresearch/dinov2/PyTorch/small/1` (DINOv2-small, a strong
  self-supervised ViT, good default choice over a plain ImageNet CNN); auxiliary dataset
  `pilkwang/rsna-knee-llm-labels` (LLM-extracted structured labels from the free-text `Report`
  — a concrete answer to the "blank vs negative" label-semantics question: at least one strong
  public approach re-derives cleaner labels from the report text via an LLM rather than trusting
  the raw sparse columns as-is); and `pilkwang/rsna-knee-weights` (pretrained checkpoint,
  presumably fine-tuned on this task). Full notebook source not yet read — worth pulling in
  detail once implementation starts.
- **Efficiency score** (confirmed by user screenshot, separate from the main AUC metric):
  `Efficiency = AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400`, objective is to **minimize**
  this score. `Benchmark` = AUC of the trivial `sample_submission.csv` (all 0.5s); `maxAUC` =
  the best AUC on the Private Leaderboard; `RuntimeSeconds` = the submission's evaluation
  runtime. Since `Benchmark − maxAUC` is negative, higher AUC pushes the first term more
  negative (better); `32400` seconds = exactly the 9-hour runtime cap, so a full-length run adds
  a flat `+1.0` penalty. There's a separate "Efficiency Leaderboard" tracked during training and
  finalized on private data after the competition ends. **Confirmed via RSNA's own press
  release**: this is a distinct award category ("for the first time, awards for the most
  efficient models"), separate from the main AUC-ranked prize — not the determinant of the
  primary win. Still worth keeping inference reasonably lean (it's a real, separate prize
  opportunity and cheap insurance against the runtime cap), but don't trade meaningful AUC for
  it. A public notebook, `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`, tracks this
  leaderboard specifically — worth reviewing if pursuing the efficiency award seriously.

**Verified local environment** (via direct inspection — this is a fresh machine for this task):

- Python 3.11.9 (Microsoft Store distribution). Already installed: `numpy`, `pandas`,
  `scikit-learn`, `opencv-python`, `pillow`, `xgboost`, `jupyter`/`jupyterlab`, `kaggle`
  (1.7.4.5), `kagglehub`. `~/.kaggle/kaggle.json` credentials present and working.
- `kaggle.exe` exists but its `Scripts` dir is **not on PATH** — needs fixing or full-path
  invocation.
- **No local GPU** (no `nvidia-smi`, no NVIDIA install). CPU-only machine.
- **109 GB free disk** — not enough to casually pull the full DICOM corpus (medical-imaging
  competitions routinely run 50–300+ GB).
- Missing: `torch`, `transformers`/`sentence-transformers`, `pydicom` (+`pylibjpeg`/`gdcm` for
  compressed DICOM transfer syntaxes), `timm`, `iterstrat` (multilabel stratified CV).

## Strategic approach: hybrid local + Kaggle-Kernel workflow, image-only inference model

Because this is a code competition, the full dataset is already mounted read-only inside every
Kaggle kernel — the full corpus never needs to live on this machine. Given no local GPU and
109 GB free:

- **Local machine**: code development, EDA on a small downloaded sample, CPU-feasible baselines
  (e.g. classical ML on tabular features), unit tests, experiment logging, notebooks prepped for
  Kaggle.
- **Kaggle Kernels**: full-data EDA validation, GPU model training, and the mandatory final
  submission kernel. A train-kernel → weights-Dataset → inference-kernel split keeps GPU-heavy
  work off this machine and satisfies the "hand over weights as a public Dataset" winner
  obligation for free.

**Model design must reflect the confirmed data split**: since `Report` text is absent from
`test.csv`, it **cannot** be used as a live input feature for the scored model. The correct use
of `Report` is as a **training-only auxiliary signal** — e.g. image–text contrastive
pretraining, multi-task distillation, or simply as an EDA tool to sanity-check the 12 label
columns — never as a fusion input the final image-only inference path depends on. (This corrects
a naive "late-fusion of image+text as input" design, which would be unsubmittable.)

**Baseline first, iterate second**, timeboxed against the Oct 15 / Oct 22 deadlines.

### Concrete techniques from the public baseline notebook (read in full via Kaggle API)

`pilkwang/rsna-knee-baseline-v1` is unusually rigorous for a "baseline" — its reasoning is
worth adopting wholesale rather than re-deriving:

- **`Fluid_Sensitive`/`Fat_Suppression` in `train_series.csv` are degenerate** — they agree on
  every training row, so they carry one axis, not two. Recover the true weighting (T1/T2/PD,
  via `ScanningSequence`/`SeriesDescription`/`SequenceName`/`ScanOptions`/TR/TE) and fat-sat
  from DICOM headers directly, then bucket every series into one of six canonical **slots**:
  `SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1` — chosen so every
  one of the 12 findings has at least one slot that shows it well. Feed the model per-slot
  embeddings plus a presence mask (studies rarely have all six).
- **Don't sort slices by filename** (SOP Instance UID is unordered, ~0 rank correlation with
  anatomy in this corpus). Compute true spatial order from
  `ImageOrientationPatient × ImagePositionPatient` (slice normal, then project position onto
  it); fall back to `InstanceNumber` only when geometry tags are missing.
- **Resolution must clear a Nyquist bound**, not just "look reasonable": crop to a fixed
  physical field of view (~130mm, below the smallest FOV in this corpus but still containing
  the joint) then resize so effective pixel pitch resolves a ~1mm meniscal-tear feature
  (pitch ≤ d/2) — e.g. 336px over a 130mm crop ≈ 0.387mm/pixel. Normalize intensity per series
  by its own 1st/99th percentile over the sampled stack (MR has no absolute intensity scale).
- **Normalize knee laterality**: medial/lateral labels (4 of the 12 targets, plus MCL) are
  meaningless unless every study is mapped to a consistent side. `Laterality` DICOM tag is
  often missing (by vendor, not randomly) — derive side from the sign of the image center's
  patient-coordinate x, using position + orientation + spacing; flip the in-plane axis for
  coronal/axial, reverse slice order for sagittal.
- **Decode once, cache as `uint8`, budget the cache against free memory** (cache size scales
  with resolution², only linearly with slice count — resolution is the expensive axis).
  Directly relevant to the efficiency score's runtime term. Also adopt the defensive pattern of
  **writing a trivial valid `submission.csv` before any real inference runs**, overwritten once
  real predictions exist — guarantees a scoreable file even if the run fails partway.
- **Aggregate slots with per-diagnosis attention, not mean-pooling**: each of the 12 findings
  gets its own learned query attending over the (masked) slot embeddings, since different
  findings are best read from different sequences — a mean pooling step would dilute the
  informative slot with several uninformative ones.
- **Fine-tune only the last blocks of the vision encoder**, with the encoder learning rate far
  below the head's (discriminative LR) — a frozen ImageNet/self-supervised encoder has no
  vocabulary for MRI signal (e.g. a meniscal tear on a PD sequence), but full fine-tuning from
  random-head gradients would either leave the head undertrained or wreck the encoder.
- **Weight training samples by label-source confidence**: report-derived (silver) labels are
  weighted by extraction confidence (a report that never mentions a finding pulls weakly, not
  as a hard negative); the small directly-annotated (gold) subset gets elevated weight since
  it's the only signal read from images rather than text.
- **Train once, submit many**: since submissions are scored kernels with a runtime cap,
  separate training (produces a weights Dataset, "members") from the scored inference kernel,
  which just loads and predicts. Each member should carry its own preprocessing fingerprint
  (resolution/crop/etc.) so a member is never silently run under a mismatched preprocessing
  recipe (shape-compatible weight loading hides this kind of bug) — verify with a small seeded
  self-check before trusting a loaded checkpoint.
- **Ensemble by rank, not by raw probability or logit averaging**: since AUC only reads order,
  averaging raw probabilities lets the most confident member dominate; rank-averaging combines
  exactly the information the metric uses.
- **Two CV leaks to guard against beyond grouping by `StudyInstanceUID`**: (1) some reports are
  byte-identical templates across studies (e.g. unremarkable-knee boilerplate) — group folds by
  a hash of the report text too, so duplicate-report groups never split across train/holdout;
  (2) keep a **two-tier validation**: a ~20% holdout (on derived/silver labels) drives model and
  epoch selection since it has enough samples per class to be statistically meaningful, while a
  much smaller "annotation check" (only the gold-labeled studies that land in that holdout)
  approximates the actual competition metric but is too small to arbitrate alone — report it,
  don't tune on it.

## Repo structure

New project at `C:\Users\aaroh\rsna-knee-abnormality-detection\`:

```
.venv/                              (gitignored)
.gitignore                          (data/, .venv/, *.dcm, weights/, __pycache__/, checkpoints)
README.md
requirements.txt
config/
  data.yaml                         (paths, label columns, seed)
  model_baseline.yaml
data/                               (gitignored)
  raw/                              (train.csv, train_series.csv, sample_submission.csv, small DICOM sample)
  interim/                          (cached DICOM metadata, resized image caches)
  processed/                        (fold assignments, cached embeddings)
notebooks/                          (EDA only, not production code)
  01_eda_labels.ipynb
  02_eda_reports_language.ipynb
  03_eda_dicom_series.ipynb
src/rsna_knee/                      (installable package: `pip install -e .`)
  data/labels.py                    (load/clean train.csv, blank-vs-NaN handling)
  data/dicom_io.py                  (pydicom loading, slice sorting, windowing)
  data/series.py                    (join train_series.csv, per-plane/sequence grouping)
  models/vision.py                  (image backbone + slice/series aggregation)
  models/report_pretrain.py         (optional: image-text contrastive/auxiliary pretraining)
  train.py
  validate.py                       (grouped multilabel-stratified CV)
  metrics.py                        (macro AUC, matching the exact competition formula)
  infer.py                          (shared by local eval AND the submission kernel — image-only path)
experiments/runs.csv                (run_id, date, commit, description, CV mean AUC, per-class AUC)
submission/
  submission_kernel.ipynb           (image-only inference notebook pushed to Kaggle)
  kernel-metadata.json
scripts/
  download_data.py                  (selective/sampled download, disk-budget aware)
  push_kernel.ps1
```

Principle: all real logic lives in `src/rsna_knee/`, tested locally; both local EDA and the
Kaggle submission notebook just call into it. Directly satisfies the reproducibility/handover
rule.

## Build steps

1. **Environment**: create `.venv`, add the Kaggle `Scripts` dir to PATH (or invoke by full
   path), install the missing packages (CPU build of `torch` locally — heavy training happens
   on Kaggle GPUs), verify auth with `kaggle competitions list -s rsna-knee-abnormality-detection`.
2. **Disk-aware data pull**: `kaggle competitions files -v` to check real sizes before
   downloading; grab the small CSVs in full; pull only a stratified sample (~20–50 studies
   across the three anatomical planes) of DICOMs locally, budgeted under ~10–15 GB. Also pull
   the public baseline notebook (`kaggle kernels pull pilkwang/rsna-knee-baseline-v1`) to avoid
   reinventing DICOM-loading boilerplate.
3. **EDA** (in order): (a) quantify exactly what fraction of `train.csv` rows have any of the 12
   labels populated vs fully blank (confirmed to be a minority per the official docs — get the
   real number); (b) per-class positive rate / imbalance among the labeled subset; (c) `Report`
   language distribution; (d) DICOM consistency (spacing, transfer syntax mix, decodability with
   `pylibjpeg`/`gdcm`); (e) series-per-study / per-plane breakdown, slice-count distribution;
   (f) sanity-check that `test.csv`/`test_series.csv` placeholders match the documented real
   test set shape (~1,300 studies) once the actual scoring data is visible.
4. **Label-expansion pipeline (high priority — this is the main differentiator)**: since most
   training studies lack direct labels, build a pipeline to derive silver labels for the
   unlabeled majority from `Report` text — e.g. an LLM-based or rule/NER-based extractor per the
   12 conditions, run only on training data (never needed at inference since `Report` is
   train-only). Validate this extractor against the small labeled subset (treat it as
   ground truth to measure extraction precision/recall per class) before trusting it to expand
   training signal. This mirrors the public baseline's `rsna-knee-llm-labels` approach and is
   likely the single highest-leverage piece of this competition.
5. **Metrics + CV**: implement `metrics.py` to exactly match `Final Score = (1/12) Σ AUC_i`;
   implement grouped multilabel-stratified K-fold (`iterstrat`), grouping by **both**
   `StudyInstanceUID` and a hash of `Report` text (duplicate-report leak, per the baseline
   notebook above) so rare classes are represented in every fold and no template report leaks
   across the train/holdout boundary. Local CV is the primary signal — leaderboard submissions
   (capped at 5/day) are for confirming CV↔LB correlation, not for tuning.
6. **Baseline model (image-only)**: start with a cheap correctness check — `timm`
   ImageNet CNN (e.g. `efficientnet_b0`) with frozen features, naive mean-pooled slices, run
   locally on CPU. Then build the real model per the baseline-notebook techniques above:
   header-derived series slots, geometry-based slice ordering, Nyquist-driven resolution,
   laterality normalization, DINOv2-small backbone with only-last-blocks fine-tuning
   (discriminative LR), per-diagnosis attention pooling over slots, confidence-weighted silver
   labels with elevated gold-label weight — trained on Kaggle GPU kernels against the expanded
   (silver+gold) label set from step 4.
7. **Push an early end-to-end submission** (data → model → `submission_kernel.ipynb` →
   scored leaderboard entry) well before iterating further, to surface kernel runtime/GPU-quota
   surprises with time to fix them.
8. **Iterate** (priority order): improve the label-expansion extractor's precision/recall →
   tune fine-tuning depth/LR schedule on the vision backbone → refine slot presence handling and
   attention pooling → volumetric/3D modeling as a stretch goal → ensemble multiple validated
   "members" (checkpoints, per the weights-as-dataset pattern above) combined by **rank
   averaging, not raw probability/logit averaging** — watching the efficiency-score runtime cost
   of any ensemble/TTA additions against the `RuntimeSeconds/32400` penalty term.
9. **Submission kernel discipline**: keep training and inference as separate kernels (train →
   upload weights as a Kaggle Dataset → inference kernel loads that dataset and writes
   `submission.csv` matching the sample format exactly, including column order and the exact
   filename `submission.csv`). Since the scored run has **internet disabled and a 9-hour cap**:
   set the inference kernel's internet toggle off during dev so failures surface early, download
   any pretrained backbone/tokenizer weights once and attach them as a Kaggle Dataset input
   (never `from_pretrained(..., resume_download=True)` against the live hub inside the scored
   kernel), and keep inference-kernel runtime well under 9 hours with margin for slower shared
   hardware on submission day.
10. **Tracking + deadlines**: log every run in `experiments/runs.csv`; target a working
   end-to-end submission within the first 1–2 weeks; freeze architecture a few days before
   Oct 22 to focus on ensembling and a dry-run of the exact final kernel.

## Verification

- `kaggle competitions list -s rsna-knee-abnormality-detection` succeeds (auth works end-to-end).
- `pytest` over `src/rsna_knee` unit tests for `metrics.py` (hand-check macro-AUC formula) and
  `labels.py` (blank/NaN handling) pass.
- A full local dry run on the small DICOM sample: `train.py` → `validate.py` produces per-fold,
  per-class, and mean AUC without errors.
- `submission_kernel.ipynb`, run via "Save & Run All (Commit)" **with internet access turned
  off** on Kaggle (does not count against the 5/day submission cap), completes in well under
  9 hours and produces a file named exactly `submission.csv` whose header matches
  `sample_submission.csv` and has one row per test `StudyInstanceUID`.
- First real "Submit to Competition" click confirms a public-leaderboard score and that local CV
  is in a sane range of it.
