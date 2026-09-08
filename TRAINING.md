# Training runbook (Linux box)

Everything here is written to be copy-paste. The goal of the first day is **one
number**: our member's macro AUC on the community's 881-study fold-0 holdout,
next to the 0.9178 a single public member scores on those same studies.

## Why this exists

We are at 0.938 public (rank ~163 of 3,243, top 5%). Reproducing public notebooks
saturates around **0.941** — the notebook authors themselves rank 95–910. Only 35
teams exceed 0.945. Both free levers (cross-lineage blending, better silver
labels) were measured and produced nothing. So training is the only remaining
path to top 10, and this runbook is that path.

## The gate — read this before spending a week

**Standalone AUC is not the gate.** Measured on this project's own data, a member
scoring **0.8963** — well *below* the 0.9202 base — still added **+0.0026** to the
blend, because its rank correlation with the base was only 0.843. What decides a
member's worth is **decorrelation**, not raw score.

So the real gate is the blend delta, cross-fitted:

```bash
python training/blend_member.py choose-weight     --base  <base pipeline's holdout predictions.csv>     --member runs/member/holdout_predictions.csv     --labels data/interim/label_sources/steven/llm_labels_v4_blend.csv
```

| honest delta | verdict |
|---|---|
| **> +0.002** | Ship it. Scale to 5-fold and more backbones. |
| **≤ +0.002** | Stop the training track, keep the banked 0.938. |

It splits the holdout, picks the weight on one half and scores it on the other,
both directions, so the number it reports is not the one it fitted. Our previous
attempt scored 0.7735 standalone, which is almost certainly too weak to help even
decorrelated — but the blend test, not the standalone number, is what decides.

## Paths

**Nothing here assumes a username or a drive.** Every path is either a
command-line argument or derived from the script's own location, so the repo works
from any directory on any machine. `/data/rsna` below is only an example — use
whatever has ~600 GB free (`~/rsna`, `/mnt/big/rsna`, anything). Pick it once and
pass the same value to all three commands.

The only fixed location is `~/.kaggle/kaggle.json`, which the Kaggle CLI requires
by convention. Copy it over from the Windows box (`C:\Users\aaroh\.kaggle\kaggle.json`):

```bash
mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

## Steps

```bash
git clone https://github.com/aarohCodes/rsna-knee-abnormality-detection.git
cd rsna-knee-abnormality-detection
bash training/setup_linux.sh /data/rsna          # env + corpus + silver labels
source .venv-linux/bin/activate
```

The repo is private, so `git clone` will ask for GitHub credentials (use a
personal access token, not your password).

Run the later commands **from the repo root**, since `--silver` is a repo-relative
path. `setup_linux.sh` downloads the silver labels there for you — they are not in
git, because `/data/` is gitignored.

`setup_linux.sh` refuses to continue if `nvidia-smi` fails or torch cannot see
CUDA — without a working GPU this box is no better than Kaggle's T4s.

```bash
# 1. smoke test the cache on 20 studies (~1 min). Confirms DICOM decoding works.
python training/prep_cache.py --data-root /data/rsna --out /data/rsna/cache --limit 20

# 2. full cache: ~28 GB, a few hours, parallel over cores. Resumable — re-running
#    skips studies already cached, so Ctrl-C is safe.
python training/prep_cache.py --data-root /data/rsna --out /data/rsna/cache

# 3. train one member. Prints the gate verdict at the end.
python training/train_member.py \
    --cache /data/rsna/cache --data-root /data/rsna \
    --silver data/interim/label_sources/steven/llm_labels_v4_blend.csv \
    --arch vit_small_patch14_dinov2.lvd142m --px 336 --epochs 16 --batch 8
```

Tune `--batch` to VRAM (each sample is `k_train` windows, so effective batch is
`batch × k_train` images): 24GB → `--batch 8`; 16GB → `--batch 4`; 48GB+ →
`--batch 16`. If you hit OOM, halve `--batch` before touching anything else.

## What the pipeline does, and why

- **`prep_cache.py`** decodes the corpus once into 64 × 384 × 384 uint8 volumes
  (~28 GB total). Training reads each study many times; decoding JPEG-2000 DICOM
  every epoch would dominate runtime. Cached at 384 even though we train at 336 —
  downsampling is free, re-decoding is not.
- **Slice order comes from geometry** (`ImageOrientationPatient` × `ImagePositionPatient`),
  never filenames. SOP Instance UIDs have ~zero rank correlation with anatomy here.
- **Crop is physical (140 mm), not pixel-based**, so a knee is the same size across
  scanners with different pixel spacing.
- **Normalisation is per sampled stack** (2nd/98th percentile) because MR has no
  absolute intensity scale.
- **Attention pooling per finding**, not mean pooling: a Baker's cyst and an ACL
  tear are not visible on the same slice, and mean pooling dilutes the one
  informative window among dozens of uninformative ones. We measured that failure
  earlier in this project.
- **Discriminative LR** — encoder at `3e-5`, head at `1e-3`. A self-supervised
  encoder has no vocabulary for MR signal so it must adapt, but driving it at the
  head's LR while the head is still random destroys the pretrained features.
- **The split is the community's** (`md5(report) % 5`, holdout fold 0). Verified to
  reproduce their fold sizes exactly: 881 / 834 / 937 / 885 / 870. Any other split
  makes the 0.9178 comparison meaningless.


## Where these settings come from

The defaults are **the raptor author's published training recipe**
(`dreaddevelopment/knee-mri-training-the-twelve-finding-model`). They are the
best-placed public author on the leaderboard (0.941, rank ~81), so their numbers
beat anything we would guess: 16 epochs, head LR 1e-3, backbone LR 3e-5, ~3 hours
on a single 4090 at 384px, and weight-averaging the best 3 epochs.

Two things we took from reading it that are easy to get wrong:

- **No flip augmentation.** Mirroring a knee swaps the medial and lateral
  compartments, and four of the twelve findings are defined by which compartment
  they sit in (Medial/Lateral Meniscus, Medial/Lateral OA). Window-sampling jitter
  is the only augmentation, by design. An earlier version of this script had a
  random horizontal flip; it was removed.
- **Soft labels, not hard.** Silver labels are probabilities and the loss expects
  them; forcing every hedged sentence to 0/1 throws away the hedging.

Known gap: the cache does **not** normalise knee laterality (left vs right). The
raptor recipe we copied does not either, and it reaches 0.941, so this is not
fatal -- but it is a real candidate lever, since consistent laterality is what
makes the medial/lateral findings well-defined. Worth an A/B once the gate passes.

## Rules compliance

Training off-Kaggle is explicitly allowed — only the *submission* must be a Kaggle
notebook. Weights ship as a Kaggle Dataset attached to the inference kernel; this
is exactly what the top public teams do (their manifests name A6000 and H100
trainers). Do not add any data source that is not publicly available to all
entrants at no cost.



## Verified backbones

Each of these was built with pretrained weights and pushed through a real forward
pass at the stated resolution on this machine, so none of them will fail on
arrival. Verified rather than assumed, because a patch-14 ViT at 336px is not its
native geometry and would have been a plausible first-run crash.

| arch | px | feat_dim | note |
|---|---|---|---|
| `vit_small_patch14_dinov2.lvd142m` | 336 | 384 | start here; closest to the public members |
| `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` | 384 | 1024 | **the 0.941 author's own backbone**; heaviest |
| `convnext_small.in12k_ft_in1k` | 336 | 768 | third family, for decorrelation |

They are three different architecture families (ViT / hybrid / pure conv), which
matters: the blend test showed a member's value comes from **decorrelation**, not
from its standalone score. Two ViTs that agree with each other add far less than a
ViT and a conv net that disagree.



## Storage: you never need 500 GB

The raw corpus is ~500 GB–1 TB, but nothing trains on raw DICOM. Everything decodes
to a fixed cache first:

| what | size |
|---|---|
| raw corpus | ~500 GB–1 TB — **never needed** |
| cache, `raptor` (64 slices) | **27.7 GB** |
| cache, `lean` (32 slices) | **13.5 GB** |
| checkpoints + working space | ~5 GB |

So the training box needs roughly **20–35 GB**, which matters on a shared machine
with a quota'd home directory.

**Two ways to get the cache there.** The second avoids downloading the corpus at all:

| path | transfer | needs 500 GB scratch? |
|---|---|---|
| download raw → build cache locally | ~500 GB+ | yes |
| **build cache on Kaggle → download the cache** | **13.5–27.7 GB** | **no** |

`prep_cache.py` is **CPU-only** (pydicom + OpenCV), so the Kaggle route runs on a
CPU notebook and **does not touch the 30h GPU quota**. Kaggle caps notebook output
at ~20 GB, so:

```bash
# lean cache fits in ONE run
python training/prep_cache.py --data-root /kaggle/input/... --out /kaggle/working/cache --slot-preset lean

# the full 27.7 GB cache needs two, then publish each as a Dataset
python training/prep_cache.py --data-root ... --out /kaggle/working/cache --shard 1/2
python training/prep_cache.py --data-root ... --out /kaggle/working/cache --shard 2/2
```

`--shard I/N` takes a deterministic slice of the sorted study list, so shards never
overlap and never leave gaps (verified: 1/2+2/2 covers [0:2]+[2:4] of 4).

## Three slot presets

| preset | slices | cache | purpose |
|---|---|---|---|
| `raptor` | 64 (32/20/12) | 27.7 GB | the public recipe, reproduced — the baseline |
| `balanced` | 64 (24/24/16) | 27.6 GB | targets the weak findings (see below) |
| `lean` | 32 (12/12/8) | **13.5 GB** | **for the efficiency prize** — halves decode and inference |

**`balanced`** exists because measured per-label AUC shows the two weakest findings
are the ones read on the two most under-sampled planes: PF OA (0.8942, axial) and
MCL (0.9055, coronal). Same 64 slices, so it is a clean single-variable test.

**`lean`** exists because the efficiency prize is a separate award where runtime
dominates: one hour is worth +0.0504 AUC, more than the entire leaderboard's
accuracy spread (0.032). Our best public-weights efficiency entry scores ≈ −1.909
(0.898 in ≤37 min) and cannot be made faster — its slice count is fixed by each
member's verified preprocessing fingerprint. **A member we train ourselves has no
such constraint.** At the measured marginal rate, a member running in 15 minutes
at 0.89 beats that entry outright, which is a far easier target than the +0.010
AUC main-LB top 10 requires.

## A second lever: `--within-slot`

The cached volume is five concatenated slots from *different series*. A 3-slice
window centred on a slot boundary stacks unrelated acquisitions into one RGB image
— measured at **~13% of all windows**. `--within-slot` forbids that. The public
recipe has the same defect; inheriting it is not a reason to keep it.

```bash
# baseline (public recipe, reproduced)
python training/train_member.py --cache /data/rsna/cache ... --out runs/base
# within-slot windows (no new cache needed)
python training/train_member.py --cache /data/rsna/cache ... --within-slot --out runs/withinslot
# a different slot budget (needs its own cache)
python training/prep_cache.py --data-root /data/rsna --out /data/rsna/cache_lean --slot-preset lean
python training/train_member.py --cache /data/rsna/cache_lean ... --out runs/lean
```

Judge each by the **blend delta** from `blend_member.py`, not by standalone AUC.

## Getting a trained member into a submission

Training alone changes nothing; the weights have to reach a scored kernel.

```bash
# 1. predictions on the clean holdout (for choosing the weight).
#    --holdout-fold 0 derives the 881-study community holdout in-process, so
#    there is no id file to create or keep in sync.
python training/infer_member.py --checkpoint runs/member/best.pt     --data-root /data/rsna --split train --holdout-fold 0     --out member_holdout.csv

# 2. choose the blend weight honestly  (see "The gate" above)
python training/blend_member.py choose-weight --base <base_holdout.csv>     --member member_holdout.csv --labels data/.../llm_labels_v4_blend.csv

# 3. in the scored kernel: predict on test, then mix at the chosen weight
python training/infer_member.py --checkpoint best.pt --out member_test.csv
python training/blend_member.py apply --base submission.csv     --member member_test.csv --weight 0.25 --out submission.csv
```

Blending is by **rank**, not probability: AUC reads only the ordering, so
probability averaging lets the most confident member dominate regardless of
whether it is more correct.

**Where it goes matters.** `bee-v2` spends ~8.2h of the 9h cap and cannot absorb a
member. `dinosaurs-v2` runs ~4h and has ~5h spare, so it is the base to blend
into. See STATUS.md.

## After a PASS

1. Train 5 folds (`--holdout-fold 0..4`) — the biggest variance-reduction lever we
   have, and it directly targets private-LB risk.
2. Vary backbones for decorrelation (DINOv2/v3 ViT, CoAtNet, ConvNeXt). Decorrelated
   members are what actually move an ensemble; more of the same lineage buys ~0.0016.
3. Upload weights as a Kaggle Dataset, blend into the submission kernel by **rank
   averaging** (AUC reads order only, so probability averaging lets the most
   confident member dominate).
4. Select the final two on the clean holdout — **never** on the public LB. That
   discipline is our main private-LB edge.
