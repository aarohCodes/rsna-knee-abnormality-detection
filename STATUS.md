# Status (last updated 2026-09-05)

Read this file first in any new session/after context compaction to get oriented.
Full strategy/reasoning is in PLAN.md; this file is the living "where are we now."
Full commit history is also mirrored to a private GitHub repo:
https://github.com/aarohCodes/rsna-knee-abnormality-detection (private -- kept
private deliberately while the competition is active, per the rules requiring
code disclosure only if we win).

## >>> START HERE (updated 2026-09-06, ~20:30) <<<

### Current state
- **Banked public score: 0.938** (bee-v2; rank ~163 of 3,243). Rank 10 on the public LB is 0.950. Deadline 2026-10-22.
- **THREE SUBMISSIONS ARE PENDING** (submitted by the user 20:14-20:15):
  refs 56062615, 56062620, 56062626. They correspond to the three kernels below.
  Scoring re-runs each pipeline against the real 1,322-study test set, so it takes
  time. A background watcher is running; if this session is gone, just re-check
  with `kaggle competitions submissions rsna-knee-abnormality-detection`.

| kernel | basis | expectation |
|---|---|---|
| `rsna-knee-bee` | prvsiyan "bee's knees" V8, a documented 0.937 | **~0.937** (strongest) |
| `rsna-knee-dinosaurs` | author's newer notebook + 4 raptor families | unknown, unvalidated |
| `rsna-knee-final-v1` | our fork + corrected DINOv3 blend weight | ~0.925-0.927 (holdout said 0.9242) |

### What to do when the scores land
1. Compare the three. Our clean holdout has predicted real scores within 0.003
   (0.9194 predicted -> 0.922 actual), so public scores are a trustworthy
   selection signal here -- no separate holdout run needed per candidate.
2. Keep the best two for final selection in October. Never select on a number we
   have not seen; never select an unverified config.
3. If bee lands ~0.937, the gap to rank 10 is ~0.013.

### BREAKTHROUGH LEAD (2026-09-06, ~21:00): we were missing a whole model arm

Applied the "look outward first" lesson immediately after the compaction and it
paid off within minutes. Two things published TODAY that we had never seen:

**1. `mattiaangeli/bend-the-knee-to-the-dinosaurs-all-public` (58 votes, run
16:32 UTC today).** The author republished the dinosaurs notebook with **every
dataset source public** -- that is what "ALL PUBLIC" means in the title.

Diffed it against our `submission_dinosaurs` fork:
- the first 26 code cells are **byte-identical** to ours (ours has one extra
  cell: our integrity guard). So the code was never the problem.
- their metadata lists **12** dataset sources; ours listed **11**. The one we
  lacked is **`mattiaangeli/rsna-knee-coat-resgated-ep10-top3`** -- a CoAtNet
  RMLP-2 + finding-gated-residual arm with three checkpoints (e04/e06/e08).

**Why this matters and why it was invisible.** In cell 25 the coat arm runs
inside `try: _coat_substitute() except Exception: restore public raptor arm`.
With the dataset unattached the manifest lookup raises, the notebook prints
`[coat-arm] unavailable`, and it carries on producing a perfectly healthy-looking
submission. That arm is blended into `_raptor.csv` at `private_alpha=0.40`, and
`_raptor.csv` then enters the final blend at weight 0.60 -- so **the missing arm
was worth 0.60 x 0.40 = 24% of the final ranking**. Textbook silent degradation:
exactly the failure mode the 0.500 bug taught us to hunt for.

**Verified before dispatch (per the pinned remote-artifact lesson):**
- `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` is public and listable, CC0-1.0.
- Downloaded `coat_resgated_ep10_top3_manifest.json` and hashed it:
  `98511a8f...3f284e` == the notebook's `MAN_SHA`. **Match.**
- Downloaded the pinned wheel and hashed it: `236c8df5...75a7a9` == `WHL_SHA`.
  **Match.** Both hash gates in `_coat_substitute` will therefore pass.

**Built and pushed `aarohnanoti/rsna-knee-dinosaurs-v2`**: identical notebook,
`dataset_sources` now matching the author's 12 exactly (verified by list
equality), plus a new guard line that prints `COAT_ARM_STATUS=OK|MISSING` by
checking for `/kaggle/working/_coat_raptor_blend_receipt.json`. Never again
let that arm vanish quietly. All code cells `ast.parse`-checked before push.

**2. `yosukeinada/rsna-knee-0-937-22-submissions-6-lessons`** -- another team's
full 22-submission ledger. Independent corroboration of three of our findings,
plus one that reframes the 0.028 gap:
- They **reproduced the public base at 0.936**. We reproduced the same lineage at
  **0.922**. So a faithful reproduction really does score ~0.936 -- our shortfall
  is a *defect in our reproduction*, mechanical and fixable, not a capability gap.
- They ran the bee/V6 route with the **same private-checkpoint -> public-copy
  substitution we did** and scored **0.937**. Direct corroboration for our
  pending `rsna-knee-bee`.
- Their own trained models added **nothing** to the public base (0.874 -> harmful,
  0.910 -> neutral at every weight). Matches our three negative tests exactly.
- Their Gold58-based forecast of 0.941 came in at 0.937. Gold58 deltas do not
  predict LB -- another reason to trust only the clean 881-study holdout.

**Related: the 0.922 kernel's own missing source.** `bend-the-knee-to-dinov3-ensembled`
still lists **13** dataset sources whose **first is an empty string** -- Kaggle
blanks references the caller cannot see, i.e. a private/deleted dataset. That
fork can therefore never be reproduced faithfully by us, which is very likely the
real 0.028 gap for that lineage. The dinosaurs ALL-PUBLIC line has no blanks, so
**it is the reproducible horse and v2 is our clean shot at it.**

### RUNTIME BUDGET DECIDES WHERE OUR MEMBER CAN GO (2026-09-07, ~07:00)

Measured the stage costs from the v2 save-run logs and projected to the real
1,322-study test set. This settles a planning question that would otherwise have
bitten us in October.

**bee-v2 (0.938) is nearly out of runtime.** Its raptor stage logs cumulative
26 / 36 / 44 / 64s over 3 studies for the 4 arms. Excluding first-arm warmup that
is ~4.2s per study per arm:

| stage | projected on 1,322 studies |
|---|---|
| 4 raptor arms | ~6.2h |
| 20-member DINOv2 package | ~0.7h |
| coat resgated arm | ~0.9h |
| audits + overhead | ~0.4h |
| **total** | **~8.2h vs the 9h cap** |

That matches the observed ~8h15m end to end. **Headroom is under an hour.**

**Consequence: our own trained member CANNOT be added to bee-v2.** A member needs
its own forward pass over 1,322 studies (~1-2h), which would push bee-v2 past the
cap and produce exactly the silent no-score failure that killed `56062615`.

**Our member must go into `dinosaurs-v2` instead.** The plain dinosaurs run
completed in ~3h05m; plus the coat arm that is ~4h, leaving **~5h of headroom** --
ample for one or more of our own members.

**Revised endgame shape:**
| final slot | candidate | ceiling |
|---|---|---|
| A | `bee-v2` = 0.938 | fixed; no room to extend |
| B | `dinosaurs-v2` + our trained member(s) | the one that can actually climb |

So the training track's output has a specific destination, and it is the lighter
kernel. Build the blend against dinosaurs-v2 from the start rather than
discovering the runtime wall after the weights exist.

**If more headroom is ever needed**: each raptor arm calls `build_study` and
re-decodes the same DICOMs from scratch. Decoding once per study and reusing
across the 4 arms would cut a large fraction of that 6.2h. That is a real
optimisation, but it means editing their proven inference code, so it is a
fallback rather than a first move.

### OVERNIGHT SWEEP: 3 NEGATIVES AND A REAL BUG CAUGHT (2026-09-07, ~06:45)

Re-ran the ecosystem check (the lesson that keeps paying) and found three new
notebooks. Tested each claim rather than adopting it.

**1. Label-correlation stacking: NEGATIVE (-0.0161 macro).**
`busyaprime/the-rsna-knee-labels-are-not-independent` shows the 12 labels
co-occur far beyond chance, and argues 12 independent heads leave signal on the
table. The co-occurrence is REAL on our data (Contusion x Fracture lift 2.79,
Lateral Meniscus x Lateral OA 2.17). But cross-fitted stacking -- predicting each
label from all 12 model outputs -- made every label worse except Contusion
(+0.0012, noise). Macro 0.9202 -> 0.9041. **The model already encodes the
correlations; re-mixing its own outputs only adds overfitting.** A joint head
trained end-to-end is a different (untested) proposition.

**2. OA-aware rule labeler: NEGATIVE, badly.**
`busyaprime/osteoarthritis-is-almost-never-written-as-oa` argues OA labels are
near chance because radiologists write osteophytes / joint space narrowing /
gonarthrose, never "OA". Correct in principle, and directly aimed at PF OA (our
2nd weakest). Graded their emitted `silver_labels.csv` against the 58 gold:

| | macro AUC vs gold |
|---|---|
| **steven/v4_blend (in use)** | **0.8927** |
| their rule labeler | 0.7266 |

**Steven wins on all 12 labels**, including OA specifically (0.8887 vs 0.7640).
An LLM reading the report already handles the consequence vocabulary; a
hand-built multilingual lexicon does not catch up. Also a useful cross-check:
steven's 0.8927 matches the 0.8932 recorded in this file long ago.
**That is now THREE label sources tested and rejected. The label track is closed.**

**3. The raptor author's TRAINING notebook -- and it caught a real bug of mine.**
`dreaddevelopment/knee-mri-training-the-twelve-finding-model` publishes the actual
recipe of the best-placed public author (0.941, rank ~81):
- 16 epochs, head LR **1e-3**, backbone LR **3e-5**, ~3h on ONE 4090 at 384px
- weight-average the best 3 epochs ("costs nothing at inference, unlike an ensemble")
- soft labels, not hard
- **"each epoch sees a different view of the same study. Nothing else is augmented."**

**Our trainer had a random horizontal flip.** That is actively harmful here:
mirroring a knee swaps medial and lateral compartments, and 4 of the 12 findings
are defined by compartment (Medial/Lateral Meniscus, Medial/Lateral OA) --
including Lateral Meniscus, our 3rd weakest. Removed. Also aligned LRs (was
3e-4/2e-5), epochs (was 8), and implemented top-k weight averaging, which
evaluates the average and only adopts it if it beats the best single epoch.

**Known gap recorded**: our cache does not normalise knee laterality. Neither does
the raptor recipe, and it reaches 0.941, so it is not fatal -- but it is a real
candidate lever and worth an A/B once the gate passes.

### *** NEW BEST: bee-v2 = 0.938, RANK ~163 *** (2026-09-07, ~06:20)

`56063873` (bee-v2 = author's V9 four-view contract + the recovered CoAtNet
resgated arm) scored **0.938**, completing in ~8h15m -- inside the 9h cap.

| | score | rank of 3,243 |
|---|---|---|
| yesterday morning | 0.922 | ~1,173 |
| last night | 0.936 | ~371 |
| **bee-v2** | **0.938** | **~163 (top 5.0%)** |
| prvsiyan (bee's own author) | 0.937 | ~197 |
| Mattia Angeli | 0.939 | ~123 |
| Dread Development | 0.941 | ~81 |

**We are now ABOVE the author of the notebook we forked**, and within 0.003 of the
best-placed notebook author. Rank moved **371 -> 163, up 208 places**.

**Three things this settles:**
1. **The recovered CoAtNet arm was real signal.** The silently-missing dataset was
   worth finding; the whole +0.002 over bee V8 comes from it plus V9.
2. **The author was right to revert V8.** V9's four-view contract beats the
   five-view experiment, reproduced independently by us.
3. **My runtime worry did not materialise.** I flagged honestly that bee-v2 at
   7h35m was running past my "roughly neutral" estimate and might hit the wall
   that killed 56062615. It completed at ~8h15m. The estimate was too optimistic
   but the margin held. Treat all my 3-study runtime extrapolations as rough.

**Gap to rank 10 (0.950) is now 0.012**, down from 0.028 yesterday morning.

**Immediate next**: submit `dinosaurs-v2` -- validated, cap reset, and its ~4h
projection now has evidence behind it since the coat arm demonstrably fits.

### ONE SUBMISSION FAILED OUTRIGHT -- THE 9h CAP IS REAL (2026-09-07, ~05:30)

`56062615` came back **COMPLETE with an EMPTY score**: the kernel died during the
scored run and produced no valid submission.csv. Elapsed to failure ~9h05m, which
points at the **9-hour runtime cap**.

| ref | elapsed | result |
|---|---|---|
| 56062626 | ~3h05m | 0.909 |
| 56062620 | ~6h30m | **0.936** |
| 56062615 | ~9h05m | **FAILED, no score** |
| 56063873 (bee-v2) | pending | -- |

**Attribution still not established** (the API exposes no notebook reference), but
the completion times constrain it: a ~3h pipeline, a ~6.5h pipeline, and one that
hit the wall. `final-v1` runs 24 members plus a DINOv3 stage and is the most
likely timeout candidate. CHECK THE SUBMISSIONS PAGE to settle it.

**Consequences:**
1. **An unscored submission is worth nothing on the private LB.** Never select one
   of the final two from a run that did not complete with a score.
2. The runtime risk flagged when adding the CoAtNet arm was real, but the
   arithmetic still clears for both v2 kernels:
   - `bee-v2` = V9 DROPS a raptor arm (~1.2h) and ADDS coat (~1h) -> ~neutral vs
     bee V8's 6h30m.
   - `dinosaurs-v2` = the ~3h pipeline + ~1h coat -> ~4h, comfortable.
3. Any future candidate must be judged on runtime headroom, not just holdout AUC.

### *** WHERE THE POINTS ACTUALLY ARE: PER-LABEL WEAKNESS *** (2026-09-07, ~05:30)

Macro AUC weights all 12 findings equally, so a weak finding costs exactly what a
strong one does. Measured per-label AUC of the best member on the clean holdout:

| finding | AUC | prev | SE | headroom to 0.95 |
|---|---|---|---|---|
| **Contusion** | 0.8939 | 19.4% | 0.0166 | +0.056 |
| **PF OA** | 0.8942 | 48.4% | 0.0112 | +0.056 |
| **Lateral Meniscus** | 0.9047 | 13.4% | 0.0189 | +0.045 |
| **MCL** | 0.9055 | 18.0% | 0.0163 | +0.045 |
| Lateral OA | 0.9126 | 26.9% | 0.0131 | +0.037 |
| Effusion | 0.9150 | 60.6% | 0.0093 | +0.035 |
| Synovitis | 0.9237 | 12.8% | 0.0175 | +0.026 |
| Medial Meniscus | 0.9268 | 40.9% | 0.0100 | +0.023 |
| Medial OA | 0.9327 | 36.3% | 0.0101 | +0.017 |
| ACL | 0.9349 | 19.8% | 0.0132 | +0.015 |
| Fracture | 0.9432 | 6.7% | 0.0211 | +0.007 |
| Baker's | 0.9549 | 25.9% | 0.0098 | -0.005 |

**Lifting only the four weakest to 0.95 is worth +0.0168 macro (0.9202 -> 0.9370)
-- larger than our entire gap to top 10.** Improving the four strongest by 0.01
each is worth only +0.0033. PF OA's SE is 0.0112, so its weakness is real signal,
not sampling noise.

**AND THERE IS A MECHANISM.** Compare the weak findings against the raptor slot
budget's plane allocation:

| plane | slices of 64 | findings that need it |
|---|---|---|
| Sagittal | 32 (50%) | ACL, menisci -- all scoring WELL |
| Coronal | 20 (31%) | **MCL 0.9055** |
| Axial | 12 (19%) | **PF OA 0.8942** (patellofemoral joint) |

**The two most under-sampled planes carry the two weakest findings.** PF OA is
read on axial slices through the patella and gets 19% of the budget; MCL is a
coronal structure. Sagittal takes half the budget and carries the findings that
already score best.

**This is a targeted, testable edge that no forker will try**, because everyone
inherits the same slot table from the same notebook. Implemented as
`--slot-preset balanced` in `prep_cache.py`
(sagittal 24 / coronal 24 / axial 16 vs raptor's 32/20/12), same 64 slices so the
comparison is single-variable. Smoke-tested on 4 studies: valid cache, and it
fills slightly MORE slots on a sparse study (40 vs 38) because it leans less on
sagittal series some studies lack.

**Caveat to hold**: these AUCs are measured against SILVER labels. Contusion,
Effusion and Synovitis also have the worst silver-vs-gold agreement (0.737, 0.719,
0.702), so part of their apparent weakness may be label noise rather than model
weakness -- a model that denoises could score better on the real test set than on
our holdout. PF OA and MCL do NOT have that excuse: their silver agreement is
0.789 and 0.877, so their weakness is more likely real.

### PHASE 1 COMPLETE -- BOTH FREE LEVERS ARE NEGATIVE (2026-09-07, ~04:50)

Ran the two zero-cost levers to closure. Both are honest negatives, which
settles where the remaining value is.

**1. Cross-lineage blending: NO NEW GAIN.** Scored all 27 local holdout
prediction files, de-duplicated to 10 distinct clean members (excluded
`their_fold1-4` -- those models may have trained on these holdout studies and
would silently inflate any blend). Ran an honest protocol: select blend members
on one half of the 881 studies, score on the other, both directions.

| direction | blend | best single | delta |
|---|---|---|---|
| A->B | 0.9260 | 0.9219 | +0.0041 |
| B->A | 0.9226 | 0.9190 | +0.0035 |
| mean | **0.9243** | 0.9205 | +0.0038 |

The two strongest members correlate at only 0.930, so blending genuinely helps.
**BUT `final-v1` already scores 0.9242 on this same holdout** -- its 50/50 DINOv3
rank blend IS this optimum. So this is a CONFIRMATION that 0.9242 is real and not
an artifact of fitting weights on the evaluation set, **not new signal**. The
community-lineage blend space is genuinely exhausted, as recorded earlier.

**2. Better silver labels: NO GAIN AVAILABLE.** Agreement with the 58 gold
studies (REJECTED leaked sources merged4/refined excluded throughout):

| source | agreement with gold |
|---|---|
| **steven/v4_blend (already in use)** | **0.8173** |
| pilkwang/v2 | 0.7953 |
| 2-source consensus | 0.7939 (worse) |
| 5-source majority vote | 0.6111 (much worse) |

The two good sources disagree on only **4.8%** of cells, so consensus had little
room to help. And with 58 gold studies, SE(agreement) = **0.0153** -- we could not
detect a genuinely better source even if we built one. The label lever is blocked
by VALIDATION CAPACITY, not by ideas.

**CONCLUSION: the free levers are exhausted. The training track is the only
remaining path to top 10.** Without it we sit at 0.936-0.941 (rank ~95-395) and
P(top 10) stays ~5%, entirely dependent on other teams regressing off their
public-set overfitting.

**Therefore the GPU spec on the user's Linux box is the critical path.** Nothing
else productive can start without it.

### MEASURED: THE SHAKEUP NOISE FLOOR (2026-09-07, ~03:50)

User directive: **focus purely on private-LB robustness** -- accept ~0.941
public, and bet on top teams collapsing from public-set overfitting rather than
on us climbing. Bootstrapped the real noise using the 881-study clean holdout
(public DINOv3 member predictions vs steven silver labels, full macro AUC 0.9178,
600 resamples, Mann-Whitney AUC):

| test set | size | SD of macro AUC | 95% range width |
|---|---|---|---|
| public LB | ~397 | **0.0051** | 0.0197 |
| private LB | ~925 | **0.0033** | 0.0127 |

**What this justifies.** A team that made 22 submissions and kept the best sits
about **+0.010 above its true skill** from selecting the max of 22 noisy draws
(E[max of 22] ~ 2.0 SD x 0.0051). The `yosukeinada` ledger documents exactly 22;
the notebook author describes chasing 0.001-0.003 movements. We have made 9,
mostly genuinely different approaches rather than blend-weight tweaks, so we
carry much less inflation into the private split.

**Honest arithmetic.** That regression narrows our 0.014 gap to rank 10 down to
~0.005-0.009. But **102 teams sit at 0.940+** and we would need to pass most of
them. Robustness alone: likely silver, plausible gold, **top 10 ~5%** -- revised
DOWN from ~10% because reproduction is now known to cap at ~0.941.

**Consequences to hold to:**
1. Select the final two on the clean 881-study holdout, NEVER on public LB.
2. Do not spend submissions on blend-weight micro-tuning -- that is precisely the
   behaviour that manufactures the +0.010 inflation we are betting against.
3. Prefer more diverse ensembles over any single tuned config (variance
   reduction is the whole thesis).

**OPEN: the user has a second machine with 128GB RAM.** Training off-Kaggle is
fully rules-legal (only the SUBMISSION must be a kernel; weights ship as a
Dataset -- exactly what dreaddevelopment does per the A6000/H100 trainer names in
the coat manifest). RAM is not the bottleneck; **GPU VRAM is**. If that machine
has 24GB+ the training path reopens and this robustness-only strategy should be
revisited. Awaiting the GPU spec.

### *** STRATEGIC PIVOT: THE PUBLIC-FORK CEILING IS ~0.941 *** (2026-09-07, ~03:30)

Pulled the full public leaderboard (3,243 teams) and looked up the authors of
every notebook we have been forking. **They are not in the top 10. They are next
to us.**

| team | rank | score |
|---|---|---|
| Dread Development (raptor author) | 95 | 0.941 |
| Mattia Angeli (dinosaurs / ALL PUBLIC author) | 159 | 0.939 |
| renta.k (0.937 weak-label author) | 176 | 0.938 |
| prvsiyan (bee author) | 213 | 0.937 |
| **us** | **395** | **0.936** |
| Pilkwang Kim (original baseline) | 910 | 0.935 |

**This FALSIFIES the claim recorded earlier in this file** that the 0.95 cluster
is "~17 forks of the same lineage". It is not. The whole public-fork ecosystem
saturates at **0.937-0.941**, and we are already at the top of it. Only **35
teams exceed 0.945** and **7 exceed 0.950**. Those teams are running something
that has never been published.

**Consequence: careful reproduction cannot reach top 10.** It can reach ~0.941
(author parity) and roughly rank 95-160. Everything beyond that requires signal
the public ecosystem does not contain.

**Hard constraint on out-training them**: the coat package manifest records its
own trainers as `trainer_a6000_epochs01_06.py` and `trainer_h100_epochs07_10.py`
on RTX A6000 and H100 80GB. Top teams train off-Kaggle on hardware we do not
have. We have ~78 T4-hours to the deadline. Brute force is not available.

**Score distribution for calibration:**
| score | rank | pct |
|---|---|---|
| 0.954 | 1 | -- |
| 0.950 | 7 | 0.2% |
| 0.945 | 35 | 1.1% |
| 0.940 | 102 | 3.1% |
| 0.937 | 197 | 6.1% |
| 0.936 | 371 | 11.4% |
| 0.922 | 1173 | 36.2% |
Note the density: +0.001 from 0.936 to 0.937 is worth **174 ranks**.

**THE REVISED PLAN**
1. Bank the reproduction gains (bee-v2, dinosaurs-v2) -> target author parity ~0.941.
2. **Cross-lineage rank blending** -- the one untapped edge. Each author runs
   mostly their OWN lineage; nobody blends bee x dinosaurs x raptor x renta and
   validates it honestly. **The earlier "our contribution is CLOSED" finding does
   NOT apply here**: that measured OUR WEAK member (0.77 clean) against their
   0.92 ensemble. Blending two independently-strong public ensembles is a
   different question and is untested. Constraint: the 9h cap -- bee alone runs
   ~5-6h, so a merged kernel must deduplicate shared stages (both run the raptor
   arms and the coat arm).
3. Select the final two on the **clean 881-study holdout, never the public LB**.
   Our real private-LB edge is that we have made 9 submissions where the
   fork-race teams made 22+, so we carry far less public-set overfitting.

### *** NEW BEST: 0.936 *** (2026-09-07, ~02:45)

`56062620` scored **0.936**, beating the banked 0.922 by **+0.014**. This is now
our best public score and the first real move since 0.922.

| ref | score |
|---|---|
| 56062626 | 0.909 (regression, do not select) |
| **56062620** | **0.936 <- new best** |
| 56062615 | pending |
| 56063873 (bee-v2) | pending |

**Attribution (INFERRED, not confirmed)**: 0.936 is almost certainly
`rsna-knee-bee` (V8). The bee lineage was the documented 0.937, and the
independent `yosukeinada` ledger records reproducing "a public **0.936**
baseline" -- an exact match to what we just scored. That also implies 0.909 is
the degraded `rsna-knee-dinosaurs` (missing the CoAtNet arm) rather than
`rsna-knee-final-v1`, whose holdout predicted 0.9242. If so, 56062615 is
final-v1 and should land near 0.925. **Confirm on the Kaggle Submissions page
before relying on this.**

**What this establishes**: reproducing the strongest published work faithfully is
worth far more than optimising our own. 0.922 -> 0.936 came entirely from running
someone else's better notebook correctly, not from any modelling of ours -- the
same lesson recorded as this project's most important, now with a number on it.

**Gap to rank 10 (0.950) is now 0.014**, down from 0.028.

**Still pending and potentially better**: `bee-v2` is V9 = the exact scored
0.937 four-view contract PLUS the CoAtNet resgated arm at alpha 0.40. If V9
beats V8, it lands above 0.936.

### FIRST SCORE LANDED: 0.909 (2026-09-06, ~23:20)

`56062626` (submitted 20:15:33, the last of the three) scored **0.909** --
**below the banked 0.922**, so it is a regression and must not be among the two
selected for the private LB.

**ATTRIBUTION IS NOT ESTABLISHED.** The submissions API exposes only
ref/date/status/score with no notebook reference, and `kernels list` tracks only
save-runs, not submission runs. Note also that the ref->kernel mapping recorded
earlier in this file was an ASSUMPTION from submission ordering, never verified.
The Kaggle Submissions page shows the source notebook per row; check there.

What each reading would mean:
| if 0.909 is | interpretation |
|---|---|
| `rsna-knee-dinosaurs` (most likely) | confirms the missing-CoAtNet-arm diagnosis; a blend missing 24% of its weight landing 0.013 under baseline fits. Makes **dinosaurs-v2 more valuable**. |
| `rsna-knee-bee` (V8) | supports the author reverting the five-view config; V9/bee-v2 is then clearly the right line |
| `rsna-knee-final-v1` | the surprise: holdout predicted 0.9242 and has tracked within 0.003 all project. A 0.016 miss would be its first real failure and would undermine our main selection instrument. |

Three submissions still pending (`56062615`, `56062620`, `56063873`=bee-v2,
submitted 22:30). Their magnitudes should disambiguate without needing the UI.

### bee-v2 VALIDATED, AND A METADATA BUG I CAUSED (2026-09-06, ~22:40)

**First bee-v2 run FAILED** -- `FileNotFoundError:
/kaggle/input/models/metaresearch/dinov2/pytorch/small/1`. All 20 DINOv2 members
were dropped one by one ("costs one vote, not the run") until none remained, and
the kernel died on `WeightsError: no member produced predictions`.

**Cause was mine**: I hand-wrote the kernel-metadata dict instead of deriving it
from a known-good file, which silently dropped **`model_sources`**. Kaggle Models
are a SEPARATE metadata key from `dataset_sources`, so attaching all seven
datasets still left the DINOv2 backbone unattached. Fixed by rebuilding the
metadata from our V8 fork's file, changing only the id and dataset list.

**Permanent guard added**: a parity assertion that every load-bearing key the
author declares (`model_sources`, `kernel_sources`, `competition_sources`, the
enable flags, `machine_shape`) is present and complete in our fork. Ran it
retroactively against `dinosaurs-v2`: **PASS** (that one was built by editing
existing metadata rather than retyping, which is why it was unaffected).
**Rule: never retype a fork's metadata; derive it and assert parity.**

**Second run: COMPLETE and fully validated.**
```
weights package: 20 member(s)                 <- 0 dropped
[arm 0..3] done + freed | 0/3 study fallback(s)
[resgated] blended CoAtNet resgated e4/e6/e8 into the Raptor arm (alpha 0.40)
infra0021 audit PASSED | public0033 overlay PASSED
BEES_KNEES_V1_PASS / V3_QUALITY_PASS / V4_FRONTIER_AUDIT_PASS / V6_WEIGHT_OVERLAY_PASS
integrity check PASSED: no NaNs, no constant columns
BEE_RESGATED_APPLIED=True alpha=0.4 error=None
```
The checkpoint SHA pin again resolved to `e1443bb0...`, re-confirming the public
copy is byte-identical to the private one we cannot access.

**Runtime, correctly attributed**: the log shows 371s between the raptor stage and
the resgated line, but the arm's own receipt says `elapsed_seconds = 34.5` --
essentially identical to dinosaurs-v2's 35.0s. The difference is fixed overhead
(isolated pip install of the pinned OpenCV, subprocess startup), NOT per-study
cost. Extrapolated: ~51 min on 1,322 studies.
**So V9 drops one raptor arm (widedense, ~1h) and adds the resgated arm (~1h) --
roughly runtime-neutral versus the V8 bee already pending.**

### THE BEE AUTHOR SHIPPED V9, AND OUR PENDING RUN IS THE ABANDONED V8 (2026-09-06, ~22:15)

Having found one missing-dataset defect, I checked the other candidates for the
same bug class. Our bee fork is **not** degraded -- its `coat` hits are only the
CoAtNet *backbone* used by the raptor arms, and V8 genuinely does not need the
coat dataset. But the diff turned up something more valuable.

**The author published a new bee version (13:27 UTC) after we forked at ~10:12.**
Cell-level diff of our fork against their current source:

| cells | verdict |
|---|---|
| 0-5 | byte-identical |
| **6** | **REPLACED**: ours `RAPTOR_FIVE_VIEW_..._V8`, theirs `RAPTOR_FOUR_VIEW_..._V9` |
| 7 | differs: the arm weight table (V8 five-view vs V9 four-view) |
| 8-9, 11-12 | byte-identical |
| 10 | differs by exactly our private->public checkpoint glob patch |
| 13 | our integrity guard (extra) |

**What V9 changes, in the author's own words**: *"V9 restores the exact scored
0.937 four-view contract; the new diversity is the resgated head blended into
the Raptor output, not another entry here."* And at the top of cell 6:
*"Exactly the scored 0.937 blend. V8's fifth view (widedense-v4) is reverted."*

So V8 -- **the version we currently have pending** -- is an experiment that added
a fifth view (`widedense-v4` at weight 0.25) which **the author has since backed
out**. Its weights were a "stated prior rather than a fitted" value, by their own
comment. V9 returns to the documented 0.937 four-view blend and adds the same
CoAtNet resgated arm we just recovered for dinosaurs, at the same alpha 0.40.

**Consequence: our pending bee run is the author's abandoned branch.** It may
still score well, but the configuration with actual evidence behind it is V9.

**Built and pushed `aarohnanoti/rsna-knee-bee-v2`** = author's V9 + our checkpoint
glob patch re-applied + integrity guard + a `BEE_RESGATED_APPLIED=` report.
Verifications done BEFORE dispatch, per the pinned lesson:
- The patch was located **by content** (unique anchor assertion), not by cell
  index -- the index moved from 10 to 12 between versions, so an index-based
  insert would have silently corrupted the wrong cell.
- Every code cell `ast.parse`d after assembly.
- V9's `MAN_SHA`/`WHL_SHA` are the **same two hashes we already verified** against
  the public copies, so both gates will pass.
- Checked V9's "expected exactly one public0033 bundle manifest" assertion cannot
  trip: the coat dataset contains no `bundle_manifest.json` (its file listing
  begins at `cnx_...`, and any `b...` name would sort first).
- Confirmed `raptor-knee-widedense` is correctly dropped: all four widedense
  mentions in V9 are comments/URLs, and only four `raptor_ft_*.pt` files are
  referenced.

Its graft is **fail-open**: on error it restores the four-view bytes and records
`applied=false`. Same silent-degradation shape as the dinosaurs arm, hence the
explicit receipt check in our guard.

### v2 VALIDATED: the recovered arm runs, and it fits the 9h cap (2026-09-06, ~21:30)

`rsna-knee-dinosaurs-v2` save-run **COMPLETE**, and the log says exactly what we
needed:

```
[coat-arm] blended OUR resgated e4/e6/e8 into the public Raptor arm (alpha 0.400)
integrity check PASSED: 3 rows, no NaNs, no constant columns
COAT_ARM_STATUS=OK {'private_alpha': 0.4, 'public_raptor_alpha': 0.6, ...}
```

Both hash gates passed on the public copies, the pinned wheel installed, the
subprocess ran, the blend applied, and `_coat_raptor_blend_receipt.json` was
written. **The arm we were silently missing now runs end to end.**

**Runtime budget (the one real risk with v2), from the receipt's own numbers:**

| stage | measured | extrapolated to 1,322 studies |
|---|---|---|
| DINOv3 / 20-member package | ~2s/study | ~0.7h |
| 4 raptor arms | ~10.5s/study total (each arm re-decodes DICOM) | ~3.9h |
| coat arm | 4.7s/study/shard, **2 T4 shards in parallel** | ~0.9h |
| fixed overhead (loads, pip) | ~80s | ~0.1h |
| **total** | | **~5.6h vs the 9h cap** |

The coat arm is cheaper than feared because its runtime contract is
`..._t4x2_v2`: it shards across both T4s (shard 0 took 1 study, shard 1 took 2,
17.6s and 17.9s wall including ~8s of model load each). So it costs under an
hour, not the ~3h a naive single-GPU extrapolation suggested.

**Caveat, stated plainly**: this extrapolates from the 3-study placeholder test
set, so it is an estimate, not a measurement. The real gate is the pending
`rsna-knee-dinosaurs` submission -- identical pipeline minus the coat arm. If it
scores, v2 is the same run plus ~0.9h, and the margin holds.

### RESOLVED: RadImageNet licence -- it is MIT, no conflict (2026-09-06)

Checked three independent sources; all agree the **pretrained weights** are MIT:

| source | licence | copyright |
|---|---|---|
| official GitHub `BMEII-AI/RadImageNet` LICENSE | **MIT** | Copyright (c) 2021 BMEII-AI |
| HuggingFace `Lab-Rasool/RadImageNet` | **MIT** | Copyright (c) 2021 BMEII-AI |
| `shigechan/radimagenet-resnet50-official` description | MIT, citing the above | -- |

The identical copyright line in both official sources shows the HF mirror derives
from the authors' own release, not a third party relabelling it.

**The distinction that resolves the confusion**: RadImageNet has TWO things under
different terms.
- The **dataset** (1.35M images) is access-controlled; radimagenet.com states it
  "can be licensed for commercial use", i.e. commercial use of the IMAGES needs an
  agreement.
- The **pretrained weights** are released separately under MIT.
**We use only the weights, never the dataset.**

**Conclusion: no ShareAlike problem and no NonCommercial problem.** MIT is
permissive and compatible with granting the CC-BY-NC 4.0 the competition requires
from a winner. The `marwanmath/resnet-50-radimagenet-marwan` tag of
CC-BY-NC-SA-4.0 appears to be that uploader's own mistaken selection; an
uploader's Kaggle tag does not override the licence the copyright holder actually
granted, and we can rely on the upstream MIT grant directly.

**REMAINING OBLIGATION (small but real)**: MIT requires the copyright notice
accompany distributions. If we place and publish a solution, it MUST include the
MIT licence text and the "Copyright (c) 2021 BMEII-AI" notice. Add this to any
writeup checklist.

### Compliance status overall (audited 2026-09-06)
Clean: forking public Apache-2.0 notebooks; public datasets meeting "publicly
available and equally accessible at no cost"; rejecting two leaked label datasets
that bitwise-matched gold; declining the second-account workaround twice; repo
kept private during the competition.
Open: (1) RadImageNet licence, above. (2) The bee notebook depends on a PRIVATE
dataset (`prvsiyan/bees-knees-rsna-public0033-weights-v1`, 403/unlisted) -- we did
NOT access it, we used the byte-identical public copy in renta0426's bundle and
left the author's SHA256 pin active to prove it. (3) If we place, attribution
obligations are substantial since nearly all the work is other people's.

### Budget
GPU ~13.5h remaining this week (30h/week, refreshes Saturdays). TPU 20h entirely
unused. ~6 weeks to the deadline.

### The single most important process lesson from this session
**Checking what the ecosystem has published beats optimising what you already
hold.** Most of the day went into careful optimisation of one forked snapshot,
worth at most +0.005, ending in a premature declaration that the space was
"exhausted". Two rounds of simply looking at recently-published kernels then found
the raptor families and the 0.937 notebook -- worth far more than every modelling
experiment combined. Look outward first, and re-check periodically, because in a
live competition the ecosystem moves under you.


## RESOLVED: GPU quota reset on schedule (2026-09-05T00:00:00)
The prior blocker (weekly 30h GPU quota exhausted 2026-08-31) is over. Confirmed
by successfully pushing `rsna-knee-base-multislice-fresh` today.

## RESOLVED: multi-slice's 0.5 score was a real bug, NOT quota exhaustion
Resubmitted the ORIGINAL `rsna-knee-base-multislice` kernel (unchanged) after
quota reset specifically to test this. **It scored 0.500 again** -- with quota
confirmed available, this rules out the quota hypothesis. Found and fixed the
real bug: `infer.py`'s `run_inference` never passed `cache_images=False` to its
internally-constructed `StudyDataset`, so it always used the K=1-calibrated
default cache size (1500 studies) regardless of `slices_per_slot`. At K=3, each
study's cached array is ~3x bigger (~7.75MB vs ~2.7MB), and with ~1,300 real
test studies that cache could balloon to ~10.5GB -- a real OOM risk that K=1
never exposed (and that the 871-study holdout, evaluated in the SAME kernel run,
apparently didn't hit either, only the larger real test set). Caching is also
pure downside during inference regardless of K or scale: it exists to avoid
re-decoding across multiple training epochs, but inference makes exactly one
pass per study, so it can never pay off, only cost memory. Fixed by passing
`cache_images=False` unconditionally in `run_inference`. Added a regression
test (`test_run_inference_disables_image_cache`), full suite 103/103 passing,
synced to `rsna-knee-src` (remember `--dir-mode zip`).

**Important consequence**: `rsna-knee-base-multislice-fresh` (currently
running, pushed BEFORE this fix) has the OLD buggy code baked into its
`rsna-knee-src` snapshot from when it was pushed -- syncing the fix now does
NOT retroactively help an already-running kernel. Its tier1/tier2 holdout
numbers will still be valid and useful (holdout apparently doesn't trigger
this bug, at least not at 871 studies), but **do not trust or submit its
actual test-set submission.csv without first re-pushing with the fix and
re-running**, even if tier1 looks great. Treat this run as informative for
the "does multi-slice help" question, not as a candidate to submit as-is.

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

   **Pre-flight check done while blocked (no GPU cost)**: re-verified the
   queued notebook during the wait -- correctly isolates K=3 as the only
   variable vs v2's own recipe, `rsna-knee-src` still fully in sync with
   local code (no drift since last push), full test suite 102/102 passing.
   No issues found; genuinely ready to fire the moment quota resets.

   **New candidate (item 7?) found while blocked, NOT started -- analysis
   only, zero GPU cost**: re-analyzed the already-downloaded
   `train_series_with_slots.csv` (from the slot-coverage diagnostic, no new
   kernel needed) to characterize the 5,025 series that never match any of
   our 6 slots. Found a clean, specific gap: **1,111 coronal
   fluid-sensitive series have no fat-suppression and therefore no slot at
   all**, even though the exact same combination already has a working slot
   for sagittal (`SAG_FLUID_NOFS`) -- this isn't a new architectural
   pattern, just filling in an inconsistently-applied one. Axial is even
   less developed: no T1 slot (572 series unused) and no non-fat-sat fluid
   slot (488 series unused) exist for that plane at all, only AX_FLUID_FS.

   Architecturally this is a cleaner change than the failed COR_T1
   fallback (roadmap item 5): that failure substituted one slot's content
   INTO another slot's position (mask=1.0 for two different meanings,
   likely the actual cause of that regression), whereas adding a genuine
   new slot (e.g. `COR_FLUID_NOFS`) with its own consistent meaning and its
   own attention query mirrors exactly how `SAG_FLUID_NOFS` already works
   successfully -- not the same failure mode. Still, NOT starting this now:
   don't want to add another variable while the multi-slice (K=3) test is
   still mid-flight, and it needs its own clean single-variable validation
   before trusting it, same discipline as everything else this session.
   Worth raising with the user as a candidate next item once multi-slice
   resolves.

**DEFERRAL LIFTED 2026-09-05 for RadImageNet** -- user explicitly approved
pursuing it after seeing that it is central to the top ensemble (appears at
multiple stages/slot layouts in the 0.95 cluster's pipeline; official
ResNet-50 weights published by `marwanmath`). External MRI datasets
(MRNet/OAI) remain deferred -- still ask before starting those.

## THREE CANDIDATES READY -- best reproduces a documented 0.937 (2026-09-06)

| kernel | basis | status |
|---|---|---|
| `rsna-knee-final-v1` | our fork + corrected DINOv3 blend weight | holdout-validated **0.9242** -> predicted ~0.925-0.927 |
| `rsna-knee-dinosaurs` | author's newer notebook + 3 raptor families | complete, unvalidated |
| **`rsna-knee-bee`** | **prvsiyan "bee's knees" V8, a documented 0.937** | **complete, all author audits PASSED** |

**`rsna-knee-bee` is the strongest candidate.** The author documents three
consecutive submissions (V2, V6, V7) all scoring exactly 0.937.

**It required real recovery work**: the notebook depends on
`prvsiyan/bees-knees-rsna-public0033-weights-v1`, which is **PRIVATE** (403,
unlisted), so it cannot run as published. Two fixes:
1. Its metadata omits `renta0426/rsna-knee-public0033-meniscus-bag-v1`, needed for
   the `public0033_meniscus10_bundle_v1` manifest. Attached the ORIGINAL only --
   the loader asserts exactly one matching manifest, so also attaching the
   `mattiaangeli` mirror would fail with "got 2" instead of "got 0".
2. The pinned `checkpoint_epoch30.safetensors` lives in that private dataset, but
   the identical file ships in the public renta0426 bundle. Patched the candidate
   list to glob `/kaggle/input/**` for it rather than hardcoding a mount layout,
   and **left their SHA256 pin intact**.

**The pin then CONFIRMED the public copy is byte-identical**:
`checkpoint_sha256 = e1443bb0f518418a31428cd7eb153f15af4c48c7175baf1f9d21c84f1c2a93e7`,
matching `_B3_CKPT_SHA` exactly. All their audits passed (infra0021, public0033
overlay, BEES_KNEES V1/V3/V4/V6), plus our own guard.

**Generalisable lesson**: some published "reproducible" results depend on private
assets and cannot actually be reproduced by anyone else. When a public mirror of
the same artefact exists, redirect to it but keep the author's integrity checks
running -- they then prove the substitution was sound rather than papering over it.

**Recommendation: submit all three.** Independent shots, daily cap allows it, and
the banked 0.922 is unaffected by any of them.

## TWO SUBMISSION CANDIDATES READY (2026-09-06) -- both integrity-checked

| kernel | basis | validation |
|---|---|---|
| `rsna-knee-final-v1` | our fork + corrected DINOv3 blend weight (50/50 rank) | **holdout-validated 0.9242** -> predicted real ~0.925-0.927 |
| `rsna-knee-dinosaurs` | the author's NEWER notebook + 3 raptor model families | **UNVALIDATED** -- see below |

Both passed the integrity guard (no NaNs, no constant columns), which exists
because two earlier submissions scored exactly 0.500 from a silent fallback to the
all-0.5 file.

**Recommendation: submit BOTH.** They are independent shots, the daily cap allows
it, and the banked 0.922 is unaffected by a bad result. Then select the final two
by what actually scored.

## IMPORTANT CORRECTION: the optimisation space was NOT exhausted

Earlier in this file I recorded that the optimisation space was closed. **That was
wrong in an important way.** It was closed for *the notebook we forked*, while the
public ecosystem kept adding model families. The "24 members buy only +0.0016"
finding was about more of the SAME lineage, and I over-generalised it into
"ensembling is saturated".

Searching recently-updated public kernels found the same author has since
published `mattiaangeli/bend-the-knee-to-the-dinosaurs`, attaching three
`dreaddevelopment/raptor-knee-*` families (maxspan, native384, native384dense)
that did not exist in our version, and crediting a notebook titled 0.937. **Check
what has been newly published BEFORE concluding a space is exhausted** -- in a
live competition the ecosystem moves, and this should have been checked far
earlier than hour twelve.

**Why the newer notebook could not be holdout-validated**: one of its raptor
stages executes embedded source via `exec(compile(_KE_SRC, ...))` which resolves
its own data paths -- effectively a THIRD root beyond ROOT and COMP -- so our
fake-root redirect could not reach it, and it failed asserting 'study identity
drift'. Unpicking generated code inside a string literal was judged not worth it
against simply running it on the real test set, where the downside is bounded.

## LABEL-CEILING HYPOTHESIS IS LIKELY WRONG (2026-09-06) -- models denoise

Free test, no GPU. `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` contains
`v52_e11_oof.csv`: **out-of-fold predictions for all 4,407 studies**, with `fold`
and `is_gold` columns. OOF predictions are clean by construction, which finally
allowed the gold evaluation at **n=58** instead of the n=11 we were stuck with.

| measured out-of-fold | macro AUC |
|---|---|
| OOF model vs SILVER (n=4407) | 0.8401 |
| **OOF model vs GOLD (n=58)** | **0.8576** |
| SILVER labels themselves vs GOLD (n=58) | 0.8927 |

**The model scores HIGHER against gold (0.8576) than against the silver labels it
trained on (0.8401), a lift of +0.018.** That is genuine denoising: unbiased label
noise averages out, so a model can recover true signal better than its teacher.

**This contradicts the hypothesis recorded earlier in this file that silver labels
cap the competition near 0.89.** They do not -- models exceed their teacher's
agreement with truth. Do not plan around a label ceiling.

If the same silver->gold lift held for our strong blend (0.9242 vs silver), it
would correspond to roughly 0.94 against gold-quality labels, i.e. the governing
target, and would mean the public leaderboard reflects genuine capability rather
than inflation.

**Caveats, which are substantial**: n=58 is small, and extrapolating a +0.018
lift measured on a WEAK model (0.84) to a strong one (0.92) is unsafe -- denoising
headroom generally shrinks as models improve. Suggestive, not established.

## ALSO TESTED: a different model FAMILY adds nothing either

Blending the RadImageNet E11 OOF (0.8422 alone, full 881-study coverage) into our
50/50 blend: every weight from 0.05 to 0.40 DECREASED the score (0.9232 down to
0.9056). So even a genuinely decorrelated family -- RadImageNet rather than
DINOv2/DINOv3 -- does not help. The ensemble is saturated, which strengthens the
conclusion that the optimisation space is closed.

## OPTIMISATION SPACE EXHAUSTED (2026-09-06). Submission kernel is at the optimum.

Two further free offline searches, both on the 881 clean studies:

**Blend weight sweep** -- `full24*(1-w) + dinov3*w`: 0.9194 at w=0, rising to
**0.9242 across a broad plateau at w=0.4-0.6**, falling to 0.9178 at w=1.0. The
plateau matters: our 50/50 choice is robust, not knife-edge. A three-way mix with
`e10_v2` reaches 0.9245, only +0.0003, inside selection noise and not worth the
complexity.

**Per-label variant selection** -- our 50/50 blend WINS OUTRIGHT on 6 of 12
labels (ACL, MCL, Medial Meniscus, Medial OA, Lateral OA, Fracture) and on macro.
A per-label best-of-7 oracle reaches 0.9253, just +0.0010, and with 12 labels x 7
candidates selected on the scoring data itself that will shrink to roughly zero
out of sample. Not worth pursuing.

**Every lever, measured honestly:**

| lever | value |
|---|---|
| our own model (3 independent tests) | 0.000 |
| ensemble breadth, 1 -> 24 members + RadImageNet | +0.0016 |
| **blend weight correction** | **+0.0048 -- CAPTURED in submission_final_v1** |
| per-label blending of our model | 0.000 |
| per-label variant selection | +0.0010 oracle, ~0 real |
| combining silver label sources (earlier) | negative |

**Conclusion: `submission_final_v1` is at the practical optimum of this
pipeline.** Clean holdout 0.9242, predicted real ~0.925-0.927 using the observed
+0.003 offset. Further gain requires something qualitatively different, not more
search over this space -- do not spend GPU re-searching blends, weights, members
or per-label schemes. They are all measured and closed.

**Remaining honest gap**: rank 10 is 0.950, we project ~0.927 -- about 0.023.
Best available reading is that a substantial part of that is accumulated
public-set fitting by teams tweaking against the public 30% (author-documented
behaviour; consistent with 24 members buying only +0.0016 and with our verbatim
run of their code scoring 0.922). If so it should not transfer to the private
70%. **This remains a hypothesis, not a measurement** -- we cannot see what the
0.950 teams actually run.

## FOUND A REAL GAIN: upweight their DINOv3 member (2026-09-06)

Free offline search over cached holdout predictions (zero GPU), on the 881 clean
studies:

| | clean holdout |
|---|---|
| best single variant (`e10_v2`) | 0.9202 |
| `full24` alone | 0.9194 |
| `dinov3` alone | 0.9178 |
| **`full24` + `dinov3`, equal-weight rank blend** | **0.9242 (+0.0041)** |
| `e10_v2` + `dinov3` | 0.9242 |
| `full24` + `e10_v2` + `dinov3` | 0.9242 |

**Robust, not a fluke: three different combinations all land at 0.9242**, so this
is the underlying "weight DINOv3 higher" effect rather than one lucky pairing.

**Interpretation:** their pipeline ALREADY blends the DINOv3 member in, at
`A5_W = 0.45` (set at the end of their cell 24). Re-blending 50/50 raises the
effective DINOv3 weight to roughly 0.72. **Their chosen weight is suboptimal on
unseen data** -- precisely the kind of error honest validation catches and
public-LB tuning does not, since a 0.004 move is inside the noise band they
describe chasing.

Predicted real score using the observed holdout->real offset of +0.003: ~0.927.
**Honest caveat**: this was best-of-50 selected on the same holdout, so expect
shrinkage; the true gain is more likely +0.002-0.003 than +0.0041.

**Actionable**: build a submission kernel that runs their pipeline on the real
test set and blends `submission.csv` with the DINOv3 predictions by rank, rather
than accepting their A5_W=0.45. Alternatively sweep A5_W directly and pick on the
clean holdout.

## SCORE JUMPED TO 0.922 -- and our validation predicted it within 0.003

Submission 56047453 (the verbatim community fork's `submission.csv`) scored
**0.922** on the public leaderboard, up from our previous best of 0.812.

**Our clean-holdout measurement for that exact file was 0.9194. It scored 0.922.
Accurate to 0.003.** This substantially resolves the comparability worry recorded
earlier: despite scoring silver labels on training studies rather than true labels
on test studies, the 881-study clean holdout is empirically a reliable proxy for
real test performance. **Trust it to rank candidates before spending submissions.**
By that mapping, `e10_v2` (0.9202 clean) should score ~0.923.

## THE 0.028 GAP IS NOW THE WHOLE GAME (open question, NOT diagnosed)

Rank 10 is 0.950. We are at 0.922.

**We ran their published notebook verbatim, with the same attached data sources,
and got 0.922. The author and ~17 forks score 0.949-0.954.** That 0.028 is an
order of magnitude larger than the 0.001-0.003 fork-race movements the author
himself describes, so something substantive differs between running their
published code and their reported result.

Unverified candidate explanations (do NOT treat any as established):
1. Published-vs-submitted mismatch -- his leaderboard entry may come from a
   version other than the one currently published. His own cell-0 warning
   ("use an earlier version as a baseline") hints the published one is the
   fork-race-tweaked one.
2. **A missing data source.** Their kernel metadata listed 13 dataset_sources and
   the FIRST WAS AN EMPTY STRING, which we filtered out when building our fork's
   metadata. If that was a real dataset mangled in the API response, we may be
   running without a whole member set. **This is the cheapest lead to check.**
3. Accumulated public-set fitting across many submissions by those teams.

**Why this matters more than any modelling work**: the gap is a discrepancy
between their code and their result, which is investigable, rather than a
capability gap requiring us to out-engineer anyone. Every modelling lever we
tested is worth <=0.002; this is worth 0.028.

## PER-LABEL BLEND: ALSO NIL (2026-09-06). Our contribution path is CLOSED.

Free offline search (all prediction files cached locally, zero GPU): per-label
blend weights of our clean member into `e10_v2`, the strongest ensemble variant.

Result: **1 of 12 labels improves, by +0.0003** (Synovitis, w=0.08). Macro
unchanged at 0.9202. And those weights were fitted ON the same holdout used to
score them, so even that is an optimistic upper bound.

**That is three independent tests, all negative:**
1. global rank blending -- monotonically harmful at every weight
2. clean retrain on their split -- our member 0.7735 vs their 0.9178
3. per-label blending -- +0.0003 on one label, fitted optimistically

**STOP TESTING THIS.** Our own modelling contributes nothing to this ensemble. Do
not train another member, do not try another blending scheme, do not revisit.

## WHERE THINGS ACTUALLY STAND (honest summary, 2026-09-06)

Everything measured on 881 studies no scored model trained on:

| | clean macro AUC |
|---|---|
| our best clean member | 0.7735 |
| their DINOv3 component | 0.9178 |
| full 24-member ensemble (default submission.csv) | 0.9194 |
| **best variant: `submission_e10_v2.csv`** | **0.9202** |

Levers tried and their honest value:
- our own member, any blending scheme: **0.000**
- ensemble breadth, 1 component -> 24 members + RadImageNet: **+0.0016**
- blend weight tuning: **0.000**
- combining silver label sources (tested earlier): **negative**

**The only free win found: `e10_v2` beats the default `submission.csv` by
+0.0008 on clean data.** Small, but it was found by honest validation rather than
public-LB guessing, which is the one discipline that has actually paid here.

**Untested and plausibly the real ceiling: LABEL QUALITY.** Every model in this
competition, theirs and ours, trains on silver labels that agree with gold at only
~0.8932. Their DINOv3 member scores 0.9178 vs silver but only ~0.8931 vs gold
(n=11, far too noisy to rely on). If the true bound is the label source rather
than the architecture, then no amount of ensembling moves it -- which is
consistent with 24 members buying +0.0016. Nobody in the fork-race is working on
this. A clean test would train holding out ALL 58 gold studies and evaluate
against them (n=58 rather than n=11), which is one ~3h run.

**Operational risk that deserves real attention:** two of our submissions scored
0.500 because the pipeline silently fell back to the all-0.5 file. Whatever we
submit finally MUST be verified to actually produce real predictions on the true
test set. That is worth more than any remaining marginal modelling gain.

## FULL ENSEMBLE MEASURED ON CLEAN DATA (2026-09-06) -- +0.0016 over one component

`rsna-knee-full-eval` v2. Redirected BOTH of their pipeline roots (ROOT for the
24-member stage, COMP for the DINOv3 stage) at a fake competition directory
holding only the 881 fold-0 studies as symlinks, so every stage scored the same
unseen studies with no code patching.

| on the clean 881-study holdout, vs silver | macro AUC |
|---|---|
| our best clean member | 0.7735 |
| their DINOv3 component alone | 0.9178 |
| **full 24-member ensemble + RadImageNet** | **0.9194** |
| best variant (`submission_e10_v2.csv`) | **0.9202** |
| legacy fold blend | 0.8971 |
| native v38 | 0.8913 |
| public_0899 variant | 0.8967 |

**FINDING: the full ensemble buys +0.0016 over its DINOv3 component alone.**
Twenty additional members plus two pinned RadImageNet stages are worth about
0.002 on unseen data. The entire fork-race is optimising a layer that contributes
almost nothing measurable off the public set.

**CAVEAT -- DO NOT SKIP THIS.** These numbers are NOT comparable to the public
leaderboard's 0.949, and it is wrong to conclude "the public LB is inflated by
0.03" from them. The leaderboard scores against the competition's REAL labels on
TEST studies; this scores against steven's SILVER labels on TRAINING studies.
Silver agrees with gold only at ~0.8932, so "agreement with silver" is a
different quantity from "agreement with truth", and models trained on silver may
share its biases. Whether anyone reaches 0.94 against the true test labels is
still unmeasured and cannot be settled from this holdout.

(Recording this restraint deliberately: three times today I treated a single
measurement as settling a broader question and had to retract -- the leakage
theory, the multi-slice verdict, and the overfitting thesis. This measurement is
real and useful; its scope is narrow.)

**What it does establish, and what follows:**
1. Ensemble breadth is not the lever. Going from 1 component to 24 members gains
   ~0.002. Do not spend GPU chasing more members or blend-weight tweaks.
2. `submission_e10_v2.csv` (0.9202) is the strongest variant on unseen data and a
   better candidate than the default `submission.csv` (0.9194).
3. The remaining honest gap to any higher target is not closable by the levers
   tried so far: our member (worse, and harmful in blend), more members (+0.002),
   or blend weights.

## OUR MEMBER ADDS NOTHING -- DEFINITIVE (2026-09-06). Do not train another.

Trained DINOv2-base on the community split (groups 1-4) so it never saw the 881
fold-0 studies, then blended offline against their DINOv3 five-fold predictions.

| | clean-holdout macro AUC |
|---|---|
| theirs alone (DINOv3 5-fold) | **0.9178** |
| ours alone (clean) | 0.7735 |
| blend ours@0.02 | 0.9175 |
| blend ours@0.05 | 0.9168 |
| blend ours@0.10 | 0.9151 |
| blend ours@0.30 | 0.9010 |

**Every blend weight is WORSE than theirs alone, monotonically decreasing.** The
earlier +0.0063 "gain" at w=0.30 was entirely a memorisation artifact: base v2 had
trained on 706 of those 881 studies, and removing that contamination dropped our
solo score from 0.8854 to 0.7735 (a 0.111 inflation) and erased the blend benefit
completely.

**CONSEQUENCES:**
1. **Do not train another member of ours.** The 12-15h plan is cancelled. This
   3h run existed precisely to gate that decision and it returned a clear no.
2. Our own modelling is not a path to 0.94. Six weeks of our experiments produced
   0.7735 clean; their single member is 0.9178.
3. I recommended that training run on the strength of the contaminated number.
   That recommendation was wrong even though I had flagged the number as
   unreliable -- flagging a caveat is not the same as acting on it.

**WHERE 0.94 MIGHT ACTUALLY LIVE -- the key open question:**
0.9178 is only their **DINOv3 component**. The deployed pipeline scoring ~0.95
publicly blends **24 members plus two RadImageNet stages**, and that full ensemble
has NEVER been measured on clean data. That is now the most important unanswered
question in this project.

Obstacle (known, already broke two attempts): their full pipeline's blending
stages read from an earlier stage that still holds test-set study IDs, so
redirecting only the DINOv3 stage is not enough. All stages must be redirected.
Assets that make this cheaper: `their_holdout_predictions.csv` and
`their_fold{0..4}_predictions.csv` are now saved locally, so any blending maths
over the DINOv3 members is free and offline from here on.

## CONTAMINATION ALARM RESOLVED: 0.9178 IS HONEST (2026-09-06)

Per-fold scores of the five public DINOv3 models, each scored alone on the 881
fold-0 studies:

| model idx | stored fold | macro AUC |
|---|---|---|
| 0 | 0 | 0.8963 |
| 1 | 1 | 0.8841 |
| 2 | 2 | 0.8952 |
| 3 | 3 | **0.8669** (lowest) |
| 4 | 4 | **0.9125** (highest) |

**The automated verdict in that cell was MISCODED and said "CONTAMINATED". It is
wrong -- ignore it.** It only tested whether the spread exceeded 0.02, which is
too crude. The correct test is whether the fold-0 model is SYSTEMATICALLY LOWEST,
because if folds 1-4 had memorised these studies they would all score above the
clean fold-0 model.

They do not. Fold 0 ranks THIRD of five, with folds 1 and 3 scoring below it.
That is ordinary model-to-model variance, not a memorisation signature.

This fits their training code exactly: it **hardcodes `va = grp == 0`** as the
holdout, so all five variants most likely held out group 0 and "fold" is a
seed/variant index rather than a rotating holdout. All five are clean here.

**CONCLUSION: 0.9178 is an honest, uncontaminated measurement of the public
DINOv3 member on unseen studies.** The five-model mean exceeding every individual
member (0.867-0.913) is exactly the variance reduction genuine ensembling
produces. The earlier alarm was worth raising and worth checking, but it was
unfounded.

Consequences:
- The strategic conclusions drawn from 0.9178 stand: the cluster's strength is
  real generalisation, not public-LB overfitting.
- The blend number 0.9241 rests on an honest baseline. What remains confounded is
  OUR side of it (base v2 trained on 706 of the 881), so the +0.0063 is still an
  upper bound pending a clean retrain on their split.
- Distance to the 0.94 governing target from the honest blend: about 0.016.

## BLEND TEST RESULT (2026-09-06): our member helps, but the number is confounded

`rsna-knee-blend-test` v3, all on the 881-study clean holdout, vs silver:

| | macro AUC |
|---|---|
| theirs alone (DINOv3 5-fold) | 0.9178 |
| ours alone (base v2) | 0.8854 **contaminated** |
| ours @ 0.05 / 0.10 / 0.15 | 0.9198 / 0.9215 / 0.9227 |
| **ours @ 0.30 (best)** | **0.9241  (+0.0063)** |
| ours @ 0.40 / 0.50 | 0.9230 / 0.9204 |

Smooth single-peaked curve at w=0.30, which is what a genuine decorrelation
benefit looks like rather than noise.

**DO NOT QUOTE +0.0063 AS A CLEAN RESULT.** Base v2 trained on 706 of these 881
studies (our split differs from theirs), so its 0.8854 solo score is inflated by
memorisation -- and therefore part of its blend contribution is memorisation too,
not decorrelation. On genuinely unseen data our member is weaker and the gain may
shrink toward zero or reverse. The contamination was deliberately built into the
test and flagged up front; it biases TOWARD finding value, so this is an upper
bound on the benefit, not a measurement of it.

Also note: 0.9241 is still **below the 0.94 governing target**, even inflated.
And the 0.9178 baseline is itself pending the per-fold contamination check.

**What this justifies**: training our own member on THEIR split (groups 1-4,
never touching the 881). That converts a suggestive result into an honest one and
yields a member legitimately composable with theirs. ~12-15h GPU, now warranted.

Timing datum: our K=1 inference over 881 studies took 21.8 min (theirs took ~12).

## GOVERNING TARGET (user directive, 2026-09-06): >=0.94 on the CLEAN holdout

User correction, verbatim in intent: **"Dont optimise for ranking, optimise to be
able to get a 0.94+ score on the remaining 70% of the test data."**

Adopt this as the success criterion for everything downstream. The distinction is
substantive, not semantic:
- Optimising RANK means partly benefiting from other teams overfitting the public
  30%. That is outside our control and is not a plan.
- Optimising SCORE ON UNSEEN DATA is an engineering target we can measure directly
  and drive.

**Operational form: >=0.94 macro AUC on the 881-study clean holdout** (their fold
0, md5(report)%5==0), honestly measured -- i.e. scored with models that never
trained on those studies. That number is a direct prediction of private-set
performance rather than a leaderboard artifact.

Consequences to hold to:
1. Every candidate is measured on the clean holdout BEFORE it is submitted.
2. The public LB is a sanity check that clean-holdout numbers track reality --
   never a target. Do not chase 0.001-0.003 public movements; that is exactly the
   documented trap the rest of the cluster is in.
3. **This makes the per-fold contamination test the most important measurement in
   flight.** If the public five-model mean (0.9178) is inflated because four of
   five members trained on these studies, then their GENUINE unseen performance
   may be well below 0.94 -- meaning possibly nobody in the visible cluster is
   truly at 0.94 on unseen data, and the target is more reachable than the public
   leaderboard suggests.
4. Do not report a number as evidence of progress unless it was measured on data
   the scored model never trained on.

## DECISIVE MEASUREMENT (2026-09-06): the cluster's strength is REAL, not overfitting

Scored the public DINOv3 5-fold member on **881 studies in their own fold-0
holdout that those weights never trained on** (`rsna-knee-community-eval` v4).

| | our base v2 | public DINOv3 member |
|---|---|---|
| tier1 (vs silver, held out) | 0.7931 | **0.9178** |
| tier2 (vs gold) | 0.7774 | 0.8931 |

**A SINGLE public member beats our entire pipeline by 0.125 on honest,
uncontaminated validation.**

**This OVERTURNS the hypothesis repeated throughout this session** -- that much of
the 0.949 cluster was public-LB overfitting that would shuffle on the private
70%. It is not. No leaderboard feedback could tune performance on studies that
were never in training. Their members generalise genuinely. The author's
overfitting warning applies to blend-weight tweaking BETWEEN forks (movements of
0.001-0.003), not to underlying member quality. **Stop repeating the
"cluster will collapse on private" theory -- it has now been measured and is
false.**

Caveat on tier2: n=881 is the study count, but gold labels exist for only ~11 of
them, so each label's gold AUC rests on ~11 samples (hence several 1.000s).
Do not draw conclusions from that row. tier1 on 881 studies is the solid number.

**Strategic consequences -- these matter more than the number itself:**
1. The gap is a CAPABILITY gap, not a tuning gap. Six weeks of our own
   experiments produced 0.7931; one public member is at 0.9178. We will not
   out-engineer that from scratch before 2026-10-22.
2. The realistic path to top 10 runs through USING these members -- which the
   notebook's own author explicitly endorses ("use the public DINOv3 member from
   knee-mri-fold-weights and validate it locally; it was selected on held-out
   labelled studies"). That is exactly what this run did.
3. **Downgrade the "honest validation is our edge" thesis.** It was premised on
   the field flying blind on inflated numbers. The members are genuinely strong,
   so clean validation is useful for CHOOSING between candidates, not a
   substitute for capability. Do not oversell it again.
4. Our own contribution should now aim at DIVERSITY (a member decorrelated from
   theirs) rather than at beating them outright, plus disciplined final
   selection.

## COMMUNITY BASELINE FORK RUNS CLEANLY IN OUR ACCOUNT (2026-09-05)

`aarohnanoti/rsna-knee-community-baseline` (verbatim fork, all 12 public dataset
sources reproduced) pushed and COMPLETED with no errors. It ran fast only
because dev-time `test.csv` is the 3-row placeholder; real cost lands at scoring.

What loaded and verified:
- **24 ensemble members** banked: 20 from `pilkwang/rsna-knee-weights` + 4 legacy
  folds (accepted at reduced weight, they carry no stored fingerprint).
- **All 5 DINOv3 folds** loaded: `vit_small_patch16_dinov3.lvd1689m pool=xcodex`.
- Two decode groups: 336px x 12 slices @130mm crop (20 members); 224px x 9
  slices @160mm (4 legacy members).
- Outputs: `submission.csv` (24-member weighted rank mean),
  `submission_public_0899.csv` ("exact no-jitter public-frontier rank mean of 20
  members" -- the 0899 in the name implies a 0.899 public score for that
  variant), plus e10/native/legacy-blend variants and a repro receipt.

**Engineering worth stealing: they FINGERPRINT every member.** Each member's
weights are checked against a stored fingerprint on a seeded input ("fingerprint
matches within 2.8e-06") before it is allowed into the blend, so a member can
never silently run under mismatched preprocessing. Our own PLAN.md called for
exactly this ("each member should carry its own preprocessing fingerprint") and
we never built it -- and the 0.5 submission is precisely the failure mode it
prevents. **Build this before trusting any multi-member blend of ours.**

**License trail partly closed**: the receipt's `encoder_sha256` is
`08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`, byte-identical
to the SHA256 in `shigechan/radimagenet-resnet50-official`'s description. So the
deployed RadImageNet encoder is the pinned `Lab-Rasool/RadImageNet` revision with
real provenance, not an unidentified copy.

**Why this matters**: their machinery now runs in our account with every member
loaded and verified. Redirecting it at the 881-study clean holdout is now a small,
contained change -- exactly the "surgical, not a rewrite" path, and the
prerequisite for the honest validation that actually differentiates us.

## THEIR PREPROCESSING IS SIMPLER THAN OURS (2026-09-05) -- uncomfortable finding

Read their `build_study` (notebook cell 22). Side by side:

| | theirs | ours |
|---|---|---|
| slot definition | `Anatomical_Plane` x `Fat_Suppression` straight from train_series.csv | weighting DERIVED from DICOM headers (TR/TE -> T1/T2/PD), 6 canonical slots |
| slice ordering | `InstanceNumber` | geometry: ImageOrientationPatient x ImagePositionPatient projected on the slice normal |
| laterality | sign of `ImagePositionPatient[0]`, flip non-sagittal | full patient-coordinate image-centre computation |
| intensity | percentile [1,99] on a 4x-subsampled crop | percentile over an n-slice sample |
| crop | centre crop of CROP_MM at pixel spacing, cv2 resize | Nyquist-argued physical crop + resize |

**We built the sophisticated version of nearly every step, and the pipeline that
is actually in the top cluster does the simple thing.** Notably we adopted
geometry-based ordering specifically because the original pilkwang baseline
argued SOP-UID/filename order is unreliable -- and they just use InstanceNumber.

Two readings, and we cannot yet tell which is right: either (a) our extra
sophistication is not buying anything, and the engineering went somewhere that
does not matter, or (b) they are leaving something on the table and our
preprocessing is a genuine (if unproven) edge. Worth keeping in mind before
attributing any future gap to preprocessing. It also means their pipeline is
CHEAPER to run than ours per study, which partly explains how they afford
n_slice=16.

Their cell 22 already falls back to `train_series` when `test_series` is absent,
which makes pointing their pipeline at holdout studies a small change rather
than a rewrite -- relevant for evaluating their members honestly.

## PRACTICAL ASSESSMENT: run their notebook, don't reimplement it (2026-09-05)

Their readout IS available: notebook cell 23 defines `TokenXAttnPool`, which
matches the checkpoint keys exactly (`q (12,384)`, `kv_norm`, MultiheadAttention
`in_proj_weight (1152,384)`, `slot_emb` with N_SLOT_TYPES+1=7). Cell 23 also has
`MeanMaxPool`, `LabelAttentionPool`, and a `segment_softmax` over a ragged
(sidx) layout.

**Worth noting**: the `SliceCrossAttentionHead` written here independently from
the state_dict shapes came out nearly identical to their `TokenXAttnPool` -- same
LayerNorm on keys/values, twelve learned per-finding queries, the same
`nn.MultiheadAttention` with dropout, a slot-type embedding. Converging on their
design without having read it first is decent evidence the reasoning was sound.
Theirs has extra machinery ours lacks (`readout.pool.dw (12,384)`, `db (12,)`,
`gate (12,)` -- some per-label delta/gating mechanism) and uses a ragged segment
layout rather than a dense (B,S,K) one.

**DECISION -- do NOT reimplement their inference pipeline from scratch.**
Faithfully reproducing it would require: TokenXAttnPool plus the dw/db/gate
parts, the `enc.vit`/`enc.tok` wrapper, timm 1.0.22 with
`vit_small_patch16_dinov3.lvd1689m`, and their exact preprocessing (n_slice=16,
stem=native, pe_init=tiled, img=336, their slot assignment, their slice
ordering, uint8/255 scaling). That is a large surface area where **a silent
mismatch produces plausible-but-wrong numbers** -- precisely the failure mode
that just cost us a 0.5 submission. Forking and RUNNING their notebook is what
forking is for; reimplementing it is all of the risk for none of the benefit.

Our added value therefore comes from the three things that are genuinely ours:
1. our own trained member (now with the corrected cross-attention aggregation),
2. honest out-of-fold validation on their clean group-0 split (which the rest of
   the field is not doing -- they validate against the public LB),
3. final-submission selection discipline driven by (2) rather than public score.

## PUBLIC DINOv3 MEMBER DISSECTED (2026-09-05) -- and it CORRECTS our item-6 verdict

Downloaded `mattiaangeli/knee-mri-fold-weights` (CC0) and inspected the
checkpoints directly. Each `m_f{k}.pt` = {state_dict, cfg, fold}, and **the fold
index IS stored** (m_f0->0 ... m_f4->4), confirming the split mapping.

**Their config (verbatim):** backbone `vit_small_patch16_dinov3.lvd1689m` (DINOv3
ViT-S/16 via **timm**, not HuggingFace -- the dataset bundles a timm-1.0.22
wheel), `img: 336` (same resolution as ours), **`n_slice: 16`**, `pool: xcodex`,
`cond: token`, `stem: native`, `pe_init: tiled`, `n_sites: 109`, labels exactly
our 12 in our order.

**Their readout architecture** (from state_dict shapes):
- `enc.tok.weight (7, 384)` -- 7 learned type tokens added at the encoder
- `readout.pool.q (12, 384)` -- twelve learned queries, one per finding
- `readout.pool.attn.in_proj_weight (1152, 384)` -- a real `nn.MultiheadAttention`
  (1152 = 3x384 for q/k/v), i.e. genuine cross-attention over all tokens
- `readout.pres_emb.weight (7, 64)` -- presence embedding
- `readout.pool.base.0 (832,)` -> `base.2 (12, 832)`; 832 = 384 + 384 + 64
  (query output + pooled CLS + presence embedding) -> classifier

**THIS CORRECTS OUR ITEM-6 CONCLUSION.** We closed item 6 as "multi-slice does
not help" based on a K=3 test -- but our implementation **mean-pools over the K
dimension** before the slot attention runs (`KneeMultiLabelModel.forward`:
`.view(b, s, k, -1).mean(dim=2)`). Averaging slices destroys exactly the
information extra slices provide. They use SIXTEEN slices with **cross-attention
over all tokens**, never averaging them away. So the honest verdict is narrower:
**"multi-slice with mean-pooling aggregation does not help"** -- the aggregation
was very likely the flaw, not the premise. Item 6 should be considered REOPENED
under a proper attention-based aggregation, not closed.

This is now the most promising model-side lever we have identified, and unlike
the six failed attempts it is motivated by a working reference implementation
rather than a guess.

## ITEM 6 (multi-slice) CLOSED -- no clear benefit (2026-09-05)

`rsna-knee-base-multislice-fresh` finished: K=3, 6 epochs, FRESH from pretrained
DINOv2-base, matching base v2's recipe exactly so K was the only variable.

| | tier1 (n=871) | tier2 (n=11) |
|---|---|---|
| base v2 (K=1, 6ep) | **0.7931** | 0.7774 |
| multislice fresh (K=3, 6ep) | 0.7866 | **0.8028** |

tier1 slightly WORSE; tier2 the best ever recorded, but tier2 is n=11 -- a single
study flipping swings it hugely, so it cannot carry a decision. **This is the
SIXTH attempt to beat v2's tier1 and, like the five before it, it lands in the
0.77-0.79 band. v2's 0.7931 remains unbeaten.** Verdict: multi-slice does not
clearly help; do not spend more GPU on it as a solo-accuracy lever.

Caveat worth keeping: for ENSEMBLE purposes a member can add value by being
*different* even when not individually better, so the K=3 checkpoint may still
earn a place as a diversity member later. Just not worth further tuning now.

Cost note: ran 6h25m, not the estimated 5h45m (epochs ~3,465s vs ~3,040s
projected). Budget estimates for K>1 runs should use ~3,500s/epoch.

**GPU budget after this run: 6.42h used, 23.58h remaining, refresh
2026-09-12T00:00:00. TPU: 20.00h COMPLETELY UNUSED** -- the public notebook has a
TPU stage (raidium/curia-2), so TPU is a genuinely untapped second budget.

## ASSET LICENSE AUDIT (2026-09-05) -- resolve before shipping anything

Same discipline that caught the two leaked label datasets. Checked via
`kaggle datasets metadata <ref> -p <dir>` (note: fields are nested under an
`info` key, not top level).

| asset | Kaggle license | verdict |
|---|---|---|
| `mattiaangeli/knee-mri-fold-weights` | **CC0-1.0** | clean, public domain, no restrictions |
| `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` | **"other"**, NO description | **unverified -- do not ship without resolving** |
| `marwanmath/resnet-50-radimagenet-marwan` | **CC-BY-NC-SA-4.0** | see conflict below |
| `shigechan/radimagenet-resnet50-official` | "other", but documented | claims upstream **MIT** |

**CONFLICT TO RESOLVE -- two copies of the same RadImageNet ResNet-50 carry
contradictory license claims:**
- `marwanmath/...` is tagged **CC-BY-NC-SA-4.0** with no description.
- `shigechan/...` description states: "Original model is MIT licensed. Source:
  Lab-Rasool/RadImageNet on Hugging Face, pinned revision
  1c94f2700886f822c953989255e039352fba6c56. SHA256:
  08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734."
  (Far better provenance: pinned revision + checksum.)

Why this matters beyond pedantry: **CC-BY-NC-SA's ShareAlike term requires
derivatives to carry the SAME license, but the competition requires a winner to
license their solution under CC-BY-NC 4.0 (no SA).** Those are not obviously
compatible. If we win using a CC-BY-NC-SA component, we may be unable to grant
what the competition requires. Also NC (NonCommercial) is at least arguable when
competing for a $77,000 prize pool.

**RESOLVED 2026-09-05 (as far as it can be) -- decision: PROCEED, eyes open.**
Checked the authoritative source (github.com/BMEII-AI/RadImageNet). Findings:
the GitHub repo carries an **MIT license file, but that covers the repo CODE**;
the pretrained **weights are distributed separately via Google Drive**, and
dataset access requires a request through radimagenet.com. So the "MIT" claim in
shigechan's description most likely conflates the repo's code license with the
weights' terms. **Neither uploader's tag is actually verified.**

Practical read, stated as a judgment call rather than a clean resolution:
- The competition's own test for external resources is "publicly available and
  equally accessible at no cost." RadImageNet weights plainly meet that, and
  effectively the entire top of the leaderboard uses them openly.
- The residual risk is narrow and only bites in the WINNING scenario: whether we
  could grant CC-BY-NC 4.0 over a solution containing a possibly-ShareAlike
  component. That risk applies identically to every team at the top.
- Therefore: proceed, but (a) prefer `shigechan/...` (pinned HF revision +
  SHA256 = real provenance) over `marwanmath/...` (no description), (b) record
  the dependency and its license ambiguity in any writeup, and (c) if we place,
  raise the licensing question with the hosts openly rather than hiding it.
Do NOT silently assume MIT.

**`tonylica/...0917-repro-assets`** (2.6GB, 3,778 downloads) is the largest
shortcut available -- a verified 0.917 reproduction -- but has an unspecified
license and no description. Treat as unverified until clarified.

## VALIDATION CONTAMINATION -- SOLVED (2026-09-05). READ BEFORE USING PUBLIC WEIGHTS.

**The public fold weights (`mattiaangeli/knee-mri-fold-weights`: m_f0.pt..m_f4.pt)
ship NO fold-assignment file.** But their split is fully DETERMINISTIC and
reproducible -- no random seed at all. From the public notebook's training code:

    grp = md5(report_text).hexdigest()[:8] as int % 5     # group by report text
    va  = [i for i in keep if grp[i] == 0]                # holdout = group 0
    tr  = [i for i in keep if grp[i] != 0]                # train   = groups 1-4

Reproduced locally against our train.csv (measured, not assumed):
- THEIR holdout (md5(report)%5==0) = **881 studies**; their train = 3,526.
- OUR holdout (seed=42, holdout_fold=0, iterative stratification) = 871 studies.
- **696 of our 871 holdout studies (79.9%) are in THEIR TRAINING data.**
- Only **175** studies are clean under BOTH definitions.
- Gold studies: 11 in their holdout, 11 in ours, only **3 in both**.

**CONSEQUENCE -- do not skip this.** Evaluating any public member on OUR holdout
is ~80% contaminated and will produce a badly inflated number. Symmetrically,
our own checkpoints (base v2 etc.) trained on 706 of the 881 studies in THEIR
holdout, so they are contaminated on that set too. There is no free lunch:

1. **For anything we TRAIN from here on in the ensemble line: adopt THEIR split**
   (train on md5(report)%5 != 0, validate on == 0). That makes our members and
   the public members directly comparable on one clean 881-study set.
2. **For an immediate zero-retraining comparison**: the 175-study intersection is
   clean for both sides. Small (n=175) but usable as a sanity check to compare
   base v2 against public members before committing GPU to retraining.
3. Historical tier1 numbers in this file are all on OUR fold 0 and are NOT
   comparable to numbers computed on their group 0. Re-baseline before comparing.

This is exactly the trap that pushes the whole cluster onto public-LB
noise-chasing (they cannot validate honestly, so they chase 0.001-0.003 public
movements). Being able to validate cleanly is our actual edge for the PRIVATE
leaderboard -- it is the concrete substance behind the strategy the user chose.

## WHAT THE 0.95 CLUSTER ACTUALLY IS (read 2026-09-05, notebooks pulled)

Pulled and read both public notebooks (source in
`data/interim/public_notebooks/`, outputs stripped into `*.src.py`).

**It is not leakage and not a secret technique. It is ONE shared community
ensemble that ~17 teams fork and re-tweak.** `mattiaangeli/bend-the-knee-to-
dinov3-ensembled` is a rank-mean of **twenty DINOv2 models + one DINOv3
ViT-S/16 + RadImageNet ResNet-50 heads** run at two different slot layouts,
pooling publicly-shared trained weights from ~8 people, plus **three**
report-label sources (pilkwang's, lixin73's, and stevenleehans' -- the one we
already use) cross-checked against each other. Public fold weights live in a
dataset called `knee-mri-fold-weights`.

**The author explicitly warns it is public-LB overfit** (quoting cell 0):
"This notebook is now likely overfit to the public leaderboard: after users
repeatedly forked and republished it with one or two parameter tweaks, I also
followed movements of only 0.001-0.003." He calls the fork-and-republish race
"opportunistic reuse, not original work" and recommends using an earlier
version, or the DINOv3 member, **validated locally on held-out labelled
studies**. This is highly relevant to our PRIVATE-LB goal: much of the
0.949-0.954 spread is public-30% noise-chasing and should shuffle on private.
(This also refines the earlier leakage theory: leakage was the wrong
explanation, but public-LB overfitting via a community fork race is real and
documented by the author himself.)

**Their architecture is genuinely more expressive than ours.** They add a
learned type embedding (plane x fat-suppression) to every token, concatenate
the tokens of EVERY series in a study into one key/value sequence, and have
twelve learned queries (one per finding) cross-attend over all of it with
multi-head attention; each query output is concatenated with the mean AND max
of per-series CLS embeddings. Missing series are handled by key-padding mask,
nothing imputed. Ours computes per-slot summaries FIRST and attends over those
-- strictly less information reaches each finding. This is a concrete,
copyable architectural upgrade.

**Other calibration**: `prvsiyan/rsna-knee-read-the-report-then-the-knee`
(251 votes) has a stated verified score of **0.906** -- even that elaborate
single-author pipeline sits well below the cluster. The 0.95s come from
POOLING, not from one better idea. That notebook also contains a nine-language
report->twelve-graded-targets extractor (cell 4), an independent Synovitis
specialist, PCA exact-label specialists, a "leakage-safe DICOM-header branch",
and a `raidium/curia-2` DINOv2-base frozen *radiology* encoder (TPU stage).

**DINO generations (checked Kaggle model catalog 2026-09-05)**: there is **no
DINOv4** (search returns nothing; does not exist). DINOv3 IS available:
`keras/dinov3` (ViT + ConvNeXt), `mdabrarhossainzahin/dinov3` (ViT-B/16),
`unmol03/dinov3-vit-large`, `pr4deepr/dinov3-vith16plus`. We have been on
DINOv2-base all along, so DINOv3 is a real upgrade path.

**STRATEGY DECISION (user, 2026-09-05)**: **use public components AND add our
own value** -- build on the public Apache-2.0 ensemble like the rest of the
field, but differentiate by validating properly (5-fold CV on held-out labels)
instead of noise-chasing the public LB, exactly as the notebook author himself
recommends. RadImageNet deferral LIFTED (see below). MRNet/OAI still deferred.

## CRITICAL REALITY CHECK (2026-09-05) -- READ BEFORE PLANNING ANYTHING

**Checked the actual public leaderboard for the first time this session.
Rank 10 = 0.949. We are at 0.812. Gap = 0.137 AUC.**

The top is DENSELY PACKED: 0.954, 0.953, 0.953, 0.952, 0.951, 0.950, 0.950,
0.950, 0.949 x6, 0.948... ~17 teams within 0.006 of each other.

**This CORRECTS a hypothesis repeated many times earlier in this session**:
that the ~0.952 leader might be leakage-inflated and could collapse on
private. One team at 0.95 while the field sits at 0.80 would look like
leakage. SEVENTEEN teams clustered at 0.95 looks like a known, shared,
effective approach we have not found. Do not keep repeating the
leakage-collapse theory -- the distribution does not support it.

**THE CEILING PROBLEM (most important strategic fact in this file):**
Our silver labels (steven's) score **0.8932 vs the 58 gold studies**. That is
what our model trains to reproduce. A model that learns those targets
PERFECTLY lands around ~0.89, not 0.95. Every queued roadmap item
(multi-slice, slot gaps, 5-fold CV, 100%-data) is a MODEL-side improvement
fighting a LABEL-side ceiling, worth maybe +0.01-0.02 combined. **None of
them can reach top 10.** Teams at 0.95 must have something structurally
different -- better label extraction, different use of gold, or a shared
public approach.
(Caveat, stated honestly: n=58 gold is small so 0.8932 has real uncertainty,
and models can somewhat exceed noisy teachers since label noise partly
averages out. But 0.89-ceiling labels -> 0.95 is not a plausible jump.)

**PUBLIC NOTEBOOKS THAT LIKELY EXPLAIN THE CLUSTER (found 2026-09-05, not
yet read -- highest-value next action, zero GPU cost):**
- `mattiaangeli/bend-the-knee-to-dinov3-ensembled` (207 votes) -- **DINOv3**,
  ensembled. We have been on DINOv2-base this whole time. Newer backbone
  generation; the DINOv2 small->base swap already gave us a real gain, so a
  DINOv3 swap is mechanically familiar and plausibly significant.
- `prvsiyan/rsna-knee-read-the-report-then-the-knee` (251 votes) -- title
  implies extracting signal from Report text = directly aimed at our label
  ceiling.
- `prvsiyan/head-and-shoulders-knees-and-toes` (176 votes)
Reading public notebooks is explicitly allowed (same category as
pilkwang/rsna-knee-baseline-v1, the source of our slot/geometry/laterality
techniques) and the user confirmed "use anything you want, but make sure
it's allowed in the rules."

**REVISED PRIORITY ORDER (supersedes the phase table below until resolved):**
1. Read both notebooks; determine what the 0.95 cluster actually does.
2. Fix the LABEL ceiling first -- nothing else matters until it moves.
3. Test DINOv3 as a backbone swap.
4. Only then return to the 5-fold CV / 100%-data plan below (still valid,
   but a +0.01-0.02 refinement, not a path to top 10 on its own).

**Process lesson worth keeping**: our engineering rigor was real (rejected
leaked datasets, reported failed experiments honestly, caught a real
submission bug), but we optimized in isolation for days without ever
checking what the field was doing. Check the leaderboard distribution and
public work EARLY, not after a week of local iteration.

## ENDGAME PLAN (written 2026-09-05, ~6.7 weeks to the 2026-10-22 deadline)

**Budget framing**: ~6.7 weeks x 30 GPU-h/week = **~200 GPU-hours total left**.
That, not ideas, is the binding constraint -- plan against it.

**The biggest untried lever, and it's specifically a PRIVATE-LB one:
proper 5-fold CV ensembling.** Everything so far optimizes ONE model on ONE
train/holdout split (seed=42, holdout_fold=0, unchanged all session). This is
NOT the same as the ensemble that failed in roadmap item 2: that averaged
three checkpoints from a SINGLE trajectory on a SINGLE split (highly
correlated -- added nothing, regressed 0.782->0.775 on real LB). A 5-fold
ensemble trains 5 models on 5 DIFFERENT 80% splits, which are genuinely
decorrelated. It is the most reliable variance-reduction technique available
and directly targets the private-LB risk: that our single-split result is a
lucky artifact. It also fixes our least-trustworthy instrument -- tier1 has
misled us before (the item-2 ensemble case); a 5-fold mean is far
better-calibrated than one n=871 split. `make_grouped_multilabel_folds()` in
validate.py already builds all 5 folds; we have only ever used fold 0.

**Second, near-free win: we train on only 80% of the data** (3,536 of 4,407
studies; 871 permanently held out). Correct during development, wasteful for
the FINAL model. Once the recipe is frozen, retrain on 100%.

| Phase | Weeks | GPU | What |
|---|---|---|---|
| Finish current tests | now | ~10h | multislice (in flight), then COR_FLUID_NOFS slot test (item 7) -- cheap single-variable checks |
| **5-fold CV ensemble** | 2-3 | ~40h | 5 folds x ~4h on whichever recipe wins. Main structural investment. |
| Validate + freeze | 4 | ~15h | Real submission confirming the ensemble beats 0.812. Freeze recipe. |
| Final model | 5 | ~20h | Retrain on 100% of data; multi-seed if budget allows |
| **Buffer** | 6 | reserve | Dry-run the exact final kernel, submit EARLY |

**Three risks to keep front of mind:**
1. **The submission path can silently fail** -- proven by the 0.5 multislice
   bug (a config change produced a garbage submission that looked perfectly
   healthy locally, tier1=0.7793). EVERY new configuration needs a real
   confirmed submission before being trusted. We got lucky that v2 works.
2. **Submission selection**: Kaggle usually lets you choose which submissions
   count for private LB (typically 2). Make sure the selected ones are
   genuinely our best -- never let a 0.5 or an unverified config be selected.
3. **Do not burn the final week's quota.** If a fix is needed in week 6 with
   no GPU-hours left, we ship whatever is already submitted.

**Honest calibration**: the visible ~0.952 leader may be leakage-inflated and
could collapse on private. First place is not promisable. But the moves above
(variance reduction, using all the data, not shipping a broken submission) are
what actually move PRIVATE rank, versus chasing public-LB points -- which is
the stated goal.

**Current best (already submitted, confirmed real): base v2 = 0.812 public.**
Nothing has beaten it. Do NOT submit `rsna-knee-base-continue-2` (base v3,
regression) or the ensemble kernel (0.775 regression).

## Submission mechanics reminder
I can push/run kernels via the Kaggle API, but **clicking "Submit to Competition"
on kaggle.com is the user's action** -- no API for that on code competitions. After
any kernel completes with a good result, tell the user the kernel URL and ask them
to submit it.
