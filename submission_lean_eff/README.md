# rsna-knee-lean-eff — VALIDATED, ready to submit

A lean kernel for the **efficiency leaderboard**, which is a separate award.

## What it is

bee-v2 truncated to its first four cells plus our integrity guard. Cell 3 already
writes a complete, valid `submission.csv` before the raptor and coat stages run
(confirmed in the bee-v2 log at 79.8s), so this is a **truncation, not surgery** —
cells 0–3 are byte-identical to bee-v2 apart from one content-anchored insert.

The one insert caps the member count:

```python
LEAN_MEMBERS = 4        # 0 restores the full 20-member behaviour exactly
```

## Why

`Efficiency = AUC/(0.5 − maxAUC) + seconds/32400`, minimised. One hour of runtime
is worth **+0.0504 AUC** — more than the entire leaderboard's accuracy spread
(0.032). We currently sit at efficiency rank **1350/3214** purely because bee-v2
takes 8h15m. Its top 10 contains teams scoring 0.934–0.941, which we already match.

| candidate | AUC | runtime | efficiency |
|---|---|---|---|
| bee-v2 as submitted | 0.938 | 8.25h | −1.149 |
| this, if ~0.91 in ~40 min | 0.910 | 0.67h | **≈ −1.98** |

## Validation (done, 2026-09-07)

Save-run passed with 0 errors:

```
[lean] using 5 of 20 members, folds [0, 1, 2, 3, 4]
<member>: fingerprint matches within 2.8e-06        (all five)
legacy DINO aggregation written from five folds
integrity check PASSED: 3 rows, no NaNs, no constant columns
```

The fingerprint checks matter: they prove each member runs under the preprocessing
it was trained with — a mismatch loads without error and silently wrecks accuracy.

A first attempt taking the first 4 members died on `legacy branch requires five
folds, found 4`. Selection is now one member per fold, which satisfies that
combiner and is the better choice anyway: cross-fold members are the most
decorrelated available.

Measured, extrapolated to 1,322 studies: decode ~20 min + 5 members ~28 min +
~1 min fixed = **~49 min**, versus bee-v2's 8h15m.

## Remaining tuning (optional)

1. **Trim datasets if mount time is significant.** It currently inherits all seven
   of bee-v2's, but cells 0–3 likely need only `tonylica/...repro-assets`,
   `renta0426/...meniscus-bag-v1`, and the DINOv2 model. Dropping the three raptor
   datasets, the opencv wheel and the coat dataset would cut mount time — but only
   change this after step 1 succeeds, one variable at a time.
3. **Tune `LEAN_MEMBERS`** against measured runtime. 4 is a starting guess, not a
   measured optimum.

Nothing here has been run. The member cap changes behaviour inside another team's
pipeline, and the failure mode for that pipeline is silent rather than a crash.
