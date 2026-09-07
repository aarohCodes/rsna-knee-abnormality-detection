"""Choose a blend weight on the clean holdout, then apply it to a submission.

Two modes, deliberately separated:

  choose-weight  offline, where labels exist: pick w by cross-fitting, and report
                 the honest expected gain rather than the fitted one
  apply          inside the kernel, where no labels exist: mix at the chosen w

Blending is by RANK, not probability. AUC reads only the order of predictions, so
averaging raw probabilities lets whichever member is most confident dominate the
ordering regardless of whether it is more correct. Rank averaging combines exactly
the information the metric uses.

The weight is chosen by cross-fitting: split the holdout, pick w on one half,
score it on the other, both directions. Picking w on the same studies you then
report is how a +0.004 that does not exist gets believed -- we already caught one
of those in this project.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
WEIGHTS = np.round(np.arange(0.0, 0.55, 0.05), 2)


def read_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing label columns {missing}")
    return df[["StudyInstanceUID"] + LABELS]


def rank_pct(a: np.ndarray) -> np.ndarray:
    return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])


def macro_auc(y: np.ndarray, p: np.ndarray, rows=None) -> float:
    if rows is not None:
        y, p = y[rows], p[rows]
    out = []
    for j in range(y.shape[1]):
        n1 = int(y[:, j].sum())
        n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(p[:, j])
        out.append((r[y[:, j] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out)) if out else float("nan")


def cmd_choose(args) -> int:
    base = read_preds(Path(args.base))
    member = read_preds(Path(args.member))
    labels = pd.read_csv(args.labels, dtype={"StudyInstanceUID": str})

    m = base.merge(member, on="StudyInstanceUID", suffixes=("_b", "_m")) \
            .merge(labels[["StudyInstanceUID"] + LABELS], on="StudyInstanceUID")
    if len(m) < 100:
        raise SystemExit(f"only {len(m)} studies overlap all three files; refusing to fit a weight")
    B = rank_pct(m[[c + "_b" for c in LABELS]].to_numpy(float))
    M = rank_pct(m[[c + "_m" for c in LABELS]].to_numpy(float))
    Y = (m[LABELS].to_numpy(float) >= 0.5).astype(int)
    n = len(m)
    print(f"studies: {n}   base {macro_auc(Y, B):.4f}   member {macro_auc(Y, M):.4f}")

    corr = np.mean([np.corrcoef(B[:, j], M[:, j])[0, 1] for j in range(12)])
    print(f"mean rank correlation base vs member: {corr:.3f}"
          f"{'  (highly correlated -- little to gain)' if corr > 0.97 else ''}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    halves = (perm[:n // 2], perm[n // 2:])

    print(f"\n{'w':>5s} {'full-holdout (optimistic)':>26s}")
    for w in WEIGHTS:
        print(f"{w:5.2f} {macro_auc(Y, (1 - w) * B + w * M):26.4f}")

    honest, picks = [], []
    for fit, ev in (halves, halves[::-1]):
        scores = [macro_auc(Y, (1 - w) * B + w * M, fit) for w in WEIGHTS]
        w = WEIGHTS[int(np.argmax(scores))]
        picks.append(w)
        honest.append((macro_auc(Y, (1 - w) * B + w * M, ev), macro_auc(Y, B, ev)))
    blend = float(np.mean([h[0] for h in honest]))
    base_only = float(np.mean([h[1] for h in honest]))
    w_final = float(np.mean(picks))

    print(f"\ncross-fitted: picked w={picks} -> mean w={w_final:.2f}")
    print(f"  blend {blend:.4f}   base alone {base_only:.4f}   HONEST delta {blend - base_only:+.4f}")
    if blend - base_only <= 0.0005:
        print("\nVERDICT: no real gain. Ship the base alone; do NOT add this member.")
    else:
        print(f"\nVERDICT: use --weight {w_final:.2f} (expected ~{blend - base_only:+.4f})")
    return 0


def cmd_apply(args) -> int:
    base = read_preds(Path(args.base))
    member = read_preds(Path(args.member))
    order = base["StudyInstanceUID"].tolist()
    member = member.set_index("StudyInstanceUID").reindex(order)
    if member[LABELS].isna().any().any():
        raise SystemExit("member predictions do not cover every study in the base submission")

    w = float(args.weight)
    if not 0.0 <= w <= 1.0:
        raise SystemExit(f"weight must be in [0,1], got {w}")
    B = rank_pct(base[LABELS].to_numpy(float))
    M = rank_pct(member[LABELS].to_numpy(float))
    blended = rank_pct((1 - w) * B + w * M)      # re-rank so the output is a clean 0-1 ordering

    out = pd.DataFrame(blended, columns=LABELS)
    out.insert(0, "StudyInstanceUID", order)
    if not np.isfinite(out[LABELS].to_numpy(float)).all():
        raise SystemExit("blended predictions contain non-finite values")
    const = [c for c in LABELS if out[c].nunique() <= 1]
    if const:
        raise SystemExit(f"constant columns would score 0.5: {const}")
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  {out.shape}  w={w}  (base {1-w:.2f} / member {w:.2f})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("choose-weight", help="pick w on the clean holdout (needs labels)")
    c.add_argument("--base", required=True, help="base pipeline's holdout predictions")
    c.add_argument("--member", required=True, help="our member's holdout predictions")
    c.add_argument("--labels", required=True, help="silver label csv")
    c.set_defaults(func=cmd_choose)

    a = sub.add_parser("apply", help="mix at a fixed w (no labels needed)")
    a.add_argument("--base", required=True)
    a.add_argument("--member", required=True)
    a.add_argument("--weight", required=True)
    a.add_argument("--out", default="submission.csv")
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
