"""Run a trained member over a study set and write predictions.

This is the missing half of the training track. `train_member.py` produces
`best.pt`; nothing could consume it. This script does, and it is written to run in
two places without modification:

  * locally, against a downloaded copy of the data, to sanity-check a checkpoint
  * inside the scored Kaggle kernel, against /kaggle/input, with internet disabled

It deliberately decodes on the fly rather than reading the training cache: the
kernel has no cache, makes exactly one pass per study, and caching would only cost
memory. That is the same bug that produced two 0.500 submissions earlier in this
project -- a cache sized for one configuration silently ballooning under another.

Preprocessing is imported from prep_cache rather than reimplemented, so a member
can never be run under a recipe different from the one it trained on. Mismatched
preprocessing loads without error and quietly destroys accuracy, which is exactly
the failure that is hardest to notice from a submission file.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_cache  # noqa: E402  (also gives us SLOT_PRESETS / build_study)
from member import IMAGENET_MEAN, IMAGENET_STD, KneeClassifier, LABELS, build_backbone  # noqa: E402


def find_competition_root(explicit: str | None = None) -> Path:
    """Locate the competition data. Kaggle mounts it at one of a couple of paths
    and the layout has changed before, so search rather than hardcode."""
    if explicit:
        return Path(explicit)
    candidates = [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                  Path("/kaggle/input/rsna-knee-abnormality-detection")]
    for c in candidates:
        if (c / "test.csv").is_file():
            return c
    base = Path("/kaggle/input")
    if base.is_dir():
        for d, _, files in __import__("os").walk(base):
            if "test.csv" in files and (Path(d) / "test_series").is_dir():
                return Path(d)
    raise RuntimeError("could not locate the competition root; pass --data-root")


def load_member(ckpt_path: Path, device: torch.device):
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    arch, px = ck["arch"], int(ck["px"])
    backbone, feat_dim = build_backbone(arch, pretrained=False)
    if int(ck.get("feat_dim", feat_dim)) != feat_dim:
        raise RuntimeError(f"feature dim mismatch: checkpoint {ck['feat_dim']} vs built {feat_dim}")
    model = KneeClassifier(backbone, feat_dim)
    model.load_state_dict(ck["model"], strict=True)   # strict: a silent partial load is worse than a crash
    model.eval().to(device)
    print(f"member: {arch} @ {px}px  (holdout_auc={ck.get('holdout_auc', float('nan')):.4f}, "
          f"epoch={ck.get('epoch')})", flush=True)
    return model, px


def windows_for(vol: np.ndarray, mask: np.ndarray, k: int, px: int) -> torch.Tensor:
    """Evenly spaced adjacent-slice triplets, matching the eval path in training."""
    import torch.nn.functional as F

    depth = vol.shape[0]
    valid = np.flatnonzero(mask > 0)
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    cand = [c for c in range(lo + 1, hi) if lo <= c - 1 and c + 1 <= hi]
    if not cand:
        cand = [max(1, min((lo + hi) // 2, depth - 2))]
    idx = np.linspace(0, len(cand) - 1, k).round().astype(int)
    centres = [cand[i] for i in idx]

    wins = np.empty((k, 3, vol.shape[1], vol.shape[2]), np.float32)
    for j, c in enumerate(centres):
        c = max(1, min(int(c), depth - 2))
        wins[j] = vol[c - 1:c + 2].astype(np.float32) / 255.0
    x = torch.from_numpy(wins)
    if x.shape[-1] != px:
        x = F.interpolate(x, size=(px, px), mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=None, help="competition root; auto-detected on Kaggle")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--out", default="member_predictions.csv")
    ap.add_argument("--k-eval", type=int, default=48)
    ap.add_argument("--slot-preset", default="raptor", choices=sorted(prep_cache.SLOT_PRESETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--study-ids", default=None,
                    help="optional newline-delimited file of study ids (e.g. a holdout list)")
    args = ap.parse_args(argv)

    # Must match the cache the member trained on, or the member sees a different
    # world at inference than it did at training.
    prep_cache.SLOTS = prep_cache.SLOT_PRESETS[args.slot_preset]
    prep_cache.MAXS = sum(s[2] for s in prep_cache.SLOTS)

    root = find_competition_root(args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"root={root}  device={device}  slots={args.slot_preset}({prep_cache.MAXS})", flush=True)

    ser = pd.read_csv(root / f"{args.split}_series.csv", dtype=str)
    for c in ("Fluid_Sensitive", "Fat_Suppression"):
        if c in ser.columns:
            ser[c] = pd.to_numeric(ser[c], errors="coerce").fillna(0).astype(int)
    grouped = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}
    series_root = root / f"{args.split}_series"

    ids = pd.read_csv(root / f"{args.split}.csv", dtype=str)["StudyInstanceUID"].tolist()
    if args.study_ids:
        wanted = [l.strip() for l in Path(args.study_ids).read_text().splitlines() if l.strip()]
        ids = [s for s in ids if s in set(wanted)]
    if args.limit:
        ids = ids[:args.limit]

    model, px = load_member(Path(args.checkpoint), device)
    amp = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # Write a valid neutral file FIRST, so a crash midway still leaves something
    # scoreable rather than nothing. Overwritten by real predictions below.
    out = Path(args.out)
    pd.DataFrame(0.5, index=range(len(ids)), columns=LABELS).assign(
        StudyInstanceUID=ids)[["StudyInstanceUID"] + LABELS].to_csv(out, index=False)

    preds = np.full((len(ids), len(LABELS)), 0.5, np.float32)
    t0, failures = time.time(), 0
    for i, sid in enumerate(ids):
        try:
            vol, mask = prep_cache.build_study(sid, grouped.get(sid, []), series_root)
            x = windows_for(vol, mask, args.k_eval, px).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device.type, dtype=amp, enabled=device.type == "cuda"):
                    logits = model(x)
            preds[i] = torch.sigmoid(logits.float())[0].cpu().numpy()
        except Exception as e:
            failures += 1
            print(f"  study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(f"  {i+1}/{len(ids)}  {time.time()-t0:.0f}s  failures={failures}", flush=True)

    df = pd.DataFrame(preds, columns=LABELS)
    df.insert(0, "StudyInstanceUID", ids)
    df.to_csv(out, index=False)
    print(f"wrote {out}  {df.shape}  failures={failures}  {time.time()-t0:.0f}s", flush=True)

    if failures == len(ids):
        print("ERROR: every study failed; predictions are all 0.5 and would score 0.5",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
