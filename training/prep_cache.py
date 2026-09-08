"""Decode the DICOM corpus once into a compact uint8 cache.

Why this exists: training reads the same studies dozens of times across epochs, and
decoding JPEG-2000/JPEG-lossless DICOM is far slower than the forward pass. Decoding
once into fixed-size uint8 volumes turns a ~500GB-1TB corpus into ~42GB that loads
at memory-map speed.

The recipe here is deliberately NOT ours. STATUS.md records an uncomfortable finding:
the public pipelines' preprocessing is *simpler* than the one we built and scores far
better (0.9178 vs 0.7931 on the same clean holdout). So this reproduces the raptor
recipe exactly -- geometry-ordered slices, a fixed physical field of view, per-stack
percentile normalisation -- rather than reinventing it.

Cached at 384px even though most arms train at 336: downsampling in the dataloader is
free, re-decoding the corpus is not. One cache serves every resolution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- the raptor contract, copied verbatim so members stay comparable ----------
CROP_MM = 140.0
CACHE_PX = 384
SPAN_LO, SPAN_HI = 0.02, 0.98
# (anatomical plane, Fluid_Sensitive preference, slice budget); -1 = don't care
#
# "raptor" is the public recipe, reproduced exactly so our members stay comparable.
# "balanced" is our own hypothesis, and it is motivated by a measurement rather than
# a hunch. Per-label AUC of the best public member on the clean holdout shows the two
# weakest findings are PF OA (0.8942) and MCL (0.9055) -- and those are precisely the
# findings read on the two most under-sampled planes. PF OA lives on axial slices
# through the patella, which get 12 of 64 slices (19%); MCL is a coronal structure,
# and coronal gets 31%. Meanwhile sagittal takes half the budget and carries the
# findings that already score best.
#
# Because macro AUC weights all 12 findings equally, lifting the four weakest to 0.95
# is worth +0.0168 to the macro, while improving the four strongest by 0.01 each is
# worth only +0.0033. So rebalancing toward the weak planes is the highest-leverage
# change available -- and it is one no forker of the public ensemble will try, since
# they all inherit this same table.
SLOT_PRESETS = {
    "raptor": [("Sagittal", 1, 18), ("Sagittal", 0, 14),
               ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)],
    "balanced": [("Sagittal", 1, 14), ("Sagittal", 0, 10),
                 ("Coronal", 1, 14), ("Coronal", 0, 10), ("Axial", -1, 16)],
    # "lean" is for the EFFICIENCY prize, a separate award where runtime dominates:
    # one hour is worth +0.0504 AUC, more than the entire leaderboard's accuracy
    # spread (0.032). Halving the slice budget roughly halves both DICOM decode and
    # inference. At the measured marginal rate (1 min ~ 0.00084 AUC), a member
    # running in 15 min at 0.89 beats our current best efficiency entry outright.
    # Proportions follow "balanced", since the weak findings are the same ones.
    "lean": [("Sagittal", 1, 7), ("Sagittal", 0, 5),
             ("Coronal", 1, 7), ("Coronal", 0, 5), ("Axial", -1, 8)],
}
SLOTS = SLOT_PRESETS["raptor"]
MAXS = sum(s[2] for s in SLOTS)  # 64

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def order_and_meta(series_dir: Path):
    """Return ([(file, pixel_spacing)] in true spatial order, median spacing).

    Slice order must come from geometry, never filename: SOP Instance UIDs are
    unordered and have ~zero rank correlation with anatomy in this corpus. The
    normal is ImageOrientationPatient's row x col, and position is
    ImagePositionPatient projected onto it. InstanceNumber is only a fallback for
    studies whose geometry tags were stripped.
    """
    import pydicom

    records, spacings = [], []
    for f in series_dir.glob("*.dcm"):
        try:
            h = pydicom.dcmread(str(f), stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), normal))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            spacings.append(ps)
            records.append((pos, f, ps))
        except Exception:
            records.append((0.0, f, 0.5))
    records.sort(key=lambda r: r[0])
    med = float(np.median(spacings)) if spacings else 0.5
    return [(f, ps) for _, f, ps in records], med


def read_pixels(path: Path) -> np.ndarray:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    d = pydicom.dcmread(str(path))
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    # MONOCHROME1 stores inverted greyscale; without this, half the corpus trains
    # with reversed contrast against the other half.
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_resize(a: np.ndarray, pixel_spacing: float, out_px: int = CACHE_PX) -> np.ndarray:
    """Centre-crop to a fixed PHYSICAL field of view, then resize.

    Cropping by pixel count would mean different scanners contribute different
    anatomy; cropping by millimetres makes a knee the same size everywhere.
    """
    import cv2

    h, w = a.shape
    crop_px = int(round(CROP_MM / max(pixel_spacing, 1e-3)))
    crop_px = min(crop_px, h, w)
    y0, x0 = (h - crop_px) // 2, (w - crop_px) // 2
    a = a[y0:y0 + crop_px, x0:x0 + crop_px]
    return cv2.resize(a, (out_px, out_px), interpolation=cv2.INTER_AREA)


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows
             if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(study_uid: str, series_rows, series_root: Path):
    """Return (uint8 volume [MAXS, CACHE_PX, CACHE_PX], presence mask [MAXS])."""
    vol = np.zeros((MAXS, CACHE_PX, CACHE_PX), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in SLOTS:
        row = pick_series(series_rows, plane, fluid, used)
        if row is None:
            idx += k
            continue
        used.add(row["SeriesInstanceUID"])
        files, med_ps = order_and_meta(series_root / study_uid / row["SeriesInstanceUID"])
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * SPAN_LO), max(int(n * SPAN_HI) - 1, int(n * SPAN_LO))
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k

        arrays, spacings = [], []
        for p in picks:
            f, ps = files[min(int(p), n - 1)]
            try:
                arrays.append(read_pixels(f))
                spacings.append(ps)
            except Exception:
                arrays.append(None)
                spacings.append(med_ps)

        # MR has no absolute intensity scale, so normalise per sampled stack.
        valid = [a for a in arrays if a is not None]
        if valid:
            flat = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(flat, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0

        for a, ps in zip(arrays, spacings):
            if idx >= MAXS:
                break
            if a is None:
                idx += 1
                continue
            w = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            vol[idx] = (crop_resize(w, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
            idx += 1
        if idx >= MAXS:
            break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="dir holding train_series.csv and train_series/")
    ap.add_argument("--out", required=True, help="output cache directory")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="only process the first N studies (smoke test)")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="process shard I of N (e.g. 1/2). Kaggle caps notebook output "
                         "at ~20GB, so the full 27.7GB cache needs two CPU runs; the lean "
                         "preset (13.5GB) fits in one. Shards are deterministic slices of "
                         "the sorted study list, so they never overlap or leave gaps.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--slot-preset", default="raptor", choices=sorted(SLOT_PRESETS),
                    help="'raptor' reproduces the public recipe; 'balanced' shifts budget "
                         "toward coronal/axial to target the weakest findings (see above)")
    args = ap.parse_args(argv)

    global SLOTS, MAXS
    SLOTS = SLOT_PRESETS[args.slot_preset]
    MAXS = sum(s[2] for s in SLOTS)
    print(f"slot preset '{args.slot_preset}': {SLOTS}  -> {MAXS} slices/study", flush=True)

    root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    series_root = root / f"{args.split}_series"

    ser = pd.read_csv(root / f"{args.split}_series.csv", dtype=str)
    for c in ("Fluid_Sensitive", "Fat_Suppression"):
        if c in ser.columns:
            ser[c] = pd.to_numeric(ser[c], errors="coerce").fillna(0).astype(int)
    grouped = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID")}

    study_ids = [s for s in sorted(grouped) if (series_root / s).is_dir()]
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard must look like I/N, got {args.shard!r}")
        if not 1 <= i <= n:
            raise SystemExit(f"--shard I must be in 1..N, got {args.shard!r}")
        total = len(study_ids)
        lo = (i - 1) * total // n
        hi = i * total // n
        study_ids = study_ids[lo:hi]
        print(f"shard {i}/{n}: studies [{lo}:{hi}] of {total}", flush=True)
    if args.limit:
        study_ids = study_ids[:args.limit]
    print(f"studies to cache: {len(study_ids)}  ->  {out}", flush=True)

    def work(sid):
        dst = out / f"{sid}.npz"
        if dst.exists():
            return sid, True, "cached"
        try:
            vol, mask = build_study(sid, grouped[sid], series_root)
            if mask.sum() == 0:
                return sid, False, "no slices decoded"
            tmp = dst.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, vol=vol, mask=mask)
            os.replace(tmp, dst)   # atomic: a killed job never leaves a torn file
            return sid, True, f"{int(mask.sum())}/{MAXS} slots"
        except Exception as e:
            return sid, False, f"{type(e).__name__}: {e}"

    from concurrent.futures import ThreadPoolExecutor
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (sid, good, msg) in enumerate(ex.map(work, study_ids), 1):
            ok, fail = ok + good, fail + (not good)
            if not good:
                print(f"  FAIL {sid[:24]} {msg}", flush=True)
            if i % 100 == 0 or i == len(study_ids):
                print(f"  {i}/{len(study_ids)}  ok={ok} fail={fail}", flush=True)

    meta = {"crop_mm": CROP_MM, "cache_px": CACHE_PX, "slots": SLOTS, "maxs": MAXS,
            "span": [SPAN_LO, SPAN_HI], "studies": ok, "failed": fail}
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone: {ok} cached, {fail} failed -> {out}/cache_meta.json", flush=True)
    return 1 if ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
