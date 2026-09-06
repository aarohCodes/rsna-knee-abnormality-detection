"""Inference: load a trained checkpoint, run on the test set, write submission.csv.

Shared by local evaluation and the Kaggle submission kernel -- both call
`run_inference`, so the exact same code path that's tested locally is what actually
gets scored. Image-only: `Report` is never read here since it isn't available in
test.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.data.series import assign_slots
from rsna_knee.datasets import StudyDataset
from rsna_knee.metrics import LABEL_COLUMNS, rank_average
from rsna_knee.models.vision import KneeMultiLabelModel

DEFAULT_TTA_CENTER_FRACTIONS = (0.35, 0.5, 0.65)


def run_inference(
    model: torch.nn.Module,
    study_ids: list[str],
    series_with_slots: pd.DataFrame,
    series_root: Path,
    device: str,
    batch_size: int = 4,
    center_fraction: float = 0.5,
    slices_per_slot: int = 1,
) -> pd.DataFrame:
    """Run `model` over every study in `study_ids` and return a dataframe shaped
    exactly like sample_submission.csv (StudyInstanceUID + the 12 label columns,
    sigmoid-activated probabilities). `center_fraction` selects which slice-window
    each series is sampled from (0.5 = middle, the default) -- see
    run_inference_tta for combining multiple values. `slices_per_slot` must match
    whatever the model was trained with (see StudyDataset/KneeMultiLabelModel) --
    passing a different value than training still runs without error (both handle
    any K), but the input distribution would differ from what the model learned.
    """
    # StudyDataset needs a `targets` frame; inference has no real targets, so pass an
    # empty one -- every study falls through to the neutral/zero-weight branch, which
    # is fine since targets/weights aren't used at inference time.
    #
    # cache_images=False: caching exists to avoid re-decoding a study across multiple
    # epochs, but inference makes exactly one pass over each study -- it can never
    # benefit from the cache, only pay its memory cost. That cost scales with
    # slices_per_slot (each study's cached array is slices_per_slot times bigger), so
    # the K=1-calibrated default cache size becomes a real OOM risk at higher K over
    # a large test set (real hidden test set, ~1,300 studies) even though it never
    # showed a problem at K=1 or over the smaller 871-study holdout. Found this after
    # a K=3 submission scored 0.5 (all-constant fallback) on the real test set despite
    # a healthy tier1 holdout AUC -- this is the most concrete lead, though not
    # directly confirmed against the actual scoring-run's error output (not
    # accessible to us). Disabling the cache removes the risk regardless.
    empty_targets = pd.DataFrame({"StudyInstanceUID": []})
    dataset = StudyDataset(
        study_ids, series_with_slots, series_root, empty_targets, center_fraction=center_fraction,
        slices_per_slot=slices_per_slot, cache_images=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            mask = batch["presence_mask"].to(device)
            logits = model(images, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            for study_id, row_probs in zip(batch["study_id"], probs):
                rows.append({"StudyInstanceUID": study_id, **dict(zip(LABEL_COLUMNS, row_probs))})

    return pd.DataFrame(rows, columns=["StudyInstanceUID", *LABEL_COLUMNS])


def run_inference_tta(
    model: torch.nn.Module,
    study_ids: list[str],
    series_with_slots: pd.DataFrame,
    series_root: Path,
    device: str,
    batch_size: int = 4,
    center_fractions: tuple[float, ...] = DEFAULT_TTA_CENTER_FRACTIONS,
    slices_per_slot: int = 1,
) -> pd.DataFrame:
    """Test-time augmentation: run inference once per `center_fractions` value (each
    sampling a different, overlapping-but-distinct slice window per series) and
    combine by rank average -- not raw probability averaging, since AUC only reads
    order and a raw average would let whichever pass produced more extreme values
    dominate (see metrics.rank_average). Each pass reuses `run_inference`, so this is
    the same tested code path repeated with a different `center_fraction`, not a
    separate implementation. `slices_per_slot` must match training, same as
    `run_inference` -- passed through unchanged to every pass.

    Note: `slices_per_slot > 1` and TTA don't currently combine usefully.
    `load_series_multi_slice` (used whenever `slices_per_slot > 1`) always spans the
    full ordered stack (fractions 0..1 evenly spaced) and ignores `center_fraction`
    entirely, so every "TTA pass" below would load identical multi-slice input and
    this would just repeat the same forward pass `len(center_fractions)` times for no
    benefit. Use `center_fractions=(0.5,)` (a single pass) when `slices_per_slot > 1`
    until/unless TTA is redesigned to vary something multi-slice loading actually
    reads (e.g. an offset applied to the evenly-spaced fractions).
    """
    passes = [
        run_inference(
            model, study_ids, series_with_slots, series_root, device, batch_size,
            center_fraction=cf, slices_per_slot=slices_per_slot,
        )
        for cf in center_fractions
    ]
    return rank_average(passes)


def write_trivial_submission(sample_submission_path: Path, out_path: Path) -> None:
    """Copy the trivial all-0.5 sample_submission.csv to `out_path`. Written *before*
    any real inference runs so the kernel always leaves a scoreable file behind even
    if it fails partway -- overwritten once real predictions exist.
    """
    pd.read_csv(sample_submission_path).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--test-series-dirname", type=str, default="test_series")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/interim/model.pt"))
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    sample_submission_path = args.data_root / "sample_submission.csv"
    write_trivial_submission(sample_submission_path, args.out)

    test_csv = pd.read_csv(args.data_root / "test.csv")
    test_series_csv = pd.read_csv(args.data_root / f"{args.test_series_dirname}.csv")
    series_root = args.data_root / args.test_series_dirname

    study_ids = test_csv["StudyInstanceUID"].tolist()
    series_with_slots = assign_slots(test_series_csv, series_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KneeMultiLabelModel(
        num_slots=6, num_labels=12, backbone_name=args.backbone, pretrained=False
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    predictions = run_inference(model, study_ids, series_with_slots, series_root, device)
    predictions.to_csv(args.out, index=False)
    print(f"Wrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
