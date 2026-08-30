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
from rsna_knee.metrics import LABEL_COLUMNS
from rsna_knee.models.vision import KneeMultiLabelModel


def run_inference(
    model: torch.nn.Module,
    study_ids: list[str],
    series_with_slots: pd.DataFrame,
    series_root: Path,
    device: str,
    batch_size: int = 4,
) -> pd.DataFrame:
    """Run `model` over every study in `study_ids` and return a dataframe shaped
    exactly like sample_submission.csv (StudyInstanceUID + the 12 label columns,
    sigmoid-activated probabilities).
    """
    # StudyDataset needs a `targets` frame; inference has no real targets, so pass an
    # empty one -- every study falls through to the neutral/zero-weight branch, which
    # is fine since targets/weights aren't used at inference time.
    empty_targets = pd.DataFrame({"StudyInstanceUID": []})
    dataset = StudyDataset(study_ids, series_with_slots, series_root, empty_targets)
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
