"""Training loop: confidence-weighted BCE over the 12 labels.

Per-label sample weight (from label_expansion.py) is the label-source confidence: a
report that never mentions a finding pulls weakly on that head rather than being
treated as a confident negative, and the 58 gold-annotated studies get elevated
weight since they're the only labels read from images rather than text.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.data.label_expansion import build_training_targets
from rsna_knee.data.labels import load_train_labels
from rsna_knee.data.series import assign_slots
from rsna_knee.datasets import StudyDataset
from rsna_knee.models.vision import KneeMultiLabelModel
from rsna_knee.validate import make_grouped_multilabel_folds, report_hash


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean of per-element BCE-with-logits, weighted per (study, label). A weight of
    zero (no gold or silver label available for that study/label) contributes nothing;
    normalizing by the sum of weights (not element count) keeps the loss scale stable
    as label coverage varies across batches.
    """
    per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    weighted = per_element * weight
    denom = weight.sum().clamp_min(1e-8)
    return weighted.sum() / denom


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        images = batch["images"].to(device)
        mask = batch["presence_mask"].to(device)
        target = batch["target"].to(device)
        weight = batch["weight"].to(device)

        optimizer.zero_grad()
        logits = model(images, mask)
        loss = weighted_bce_loss(logits, target, weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def build_datasets(data_root: Path, series_root: Path, study_ids: list[str] | None = None):
    train_labels = load_train_labels(data_root / "train.csv")
    series_df = pd.read_csv(data_root / "train_series.csv")
    silver = pd.read_csv(data_root / "external" / "rsna-knee-llm-labels" / "report_labels_v2.csv")
    targets = build_training_targets(train_labels, silver)

    if study_ids is None:
        study_ids = [d.name for d in series_root.iterdir() if d.is_dir()]
    series_subset = series_df[series_df["StudyInstanceUID"].isin(study_ids)]
    series_with_slots = assign_slots(series_subset, series_root)

    train_labels["report_hash"] = train_labels["Report"].map(report_hash)
    return train_labels, series_with_slots, targets, study_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint-out", type=Path, default=Path("data/interim/model.pt"))
    args = parser.parse_args()

    series_root = args.data_root / "train_series"
    train_labels, series_with_slots, targets, study_ids = build_datasets(args.data_root, series_root)

    dataset = StudyDataset(study_ids, series_with_slots, series_root, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KneeMultiLabelModel(
        num_slots=6, num_labels=12, backbone_name=args.backbone, pretrained=args.pretrained
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch {epoch+1}/{args.epochs}: loss={loss:.4f}")

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_out)
    print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
