"""Dataset, model and metric for training a competition member.

Design is copied from the public raptor member rather than from our own earlier
pipeline, on purpose. Ours scored 0.7735 on the clean 881-study holdout; theirs
scored 0.9178 on the identical studies. STATUS.md records that their preprocessing
is *simpler* than ours and still much better, so the sane move is to reproduce the
thing that works and only then vary it.

The split is the community's -- `int(md5(report)[:8], 16) % 5`, holdout fold 0 --
so a member trained here is directly comparable to the 0.9178 public number on the
same 881 studies. Any other split would make that comparison meaningless.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from torch.utils.data import Dataset

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def community_fold(report: str, n_splits: int = 5) -> int:
    """The public ensemble's fold assignment. Raw report text, NO normalisation --
    matching them exactly matters more here than matching our own conventions,
    because the entire point is to hold out the studies they held out."""
    return int(hashlib.md5(str(report).encode()).hexdigest()[:8], 16) % n_splits


def load_targets(data_root: Path, silver_csv: Path):
    """Return a frame of StudyInstanceUID, 12 labels, fold, is_gold, weight.

    Silver labels come from report text; the 58 gold studies were read from images
    and override silver wherever present. Gold rows carry more weight because they
    are the only signal not filtered through a report.
    """
    train = pd.read_csv(data_root / "train.csv", dtype={"StudyInstanceUID": str})
    silver = pd.read_csv(silver_csv, dtype={"StudyInstanceUID": str})
    missing = [c for c in LABELS if c not in silver.columns]
    if missing:
        raise ValueError(f"silver label file is missing columns: {missing}")

    df = train[["StudyInstanceUID", "Report"]].merge(
        silver[["StudyInstanceUID"] + LABELS], on="StudyInstanceUID", how="inner")
    gold = train[train[LABELS].notna().any(axis=1)].set_index("StudyInstanceUID")

    y = df[LABELS].to_numpy(float)
    is_gold = np.zeros(len(df), bool)
    for i, sid in enumerate(df["StudyInstanceUID"]):
        if sid in gold.index:
            row = gold.loc[sid, LABELS].to_numpy(float)
            keep = ~np.isnan(row)
            y[i][keep] = row[keep]
            is_gold[i] = True

    out = df[["StudyInstanceUID"]].copy()
    out[LABELS] = y
    out["fold"] = [community_fold(r) for r in df["Report"].fillna("")]
    out["is_gold"] = is_gold
    out["weight"] = np.where(is_gold, 3.0, 1.0)
    return out


class StudyWindows(Dataset):
    """Yields (windows[K,3,px,px], target[12], weight[12]) for one study.

    A "window" is a triplet of spatially adjacent slices stacked as RGB, which lets
    an ImageNet/DINO-pretrained backbone see local through-plane context without a
    3D architecture.
    """

    def __init__(self, cache_dir, study_ids, targets, weights, k=24, px=336,
                 train=True, seed=0):
        self.cache = Path(cache_dir)
        self.ids = list(study_ids)
        self.y = np.asarray(targets, np.float32)
        self.w = np.asarray(weights, np.float32)
        self.k, self.px, self.train = int(k), int(px), bool(train)
        self.rng = np.random.default_rng(seed)
        if len(self.ids) != len(self.y):
            raise ValueError("study id / target length mismatch")

    def __len__(self):
        return len(self.ids)

    def _centres(self, mask, depth):
        valid = np.flatnonzero(mask > 0)
        if len(valid) < 3:
            valid = np.arange(min(3, depth))
        lo, hi = int(valid.min()), int(valid.max())
        cand = [c for c in range(lo + 1, hi) if lo <= c - 1 and c + 1 <= hi]
        if not cand:
            cand = [max(1, min((lo + hi) // 2, depth - 2))]
        if self.train and len(cand) > self.k:
            # jitter which windows are seen each epoch: cheap augmentation that
            # also stops the model memorising a fixed slice grid
            return sorted(self.rng.choice(cand, self.k, replace=False))
        idx = np.linspace(0, len(cand) - 1, self.k).round().astype(int)
        return [cand[i] for i in idx]

    def __getitem__(self, i):
        import torch.nn.functional as F

        d = np.load(self.cache / f"{self.ids[i]}.npz")
        vol, mask = d["vol"], d["mask"]
        depth = vol.shape[0]
        wins = np.empty((self.k, 3, vol.shape[1], vol.shape[2]), np.float32)
        for j, c in enumerate(self._centres(mask, depth)):
            c = max(1, min(int(c), depth - 2))
            wins[j] = vol[c - 1:c + 2].astype(np.float32) / 255.0

        x = torch.from_numpy(wins)
        if x.shape[-1] != self.px:
            x = F.interpolate(x, size=(self.px, self.px), mode="bilinear", align_corners=False)
        # NO horizontal flip. It looks like free augmentation and is actively harmful
        # here: mirroring a knee swaps the medial and lateral compartments, and four of
        # the twelve findings (Medial/Lateral Meniscus, Medial/Lateral OA) are defined by
        # which compartment they sit in. The raptor author's training notebook says the
        # same thing from the other direction -- window sampling is the ONLY augmentation
        # they use. Slice jitter in _centres() is that augmentation.
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.from_numpy(self.y[i]), torch.from_numpy(self.w[i])


class KneeClassifier(nn.Module):
    """Backbone + per-finding attention pooling over windows.

    Each of the 12 findings gets its own attention distribution over the study's
    windows, because different findings are best read from different slices: a
    Baker's cyst and an ACL tear are simply not visible on the same image. Mean
    pooling would dilute the one informative window with dozens of uninformative
    ones -- a failure we measured earlier in this project.
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, n_out: int = 12, drop: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(feat_dim)
        self.att = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n_out))
        self.cls_w = nn.Parameter(torch.zeros(n_out, feat_dim))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        nn.init.trunc_normal_(self.cls_w, std=0.02)

    def encode(self, x):
        b, k = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).view(b, k, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(self, x):
        return self.head(self.encode(x))


def build_backbone(arch: str, pretrained: bool = True):
    import timm

    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = not hybrid and any(k in arch for k in ("vit", "deit", "dinov2", "dinov3", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    kw.update(dict(global_pool="token", dynamic_img_size=True) if is_vit
              else dict(global_pool="avg"))
    model = timm.create_model(arch, **kw)
    return model, model.num_features


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition metric: unweighted mean ROC AUC over the 12 findings.

    Uses the Mann-Whitney identity so this stays fast enough to call every epoch.
    Labels are thresholded at 0.5 because silver labels are probabilities.
    """
    scores = []
    for j in range(y_true.shape[1]):
        y = (y_true[:, j] >= 0.5).astype(int)
        n1 = int(y.sum())
        n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(y_pred[:, j])
        scores.append((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(scores)) if scores else float("nan")
