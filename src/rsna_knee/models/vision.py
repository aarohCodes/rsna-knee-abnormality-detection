"""Vision backbone + per-diagnosis attention pooling over series slots.

A study contributes up to six slot embeddings (one per canonical series slot -- see
dicom_io.SERIES_SLOTS) with a presence mask for slots the study doesn't have. Rather
than mean-pooling slots (which dilutes the one slot carrying evidence for a given
finding with several that don't), each of the 12 diagnoses gets its own learned query
that attends over the present slots -- see PLAN.md section 6 for the derivation.
"""
from __future__ import annotations

import timm
import torch
from torch import nn

NEG_INF = -1e9


class SlotAttentionHead(nn.Module):
    """Per-diagnosis masked attention pooling over slot embeddings.

    Input: slot_embeddings (B, S, D), presence_mask (B, S) with 1=present, 0=absent.
    Output: logits (B, num_labels).
    """

    def __init__(self, embed_dim: int, num_slots: int, num_labels: int, attn_dim: int = 128):
        super().__init__()
        self.slot_identity = nn.Parameter(torch.zeros(num_slots, embed_dim))
        nn.init.normal_(self.slot_identity, std=0.02)
        self.project = nn.Linear(embed_dim, attn_dim)
        self.queries = nn.Parameter(torch.zeros(num_labels, attn_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.classifiers = nn.Linear(attn_dim, num_labels)
        self.attn_dim = attn_dim

    def forward(self, slot_embeddings: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        b, s, d = slot_embeddings.shape
        h = self.project(slot_embeddings + self.slot_identity.unsqueeze(0))  # (B, S, A)

        # scores[b, o, s] = <h[b, s], queries[o]> / sqrt(A)
        scores = torch.einsum("bsa,oa->bos", h, self.queries) / (self.attn_dim**0.5)
        mask = presence_mask.unsqueeze(1).bool()  # (B, 1, S) broadcasts over labels
        scores = scores.masked_fill(~mask, NEG_INF)
        attn = torch.softmax(scores, dim=-1)  # (B, num_labels, S)
        # A study with zero present slots would produce a uniform softmax over -inf
        # (nan); callers must guarantee at least one present slot per study.

        context = torch.einsum("bos,bsa->boa", attn, h)  # (B, num_labels, A)
        logits = (context * self.classifiers.weight.unsqueeze(0)).sum(-1) + self.classifiers.bias
        return logits


class SlotEncoder(nn.Module):
    """Wraps a timm image backbone to embed one representative crop per slot."""

    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B*S, C, H, W) -> (B*S, embed_dim). Caller reshapes to (B, S, D)."""
        return self.backbone(images)


class KneeMultiLabelModel(nn.Module):
    def __init__(
        self,
        num_slots: int,
        num_labels: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.encoder = SlotEncoder(backbone_name, pretrained=pretrained)
        self.head = SlotAttentionHead(
            embed_dim=self.encoder.embed_dim,
            num_slots=num_slots,
            num_labels=num_labels,
            attn_dim=attn_dim,
        )

    def forward(self, images: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        """images: (B, S, C, H, W), presence_mask: (B, S) -> logits (B, num_labels)."""
        b, s, c, h, w = images.shape
        flat = images.view(b * s, c, h, w)
        embeddings = self.encoder(flat).view(b, s, -1)
        return self.head(embeddings, presence_mask)
