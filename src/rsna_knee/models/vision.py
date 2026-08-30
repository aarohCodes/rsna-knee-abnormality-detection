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

# Standard ImageNet normalization stats DINOv2 (and most ImageNet-pretrained ViT/CNN
# backbones) were trained with. Our own preprocessing pipeline (dicom_io/preprocess.py)
# already produces percentile-normalized [0, 1] images -- this is an additional,
# backbone-specific normalization layered on top, not a replacement for it.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class HFSlotEncoder(nn.Module):
    """Wraps a HuggingFace `transformers` vision backbone (e.g. DINOv2) loaded from a
    local path -- Kaggle's DINOv2 Model source ships in `transformers` format
    (AutoModel.from_pretrained), not timm, so it needs its own loading path. Applies
    ImageNet normalization on top of our own [0, 1] preprocessing, matching what these
    backbones were trained with (see preprocessor_config.json on the model source).
    """

    def __init__(self, pretrained_path: str):
        super().__init__()
        from transformers import AutoModel  # local import: only needed for this path

        self.backbone = AutoModel.from_pretrained(pretrained_path)
        self.embed_dim = self.backbone.config.hidden_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B*S, C, H, W) in [0, 1] -> (B*S, embed_dim)."""
        normalized = (images - self.mean) / self.std
        return self.backbone(pixel_values=normalized).pooler_output

    def freeze_except_last_n_blocks(self, n: int) -> None:
        """Freeze the patch/position embeddings and all but the last `n` transformer
        blocks, per PLAN.md's rationale: a frozen self-supervised encoder has no
        vocabulary for MRI signal, but early ViT blocks are generic edge/texture
        filters where there's little supervision here to safely improve them, while
        late blocks carry the semantics worth adapting. `layernorm` (which directly
        feeds pooler_output) stays trainable regardless of `n`.
        """
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        layers = self.backbone.encoder.layer
        n_frozen = max(0, len(layers) - n)
        for layer in layers[:n_frozen]:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in layers[n_frozen:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True


class KneeMultiLabelModel(nn.Module):
    def __init__(
        self,
        num_slots: int,
        num_labels: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        attn_dim: int = 128,
        hf_pretrained_path: str | None = None,
    ):
        """If `hf_pretrained_path` is set, uses HFSlotEncoder (a local HuggingFace
        `transformers` checkpoint, e.g. a mounted Kaggle DINOv2 Model source) and
        ignores `backbone_name`/`pretrained`; otherwise uses the timm-based SlotEncoder.
        """
        super().__init__()
        if hf_pretrained_path is not None:
            self.encoder = HFSlotEncoder(hf_pretrained_path)
        else:
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

    def discriminative_param_groups(self, encoder_lr: float, head_lr: float) -> list[dict]:
        """Two optimizer param groups: a low LR for the (partially unfrozen) encoder,
        a higher LR for the randomly-initialized head. Per PLAN.md: the head has
        everything to learn from a random start, while the encoder starts from a good
        solution and only needs to be nudged off it -- one shared LR would either
        leave the head undertrained or wreck the encoder in the first few steps.
        Frozen encoder parameters (requires_grad=False) are excluded entirely, since
        handing them to the optimizer would be dead weight.
        """
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        return [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]
