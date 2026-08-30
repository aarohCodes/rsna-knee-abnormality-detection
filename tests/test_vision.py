from unittest.mock import MagicMock, patch

import torch
from torch import nn

from rsna_knee.models.vision import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    HFSlotEncoder,
    KneeMultiLabelModel,
    SlotAttentionHead,
)


class _FakeDinov2Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _FakeDinov2Backbone(nn.Module):
    """Stands in for the real Dinov2Model's structure (embeddings / encoder.layer /
    layernorm) so freeze_except_last_n_blocks can be tested without downloading real
    weights."""

    def __init__(self, n_layers: int = 12):
        super().__init__()
        self.config = MagicMock(hidden_size=4)
        self.embeddings = nn.Linear(4, 4)
        self.encoder = MagicMock()
        self.encoder.layer = nn.ModuleList([_FakeDinov2Layer() for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(4)

    def forward(self, pixel_values):
        result = MagicMock()
        result.pooler_output = torch.zeros(pixel_values.shape[0], 4)
        return result


def test_slot_attention_head_output_shape():
    torch.manual_seed(0)
    head = SlotAttentionHead(embed_dim=16, num_slots=6, num_labels=12, attn_dim=8)
    slot_embeddings = torch.randn(4, 6, 16)
    presence_mask = torch.ones(4, 6)
    logits = head(slot_embeddings, presence_mask)
    assert logits.shape == (4, 12)
    assert torch.isfinite(logits).all()


def test_slot_attention_head_ignores_masked_slots():
    """A masked-out slot's embedding must not influence the output: replacing it with
    garbage while keeping it masked should give identical logits."""
    torch.manual_seed(0)
    head = SlotAttentionHead(embed_dim=16, num_slots=6, num_labels=12, attn_dim=8)
    head.eval()

    slot_embeddings = torch.randn(2, 6, 16)
    presence_mask = torch.ones(2, 6)
    presence_mask[:, -1] = 0  # last slot absent

    with torch.no_grad():
        logits_a = head(slot_embeddings, presence_mask)

    corrupted = slot_embeddings.clone()
    corrupted[:, -1, :] = torch.randn(2, 16) * 1000  # garbage in the masked-out slot

    with torch.no_grad():
        logits_b = head(corrupted, presence_mask)

    assert torch.allclose(logits_a, logits_b, atol=1e-5)


def test_slot_attention_head_partial_presence_varies_by_mask():
    """Different presence masks (with real signal in the varying slot) should give
    different outputs -- sanity check that masking isn't a no-op."""
    torch.manual_seed(1)
    head = SlotAttentionHead(embed_dim=16, num_slots=6, num_labels=12, attn_dim=8)
    head.eval()

    slot_embeddings = torch.randn(1, 6, 16)
    mask_all = torch.ones(1, 6)
    mask_one_missing = mask_all.clone()
    mask_one_missing[0, 0] = 0

    with torch.no_grad():
        logits_all = head(slot_embeddings, mask_all)
        logits_missing = head(slot_embeddings, mask_one_missing)

    assert not torch.allclose(logits_all, logits_missing)


def test_full_model_forward_shape_no_pretrained_download():
    torch.manual_seed(0)
    model = KneeMultiLabelModel(
        num_slots=6, num_labels=12, backbone_name="resnet18", pretrained=False
    )
    model.eval()
    images = torch.randn(2, 6, 3, 64, 64)
    presence_mask = torch.ones(2, 6)
    with torch.no_grad():
        logits = model(images, presence_mask)
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


class _FakeHFBackbone(nn.Module):
    """Stands in for a `transformers` AutoModel instance without downloading or
    loading real weights -- records the pixel_values it was called with so tests can
    verify HFSlotEncoder's normalization, and returns a pooler_output of the
    requested embedding dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = MagicMock(hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.received_pixel_values: torch.Tensor | None = None

    def forward(self, pixel_values: torch.Tensor):
        self.received_pixel_values = pixel_values
        result = MagicMock()
        result.pooler_output = torch.zeros(pixel_values.shape[0], self.hidden_size)
        return result


def test_hf_slot_encoder_applies_imagenet_normalization():
    fake_backbone = _FakeHFBackbone(hidden_size=8)
    with patch("transformers.AutoModel.from_pretrained", return_value=fake_backbone):
        encoder = HFSlotEncoder(pretrained_path="fake/local/path")

    images = torch.rand(3, 3, 32, 32)  # our pipeline always produces [0, 1] images
    out = encoder(images)

    assert out.shape == (3, 8)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    expected = (images - mean) / std
    assert torch.allclose(fake_backbone.received_pixel_values, expected, atol=1e-6)


def test_hf_slot_encoder_embed_dim_from_config():
    fake_backbone = _FakeHFBackbone(hidden_size=384)  # DINOv2-small's real hidden_size
    with patch("transformers.AutoModel.from_pretrained", return_value=fake_backbone):
        encoder = HFSlotEncoder(pretrained_path="fake/local/path")
    assert encoder.embed_dim == 384


def test_full_model_uses_hf_encoder_when_path_given():
    fake_backbone = _FakeHFBackbone(hidden_size=8)
    with patch("transformers.AutoModel.from_pretrained", return_value=fake_backbone):
        model = KneeMultiLabelModel(
            num_slots=6, num_labels=12, hf_pretrained_path="fake/local/path"
        )
    model.eval()

    images = torch.rand(2, 6, 3, 32, 32)
    presence_mask = torch.ones(2, 6)
    with torch.no_grad():
        logits = model(images, presence_mask)
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()
    assert isinstance(model.encoder, HFSlotEncoder)


def _make_hf_encoder_with_fake_backbone(n_layers: int = 12) -> tuple[HFSlotEncoder, MagicMock]:
    fake_backbone = _FakeDinov2Backbone(n_layers=n_layers)
    with patch("transformers.AutoModel.from_pretrained", return_value=fake_backbone):
        encoder = HFSlotEncoder(pretrained_path="fake/local/path")
    return encoder, fake_backbone


def test_freeze_except_last_n_blocks_freezes_embeddings_and_early_layers():
    encoder, backbone = _make_hf_encoder_with_fake_backbone(n_layers=12)
    encoder.freeze_except_last_n_blocks(2)

    assert all(not p.requires_grad for p in backbone.embeddings.parameters())
    for layer in backbone.encoder.layer[:10]:
        assert all(not p.requires_grad for p in layer.parameters())
    for layer in backbone.encoder.layer[10:]:
        assert all(p.requires_grad for p in layer.parameters())
    assert all(p.requires_grad for p in backbone.layernorm.parameters())


def test_freeze_except_last_n_blocks_zero_freezes_everything_but_layernorm():
    encoder, backbone = _make_hf_encoder_with_fake_backbone(n_layers=12)
    encoder.freeze_except_last_n_blocks(0)

    assert all(not p.requires_grad for p in backbone.embeddings.parameters())
    for layer in backbone.encoder.layer:
        assert all(not p.requires_grad for p in layer.parameters())
    assert all(p.requires_grad for p in backbone.layernorm.parameters())


def test_freeze_except_last_n_blocks_n_exceeding_layer_count_unfreezes_all_layers():
    encoder, backbone = _make_hf_encoder_with_fake_backbone(n_layers=12)
    encoder.freeze_except_last_n_blocks(100)

    for layer in backbone.encoder.layer:
        assert all(p.requires_grad for p in layer.parameters())


def test_discriminative_param_groups_excludes_frozen_params():
    fake_backbone = _FakeDinov2Backbone(n_layers=12)
    with patch("transformers.AutoModel.from_pretrained", return_value=fake_backbone):
        model = KneeMultiLabelModel(num_slots=6, num_labels=12, hf_pretrained_path="fake/path")
    model.encoder.freeze_except_last_n_blocks(2)

    groups = model.discriminative_param_groups(encoder_lr=1e-5, head_lr=1e-3)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["lr"] == 1e-3

    encoder_group_params = set(id(p) for p in groups[0]["params"])
    # every param in the encoder's low-LR group must require grad (frozen ones excluded)
    assert all(p.requires_grad for p in groups[0]["params"])
    # the frozen embeddings params must NOT be in the optimizer's param groups at all
    frozen_ids = set(id(p) for p in fake_backbone.embeddings.parameters())
    assert frozen_ids.isdisjoint(encoder_group_params)

    head_group_params = set(id(p) for p in groups[1]["params"])
    assert head_group_params == set(id(p) for p in model.head.parameters())
