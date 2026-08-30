import torch

from rsna_knee.models.vision import KneeMultiLabelModel, SlotAttentionHead


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
