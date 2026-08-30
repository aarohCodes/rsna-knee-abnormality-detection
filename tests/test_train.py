import torch

from rsna_knee.train import weighted_bce_loss


def test_zero_weight_elements_dont_affect_loss():
    logits = torch.tensor([[10.0, -10.0]])  # confidently wrong on both if target flips
    target_a = torch.tensor([[0.0, 1.0]])
    weight_a = torch.tensor([[1.0, 0.0]])  # second element doesn't count

    target_b = target_a.clone()
    target_b[0, 1] = 0.0  # flip the zero-weight element's target
    weight_b = weight_a.clone()

    loss_a = weighted_bce_loss(logits, target_a, weight_a)
    loss_b = weighted_bce_loss(logits, target_b, weight_b)
    assert torch.allclose(loss_a, loss_b)


def test_all_zero_weight_returns_finite_not_nan():
    logits = torch.randn(3, 12)
    target = torch.rand(3, 12)
    weight = torch.zeros(3, 12)
    loss = weighted_bce_loss(logits, target, weight)
    assert torch.isfinite(loss)


def test_higher_weight_dominates_the_average():
    # two elements, one heavily weighted and correct, one lightly weighted and wrong
    logits = torch.tensor([[10.0, 10.0]])
    target = torch.tensor([[1.0, 0.0]])
    weight = torch.tensor([[100.0, 0.01]])
    loss = weighted_bce_loss(logits, target, weight)
    # should be close to the loss of the heavily-weighted (correct) element alone
    correct_only = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([10.0]), torch.tensor([1.0])
    )
    assert abs(loss.item() - correct_only.item()) < 0.01
