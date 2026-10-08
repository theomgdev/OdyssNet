import pytest
import torch
import torch.nn as nn
from odyssnet.core.network import OdyssNet


def compute_smooth_trajectory_score(step_scores, active_lengths, power=2.0):
    """
    Integrates step scores along the sequence dimension using a smooth power horizon.
    step_scores: (B, Q, K, T)
    active_lengths: (B, Q, K) integer active token counts
    """
    b, q_n, k_n, t_max = step_scores.shape
    device = step_scores.device
    t_idx = torch.arange(1, t_max + 1, device=device, dtype=step_scores.dtype).view(1, 1, 1, t_max)
    lens = active_lengths.unsqueeze(-1).clamp(min=1).to(step_scores.dtype)

    # Normalized continuous time tau = (t + 1) / T
    tau = (t_idx / lens).clamp(max=1.0)
    weights = tau.pow(power)

    # Mask out padded positions
    mask = t_idx <= lens
    weights = weights * mask

    weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    norm_weights = weights / weight_sum
    return (step_scores * norm_weights).sum(dim=-1)


def test_smooth_weights_properties():
    step_scores = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1, 1, 1, 4)
    active_lens = torch.tensor([[[4]]])
    score = compute_smooth_trajectory_score(step_scores, active_lens, power=2.0)

    # Manual weights for T=4, p=2: [1/16, 4/16, 9/16, 16/16] = [1, 4, 9, 16] / 30
    expected = (1.0 * 1 + 2.0 * 4 + 3.0 * 9 + 4.0 * 16) / 30.0
    assert torch.isclose(score, torch.tensor([[[10.0 / 3.0]]]), atol=1e-5)


def test_smooth_weights_padding_invariance():
    # Sequence of length 2 padded to length 4
    step_scores = torch.tensor([[[[1.0, 2.0, -999.0, -999.0]]]])
    active_lens = torch.tensor([[[2]]])
    score = compute_smooth_trajectory_score(step_scores, active_lens, power=2.0)

    # Manual weights for T=2, p=2: [1/4, 4/4] = [1, 4] / 5
    expected = (1.0 * 1 + 2.0 * 4) / 5.0  # 9 / 5 = 1.8
    assert torch.isclose(score, torch.tensor([[[1.8]]]), atol=1e-5)


def test_smooth_trajectory_gradient_flow():
    scores = torch.randn(2, 3, 4, 16, requires_grad=True)
    lens = torch.randint(4, 16, (2, 3, 4))
    integrated = compute_smooth_trajectory_score(scores, lens, power=2.0)
    loss = integrated.sum()
    loss.backward()
    assert scores.grad is not None
    assert (scores.grad.abs() > 0).any()
