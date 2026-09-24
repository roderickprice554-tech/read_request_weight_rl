import math

import torch


def compute_opd_advantage_weights(
    trajectory_advantage: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    eps: float = 0.2,
    weight_lambda: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Convert detached same-token OPD deltas into bounded advantage weights."""
    if not 0.0 < eps < 1.0:
        raise ValueError("OPD weight clipping requires 0 < eps < 1")
    if weight_lambda < 0.0:
        raise ValueError("OPD weight_lambda must be non-negative")
    expected_shape = trajectory_advantage.shape
    if teacher_logprobs.shape != expected_shape or student_logprobs.shape != expected_shape:
        raise ValueError("advantages and teacher/student log-probabilities must have equal shapes")
    if response_mask.shape != expected_shape:
        raise ValueError("response_mask must match the token advantage shape")

    with torch.no_grad():
        base_advantage = trajectory_advantage.detach().float()
        opd_delta = teacher_logprobs.detach().float() - student_logprobs.detach().float()
        aligned_delta = torch.sign(base_advantage) * opd_delta
        clipped_delta = torch.clamp(
            aligned_delta,
            min=math.log1p(-eps),
            max=math.log1p(eps),
        )
        weight = 1.0 + weight_lambda * (torch.exp(clipped_delta) - 1.0)
        valid_mask = response_mask.bool()
        weight = torch.where(valid_mask, weight, torch.ones_like(weight))
        token_advantage = base_advantage * weight

    return {
        "opd_delta": opd_delta,
        "aligned_delta": aligned_delta,
        "clipped_delta": clipped_delta,
        "opd_weight": weight,
        "token_advantage": token_advantage,
    }
