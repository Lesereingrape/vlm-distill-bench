"""Distillation objectives for categorical-answer VLMs.

`KDLoss` mixes hard cross-entropy with temperature-scaled KL against the
teacher's logits (Hinton et al., 2015). `topk_agreement` is the diagnostic
that answers the question reviewers always ask: is the student copying the
teacher's mistakes, or only its decisions?
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class KDLoss:
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.temperature = temperature
        self.alpha = alpha

    def __call__(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 labels: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        soft = F.kl_div(
            F.log_softmax(student_logits / t, dim=-1),
            F.softmax(teacher_logits / t, dim=-1),
            reduction="batchmean",
        ) * (t * t)
        hard = F.cross_entropy(student_logits, labels)
        return self.alpha * soft + (1 - self.alpha) * hard


def topk_agreement(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                   k: int = 3) -> float:
    with torch.no_grad():
        s = student_logits.topk(min(k, student_logits.shape[-1]), dim=-1).indices
        t = teacher_logits.topk(min(k, teacher_logits.shape[-1]), dim=-1).indices
        hits = sum(len(set(sr) & set(tr))
                   for sr, tr in zip(s.tolist(), t.tolist(), strict=True))
        total = s.shape[0] * min(k, student_logits.shape[-1])
        return hits / total


def argmax_match(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 labels: torch.Tensor) -> tuple[float, float]:
    """(student-vs-teacher agreement, both correctness) — mistake-transfer probe."""
    with torch.no_grad():
        sa, ta = student_logits.argmax(-1), teacher_logits.argmax(-1)
        agree = (sa == ta).float().mean().item()
        both_right = ((sa == labels) & (ta == labels)).float().mean().item()
        return agree, both_right
