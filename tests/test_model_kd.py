import pytest
import torch

from vdb.kd import KDLoss, argmax_match, topk_agreement
from vdb.model import MiniVLM, ModelSpec


def test_shapes_and_finite_logits():
    m = MiniVLM(vocab=16, n_answers=10, spec=ModelSpec(scale=0.5))
    out = m(torch.randn(4, 3, 32, 32), torch.randint(1, 16, (4, 6)))
    assert out.shape == (4, 10)
    assert torch.isfinite(out).all()


def test_padding_is_ignored_by_question_tower():
    m = MiniVLM(vocab=16, n_answers=10, spec=ModelSpec())
    img = torch.randn(1, 3, 32, 32)
    q1 = torch.tensor([[2, 3, 0, 0, 0, 0]])
    q2 = torch.tensor([[0, 0, 3, 2, 0, 0]])
    assert torch.allclose(m(img, q1), m(img, q2), atol=1e-6)


def test_kd_loss_zero_when_teacher_equals_student_uniform_case():
    kd = KDLoss(temperature=2.0, alpha=1.0)  # pure soft term
    logits = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))
    loss_same = kd(logits, logits.clone(), labels)
    loss_rand = kd(logits, torch.randn(8, 5) * 3, labels)
    assert loss_same.item() == pytest.approx(0.0, abs=1e-6)
    assert loss_rand > loss_same


def test_kd_rejects_bad_alpha():
    with pytest.raises(ValueError):
        KDLoss(alpha=1.5)


def test_diagnostics_bounds():
    s = torch.randn(16, 6)
    t = torch.randn(16, 6)
    y = s.argmax(-1)  # force full agreement scenario
    agree, both = argmax_match(s, s.clone(), y)
    assert agree == 1.0 and both == 1.0
    assert 0.0 <= topk_agreement(s, t, k=3) <= 1.0
