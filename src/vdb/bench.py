"""Training loop and benchmark metrics — the core of a reproducible run.

`fit` trains a MiniVLM either with plain cross-entropy or against a frozen
teacher via `KDLoss`; `evaluate` reports overall and per-family accuracy,
expected calibration error, and CPU forward latency. All randomness flows
through `seed`, so a config produces the same table *inside the environment the
artifact records*: averaging logits over a batch is a float reduction, and its
order depends on the CPU thread count and the torch build. Outside that
environment expect the same shape, not the same digits — `environment()` is why
the README can say that instead of promising bit-exactness it cannot deliver.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .kd import KDLoss, argmax_match, topk_agreement
from .model import MiniVLM, ModelSpec
from .synth import Example, num_classes, pad_tokens, vocab_size


def to_dataset(examples: list[Example], max_len: int = 6) -> TensorDataset:
    x = torch.from_numpy(np.stack([e.image for e in examples]))
    q = torch.from_numpy(pad_tokens(examples, max_len))
    y = torch.tensor([e.answer for e in examples], dtype=torch.long)
    return TensorDataset(x, q, y)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class TrainResult:
    model: MiniVLM
    history: list[dict] = field(default_factory=list)


def fit(examples: list[Example], spec: ModelSpec, *, epochs: int = 20, lr: float = 3e-3,
        batch_size: int = 128, seed: int = 0, teacher: MiniVLM | None = None,
        kd: KDLoss | None = None, val: list[Example] | None = None) -> TrainResult:
    _seed_everything(seed)
    model = MiniVLM(vocab_size(), num_classes(), spec)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(to_dataset(examples, spec.max_len), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    kd = kd or KDLoss()
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, qb, yb in loader:
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(xb, qb)
                loss = kd(model(xb, qb), t_logits, yb)
            else:
                loss = F.cross_entropy(model(xb, qb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        row = {"epoch": epoch, "loss": total / len(examples)}
        if val is not None:
            row["val_acc"] = evaluate(model, val)["accuracy"]
        history.append(row)
    return TrainResult(model=model, history=history)


@torch.no_grad()
def logits_of(model: MiniVLM, examples: list[Example], max_len: int = 6) -> torch.Tensor:
    model.eval()
    ds = to_dataset(examples, max_len)
    out = []
    for xb, qb, _ in DataLoader(ds, batch_size=256):
        out.append(model(xb, qb))
    return torch.cat(out)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def evaluate(model: MiniVLM, examples: list[Example], max_len: int = 6) -> dict:
    model.eval()
    logits = logits_of(model, examples, max_len)
    probs = F.softmax(logits, dim=-1).numpy()
    labels = np.array([e.answer for e in examples])
    pred = probs.argmax(-1)
    acc = float((pred == labels).mean())
    per_family: dict[str, float] = {}
    fams = np.array([e.family for e in examples])
    for fam in sorted(set(fams.tolist())):
        m = fams == fam
        per_family[fam] = float((pred[m] == labels[m]).mean())
    out = {
        "accuracy": acc,
        "per_family": per_family,
        "ece": expected_calibration_error(probs, labels),
        "params": model.n_params(),
    }
    return out


def measure_latency(model: MiniVLM, n: int = 50, img_size: int = 3) -> float:
    """Mean CPU forward latency (ms) for a batch of 64."""
    model.eval()
    x = torch.randn(64, 3, 32, 32)
    q = torch.randint(1, vocab_size(), (64, model.spec.max_len))
    with torch.no_grad():
        model(x, q)  # warmup
        t0 = time.perf_counter()
        for _ in range(n):
            model(x, q)
    return (time.perf_counter() - t0) / n * 1000


@torch.no_grad()
def distill_diagnostics(student: MiniVLM, teacher: MiniVLM,
                        examples: list[Example], max_len: int = 6) -> dict:
    s = logits_of(student, examples, max_len)
    t = logits_of(teacher, examples, max_len)
    y = torch.tensor([e.answer for e in examples])
    agree, both = argmax_match(s, t, y)
    return {"topk_agreement": topk_agreement(s, t), "argmax_agreement": agree,
            "both_correct": both}


def environment() -> dict:
    """The machine a `bench` artifact came off, recorded beside the numbers."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }
