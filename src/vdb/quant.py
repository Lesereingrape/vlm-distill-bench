"""Post-training quantization probe.

Dynamic int8 quantization of all Linear layers — the same recipe vLLM-style
serving stacks apply to small language towers — with accuracy and latency
comparison against the fp32 original.
"""

from __future__ import annotations

import copy

import torch

from .bench import evaluate
from .model import MiniVLM
from .synth import Example


def quantize_student(model: MiniVLM) -> MiniVLM:
    """Replace every Linear with a quantization-inference equivalent."""
    from torch.ao.quantization import quantize_dynamic

    q = quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
    q.spec = model.spec  # type: ignore[attr-defined]
    return q  # type: ignore[return-value]


def quant_report(model: MiniVLM, examples: list[Example], max_len: int = 6) -> dict:
    import io

    from .bench import measure_latency

    q_model = quantize_student(model)
    buf_fp = io.BytesIO()
    torch.save(model.state_dict(), buf_fp)
    buf_q = io.BytesIO()
    torch.save(q_model.state_dict(), buf_q)
    fp_eval = evaluate(model, examples, max_len)
    q_eval = evaluate(q_model, examples, max_len)
    return {
        "fp32_accuracy": fp_eval["accuracy"],
        "int8_accuracy": q_eval["accuracy"],
        "fp32_mb": buf_fp.getbuffer().nbytes / 1e6,
        "int8_mb": buf_q.getbuffer().nbytes / 1e6,
        "compression": buf_fp.getbuffer().nbytes / max(buf_q.getbuffer().nbytes, 1),
        "fp32_ms": measure_latency(model),
        "int8_ms": measure_latency(q_model),
    }
