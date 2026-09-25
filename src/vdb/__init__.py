"""vdb — vlm-distill-bench.

Reproducible distillation and quantization benchmarks for compact
vision-language models, on procedurally generated Mini-CLEVR data.
"""

from .bench import environment, evaluate, fit
from .kd import KDLoss, argmax_match, topk_agreement
from .model import MiniVLM, ModelSpec
from .quant import quant_report
from .synth import Example, generate, split

__version__ = "0.1.0"

__all__ = ["Example", "KDLoss", "MiniVLM", "ModelSpec", "argmax_match",
           "environment", "evaluate", "fit", "generate", "quant_report", "split",
           "topk_agreement"]
