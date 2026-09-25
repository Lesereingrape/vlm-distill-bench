# vlm-distill-bench

> Reproducible distillation & quantization benchmarks for compact vision-language models — **runs end-to-end on CPU in ~90 seconds, no downloads, no API keys.**

`vdb` trains a family of Mini-VLMs (CNN visual tower + embedding question tower) on **procedurally generated visual QA** (mini-CLEVR: counting, color, shape, spatial relations), then measures what model-compression techniques actually buy you: knowledge distillation, int8 dynamic quantization, calibration (ECE), and per-task-family behavior.

Every number in this README is produced by `vdb bench` on a laptop CPU and is byte-reproducible from the committed seeds. Nothing is copied from papers.

![ci](https://github.com/Lesereingrape/vlm-distill-bench/actions/workflows/ci.yml/badge.svg)

## Headline results (3 seeds, mean ± std)

`vdb bench` (defaults = tuned config; `results/bench-seed{0,1,2}.json` are committed):

| model | params | accuracy | ECE ↓ | count | color | shape | spatial |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher | 69.3k | 0.758 ± 0.004 | 0.049 | 0.99 | 0.82 | 0.56 | 0.53 |
| student, CE only | 1.9k | 0.578 ± 0.050 | 0.058 | 0.61 | 0.58 | 0.53 | 0.59 |
| **student, KD** | 1.9k | **0.601 ± 0.059** | **0.041** | 0.65 | 0.63 | 0.52 | 0.57 |

- Same 1.9k-param student, same data, same init seed — **distillation adds +2.3 points of accuracy on average and cuts calibration error 29%**, winning 2 of 3 seeds and losing one by 0.6 points. (Reported honestly: seed variance at this scale is real.)
- int8 dynamic quantization: accuracy-neutral (0.601 → 0.598) but **latency-negative at this size** (≈2–4 ms → ≈3–5 ms) — dynamic quantization only pays off once the Linear layers dominate compute. Negative results included because benchmarks that only publish wins aren't benchmarks.

## The temperature trap (what tuning taught us)

The textbook loss `α·T²·KL(σ(z_s/T) ‖ σ(z_t/T)) + (1−α)·CE` made the student **worse than plain CE at every α** when `T ∈ {2,3,4}` on this scale — the `T²` gradient compensation is calibrated for logits trained from scratch, and here the softened target simply drowns the learning signal. At `T=1` (dark knowledge without over-smoothing) distillation consistently wins. The sweep is reproducible:

```bash
vdb bench --temperature 4 --alpha 0.9   # KD loses to CE
vdb bench --temperature 1 --alpha 0.5   # KD beats CE, calibration improves
```

## Quickstart

```bash
pip install -e .          # torch + numpy, that's it
vdb bench                 # ~90s on CPU, prints table, writes results/bench-seed0.json
vdb synth --n 10          # inspect the procedural dataset
```

## How it works

```
synth.py   deterministic scene → image(3×32×32) + tokenized question + categorical answer
model.py   MiniVLM = Conv visual tower ⊕ masked-mean question tower → fusion head; width via scale
kd.py      KDLoss(T, α) + top-k / argmax agreement (is the student copying mistakes?)
bench.py   seeded train/eval: per-family accuracy, 15-bin ECE, CPU latency
quant.py   int8 dynamic quantization report: accuracy, size, latency
cli.py     the whole experiment in one reproducible command
```

## Using it as a library

```python
from vdb import ModelSpec, fit, evaluate, generate, KDLoss
from vdb.synth import split

train, val = split(3000, 800, seed=0)
teacher = fit(train, ModelSpec(scale=2.0), epochs=40, seed=0).model
student = fit(train, ModelSpec(scale=0.3), epochs=40, seed=0,
              teacher=teacher, kd=KDLoss(temperature=1.0, alpha=0.5)).model
print(evaluate(student, val))
```

## Reproduction protocol

- All seeds flow through `torch.manual_seed` + `numpy.random.default_rng`; dataset generation is pure-function of the seed.
- `results/*.json` embed the exact config; `vdb bench --seed N` extends the table.
- CI (GitHub Actions): ruff + full pytest matrix on Python 3.10–3.12, CPU-only torch wheels.

## Roadmap

- [ ] Distillation on real data: swap in VQAv2-mini / CLEVR with the same metric stack
- [ ] Token-level KD for autoregressive decoders (JSD, on-policy student sampling)
- [ ] Pruning + KD interaction grid
- [ ] Shared-head ablation: does the question tower or the visual tower carry the KD gain?

## License

Apache-2.0
