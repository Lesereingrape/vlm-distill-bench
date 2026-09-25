# vlm-distill-bench

> Reproducible distillation & quantization benchmarks for compact vision-language models — **runs end-to-end on CPU in ~60 seconds, no downloads, no API keys.**

`vdb` trains a family of Mini-VLMs (CNN visual tower + embedding question tower) on **procedurally generated visual QA** (mini-CLEVR: counting, color, shape, spatial relations), then measures what model-compression techniques actually buy you: knowledge distillation, int8 dynamic quantization, calibration (ECE), and per-task-family behavior.

Every number in the results block below is rendered from the committed `results/bench-seed*.json` by `experiments/make_report.py`, and a test fails if the README drifts from those artifacts — nothing is hand-copied, and nothing is quoted from a paper. Re-running all three seeds reproduced every accuracy, ECE and diagnostic field bit for bit against the published artifacts; the only fields that moved were the latency columns, which measure wall-clock rather than arithmetic.

![ci](https://github.com/Lesereingrape/vlm-distill-bench/actions/workflows/ci.yml/badge.svg)

## Headline results (3 seeds, mean ± sample std)

<!-- RESULTS:START -->
*Every figure below is produced by `vdb bench` on CPU and committed as `results/bench-seed*.json`, one file per seed (0, 1, 2); the table and the sentences around it are rendered by `experiments/make_report.py`, and a test pins the README to that render. All 3 seeds share the same 3000-example training set, the same 800-example held-out set and T=1.0 / alpha=0.5; only the random stream differs. Numbers are mean ± **sample** standard deviation over the seeds (`statistics.stdev`, divided by n-1) - with three seeds we quote the wider estimator, not the flattering one.*

| model | params | accuracy | ECE ↓ | count | color | shape | spatial |
|---|---:|---:|---:|---:|---:|---:|---:|
| **teacher** | 69.3k | 0.758 ± 0.004 | 0.049 | 0.99 | 0.82 | 0.56 | 0.53 |
| student, CE only | 1.9k | 0.578 ± 0.050 | 0.058 | 0.61 | 0.58 | 0.53 | 0.59 |
| student, KD | 1.9k | 0.601 ± 0.059 | 0.041 | 0.65 | 0.63 | 0.52 | 0.57 |

- The same 1.9k-parameter student, same data, same teacher: **distillation adds +2.3 points of mean accuracy and cuts calibration error 29%** (0.058 → 0.041). It wins 2 of 3 seeds and loses one by 0.6 points, so the honest reading is *better on average, not on every run*: the seed spread (±0.059 for KD against ±0.050 for CE) is the same order as the gain itself.
- int8 dynamic quantization is accuracy-neutral (0.601 → 0.598) and **latency-negative**: 1.2 to 1.5 ms per batch becomes 1.7 to 2.1 ms (+36% on the mean). At this size the conv towers dominate the clock, so quantizing the fusion head buys nothing; the technique pays only once Linear layers are the compute. Reported as a negative result because a benchmark that publishes only wins is not a benchmark.
- What the student actually copies: its argmax matches the teacher on 67.3% of held-out examples, but only 52.3% are right in *both* models. KD inherits the teacher's errors, not just its answers - which is why the two hard families still sit at 0.55 against teacher accuracy of 0.758: the ceiling is the teacher, and distillation inside a 1.9k-parameter student cannot reach past it.
<!-- RESULTS:END -->

## The temperature trap (what tuning taught us)

The textbook loss `α·T²·KL(σ(z_s/T) ‖ σ(z_t/T)) + (1−α)·CE` made the student **worse than plain CE at every α** when `T ∈ {2,3,4}` on this scale — the `T²` gradient compensation is calibrated for logits trained from scratch, and here the softened target simply drowns the learning signal. At `T=1` (dark knowledge without over-smoothing) distillation consistently wins. The sweep is reproducible:

```bash
vdb bench --temperature 4 --alpha 0.9   # KD loses to CE
vdb bench --temperature 1 --alpha 0.5   # KD beats CE, calibration improves
```

## Quickstart

```bash
pip install -e .           # torch + numpy, that's it
vdb bench --out /tmp/run.json               # ~60s on CPU, seed 0, results/ untouched
vdb bench --seed 1 --out /tmp/seed1.json    # the next seed of the same table
vdb synth --n 10           # inspect the procedural dataset
python experiments/make_report.py --write   # re-render the README block from results/
```

`vdb bench` without `--out` writes `results/bench-seed0.json`, i.e. it replaces a
published artifact: the accuracies come back identical, the latency columns do not, so
re-render the README afterwards if you mean it.

## How it works

```
synth.py   deterministic scene → image(3×32×32) + tokenized question + categorical answer
model.py   MiniVLM = Conv visual tower ⊕ masked-mean question tower → fusion head; width via scale
kd.py      KDLoss(T, α) + top-k / argmax agreement (is the student copying mistakes?)
bench.py   seeded train/eval: per-family accuracy, 15-bin ECE, CPU latency, environment()
quant.py   int8 dynamic quantization report: accuracy, size, latency
cli.py     the whole experiment in one reproducible command

experiments/make_report.py   renders the README block above from results/*.json
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

## Reproducing

Four things tie this README to the code, and CI runs all four:

1. `tests/test_readme_matches_results.py` compares the results block byte for byte with
   `experiments/make_report.py` run against the committed `results/bench-seed*.json`. The
   renderer holds no measurements of its own, so a stale table fails the build instead of
   quietly shipping.
2. `tests/test_readme_size_claims.py` checks the numbers the renderer does *not* write —
   the `~60s` wall-clock budget, the `3 seeds` in the heading, the `15-bin` ECE, the
   `3×32×32` image and the CI Python range — against the recorded artifacts and `src/`.
3. Each artifact carries the `environment` it was measured under (Python, torch, numpy,
   CPU thread count) next to a `runtime_sec`. Averaging logits over a batch is a float
   reduction whose order depends on the thread count and the build, so reproducibility is
   stated as a property of that environment, not promised about your laptop.
4. `vdb bench` derives every random stream from one integer (`torch.manual_seed` plus
   `numpy.random.default_rng` for the scene generator) and writes the config it used into
   the artifact, so `--seed N` extends the published table instead of starting an
   experiment that cannot be lined up with the old one.

Those artifacts *are* a verification run: the published numbers were reproduced from
scratch into a temporary directory and compared field by field against the run that
produced the table in this file. 15 fields out of several hundred differed — exactly the
five `*_ms` columns per seed, which time a wall clock and move with machine load — while
every accuracy, ECE, per-family score and distillation diagnostic came back identical to
the last decimal, across a newer Python and torch build than the original run used. That
is the honest shape of "reproducible" here: the science is stable, the latency columns
are not, and the table above is bolded by measured mean accuracy rather than by which row
we find most interesting.

CI (GitHub Actions) runs ruff over `src`, `tests` and `experiments`, then the full pytest
matrix on Python 3.10–3.12 with CPU-only torch wheels — which is what makes the four
guards above load-bearing rather than decorative.

```bash
for n in 0 1 2; do vdb bench --seed $n --out /tmp/vlm-rerun-$n.json; done
python experiments/make_report.py            # print what the README must contain
```

## Roadmap

- [ ] Distillation on real data: swap in VQAv2-mini / CLEVR with the same metric stack
- [ ] Token-level KD for autoregressive decoders (JSD, on-policy student sampling)
- [ ] Pruning + KD interaction grid
- [ ] Shared-head ablation: does the question tower or the visual tower carry the KD gain?

## License

Apache-2.0
