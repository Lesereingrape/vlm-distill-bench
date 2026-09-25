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

The textbook loss `α·T²·KL(σ(z_s/T) ‖ σ(z_t/T)) + (1−α)·CE` invites you to tune `α`. On
this scale `T` is the axis that moves the student — and the honest starting point is that
the sentence this section used to lead with, "worse than plain CE at every α once `T > 1`",
was a memory of tuning runs rather than a measurement. So it got measured: `T ∈ {1,2,3,4}`
against `α ∈ {0.3,0.5,0.7,0.9}`, one shared teacher, one shared seed, 16 students. The
table below and every number in the sentences under it are rendered from
`results/temperature-sweep.json`; two of those sentences exist to retire the old claim.

<!-- SWEEP:START -->
*Swept by `python experiments/run_temperature_sweep.py --out results/temperature-sweep.json` and committed as that file: 4 temperatures x 4 distillation weights = 16 students, all distilled from one teacher (0.759 held-out accuracy) and evaluated on the same 800 examples of seed 0. Each cell gives accuracy / 15-bin ECE. The student trained with plain cross-entropy and no teacher at all - the number every cell is compared against - scores 0.635 / 0.064. This block is rendered from the artifact by `experiments/make_report.py`, so a re-run moves the prose along with the table.*

| accuracy / ECE | alpha=0.3 | alpha=0.5 | alpha=0.7 | alpha=0.9 | mean over alpha |
|---|---:|---:|---:|---:|---:|
| **T=1** | 0.632 / 0.047 | 0.651 / 0.043 | 0.635 / 0.058 | 0.639 / 0.048 | 0.639 / 0.049 |
| T=2 | 0.641 / 0.068 | 0.645 / 0.064 | 0.610 / 0.062 | 0.618 / 0.092 | 0.628 / 0.072 |
| T=3 | 0.590 / 0.093 | 0.591 / 0.100 | 0.608 / 0.101 | 0.605 / 0.121 | 0.598 / 0.104 |
| T=4 | 0.598 / 0.104 | 0.588 / 0.127 | 0.603 / 0.132 | 0.565 / 0.172 | 0.588 / 0.134 |

- Temperature is the axis, not alpha: the per-T means fall from 0.639 at T=1 to 0.588 at T=4, a spread of **5.1 points**, while the per-alpha means spread 1.2 points (4.2x the temperature effect for the same sweep). The two axes also differ in *kind*: ECE rises monotonically with temperature in 4 of 4 alpha columns (at alpha=0.3 it goes 0.047 -> 0.104), while accuracy is monotone in only 3 of 4. Over-smoothing reliably costs you confidence and only sometimes costs you answers.
- 4 of 16 cells beat the CE baseline, 11 fall below it and 1 ties it exactly - and where the winners sit matters: T=1 takes 2 of 4; T=2 takes 2 of 4; T=3 takes 0 of 4; T=4 takes 0 of 4. From T=3 upward nothing recovers it, at any alpha - every cell there is below 0.635 *and* above 0.064, which is the part of 'the temperature trap' that actually reproduces.
- The version of that claim which survives its own sweep is narrower than the one this file used to make. Distillation is *not* worse than CE at every alpha from T=2 upward: at T=2 the 2 alphas 0.3 and 0.5 beat CE on accuracy anyway (0.645 against 0.635). And T=1 is not unanimous in the other direction either - alpha=0.3 lands 0.25 points *below* CE there, i.e. 2 held-out examples. Across the whole grid no temperature wins at every alpha, and only T=3 and T=4 lose at every one.
- What T really costs is visible in the confidence column instead: at T=2 all but 1 of 4 cells are worse calibrated than CE, and every T=2 cell is worse calibrated than every T=1 cell (0.062 against 0.058). Softening the target buys a smoother gradient and sells the probabilities - which is why the headline run sits at T=1: the accuracy difference is inside seed noise, the calibration difference is not.
- One of these cells *is* the headline table: seed 0 of `vdb bench` runs T=1 / alpha=0.5, and the sweep reproduces that student to the last digit (0.65125 accuracy, 0.04305 ECE and all four per-family scores), which is the check that the sweep's training loop and `vdb bench` are the same code path rather than two similar ones. The best cell in the grid is that same one (T=1 / alpha=0.5, 0.651): nothing swept here beats the setting already published.
*Read it as one seed and summaries only. The cells share a teacher, a training set and a random stream, so they are paired, but this artifact stores accuracies rather than per-example outcomes, so no cell-vs-cell paired test is possible from it; one held-out example is worth 0.12 points, and the KD student's spread across the 3 seeds of the headline table is ±0.059, i.e. 5.9 points against the 3.8-point widest alpha row here. Differences of a point or two between two cells are therefore not measurements of a mechanism. The monotone ECE trend and the collapse from T=3 are the parts that survive that standard.*
<!-- SWEEP:END -->

Reproduce it (one teacher plus 16 students on CPU):

```bash
python experiments/run_temperature_sweep.py --write   # -> results/temperature-sweep.json
python experiments/make_report.py --write             # re-render this section
```

`vdb bench --temperature 4 --alpha 0.9` reproduces the worst corner of the grid, and
`vdb bench --temperature 1 --alpha 0.5` the cell the headline table publishes.

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

experiments/make_report.py          renders both README blocks from results/*.json
experiments/run_temperature_sweep.py writes the T x alpha grid the second block reads
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

1. `tests/test_readme_matches_results.py` compares **both** measured blocks byte for byte
   with `experiments/make_report.py` run against the committed `results/*.json` - the
   headline table against the three seed artifacts, the temperature section against the
   sweep grid. The renderer holds no measurements of its own, so a stale table fails the
   build instead of quietly shipping. The same file checks that the cell the two artifacts
   share (`vdb bench` at T=1 / alpha=0.5, seed 0) is *identical* in both: the sweep builds
   its own teacher rather than calling `bench.run`, and without that check it could drift
   into measuring a similar-but-different experiment.
2. `tests/test_readme_size_claims.py` checks the numbers the renderer does *not* write —
   the `~60s` wall-clock budget, the `3 seeds` in the heading, the `15-bin` ECE, the
   `3×32×32` image, the temperature and alpha lists the sweep paragraph recites, and the
   CI Python range — against the recorded artifacts and `src/`.
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
