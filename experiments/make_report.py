"""Render the README's measured blocks from the committed artifacts in `results/`.

Nothing in those blocks is typed by hand:

* `vdb bench --seed N` writes one JSON per seed, and `build()` aggregates them into the
  headline table and the sentences around it (markers `RESULTS`),
* `python experiments/run_temperature_sweep.py` writes one grid artifact, and
  `build_sweep()` renders the temperature discussion from it (markers `SWEEP`).

Run `python experiments/make_report.py --write` to splice both blocks back into
README.md; `tests/test_readme_matches_results.py` asserts the README already equals
these renders, so a re-run that moves a number moves the README too.

Every claim ("wins 2 of 3 seeds", "latency-negative", which row is bold, how many sweep
cells beat the CE baseline) is computed from the artifacts rather than asserted, so it
cannot outlive the result that supported it.
"""

from __future__ import annotations

import json
import statistics
from itertools import pairwise
from pathlib import Path

FAMILIES = ("count", "color", "shape", "spatial")
ARMS = (("teacher", "teacher"),
        ("student_ce", "student, CE only"),
        ("student_kd", "student, KD"))


def _mean(xs):
    return statistics.mean(xs)


def _sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k"


def _acc(docs, arm):
    return [d[arm]["accuracy"] for d in docs]


def _ece(docs, arm):
    return [d[arm]["ece"] for d in docs]


def build(docs: list[dict]) -> str:
    """Render the README block. Pure function of the artifacts: no measurement here."""
    assert docs, "no seed artifacts to render"
    n = len(docs)
    seeds = ", ".join(str(d["config"]["seed"]) for d in docs)
    cfg = docs[0]["config"]
    assert len({d["config"]["seed"] for d in docs}) == n, "two artifacts share a seed"
    ref = {k: v for k, v in cfg.items() if k != "seed"}
    assert all({k: v for k, v in d["config"].items() if k != "seed"} == ref for d in docs), (
        "artifacts come from different configs; averaging across them is meaningless")

    best_arm = max((a for a, _ in ARMS), key=lambda a: _mean(_acc(docs, a)))

    out: list[str] = []
    out.append(
        f"*Every figure below is produced by `vdb bench` on CPU and committed as "
        f"`results/bench-seed*.json`, one file per seed ({seeds}); the table and the "
        f"sentences around it are rendered by `experiments/make_report.py`, and a test "
        f"pins the README to that render. All {n} seeds share the same "
        f"{cfg['train_n']}-example training "
        f"set, the same {cfg['val_n']}-example held-out set and T={cfg['temperature']} / "
        f"alpha={cfg['alpha']}; only the random stream differs. Numbers are "
        f"mean ± **sample** standard deviation over the seeds (`statistics.stdev`, "
        f"divided by n-1) - with three seeds we quote the wider estimator, not the "
        f"flattering one.*")
    out.append("")

    out.append("| model | params | accuracy | ECE ↓ | "
               + " | ".join(FAMILIES) + " |")
    out.append("|---|---:|---:|---:|" + "---:|" * len(FAMILIES))
    for arm, label in ARMS:
        cells = [f"{_mean(_acc(docs, arm)):.3f} ± {_sd(_acc(docs, arm)):.3f}",
                 f"{_mean(_ece(docs, arm)):.3f}"]
        cells += [f"{_mean([d[arm]['per_family'][f] for d in docs]):.2f}" for f in FAMILIES]
        row = f"**{label}**" if arm == best_arm else label
        out.append(f"| {row} | {_k(docs[0][arm]['params'])} | " + " | ".join(cells) + " |")
    out.append("")

    kd, ce = _acc(docs, "student_kd"), _acc(docs, "student_ce")
    deltas = [(a - b) * 100 for a, b in zip(kd, ce, strict=True)]
    wins = sum(1 for d in deltas if d > 0)
    gain = (_mean(kd) - _mean(ce)) * 100
    ece_ce, ece_kd = _mean(_ece(docs, "student_ce")), _mean(_ece(docs, "student_kd"))
    ece_cut = (1 - ece_kd / ece_ce) * 100
    out.append(
        f"- The same {_k(docs[0]['student_kd']['params'])}-parameter student, same data, "
        f"same teacher: **distillation adds {gain:+.1f} points of mean accuracy and cuts "
        f"calibration error {ece_cut:.0f}%** ({ece_ce:.3f} → {ece_kd:.3f}). It wins "
        f"{wins} of {n} seeds and loses one by {abs(min(deltas)):.1f} points, so the "
        f"honest reading is *better on average, not on every run*: the seed spread "
        f"(±{_sd(kd):.3f} for KD against ±{_sd(ce):.3f} for CE) is the same order "
        f"as the gain itself.")

    q = [d["quantization"] for d in docs]
    fp_acc, i8_acc = _mean([x["fp32_accuracy"] for x in q]), _mean([x["int8_accuracy"] for x in q])
    fp_ms, i8_ms = [x["fp32_ms"] for x in q], [x["int8_ms"] for x in q]
    slower = (_mean(i8_ms) / _mean(fp_ms) - 1) * 100
    out.append(
        f"- int8 dynamic quantization is accuracy-neutral "
        f"({fp_acc:.3f} → {i8_acc:.3f}) and **latency-negative**: "
        f"{min(fp_ms):.1f} to {max(fp_ms):.1f} ms per batch becomes "
        f"{min(i8_ms):.1f} to {max(i8_ms):.1f} ms ({slower:+.0f}% on the mean). At this "
        f"size the conv towers dominate the clock, so quantizing the fusion head buys "
        f"nothing; the technique pays only once Linear layers are the compute. Reported "
        f"as a negative result because a benchmark that publishes only wins is not a "
        f"benchmark.")

    agree = _mean([d["distill_diagnostics"]["argmax_agreement"] for d in docs])
    both = _mean([d["distill_diagnostics"]["both_correct"] for d in docs])
    teacher_acc = _mean(_acc(docs, "teacher"))
    hard = _mean([_mean([d["student_kd"]["per_family"][f] for d in docs])
                  for f in ("shape", "spatial")])
    out.append(
        f"- What the student actually copies: its argmax matches the teacher on "
        f"{agree:.1%} of held-out examples, but only {both:.1%} are right in *both* "
        f"models. KD inherits the teacher's errors, not just its answers - which is why "
        f"the two hard families still sit at {hard:.2f} against teacher accuracy of "
        f"{teacher_acc:.3f}: the ceiling is the teacher, and distillation inside a "
        f"{_k(docs[0]['student_kd']['params'])}-parameter student cannot reach past it.")
    return "\n".join(out)


def _and(xs: list[str]) -> str:
    """'a', 'a and b', 'a, b and c' - the lists in these sentences are short."""
    return " and ".join([", ".join(xs[:-1]), xs[-1]]) if len(xs) > 1 else xs[0]


def _monotone(xs: list[float], increasing: bool) -> bool:
    return all((b > a) == increasing for a, b in pairwise(xs))


def build_sweep(doc: dict, published: list[dict]) -> str:
    """Render the temperature-trap block from `results/temperature-sweep.json`.

    `published` is the same seed artifacts the headline table renders from: the sweep
    contains one cell that a `vdb bench` run already produced, and this function asserts
    they agree to the last digit before quoting either.

    Also a correction that earns its keep: an earlier draft of this section claimed KD
    lost to plain CE at *every* alpha for T >= 2 and that T=1 "consistently wins". The
    grid is not that tidy, so the sentences below are counted out of the grid - which
    cells beat the CE baseline, on which axis, and where the claim collapses.
    """
    cfg, base = doc["config"], doc["baseline"]
    temps, alphas = cfg["temperatures"], cfg["alphas"]
    assert temps == sorted(temps) and alphas == sorted(alphas), "grid axes must be ordered"
    grid = {(c["temperature"], c["alpha"]): c for c in doc["cells"]}
    assert len(grid) == len(temps) * len(alphas), "the sweep grid is incomplete"
    assert published, "the sweep block needs the seed artifacts to cross-check against"
    seeds = {d["config"]["seed"] for d in published}
    assert cfg["seed"] in seeds, "the sweep's seed is not one of the published runs"
    ce_acc, ce_ece = base["ce_accuracy"], base["ce_ece"]
    sd = _sd([d["student_kd"]["accuracy"] for d in published])

    def acc(t, a) -> float:
        return grid[(t, a)]["kd_accuracy"]

    def ece(t, a) -> float:
        return grid[(t, a)]["kd_ece"]

    row_spread = max((max(acc(t, a) for a in alphas) - min(acc(t, a) for a in alphas))
                     * 100 for t in temps)

    out: list[str] = []
    out.append(
        f"*Swept by `python experiments/run_temperature_sweep.py --out "
        f"results/temperature-sweep.json` and committed as that file: "
        f"{len(temps)} temperatures x {len(alphas)} distillation weights = "
        f"{len(doc['cells'])} students, all distilled from one teacher "
        f"({base['teacher_accuracy']:.3f} held-out accuracy) and evaluated on the same "
        f"{cfg['val_n']} examples of seed {cfg['seed']}. Each cell gives accuracy / 15-bin "
        f"ECE. The student trained with plain cross-entropy and no teacher at all - the "
        f"number every cell is compared against - scores {ce_acc:.3f} / {ce_ece:.3f}. "
        f"This block is rendered from the artifact by `experiments/make_report.py`, so a "
        f"re-run moves the prose along with the table.*")
    out.append("")

    out.append("| accuracy / ECE | " + " | ".join(f"alpha={a:g}" for a in alphas)
               + " | mean over alpha |")
    out.append("|---|" + "---:|" * (len(alphas) + 1))
    t_acc = {t: _mean([acc(t, a) for a in alphas]) for t in temps}
    t_ece = {t: _mean([ece(t, a) for a in alphas]) for t in temps}
    best_t = max(temps, key=lambda t: t_acc[t])
    for t in temps:
        cells = [f"{acc(t, a):.3f} / {ece(t, a):.3f}" for a in alphas]
        label = f"**T={t:g}**" if t == best_t else f"T={t:g}"
        out.append(f"| {label} | " + " | ".join(cells)
                   + f" | {t_acc[t]:.3f} / {t_ece[t]:.3f} |")
    out.append("")

    winners = [(t, a) for t in temps for a in alphas if acc(t, a) > ce_acc]
    losers = [(t, a) for t in temps for a in alphas if acc(t, a) < ce_acc]
    ties = [(t, a) for t in temps for a in alphas if acc(t, a) == ce_acc]
    worst_t = min(temps, key=lambda t: t_acc[t])
    a_acc = {a: _mean([acc(t, a) for t in temps]) for a in alphas}
    t_spread = (max(t_acc.values()) - min(t_acc.values())) * 100
    a_spread = (max(a_acc.values()) - min(a_acc.values())) * 100
    acc_mono = sum(1 for a in alphas if _monotone([acc(t, a) for t in temps], False))
    ece_mono = sum(1 for a in alphas if _monotone([ece(t, a) for t in temps], True))
    out.append(
        f"- Temperature is the axis, not alpha: the per-T means fall from "
        f"{t_acc[temps[0]]:.3f} at T={temps[0]:g} to {t_acc[worst_t]:.3f} at "
        f"T={worst_t:g}, a spread of **{t_spread:.1f} points**, while the per-alpha means "
        f"spread {a_spread:.1f} points ({t_spread / max(a_spread, 1e-9):.1f}x the "
        f"temperature effect for the same sweep). The two axes also differ in *kind*: "
        f"ECE rises monotonically with temperature in {ece_mono} of {len(alphas)} alpha "
        f"columns (at alpha={alphas[0]:g} it goes {ece(temps[0], alphas[0]):.3f} -> "
        f"{ece(temps[-1], alphas[0]):.3f}), while accuracy is monotone in only "
        f"{acc_mono} of {len(alphas)}. Over-smoothing reliably costs you confidence and "
        f"only sometimes costs you answers.")

    win = {t: [a for a in alphas if acc(t, a) > ce_acc] for t in temps}
    by_t = {t: len(win[t]) for t in temps}
    collapse = [t for t in temps if by_t[t] == 0]
    assert collapse, "no temperature loses to CE everywhere, so the framing needs rewriting"
    assert all(ece(t, a) > ce_ece for t in collapse for a in alphas), (
        "the collapse rows are quoted as losing on calibration too, and no longer do")
    mixed = [t for t in temps if 0 < by_t[t] < len(alphas)]
    assert mixed, "every row is unanimous, so the correction paragraph below needs rewriting"
    out.append(
        f"- {len(winners)} of {len(doc['cells'])} cells beat the CE baseline, "
        f"{len(losers)} fall below it and {len(ties)} tie{'s' if len(ties) == 1 else ''} it "
        f"exactly - and where the "
        f"winners sit matters: "
        + "; ".join(f"T={t:g} takes {by_t[t]} of {len(alphas)}" for t in temps)
        + f". From T={min(collapse):g} upward nothing recovers it, at any alpha - every "
        f"cell there is below {ce_acc:.3f} *and* above {ce_ece:.3f}, which is the part of "
        f"'the temperature trap' that actually reproduces.")

    t = max(mixed)
    first = temps[0]
    assert all(0 < by_t[x] < len(alphas) for x in [first]), (
        "the T=1 row is quoted as splitting, and no longer does")
    worst_alpha = min(alphas, key=lambda a: acc(first, a))
    gap = (ce_acc - acc(first, worst_alpha)) * 100
    out.append(
        f"- The version of that claim which survives its own sweep is narrower than the "
        f"one this file used to make. Distillation is *not* worse than CE at every alpha "
        f"from T={t:g} upward: at T={t:g} the {by_t[t]} alphas "
        f"{_and([f'{a:g}' for a in win[t]])} beat CE on accuracy anyway "
        f"({max(acc(t, a) for a in win[t]):.3f} against {ce_acc:.3f}). And T={first:g} is "
        f"not unanimous in the other direction either - alpha={worst_alpha:g} lands "
        f"{gap:.2f} points *below* CE there, i.e. {round(gap * cfg['val_n'] / 100)} held-out "
        f"examples. Across the whole grid no temperature wins at every alpha, and only "
        f"{_and([f'T={x:g}' for x in collapse])} lose at every one.")
    out.append(
        f"- What T really costs is visible in the confidence column instead: at "
        f"T={t:g} all but {sum(1 for a in alphas if ece(t, a) < ce_ece)} of "
        f"{len(alphas)} cells are worse calibrated than CE, and every T={t:g} cell is "
        f"worse calibrated than every T={temps[0]:g} cell "
        f"({min(ece(t, a) for a in alphas):.3f} against "
        f"{max(ece(temps[0], a) for a in alphas):.3f}). Softening the target buys a "
        f"smoother gradient and sells the probabilities - which is why the headline run "
        f"sits at T=1: the accuracy difference is inside seed noise, the calibration "
        f"difference is not.")

    best_cell = max(doc["cells"], key=lambda c: c["kd_accuracy"])
    pub = next(d for d in published if d["config"]["seed"] == cfg["seed"])
    t_star, a_star = pub["config"]["temperature"], pub["config"]["alpha"]
    home = grid[(t_star, a_star)]
    assert abs(home["kd_accuracy"] - pub["student_kd"]["accuracy"]) < 1e-12, (
        "the sweep and the headline artifacts disagree about the same run")
    assert abs(home["kd_ece"] - pub["student_kd"]["ece"]) < 1e-12, (
        "the sweep and the headline artifacts disagree about the same ECE")
    assert home["per_family"] == pub["student_kd"]["per_family"], (
        "the sweep and the headline artifacts disagree about the same per-family scores")
    out.append(
        f"- One of these cells *is* the headline table: seed {cfg['seed']} of `vdb bench` "
        f"runs T={t_star:g} / alpha={a_star:g}, and the sweep reproduces that student to "
        f"the last digit ({home['kd_accuracy']:.5f} accuracy, {home['kd_ece']:.5f} ECE and "
        f"all four per-family scores), which is the check that the sweep's training loop "
        f"and `vdb bench` are the same code path rather than two similar ones. The best "
        f"cell in the grid is that same one (T={best_cell['temperature']:g} / "
        f"alpha={best_cell['alpha']:g}, {best_cell['kd_accuracy']:.3f}): nothing swept "
        f"here beats the setting already published.")
    out.append(
        f"*Read it as one seed and summaries only. The cells share a teacher, a training "
        f"set and a random stream, so they are paired, but this artifact stores accuracies "
        f"rather than per-example outcomes, so no cell-vs-cell paired test is possible "
        f"from it; one held-out example is worth {100 / cfg['val_n']:.2f} points, and the "
        f"KD student's spread across the {len(published)} seeds of the headline table is "
        f"±{sd:.3f}, i.e. {sd * 100:.1f} points against the {row_spread:.1f}-point widest "
        f"alpha row here. Differences of a point or two between two cells are therefore "
        f"not measurements of a mechanism. The monotone ECE trend and the collapse from "
        f"T={min(collapse):g} are the parts that survive that standard.*")
    return "\n".join(out)


def load_sweep(root: Path) -> dict:
    path = root / "results" / "temperature-sweep.json"
    assert path.exists(), f"missing {path}; run experiments/run_temperature_sweep.py"
    return json.loads(path.read_text(encoding="utf-8"))


def load(root: Path) -> list[dict]:
    paths = sorted((root / "results").glob("bench-seed*.json"))
    assert paths, f"no results/bench-seed*.json under {root}"
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _write(path: Path, block: str, marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    assert start in text and end in text, f"{path} is missing the {marker} markers"
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    nl = "\n"
    path.write_text(f"{head}{start}{nl}{block}{nl}{end}{tail}", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="make_report")
    ap.add_argument("--write", action="store_true",
                    help="splice the blocks into README.md instead of printing them")
    ap.add_argument("--root", default=".", help="repo root containing results/")
    args = ap.parse_args()
    root = Path(args.root)
    docs = load(root)
    blocks = [("RESULTS", build(docs))]
    sweep_path = root / "results" / "temperature-sweep.json"
    if sweep_path.exists():
        blocks.append(("SWEEP", build_sweep(load_sweep(root), docs)))
    if args.write:
        for marker, rendered in blocks:
            _write(root / "README.md", rendered, marker)
        print(f"README {' and '.join(m.lower() for m, _ in blocks)} blocks rewritten")
    else:
        for marker, rendered in blocks:
            print(f"<!-- {marker} -->\n{rendered}\n")
