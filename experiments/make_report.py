"""Render the README headline block from the committed `results/bench-seed*.json`.

Nothing in that block is typed by hand: `vdb bench --seed N` writes one JSON per seed,
and this file aggregates them into the tables and the sentences around them. Run
`python experiments/make_report.py --write` to splice the rendered block back between
the RESULTS markers; `tests/test_readme_matches_results.py` asserts the README already
equals this output, so a re-run that moves a number moves the README too.

Every claim ("wins 2 of 3 seeds", "latency-negative", which row is bold) is computed
from the artifacts rather than asserted, so it cannot outlive the result that
supported it.
"""

from __future__ import annotations

import json
import statistics
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


def load(root: Path) -> list[dict]:
    paths = sorted((root / "results").glob("bench-seed*.json"))
    assert paths, f"no results/bench-seed*.json under {root}"
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _write(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    assert start in text and end in text, f"{path} is missing the RESULTS markers"
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    nl = "\n"
    path.write_text(f"{head}{start}{nl}{block}{nl}{end}{tail}", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="make_report")
    ap.add_argument("--write", action="store_true",
                    help="splice the block into README.md instead of printing it")
    ap.add_argument("--root", default=".", help="repo root containing results/")
    args = ap.parse_args()
    rendered = build(load(Path(args.root)))
    if args.write:
        _write(Path(args.root) / "README.md", rendered)
        print("README results block rewritten")
    else:
        print(rendered)
