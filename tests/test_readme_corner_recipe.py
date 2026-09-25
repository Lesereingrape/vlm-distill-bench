"""`vdb bench --temperature 4 --alpha 0.9` is a command a reader will run; check what it prints.

The sweep artifact holds 16 cells, and `test_readme_matches_results.py` proves the sweep's
loop and `vdb bench` are one code path at the cell they share (T=1/alpha=0.5). That leaves
the README's other recipe - "this command reproduces the worst corner" - pointing at a cell
no `vdb bench` run had ever produced. So one ran: `results/bench-corner.json` is the output
of exactly that command, and the tests below compare it to the grid digit for digit. That is
cheaper and stricter than a rerun in CI, which could only ever disagree with the artifact on
a different torch build without proving anything was wrong; the flag-wiring test here is the
part that *can* rot silently, because the published cell sits at the defaults.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"<!-- (?:RESULTS|SWEEP):START -->.*?<!-- (?:RESULTS|SWEEP):END -->",
                   re.DOTALL)
BENCH = re.compile(r"`vdb bench --temperature ([\d.]+) --alpha ([\d.]+)`")


def _prose() -> str:
    """The README with both generated blocks removed, i.e. the hand-written sentences."""
    return BLOCK.sub("", (ROOT / "README.md").read_text(encoding="utf-8"))


def _artifact(name: str) -> dict:
    path = ROOT / "results" / name
    assert path.exists(), f"missing committed artifact results/{name}"
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(sweep: dict, temperature: float, alpha: float) -> dict:
    return next(c for c in sweep["cells"]
                if c["temperature"] == temperature and c["alpha"] == alpha)


def test_the_corner_run_used_the_documented_command():
    """The artifact records its own config; the README must tell readers to pass the same."""
    corner, cmds = _artifact("bench-corner.json"), BENCH.findall(_prose())
    assert ("4", "0.9") in cmds, (
        f"the README no longer tells the reader to run `vdb bench --temperature 4 "
        f"--alpha 0.9` (it lists {cmds})")
    cfg, swept = corner["config"], _artifact("temperature-sweep.json")["config"]
    assert (cfg["temperature"], cfg["alpha"]) == (4.0, 0.9), cfg
    assert cfg["temperature"] in swept["temperatures"]
    assert cfg["alpha"] in swept["alphas"]
    for key in ("seed", "epochs", "train_n", "val_n", "student_scale", "teacher_scale"):
        assert cfg[key] == swept[key], (
            f"the corner run changed {key}, so it is not a cell of the published grid")


def test_the_corner_command_landed_on_the_grid_cell_it_claims():
    sweep, corner = _artifact("temperature-sweep.json"), _artifact("bench-corner.json")
    cell = _cell(sweep, 4.0, 0.9)
    assert corner["student_kd"]["accuracy"] == cell["kd_accuracy"]
    assert corner["student_kd"]["ece"] == cell["kd_ece"]
    assert corner["student_kd"]["per_family"] == cell["per_family"], (
        "the corner student differs from the swept one in at least one family")


def test_only_the_loss_differs_between_the_corner_run_and_the_grid():
    """Same seed, same teacher and same CE student - otherwise the cell is not comparable."""
    sweep, corner = _artifact("temperature-sweep.json"), _artifact("bench-corner.json")
    assert corner["teacher"]["accuracy"] == sweep["baseline"]["teacher_accuracy"]
    assert corner["student_ce"]["accuracy"] == sweep["baseline"]["ce_accuracy"]
    assert corner["student_kd"]["params"] == corner["student_ce"]["params"]


def test_the_corner_really_is_the_worst_cell_of_the_grid():
    sweep = _artifact("temperature-sweep.json")
    cells = sweep["cells"]
    worst = min(cells, key=lambda c: c["kd_accuracy"])
    least_calibrated = max(cells, key=lambda c: c["kd_ece"])
    assert (worst["temperature"], worst["alpha"]) == (4.0, 0.9), (
        "the least accurate cell is no longer T=4/alpha=0.9, so the README's corner "
        "command points at the wrong one")
    assert least_calibrated is worst, (
        "the corner is now only the least accurate cell, so calling it 'the worst' "
        "overstates what the grid says")
    assert worst["kd_accuracy"] < sweep["baseline"]["ce_accuracy"]


def test_the_quoted_corner_digits_are_the_measured_ones():
    """The prose outside the block recites the corner's accuracy and ECE to 3 decimals."""
    corner = _artifact("bench-corner.json")
    sentence = re.search(
        r"worst corner of the grid:.*?\(([\d.]+)\).*?\(([\d.]+)\)", _prose(), re.DOTALL)
    assert sentence, "the README no longer quotes the corner cell's two numbers"
    assert (float(sentence.group(1)), float(sentence.group(2))) == (
        round(corner["student_kd"]["accuracy"], 3), round(corner["student_kd"]["ece"], 3))


def test_the_reproduce_line_counts_the_trainings_the_artifact_records():
    sweep = _artifact("temperature-sweep.json")
    m = re.search(r"\((\d+) trainings on CPU", _prose())
    assert m, "the reproduce line no longer states how many models the sweep fits"
    # one teacher, one CE baseline, then one student per (T, alpha) cell
    assert int(m.group(1)) == len(sweep["cells"]) + 2, (
        f"the README counts {m.group(1)} trainings, the grid holds {len(sweep['cells'])} "
        "students plus the teacher and the CE baseline")


def test_the_corner_artifact_describes_one_coherent_run():
    """A committed file has to be plausible on its own, not just equal to a neighbour.

    The families are unbalanced, so the size-weighted per-family mean reconciles against
    the recorded accuracy only if both came from the same evaluated model - which is what
    makes this a check rather than a restatement.
    """
    from collections import Counter

    from vdb.synth import split

    corner = _artifact("bench-corner.json")
    cfg = corner["config"]
    _, val = split(cfg["train_n"], cfg["val_n"], seed=cfg["seed"])
    sizes = Counter(e.family for e in val)
    for arm in ("teacher", "student_ce", "student_kd"):
        fams = corner[arm]["per_family"]
        weighted = sum(fams[f] * sizes[f] for f in fams) / sum(sizes.values())
        assert abs(weighted - corner[arm]["accuracy"]) < 1e-4, (arm, weighted)
    q = corner["quantization"]
    assert q["fp32_accuracy"] == corner["student_kd"]["accuracy"]
    assert 0.0 < q["int8_mb"] <= q["fp32_mb"]
    assert corner["environment"]["device"] == "cpu" and corner["runtime_sec"] > 0


def test_the_cli_flags_reach_the_distillation_loss(monkeypatch, tmp_path):
    """A refactor that dropped `--temperature` would leave every artifact test green.

    The published cell sits at the defaults, so only the flags themselves can be wrong
    without anything published noticing; this runs the real CLI at a tiny budget and
    catches what the loss was actually built with.
    """
    from vdb import kd as kd_module
    from vdb.cli import main

    built: list[dict] = []
    real = kd_module.KDLoss
    monkeypatch.setattr(kd_module, "KDLoss",
                        lambda *a, **kw: (built.append(kw), real(*a, **kw))[1])
    out = tmp_path / "corner-wiring.json"
    assert main(["bench", "--train-n", "200", "--val-n", "100", "--epochs", "1",
                 "--temperature", "4", "--alpha", "0.9", "--out", str(out)]) == 0
    assert built, "the CLI never built a KD loss, so it does not distil at all"
    assert {tuple(sorted(kw.items())) for kw in built} == {
        (("alpha", 0.9), ("temperature", 4.0))}, built
    assert json.loads(out.read_text(encoding="utf-8"))["config"]["temperature"] == 4.0
