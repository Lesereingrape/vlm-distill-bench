"""Guard: the README's measured blocks must equal what make_report renders from results/.

Everything between the RESULTS markers - the table *and* the sentences around it,
including which row is bold and how many seeds the win counts - is rendered from the
committed per-seed JSONs, and everything between the SWEEP markers is rendered from the
committed temperature grid. If you re-run `vdb bench` or the sweep, regenerate the blocks
with `python experiments/make_report.py --write`; if you want different prose, change
`make_report.py`, because this test is what keeps the README honest.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import make_report  # noqa: E402

MARKER = re.compile(r"<!-- RESULTS:START -->\n(.*?)\n<!-- RESULTS:END -->", re.DOTALL)
SWEEP = re.compile(r"<!-- SWEEP:START -->\n(.*?)\n<!-- SWEEP:END -->", re.DOTALL)


def test_readme_matches_the_committed_artifacts():
    docs = make_report.load(ROOT)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = MARKER.search(readme)
    assert match, "README is missing the RESULTS markers"
    expected = make_report.build(docs).strip()
    assert match.group(1).strip() == expected, (
        "README results block is stale; run `python experiments/make_report.py --write`")


def test_the_rendered_block_reports_nothing_the_artifacts_do_not_support():
    """The renderer must not keep a row for an arm the artifact no longer has."""
    docs = make_report.load(ROOT)
    block = make_report.build(docs)
    for arm, label in make_report.ARMS:
        assert arm in docs[0], f"artifact lost {arm}"
        assert label in block, f"renderer dropped the {label} row"
    rows = [ln for ln in block.splitlines() if ln.startswith("| ")]
    assert len(rows) == len(make_report.ARMS) + 1, (
        "the table holds one row per arm - not one per seed, and no stale arms")


def test_readme_sweep_block_matches_the_grid_artifact():
    """Same rule for the temperature section: it is a render of a committed file."""
    docs = make_report.load(ROOT)
    block = SWEEP.search((ROOT / "README.md").read_text(encoding="utf-8"))
    assert block, "README is missing the SWEEP markers"
    expected = make_report.build_sweep(make_report.load_sweep(ROOT), docs).strip()
    assert block.group(1).strip() == expected, (
        "README sweep block is stale; run `python experiments/make_report.py --write`")


def test_the_sweep_and_the_headline_agree_about_the_cell_they_share():
    """One cell exists in both artifacts; the two files must not disagree about it.

    This is the guard that the sweep script and `vdb bench` really run the same training
    path: `run_temperature_sweep.py` builds its own teacher and student rather than
    calling `bench.run`, so without this check the grid could quietly drift into
    measuring a similar-but-different experiment.
    """
    docs = make_report.load(ROOT)
    sweep = make_report.load_sweep(ROOT)
    seed = sweep["config"]["seed"]
    pub = next(d for d in docs if d["config"]["seed"] == seed)
    cell = next(c for c in sweep["cells"]
                if c["temperature"] == pub["config"]["temperature"]
                and c["alpha"] == pub["config"]["alpha"])
    assert cell["kd_accuracy"] == pub["student_kd"]["accuracy"]
    assert cell["kd_ece"] == pub["student_kd"]["ece"]
    assert cell["per_family"] == pub["student_kd"]["per_family"]
    assert sweep["baseline"]["ce_accuracy"] == pub["student_ce"]["accuracy"]
    assert sweep["baseline"]["teacher_accuracy"] == pub["teacher"]["accuracy"]
