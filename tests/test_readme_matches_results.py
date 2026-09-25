"""Guard: the README results block must equal make_report.build(results/).

Everything between the RESULTS markers - the table *and* the sentences around it,
including which row is bold and how many seeds the win counts - is rendered from the
committed per-seed JSONs. If you re-run `vdb bench`, regenerate the block with
`python experiments/make_report.py --write`; if you want different prose, change
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
