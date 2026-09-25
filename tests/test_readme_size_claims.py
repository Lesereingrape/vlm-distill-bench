"""Guard the hand-written numbers that live *outside* the rendered results block.

The table is pinned to the artifact by `test_readme_matches_results.py`; these are the
figures a reader takes on trust from the prose - how long a run takes, how many seeds
sit behind the headline, which ECE bin count and which CI Python versions. Each is a
claim about the code or the artifact, so each is checked against it.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _docs() -> list[dict]:
    paths = sorted((ROOT / "results").glob("bench-seed*.json"))
    assert paths, "no committed results/bench-seed*.json to check against"
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _sweep() -> dict:
    path = ROOT / "results" / "temperature-sweep.json"
    assert path.exists(), "no committed results/temperature-sweep.json to check against"
    return json.loads(path.read_text(encoding="utf-8"))


README_PATH = ROOT / "README.md"


def _readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def test_promised_runtime_is_the_measured_one():
    """The front page advertises a wall-clock budget; the artifacts recorded it."""
    text = _readme()
    claimed = {int(m.group(1)) for m in re.finditer(r"~(\d+)\s*(?:seconds|s on CPU)", text)}
    assert claimed, "README no longer promises a runtime; drop this test with the claim"
    runtimes = [d["runtime_sec"] for d in _docs()]
    for c in claimed:
        assert all(0.5 * c <= s <= 2.0 * c for s in runtimes), (
            f"README promises ~{c}s per seed; committed runs took {runtimes}s")


def test_seed_count_and_std_convention_match_the_artifacts():
    text = _readme()
    m = re.search(r"Headline results \((\d+) seeds, mean . ([a-z]+) std\)", text)
    assert m, "the headline section no longer states its seed count and std convention"
    n_seeds, convention = int(m.group(1)), m.group(2)
    assert n_seeds == len(_docs()), f"README says {n_seeds} seeds, results/ has {len(_docs())}"
    assert convention in ("sample", "population"), convention
    block = re.search(r"<!-- RESULTS:START -->(.*?)<!-- RESULTS:END -->", text,
                      re.DOTALL).group(1)
    assert convention in block and "divided by n-1" in block.replace("n\u22121", "n-1"), (
        "the heading names a convention the rendered block does not repeat")


def test_ece_bin_count_and_image_size_match_the_code():
    from vdb.bench import expected_calibration_error
    from vdb.synth import IMG

    text = _readme()
    m = re.search(r"(\d+)-bin ECE", text)
    assert m, "README no longer names the ECE bin count"
    default_bins = inspect.signature(expected_calibration_error).parameters["bins"].default
    assert int(m.group(1)) == default_bins, (
        f"README says {m.group(1)} bins, bench.py defaults to {default_bins}")

    m = re.search(r"image\(3.(\d+).(\d+)\)", text)
    assert m, "README no longer states the synthetic image size"
    assert (int(m.group(1)), int(m.group(2))) == (IMG, IMG), (
        f"README says 3x{m.group(1)}x{m.group(2)}, synth builds {IMG}x{IMG}")


def test_sweep_prose_describes_the_grid_that_was_actually_swept():
    """The paragraph above the SWEEP block is hand-written, so pin it to the artifact."""
    text, doc = _readme(), _sweep()
    temps = ", ".join(f"{t:g}" for t in doc["config"]["temperatures"])
    alphas = ", ".join(f"{a:g}" for a in doc["config"]["alphas"])
    listed = [m.replace(" ", "") for m in re.findall(r"\u2208\s*\{([\d, .]+)\}", text)]
    assert [temps.replace(" ", ""), alphas.replace(" ", "")] == listed[:2], (
        f"README recites {listed[:2]} for the sweep, the artifact swept "
        f"[{temps}] and [{alphas}]")
    n = len(doc["cells"])
    assert f"{n} students" in text, f"README counts {n} students somewhere else now"
    rows, cols = {c["temperature"] for c in doc["cells"]}, {c["alpha"] for c in doc["cells"]}
    assert len(rows) * len(cols) == n, "the grid is no longer a full cross"


def test_ci_python_matrix_in_the_readme_matches_the_workflow():
    workflow = next((ROOT / ".github" / "workflows").glob("*.yml"))
    listed = re.search(r"python-version: \[(.*?)\]", workflow.read_text(encoding="utf-8"))
    assert listed, "CI workflow no longer declares a python-version list"
    versions = [v.strip().strip('"') for v in listed.group(1).split(",")]
    m = re.search(r"Python (\d+\.\d+).(\d+\.\d+)", _readme())
    assert m, "README no longer states the CI Python matrix"
    assert (m.group(1), m.group(2)) == (versions[0], versions[-1]), (
        f"README claims {m.group(1)}-{m.group(2)} but CI runs {versions}")
