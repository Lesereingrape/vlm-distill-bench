"""The committed artifacts have to agree with themselves, not just with the README.

`test_readme_matches_results.py` pins the published table to these files; this file is
the other direction - it checks that each `bench-seedN.json` describes one coherent run.
A `vdb bench` refactor that, say, measured the quantized model against a different arm
than `student_kd` would keep producing plausible-looking JSON that the README would then
render faithfully. These assertions fail that instead.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / "results").glob("bench-seed*.json"))]

ARMS = ("teacher", "student_ce", "student_kd")
FAMILIES = ("count", "color", "shape", "spatial")


def test_every_seed_artifact_is_present_and_distinct():
    assert len(DOCS) == 3, f"expected 3 committed seeds, found {len(DOCS)}"
    assert {d["config"]["seed"] for d in DOCS} == {0, 1, 2}
    configs = [{k: v for k, v in d["config"].items() if k != "seed"} for d in DOCS]
    assert len(configs) == 3 and all(c == configs[0] for c in configs), (
        "seeds must differ only in the seed, or the README's shared-budget sentence lies")


def test_per_family_scores_reconcile_with_the_reported_accuracy():
    """A per-family table that contradicts the headline number is the classic drift.

    The families are *not* balanced - a scene asks two color questions but one of each
    other kind - so the overall accuracy is the size-weighted mean over the four. The
    sizes come from the generator itself, which makes this a real reconciliation rather
    than a restatement.
    """
    from collections import Counter

    from vdb.synth import split

    for d in DOCS:
        cfg = d["config"]
        _, val = split(cfg["train_n"], cfg["val_n"], seed=cfg["seed"])
        sizes = Counter(e.family for e in val)
        assert tuple(sorted(sizes)) == tuple(sorted(FAMILIES)), sorted(sizes)
        for arm in ARMS:
            fams = d[arm]["per_family"]
            weighted = sum(fams[f] * sizes[f] for f in fams) / sum(sizes.values())
            assert abs(weighted - d[arm]["accuracy"]) < 1e-4, (
                f"seed {cfg['seed']} {arm}: weighted per-family {weighted:.4f} "
                f"vs recorded accuracy {d[arm]['accuracy']:.4f}")


def test_quantization_row_describes_the_model_the_table_just_scored():
    for d in DOCS:
        q = d["quantization"]
        assert q["fp32_accuracy"] == d["student_kd"]["accuracy"], (
            "the int8 report is measured on a different student than the KD row")
        assert 0.0 < q["int8_mb"] <= q["fp32_mb"], (q["fp32_mb"], q["int8_mb"])
        assert abs(q["compression"] - q["fp32_mb"] / q["int8_mb"]) < 1e-3
        assert all(v > 0 for v in (q["fp32_ms"], q["int8_ms"]))


def test_every_arm_records_the_environment_and_a_wall_clock():
    for d in DOCS:
        env = d.get("environment")
        assert env, f"seed {d['config']['seed']} has no environment block"
        for key in ("python", "platform", "torch", "numpy", "threads", "device"):
            assert env[key], key
        assert env["device"] == "cpu" and env["threads"] >= 1
        assert d["runtime_sec"] > 0


def test_diagnostics_stay_inside_the_probability_bounds_they_claim():
    for d in DOCS:
        diag = d["distill_diagnostics"]
        assert 0.0 <= diag["both_correct"] <= diag["argmax_agreement"] <= 1.0, (
            "examples both models get right cannot outnumber the agreements")
        assert 0.0 <= diag["topk_agreement"] <= 1.0
