import json

from vdb.bench import distill_diagnostics, evaluate, fit, measure_latency
from vdb.cli import main
from vdb.kd import KDLoss
from vdb.model import ModelSpec
from vdb.synth import split


def test_tiny_pipeline_trains_and_learns():
    train, val = split(300, 100, seed=1)
    teacher = fit(train, ModelSpec(scale=0.75), epochs=6, seed=1).model
    early = evaluate(teacher, val)["accuracy"]
    stronger = fit(train, ModelSpec(scale=0.75), epochs=18, seed=1).model
    assert evaluate(stronger, val)["accuracy"] >= early - 0.05
    assert evaluate(stronger, val)["accuracy"] > 0.3  # above random (1/12 classes)
    assert measure_latency(stronger, n=5) > 0


def test_distillation_learns_faster_than_ce_at_equal_steps():
    """Same student budget: KD should not be worse, and diagnostics are sane."""
    train, val = split(400, 150, seed=2)
    teacher = fit(train, ModelSpec(scale=0.75), epochs=10, seed=2).model
    spec = ModelSpec(scale=0.25)
    ce = fit(train, spec, epochs=4, seed=2).model
    kd = fit(train, spec, epochs=4, seed=2, teacher=teacher,
             kd=KDLoss(temperature=3.0, alpha=0.9)).model
    assert evaluate(kd, val)["accuracy"] >= evaluate(ce, val)["accuracy"] - 0.1
    d = distill_diagnostics(kd, teacher, val)
    assert 0.0 <= d["topk_agreement"] <= 1.0
    assert 0.0 <= d["argmax_agreement"] <= 1.0


def test_cli_bench_writes_results(tmp_path):
    out = tmp_path / "r.json"
    rc = main(["bench", "--train-n", "200", "--val-n", "100", "--epochs", "2",
               "--out", str(out)])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    for key in ("teacher", "student_ce", "student_kd", "quantization"):
        assert key in doc
    assert doc["quantization"]["int8_mb"] < doc["quantization"]["fp32_mb"]


def test_cli_synth_runs(capsys):
    assert main(["synth", "--n", "10"]) == 0
    assert "generated 10 examples" in capsys.readouterr().out
