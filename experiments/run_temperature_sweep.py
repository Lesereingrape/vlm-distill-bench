"""Measure the temperature trap the README describes, instead of asserting it.

`vdb bench` fixes T=1/alpha=0.5, so the claim "the textbook T^2-compensated loss is
worse than plain CE at every alpha once T > 1" was, until this script, a memory of
tuning runs. One shared teacher and one shared CE student are trained per seed and then
every (T, alpha) KD student starts from the same seed as the CE student did, so the only
thing that varies across the grid is the loss - which is what makes the comparison in
the README's second results block a measurement rather than an anecdote.

    python experiments/run_temperature_sweep.py            # prints, writes nothing
    python experiments/run_temperature_sweep.py --write    # -> results/temperature-sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vdb.bench import (
    environment,
    evaluate,
    fit,
    measure_latency,
)
from vdb.kd import KDLoss
from vdb.model import ModelSpec
from vdb.synth import split

SEED = 0
EPOCHS = 40
TRAIN_N = 3000
VAL_N = 800
TEACHER_SCALE = 2.0
STUDENT_SCALE = 0.3
TEMPERATURES = (1.0, 2.0, 3.0, 4.0)
ALPHAS = (0.3, 0.5, 0.7, 0.9)


def run() -> dict:
    started = time.perf_counter()
    train, val = split(TRAIN_N, VAL_N, seed=SEED)

    print("teacher + CE baseline ...", file=sys.stderr)
    teacher = fit(train, ModelSpec(scale=TEACHER_SCALE), epochs=EPOCHS, seed=SEED).model
    base = fit(train, ModelSpec(scale=STUDENT_SCALE), epochs=EPOCHS, seed=SEED).model
    ce_eval = evaluate(base, val)
    baseline = {"teacher_accuracy": evaluate(teacher, val)["accuracy"],
                "ce_accuracy": ce_eval["accuracy"], "ce_ece": ce_eval["ece"],
                "ce_ms": measure_latency(base)}

    cells = []
    for t in TEMPERATURES:
        for alpha in ALPHAS:
            print(f"KD T={t} alpha={alpha} ...", file=sys.stderr)
            res = fit(train, ModelSpec(scale=STUDENT_SCALE), epochs=EPOCHS, seed=SEED,
                      teacher=teacher, kd=KDLoss(temperature=t, alpha=alpha), val=val)
            ev = evaluate(res.model, val)
            cells.append({"temperature": t, "alpha": alpha,
                          "kd_accuracy": ev["accuracy"], "kd_ece": ev["ece"],
                          "kd_ms": measure_latency(res.model),
                          "per_family": ev["per_family"]})

    return {
        "config": {"seed": SEED, "epochs": EPOCHS, "train_n": TRAIN_N, "val_n": VAL_N,
                   "teacher_scale": TEACHER_SCALE, "student_scale": STUDENT_SCALE,
                   "temperatures": list(TEMPERATURES), "alphas": list(ALPHAS)},
        "baseline": baseline,
        "cells": cells,
        "environment": environment(),
        "runtime_sec": round(time.perf_counter() - started, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="run_temperature_sweep")
    ap.add_argument("--write", action="store_true",
                    help="commit the grid to results/temperature-sweep.json")
    ap.add_argument("--out", default="results/temperature-sweep.json")
    args = ap.parse_args()
    doc = run()
    print(json.dumps(doc, indent=2))
    if args.write:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)
