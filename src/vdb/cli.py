"""`vdb` command line: synth preview, one-shot reproducible benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model import ModelSpec


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import ANSWERS, generate

    examples = generate(args.n, seed=args.seed)
    families: dict[str, int] = {}
    for e in examples:
        families[e.family] = families.get(e.family, 0) + 1
    print(f"generated {len(examples)} examples, families: {families}")
    e = examples[0]
    print(f"first: family={e.family} answer={ANSWERS[e.answer]!r} "
          f"question={list(e.question)} image={e.image.shape}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import distill_diagnostics, evaluate, fit, measure_latency
    from .kd import KDLoss
    from .quant import quant_report
    from .synth import split

    train, val = split(args.train_n, args.val_n, seed=args.seed)
    epochs = args.epochs

    print("[1/4] training teacher ...", file=sys.stderr)
    teacher = fit(train, ModelSpec(scale=args.teacher_scale), epochs=epochs,
                  seed=args.seed).model
    t_eval = evaluate(teacher, val)

    print("[2/4] student, hard-label CE ...", file=sys.stderr)
    base = fit(train, ModelSpec(scale=args.student_scale), epochs=epochs,
               seed=args.seed).model
    b_eval = evaluate(base, val)

    print("[3/4] student, knowledge distillation ...", file=sys.stderr)
    kd_res = fit(train, ModelSpec(scale=args.student_scale), epochs=epochs,
                 seed=args.seed, teacher=teacher,
                 kd=KDLoss(temperature=args.temperature, alpha=args.alpha), val=val)
    k_eval = evaluate(kd_res.model, val)
    diag = distill_diagnostics(kd_res.model, teacher, val)

    print("[4/4] int8 dynamic quantization ...", file=sys.stderr)
    q_report = quant_report(kd_res.model, val)

    doc = {
        "config": {"seed": args.seed, "epochs": epochs, "train_n": args.train_n,
                   "val_n": args.val_n, "student_scale": args.student_scale,
                   "teacher_scale": args.teacher_scale,
                   "temperature": args.temperature, "alpha": args.alpha},
        "teacher": {**t_eval, "ms": measure_latency(teacher)},
        "student_ce": {**b_eval, "ms": measure_latency(base)},
        "student_kd": {**k_eval, "ms": measure_latency(kd_res.model)},
        "distill_diagnostics": diag,
        "quantization": q_report,
    }
    _print_table(doc)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def _print_table(doc: dict) -> None:
    print(f"\n{'model':<16}{'params':>9}{'acc':>7}{'ece':>7}{'ms':>7}   per-family acc")
    for key, label in [("teacher", "teacher"), ("student_ce", "student-CE"),
                       ("student_kd", "student-KD")]:
        r = doc[key]
        fams = "  ".join(f"{k}={v:.2f}" for k, v in r["per_family"].items())
        print(f"{label:<16}{r['params']:>9}{r['accuracy']:>7.3f}{r['ece']:>7.3f}"
              f"{r['ms']:>7.1f}   {fams}")
    q = doc["quantization"]
    print(f"\nint8 quant: acc {q['fp32_accuracy']:.3f} -> {q['int8_accuracy']:.3f}, "
          f"size {q['fp32_mb']:.2f}MB -> {q['int8_mb']:.2f}MB "
          f"({q['compression']:.1f}x), latency {q['fp32_ms']:.1f}ms -> {q['int8_ms']:.1f}ms")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vdb", description="vlm-distill-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="preview the procedural dataset")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    b = sub.add_parser("bench", help="run the full reproducible experiment")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--epochs", type=int, default=40)
    b.add_argument("--train-n", type=int, default=3000)
    b.add_argument("--val-n", type=int, default=800)
    b.add_argument("--student-scale", type=float, default=0.3)
    b.add_argument("--teacher-scale", type=float, default=2.0)
    b.add_argument("--temperature", type=float, default=1.0)
    b.add_argument("--alpha", type=float, default=0.5)
    b.add_argument("--out", default="results/bench-seed0.json")
    b.set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
