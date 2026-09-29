"""Command-line entry point; no GPU imports occur on the CPU path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .runner import preflight, run
from .registry import ALL_NAMES, REMAINING_NAMES, ML_NAMES


MAX_ROWS = 5_000_000
MAX_SEED = 2**32 - 1


def positive_rows(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rows must be an integer") from exc
    if not 1 <= value <= MAX_ROWS:
        raise argparse.ArgumentTypeError(f"rows must be between 1 and {MAX_ROWS}")
    return value


def repeat_count(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("repeat must be an integer") from exc
    if not 3 <= value <= 100:
        raise argparse.ArgumentTypeError("repeat must be between 3 and 100")
    return value


def warmup_count(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("warmup must be an integer") from exc
    if not 1 <= value <= 20:
        raise argparse.ArgumentTypeError("warmup must be between 1 and 20")
    return value


def bounded_seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if not 0 <= value <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"seed must be between 0 and {MAX_SEED}")
    return value


def positive_seconds(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 3600:
        raise argparse.ArgumentTypeError("time limits must be 1–3600 seconds")
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m nca_ads_bench",
        description="NCA-ADS CPU/GPU correctness and timing harness. GPU requires CuPy, cuDF, and a visible CUDA device.")
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="Record environment and run M02/M03 small correctness fixtures")
    pre.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    pre.add_argument("--output", type=Path, required=True, help="New JSON report path (never overwritten)")
    bench = sub.add_parser("run", help="Run three workloads; GPU also measures same-host CPU baselines")
    bench.add_argument("--workload", choices=ALL_NAMES, help="Run only this workload; omitted runs the original three")
    bench.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    bench.add_argument("--rows", nargs="+", type=positive_rows, default=[10_000, 100_000],
                       help="One or more row counts (default: 10000 100000; max each: 5000000)")
    bench.add_argument("--repeat", type=repeat_count, default=5, help="Timed repeats, 3–100 (default: 5)")
    bench.add_argument("--warmup", type=warmup_count, default=2, help="Untimed warmups, 1–20 (default: 2)")
    bench.add_argument("--seed", type=bounded_seed, default=20260928,
                       help="NumPy seed, 0–4294967295 (default: 20260928)")
    bench.add_argument("--output", type=Path, default=Path("results"), help="Parent directory for unique run folders")
    batch = sub.add_parser("course-batch", help="Bounded remaining-module tests; each case runs in an isolated process")
    batch.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    batch.add_argument("--workloads", nargs="+", choices=ALL_NAMES, default=list(REMAINING_NAMES))
    batch.add_argument("--rows", nargs="+", type=positive_rows, default=[10_000, 100_000])
    batch.add_argument("--repeat", type=repeat_count, default=3)
    batch.add_argument("--warmup", type=warmup_count, default=1)
    batch.add_argument("--seed", type=bounded_seed, default=20260928)
    batch.add_argument("--case-timeout", type=positive_seconds, default=120, help="Maximum seconds per child process (default 120)")
    batch.add_argument("--max-seconds", type=positive_seconds, default=900, help="Total batch wall-clock budget including preflight (default 900)")
    batch.add_argument("--output", type=Path, default=Path("results"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
            report = preflight(args.backend, args.output)
            print(f"{report['status']}: {args.output}")
            return 0 if report["status"] == "passed" else 1
        if len(set(args.rows)) != len(args.rows):
            parser().error("--rows values must be unique")
        if args.command == "course-batch":
            if len(set(args.workloads)) != len(args.workloads):
                parser().error("--workloads values must be unique")
            if args.rows != sorted(args.rows):
                parser().error("course-batch --rows must be ascending")
            if any(name in REMAINING_NAMES for name in args.workloads) and any(size < 100 or size > 100_000 for size in args.rows):
                parser().error("remaining-module cases require 100–100000 rows")
            if any(name in ML_NAMES for name in args.workloads) and min(args.rows) < 1000:
                parser().error("ML cases require at least 1000 rows")
            from .batch import run_batch
            directory, report = run_batch(args)
        else:
            if args.workload in REMAINING_NAMES and any(size < 100 or size > 100_000 for size in args.rows):
                parser().error("remaining-module cases require 100–100000 rows")
            if args.workload in ML_NAMES and min(args.rows) < 1000:
                parser().error("ML cases require at least 1000 rows")
            directory, report = run(args.backend, args.rows, args.repeat, args.warmup, args.seed, args.output, args.workload)
        print(f"{report['status']}: {directory}")
        return 0 if report["status"] == "passed" else 1
    except (FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
