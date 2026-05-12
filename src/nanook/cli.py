"""Command-line interface: ``nanook assess`` / ``apply`` / ``simulate``.

A thin shell around the Pipeline and metric functions. Reads/writes Parquet
through Polars; deliberately stdlib-only (argparse + json) so the wheel ships
without click/typer pulls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from nanook.metrics.risk.k_anonymity import k_anonymity
from nanook.metrics.risk.l_diversity import l_diversity
from nanook.metrics.risk.t_closeness import t_closeness
from nanook.pipeline import Pipeline

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``nanook`` script. Returns a Unix exit code."""
    parser = argparse.ArgumentParser(prog="nanook", description="Statistical disclosure control.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_assess = sub.add_parser("assess", help="Compute risk metrics on a Parquet file.")
    p_assess.add_argument("--input", type=Path, required=True)
    p_assess.add_argument("--qis", required=True, help="Comma-separated quasi-identifier columns.")
    p_assess.add_argument("--sensitive", default=None, help="Single sensitive column for l/t.")
    p_assess.add_argument("--k", type=int, default=5)
    p_assess.add_argument("--l", type=int, default=3)
    p_assess.add_argument("--t", type=float, default=0.2)
    p_assess.set_defaults(func=_cmd_assess)

    p_apply = sub.add_parser("apply", help="Apply a pipeline JSON to a Parquet file.")
    p_apply.add_argument("--input", type=Path, required=True)
    p_apply.add_argument("--pipeline", type=Path, required=True)
    p_apply.add_argument("--output", type=Path, required=True)
    p_apply.set_defaults(func=_cmd_apply)

    p_sim = sub.add_parser("simulate", help="Apply a pipeline and report risk+utility only.")
    p_sim.add_argument("--input", type=Path, required=True)
    p_sim.add_argument("--pipeline", type=Path, required=True)
    p_sim.set_defaults(func=_cmd_simulate)

    args = parser.parse_args(argv)
    return args.func(args)


def _cmd_assess(args: argparse.Namespace) -> int:
    df = pl.read_parquet(args.input)
    qis = [c.strip() for c in args.qis.split(",") if c.strip()]
    payload: dict = {"k_anonymity": k_anonymity(df, qis=qis, k=args.k).to_dict()}
    if args.sensitive:
        payload["l_diversity"] = l_diversity(df, qis=qis, sensitive=args.sensitive, l=args.l).to_dict()
        if df.schema[args.sensitive].is_numeric():
            payload["t_closeness"] = t_closeness(df, qis=qis, sensitive=args.sensitive, t=args.t).to_dict()
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    df = pl.read_parquet(args.input)
    pipeline = Pipeline.from_dict(json.loads(args.pipeline.read_text()))
    protected = pipeline.apply(df)
    protected.write_parquet(args.output)
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    df = pl.read_parquet(args.input)
    pipeline = Pipeline.from_dict(json.loads(args.pipeline.read_text()))
    protected = pipeline.apply(df)
    report = pipeline.assess(df, protected)
    json.dump(report.to_dict(), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
