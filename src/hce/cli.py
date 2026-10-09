"""Command-line entry points for audits, E1-E4, reports, and math checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hce.config import load_config
from hce.data.download import fetch_manifest
from hce.core import (
    approximate_U, build_incidence, compute_cumulants, compute_spectrum,
    compute_walk_moments, evaluate_thermodynamics,
)
from hce.experiments.approximation import run_approximation
from hce.experiments.audit import run_audit
from hce.experiments.classification import run_classification
from hce.experiments.nulls import run_nulls
from hce.experiments.temporal import run_temporal
from hce.models import HypergraphSample
from hce.plots import generate_report


def mathematical_self_check() -> dict:
    sample = HypergraphSample(
        sample_id="golden", dataset="synthetic", num_vertices=4,
        hyperedges=((0, 1, 2), (1, 2, 3), (0, 3)),
    )
    H = build_incidence(sample)
    moments = compute_walk_moments(H)
    cumulants = compute_cumulants(moments.W1, moments.W2, moments.W3, moments.M)
    spectrum = compute_spectrum(H, compress_multiplicities=False)
    thermo = evaluate_thermodynamics(spectrum, [0.1])
    expected = {
        "W": [3, 8, 34, 176], "decomposition": [22, 12, 62, 102, 12],
        "kappa": [8 / 3, 38 / 9, 160 / 27],
        "logZ": 1.387251474974264, "U": 3.113330829741711,
        "S": 1.075918392000093, "F": -13.872514749742638,
    }
    actual = {
        "W": [moments.W0, moments.W1, moments.W2, moments.W3],
        "decomposition": [moments.A2, moments.O2, moments.A3, moments.B3, moments.T3],
        "kappa": [cumulants.kappa1, cumulants.kappa2, cumulants.kappa3],
        "logZ": thermo.logZ[0], "U": thermo.U_struct[0],
        "S": thermo.entropy[0], "F": thermo.free_energy[0],
    }
    passed = all(np.allclose(actual[key], value, rtol=1e-10, atol=1e-10)
                 for key, value in expected.items())
    return {"status": "passed" if passed else "failed", "actual": actual,
            "expected": expected}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hce")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "nulls", "classify", "approximation", "temporal"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--dpi", type=int, default=300)
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--manifest", required=True)
    fetch.add_argument("--output-dir", required=True)
    subparsers.add_parser("self-check")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "self-check":
        result = mathematical_self_check()
        print(json.dumps(result, indent=2, default=float))
        return 0 if result["status"] == "passed" else 1
    if args.command == "report":
        figures = generate_report(args.run_dir, args.dpi)
        print(json.dumps({"run_dir": args.run_dir, "figures": figures}, indent=2))
        return 0
    if args.command == "fetch":
        results = fetch_manifest(args.manifest, args.output_dir)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0 if all(row.get("status") == "available" for row in results) else 2
    config = load_config(args.config)
    runners = {
        "audit": run_audit, "nulls": run_nulls, "classify": run_classification,
        "approximation": run_approximation, "temporal": run_temporal,
    }
    run_dir = runners[args.command](config)
    print(str(run_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

