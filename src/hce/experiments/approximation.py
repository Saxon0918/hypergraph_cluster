"""E3: forward/inverse local approximations and real measured timings."""

from __future__ import annotations

import time
import tracemalloc
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hce.core import (
    approximate_U, build_incidence, compute_cumulants, compute_spectrum,
    compute_walk_moments, evaluate_thermodynamics, invert_U_approx, invert_U_exact,
)
from hce.experiments.common import (
    clean_samples, load_hic_from_config, load_scholp_from_config, policy_from_config,
    prepare_run, spectrum_kwargs,
)
from hce.data.scholp import calendar_window_samples, event_table_to_sample
from hce.utils import append_jsonl, write_table


def _timed(callable_):
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    start = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1 if sys.platform == "darwin" else 1024
    return result, elapsed, peak, max(0, rss_after - rss_before) * unit


def run_approximation(config: dict) -> Path:
    run_dir = prepare_run(config, "approximation")
    settings = config.get("approximation", {})
    names = settings.get("datasets", config.get("classification", {}).get("datasets", []))
    samples, _ = load_hic_from_config(config, names)
    scholp_names = settings.get("scholp_datasets", [])
    if scholp_names:
        tables, _ = load_scholp_from_config(config, scholp_names)
        for name, table in tables.items():
            if "congress" in name.lower():
                samples.extend(calendar_window_samples(table, "year", min_unique_edges=100))
            else:
                samples.append(event_table_to_sample(table, f"{name}:all_events"))
    samples, _, _ = clean_samples(samples, policy_from_config(config))
    x_grid = np.asarray(settings.get("x_grid", [0.01, 0.03, 0.1, 0.3, 1, 3]), dtype=float)
    error_rows, runtime_rows = [], []
    repetitions = int(config.get("benchmark", {}).get("repetitions", 5))
    warmups = int(config.get("benchmark", {}).get("warmups", 1))
    queries = config.get("benchmark", {}).get("queries_J", [1, 12, 80])
    for sample in samples:
        try:
            H = build_incidence(sample)
            spectrum = compute_spectrum(H, **spectrum_kwargs(config))
            moments = compute_walk_moments(H)
            cumulants = compute_cumulants(moments.W1, moments.W2, moments.W3, moments.M)
            beta = x_grid / spectrum.lambda_max
            exact = evaluate_thermodynamics(spectrum, beta)
            for order in (1, 2):
                forward = approximate_U(cumulants, beta, order)
                for idx, (x, b, target, estimate) in enumerate(zip(
                    x_grid, beta, exact.U_struct, forward
                )):
                    approximate_inverse = invert_U_approx(
                        cumulants, float(target), order, spectrum.lambda_max
                    )
                    exact_inverse = invert_U_exact(spectrum, float(target))
                    beta_hat = approximate_inverse.beta
                    error_rows.append({
                        "dataset": sample.dataset, "sample_id": sample.sample_id,
                        "order": order, "x": x, "beta_true": b,
                        "U_exact": target, "U_approx": estimate,
                        "signed_U_error": estimate - target,
                        "NAE_U": abs(estimate - target) / spectrum.lambda_max,
                        "beta_hat": beta_hat, "inverse_status": approximate_inverse.status,
                        "RE_beta": abs(beta_hat - b) / b if beta_hat is not None and b > 0 else np.nan,
                        "AE_x": abs(beta_hat - b) * spectrum.lambda_max if beta_hat is not None else np.nan,
                        "exact_inverse_beta": exact_inverse.beta,
                        "exact_inverse_status": exact_inverse.status,
                        "spectral_variance": exact.spectral_variance[idx],
                    })
            for route in ("exact_G", "exact_small_side", "moment_3"):
                try:
                    for J in queries:
                        grid = np.logspace(-2, 1, int(J))
                        for _ in range(warmups):
                            if route in {"exact_G", "exact_small_side"}:
                                kwargs = spectrum_kwargs(config)
                                if route == "exact_G":
                                    kwargs["side"] = "edge"
                                sp = compute_spectrum(H, **kwargs)
                                evaluate_thermodynamics(sp, grid / sp.lambda_max)
                            else:
                                mo = compute_walk_moments(H)
                                cu = compute_cumulants(mo.W1, mo.W2, mo.W3, mo.M)
                                approximate_U(cu, grid / spectrum.lambda_max, 2)
                        for repetition in range(repetitions):
                            if route in {"exact_G", "exact_small_side"}:
                                kwargs = spectrum_kwargs(config)
                                if route == "exact_G":
                                    kwargs["side"] = "edge"
                                _, elapsed, peak, rss_delta = _timed(lambda: evaluate_thermodynamics(
                                    compute_spectrum(H, **kwargs),
                                    grid / spectrum.lambda_max,
                                ))
                            else:
                                def moment_route():
                                    mo = compute_walk_moments(H)
                                    cu = compute_cumulants(mo.W1, mo.W2, mo.W3, mo.M)
                                    return approximate_U(cu, grid / spectrum.lambda_max, 2)
                                _, elapsed, peak, rss_delta = _timed(moment_route)
                            runtime_rows.append({
                                "dataset": sample.dataset, "sample_id": sample.sample_id,
                                "route": route, "J": J, "repetition": repetition,
                                "seconds": elapsed, "python_peak_bytes": peak,
                                "process_peak_rss_delta_bytes": rss_delta,
                                "n": H.shape[0], "M": H.shape[1],
                                "note": "RSS is process high-water delta; tracemalloc excludes native BLAS",
                            })
                except Exception as exc:
                    append_jsonl(run_dir / "failures.jsonl", {
                        "dataset": sample.dataset, "sample_id": sample.sample_id,
                        "stage": f"benchmark:{route}", "error_type": type(exc).__name__,
                        "reason": str(exc), "recoverable": True,
                    })
        except Exception as exc:
            append_jsonl(run_dir / "failures.jsonl", {
                "dataset": sample.dataset, "sample_id": sample.sample_id,
                "stage": "approximation", "error_type": type(exc).__name__,
                "reason": str(exc), "recoverable": True,
            })
    if error_rows:
        write_table(pd.DataFrame(error_rows), run_dir / "approximation_errors.parquet")
    if runtime_rows:
        write_table(pd.DataFrame(runtime_rows), run_dir / "runtime.csv")
    return run_dir

