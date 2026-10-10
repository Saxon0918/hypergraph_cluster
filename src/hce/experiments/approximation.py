"""E3: forward/inverse local approximations and staged measured timings."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import math
import re
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hce.config import config_hash
from hce.core import (
    approximate_U, build_dense_gram_operator, build_incidence, compute_cumulants,
    compute_spectrum_and_moments, compute_spectrum_from_operator,
    compute_walk_moments_from_operator, evaluate_thermodynamics, invert_U_approx,
    invert_U_exact,
)
from hce.data.scholp import calendar_window_samples, event_table_to_sample
from hce.exceptions import MemoryBudgetError
from hce.experiments.common import (
    clean_samples, load_hic_from_config, load_scholp_from_config, policy_from_config,
    prepare_run, spectrum_kwargs,
)
from hce.utils import append_jsonl, write_json, write_table, write_table_batches


CHECKPOINT_VERSION = 3


def _timed(callable_):
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = callable_()
    except BaseException:
        tracemalloc.stop()
        raise
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1 if sys.platform == "darwin" else 1024
    return result, elapsed, peak, max(0, rss_after - rss_before) * unit


def _sample_signature(sample) -> str:
    digest = hashlib.sha256()
    digest.update(sample.dataset.encode("utf-8"))
    digest.update(b"\0")
    digest.update(sample.sample_id.encode("utf-8"))
    digest.update(np.asarray([sample.num_vertices, sample.M], dtype="<i8").tobytes())
    for edge in sample.hyperedges:
        digest.update(np.asarray([len(edge)], dtype="<i8").tobytes())
        digest.update(np.asarray(edge, dtype="<i8").tobytes())
    return digest.hexdigest()


def _checkpoint_path(run_dir: Path, sample) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sample_id).strip("_")[:80]
    suffix = hashlib.sha256(
        f"{sample.dataset}\0{sample.sample_id}".encode("utf-8")
    ).hexdigest()[:12]
    return run_dir / "checkpoints" / "e3" / f"{slug}-{suffix}.json"


def _checkpoint_identity(config: dict, sample) -> dict:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "experiment": "E3",
        "config_hash": config_hash(config),
        "sample_signature": _sample_signature(sample),
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
    }


def _load_checkpoint(path: Path, identity: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if any(payload.get(key) != value for key, value in identity.items()):
        return None
    return payload if payload.get("complete") else None


def _benchmark_selection(samples, per_dataset: int) -> set[str]:
    """Choose a deterministic dimension-spanning runtime subset per source."""
    selected = set()
    for dataset in sorted({sample.dataset for sample in samples}):
        current = [sample for sample in samples if sample.dataset == dataset]
        ordered = sorted(
            current,
            key=lambda sample: (
                min(sample.M, len({v for edge in sample.hyperedges for v in edge})),
                sample.M,
                sample.sample_id,
            ),
        )
        if len(ordered) <= per_dataset:
            chosen = ordered
        else:
            indices = np.linspace(0, len(ordered) - 1, per_dataset).round().astype(int)
            chosen = [ordered[index] for index in sorted(set(indices))]
        selected.update(sample.sample_id for sample in chosen)
    return selected


def _runtime_row_base(
    sample, H, route, status="ok", reason=None, blas_threads: int = 1
) -> dict:
    n_active = int(np.count_nonzero(np.asarray(H.getnnz(axis=1)).ravel()))
    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "route": route,
        "n": int(H.shape[0]),
        "n_active": n_active,
        "M": int(H.shape[1]),
        "status": status,
        "reason": reason,
        "blas_threads": int(blas_threads),
        "memory_note": "tracemalloc excludes native BLAS allocations",
    }


def _query_many(product, route: str, beta, loops: int):
    if route in {"exact_G", "exact_small_side"}:
        result = None
        for _ in range(loops):
            result = evaluate_thermodynamics(product, beta)
        return result
    result = None
    for _ in range(loops):
        result = approximate_U(product, beta, 2)
    return result


def _prepare_route(H, route: str, config: dict):
    values = spectrum_kwargs(config)
    side = "edge" if route == "exact_G" else "auto"
    factor = 3.0 if route == "moment_3" else 4.0
    operation = "moment computation" if route == "moment_3" else "dense eigendecomposition"
    return build_dense_gram_operator(
        H,
        side=side,
        memory_budget_gb=values["memory_budget_gb"],
        workspace_factor=factor,
        operation=operation,
    )


def _compute_route(prepared, route: str, config: dict):
    if route in {"exact_G", "exact_small_side"}:
        return compute_spectrum_from_operator(
            prepared,
            psd_relative_tolerance=spectrum_kwargs(config)["psd_relative_tolerance"],
        )
    moments = compute_walk_moments_from_operator(prepared)
    return compute_cumulants(moments.W1, moments.W2, moments.W3, moments.M)


def _benchmark_route(sample, H, route: str, lambda_reference: float, config: dict):
    benchmark = config.get("benchmark", {})
    queries = [int(value) for value in benchmark.get("queries_J", [1, 12, 80])]
    warmups = int(benchmark.get("warmups", 1))
    repetitions = int(benchmark.get("repetitions", 5))
    query_loops = max(1, int(benchmark.get("query_timing_loops", 50)))
    max_exact_g = int(benchmark.get("max_exact_g_dimension", 2000))
    blas_threads = max(1, int(benchmark.get("blas_threads", 1)))
    base = _runtime_row_base(sample, H, route, blas_threads=blas_threads)
    if route == "exact_G" and H.shape[1] > max_exact_g:
        reason = (
            f"edge-side dimension {H.shape[1]} exceeds configured exact-G "
            f"compute limit {max_exact_g}"
        )
        return [{
            **base, "J": J, "repetition": -1, "seconds": math.nan,
            "operator_seconds": math.nan, "compute_seconds": math.nan,
            "query_seconds": math.nan, "query_timing_loops": query_loops,
            "operator_side": "edge", "operator_dimension": int(H.shape[1]),
            "python_peak_bytes": math.nan, "process_peak_rss_delta_bytes": math.nan,
            "status": "not_run", "reason": reason,
            "beta_grid_source": "diagnostic_exact_lambda_max",
        } for J in queries]

    def execute(record: bool, repetition: int):
        prepared, operator_seconds, operator_peak, operator_rss = _timed(
            lambda: _prepare_route(H, route, config)
        )
        product, compute_seconds, compute_peak, compute_rss = _timed(
            lambda: _compute_route(prepared, route, config)
        )
        rows = []
        for J in queries:
            beta = np.logspace(-2, 1, J) / lambda_reference
            loops = query_loops if record else 1
            _, elapsed, query_peak, query_rss = _timed(
                lambda: _query_many(product, route, beta, loops)
            )
            query_seconds = elapsed / loops
            if record:
                rows.append({
                    **base,
                    "J": J,
                    "repetition": repetition,
                    "seconds": operator_seconds + compute_seconds + query_seconds,
                    "operator_seconds": operator_seconds,
                    "compute_seconds": compute_seconds,
                    "query_seconds": query_seconds,
                    "query_timing_loops": loops,
                    "operator_side": prepared.side,
                    "operator_dimension": prepared.dimension,
                    "python_peak_bytes": max(operator_peak, compute_peak, query_peak),
                    "process_peak_rss_delta_bytes": max(
                        operator_rss, compute_rss, query_rss
                    ),
                    "beta_grid_source": "diagnostic_exact_lambda_max",
                })
        return rows

    try:
        for _ in range(warmups):
            execute(False, -1)
        rows = []
        for repetition in range(repetitions):
            rows.extend(execute(True, repetition))
        return rows
    except MemoryBudgetError as exc:
        return [{
            **base, "J": J, "repetition": -1, "seconds": math.nan,
            "operator_seconds": math.nan, "compute_seconds": math.nan,
            "query_seconds": math.nan, "query_timing_loops": query_loops,
            "operator_side": "edge" if route == "exact_G" else "auto",
            "operator_dimension": int(H.shape[1]) if route == "exact_G" else math.nan,
            "python_peak_bytes": math.nan, "process_peak_rss_delta_bytes": math.nan,
            "status": "not_run", "reason": str(exc),
            "beta_grid_source": "diagnostic_exact_lambda_max",
        } for J in queries]


def _process_sample_task(sample, benchmark_selected: bool, config: dict) -> dict:
    settings = config.get("approximation", {})
    x_grid = np.asarray(
        settings.get("x_grid", [0.01, 0.03, 0.1, 0.3, 1, 3]), dtype=float
    )
    blas_threads = max(1, int(config.get("benchmark", {}).get("blas_threads", 1)))
    error_rows = []
    runtime_rows = []
    failures = []
    with threadpool_limits(limits=blas_threads):
        try:
            H, incidence_seconds, incidence_peak, incidence_rss = _timed(
                lambda: build_incidence(sample)
            )
            (spectrum, moments), reference_seconds, reference_peak, reference_rss = _timed(
                lambda: compute_spectrum_and_moments(H, **spectrum_kwargs(config))
            )
            cumulants = compute_cumulants(
                moments.W1, moments.W2, moments.W3, moments.M
            )
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
                        "order": order, "x": float(x), "beta_true": float(b),
                        "U_exact": float(target), "U_approx": float(estimate),
                        "signed_U_error": float(estimate - target),
                        "NAE_U": float(abs(estimate - target) / spectrum.lambda_max),
                        "beta_hat": beta_hat if beta_hat is not None else math.nan,
                        "inverse_status": approximate_inverse.status,
                        "RE_beta": (
                            abs(beta_hat - b) / b
                            if beta_hat is not None and b > 0 else math.nan
                        ),
                        "AE_x": (
                            abs(beta_hat - b) * spectrum.lambda_max
                            if beta_hat is not None else math.nan
                        ),
                        "exact_inverse_beta": (
                            exact_inverse.beta
                            if exact_inverse.beta is not None else math.nan
                        ),
                        "exact_inverse_status": exact_inverse.status,
                        "spectral_variance": float(exact.spectral_variance[idx]),
                    })
            runtime_rows.append({
                **_runtime_row_base(
                    sample, H, "diagnostic_reference", blas_threads=blas_threads
                ),
                "J": len(x_grid), "repetition": 0,
                "seconds": incidence_seconds + reference_seconds,
                "operator_seconds": incidence_seconds,
                "compute_seconds": reference_seconds,
                "query_seconds": math.nan,
                "query_timing_loops": 1,
                "operator_side": spectrum.side,
                "operator_dimension": spectrum.operator_dimension,
                "python_peak_bytes": max(incidence_peak, reference_peak),
                "process_peak_rss_delta_bytes": max(incidence_rss, reference_rss),
                "beta_grid_source": "sample_exact_lambda_max",
            })
            if benchmark_selected:
                for route in ("exact_G", "exact_small_side", "moment_3"):
                    try:
                        runtime_rows.extend(
                            _benchmark_route(
                                sample, H, route, spectrum.lambda_max, config
                            )
                        )
                    except Exception as exc:
                        failures.append({
                            "dataset": sample.dataset, "sample_id": sample.sample_id,
                            "stage": f"benchmark:{route}",
                            "error_type": type(exc).__name__, "reason": str(exc),
                            "recoverable": True,
                        })
        except Exception as exc:
            failures.append({
                "dataset": sample.dataset, "sample_id": sample.sample_id,
                "stage": "approximation", "error_type": type(exc).__name__,
                "reason": str(exc), "recoverable": True,
            })
    return {
        "complete": True,
        "status": "failed" if not error_rows else "ok",
        "benchmark_selected": benchmark_selected,
        "error_rows": error_rows,
        "runtime_rows": runtime_rows,
        "failures": failures,
    }


def _checkpoint_frames(paths: list[Path], key: str):
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get(key, [])
        if rows:
            yield pd.DataFrame(rows)


def _load_samples(config: dict):
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
    selected_ids = set(settings.get("sample_ids", []))
    if selected_ids:
        samples = [sample for sample in samples if sample.sample_id in selected_ids]
    elif "max_objects" in settings:
        samples = samples[: int(settings["max_objects"])]
    return samples


def _persist_outputs(run_dir: Path, paths: list[Path]) -> None:
    write_table_batches(
        _checkpoint_frames(paths, "error_rows"),
        run_dir / "approximation_errors.parquet",
    )
    write_table_batches(
        _checkpoint_frames(paths, "runtime_rows"), run_dir / "runtime.csv"
    )


def run_approximation(config: dict) -> Path:
    run_dir = prepare_run(config, "approximation")
    settings = config.get("approximation", {})
    benchmark = config.get("benchmark", {})
    samples = _load_samples(config)
    benchmark_limit = max(1, int(benchmark.get("samples_per_dataset", 40)))
    benchmark_ids = _benchmark_selection(samples, benchmark_limit)
    manifest_rows = []
    for sample in samples:
        n_active = len({v for edge in sample.hyperedges for v in edge})
        manifest_rows.append({
            "dataset": sample.dataset, "sample_id": sample.sample_id,
            "n": sample.num_vertices, "n_active": n_active, "M": sample.M,
            "small_side_dimension": min(n_active, sample.M),
            "benchmark_selected": sample.sample_id in benchmark_ids,
            "selection_rule": "dimension_spanning_even_indices_per_dataset",
        })
    write_table(pd.DataFrame(manifest_rows), run_dir / "benchmark_manifest.csv")

    workers = max(1, int(settings.get("parallel_workers", 2)))
    resume = bool(settings.get("resume", True))
    progress_every = max(1, int(settings.get("progress_every_samples", 25)))
    print(
        f"[E3] samples={len(samples)} benchmark_samples={len(benchmark_ids)} "
        f"workers={workers} blas_threads_per_worker="
        f"{benchmark.get('blas_threads', 1)}", flush=True,
    )

    identities = {
        sample.sample_id: _checkpoint_identity(config, sample) for sample in samples
    }
    paths = {
        sample.sample_id: _checkpoint_path(run_dir, sample) for sample in samples
    }
    completed_paths: dict[str, Path] = {}
    pending_samples = []
    for sample in samples:
        checkpoint = (
            _load_checkpoint(paths[sample.sample_id], identities[sample.sample_id])
            if resume else None
        )
        if checkpoint is not None:
            completed_paths[sample.sample_id] = paths[sample.sample_id]
        else:
            pending_samples.append(sample)
    if completed_paths:
        print(f"[E3] resumed {len(completed_paths)} completed checkpoints", flush=True)

    failures_path = run_dir / "failures.jsonl"
    started = time.perf_counter()
    completed_count = len(completed_paths)

    def save_result(sample, payload):
        nonlocal completed_count
        payload.update(identities[sample.sample_id])
        write_json(paths[sample.sample_id], payload)
        completed_paths[sample.sample_id] = paths[sample.sample_id]
        for failure in payload.get("failures", []):
            append_jsonl(failures_path, failure)
        completed_count += 1
        if (
            completed_count == 1
            or completed_count % progress_every == 0
            or completed_count == len(samples)
            or payload.get("benchmark_selected")
        ):
            elapsed = time.perf_counter() - started
            newly_completed = max(1, completed_count - (len(samples) - len(pending_samples)))
            average = elapsed / newly_completed
            remaining = len(samples) - completed_count
            print(
                f"[E3] {completed_count}/{len(samples)} sample={sample.sample_id} "
                f"status={payload.get('status')} benchmark="
                f"{payload.get('benchmark_selected')} ETA≈{average * remaining / 3600:.2f}h",
                flush=True,
            )

    if workers == 1:
        for sample in pending_samples:
            save_result(
                sample,
                _process_sample_task(sample, sample.sample_id in benchmark_ids, config),
            )
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        sample_iter = iter(pending_samples)
        in_flight = {}
        try:
            for _ in range(min(len(pending_samples), workers * 2)):
                sample = next(sample_iter, None)
                if sample is None:
                    break
                future = executor.submit(
                    _process_sample_task,
                    sample,
                    sample.sample_id in benchmark_ids,
                    config,
                )
                in_flight[future] = sample
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    sample = in_flight.pop(future)
                    try:
                        payload = future.result()
                    except Exception as exc:
                        payload = {
                            "complete": True,
                            "status": "failed",
                            "benchmark_selected": sample.sample_id in benchmark_ids,
                            "error_rows": [], "runtime_rows": [],
                            "failures": [{
                                "dataset": sample.dataset,
                                "sample_id": sample.sample_id,
                                "stage": "approximation_worker",
                                "error_type": type(exc).__name__,
                                "reason": str(exc), "recoverable": True,
                            }],
                        }
                    save_result(sample, payload)
                    next_sample = next(sample_iter, None)
                    if next_sample is not None:
                        next_future = executor.submit(
                            _process_sample_task,
                            next_sample,
                            next_sample.sample_id in benchmark_ids,
                            config,
                        )
                        in_flight[next_future] = next_sample
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    ordered_paths = [
        completed_paths[sample.sample_id]
        for sample in samples
        if sample.sample_id in completed_paths
    ]
    _persist_outputs(run_dir, ordered_paths)
    return run_dir
