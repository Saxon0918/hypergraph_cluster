"""E1: real structures versus size- and margin-constrained null ensembles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hce.config import config_hash
from hce.core import (
    build_incidence, compute_cumulants, compute_spectrum_and_moments,
    evaluate_thermodynamics,
)
from hce.data.scholp import calendar_window_samples, event_table_to_sample
from hce.evaluation.statistics import (
    monte_carlo_two_sided_p, normalized_curve_integral, standardized_difference,
)
from hce.experiments.common import (
    clean_samples, load_hic_from_config, load_scholp_from_config, policy_from_config,
    prepare_run, spectrum_kwargs,
)
from hce.nulls import effective_sample_size, rhat, run_fixed_margins_chain, sample_size_null
from hce.utils import append_jsonl, stable_seed, write_json, write_table_batches


CHECKPOINT_VERSION = 2


def _measure(sample, x_grid, c_scale, config, model, draw, seed):
    """Measure one object while sharing its dense Gram construction."""
    H = build_incidence(sample)
    spectrum, moments = compute_spectrum_and_moments(H, **spectrum_kwargs(config))
    cumulants = compute_cumulants(moments.W1, moments.W2, moments.W3, moments.M)
    curve = evaluate_thermodynamics(spectrum, np.asarray(x_grid) / c_scale)
    moment_row = {
        "dataset": sample.dataset, "sample_id": sample.sample_id,
        "model": model, "draw": draw, "seed": seed,
        **{key: value for key, value in asdict(moments).items() if key != "diagnostics"},
        **asdict(cumulants), "lambda_max": spectrum.lambda_max,
    }
    curve_rows = [{
        "dataset": sample.dataset, "sample_id": sample.sample_id,
        "model": model, "draw": draw, "seed": seed,
        "x": float(x), "beta": float(beta), "c_scale": c_scale,
        "U": float(U), "S": float(S), "S_norm": float(Snorm),
    } for x, beta, U, S, Snorm in zip(
        x_grid, curve.beta, curve.U_struct, curve.entropy, curve.normalized_entropy
    )]
    return moment_row, curve_rows


def _decorate_measure(result, parent_sample_id: str, chain) -> dict:
    row, curves = result
    row.update({"parent_sample_id": parent_sample_id, "chain": chain})
    for curve_row in curves:
        curve_row.update({"parent_sample_id": parent_sample_id, "chain": chain})
    return {"moment": row, "curves": curves}


def _run_n1_batch_task(real, draws, x_grid, c_scale, config, root_seed, blas_threads):
    results = []
    with threadpool_limits(limits=int(blas_threads)):
        for draw in draws:
            seed = stable_seed(root_seed, real.dataset, real.sample_id, "N1", draw)
            null_sample, diag = sample_size_null(real, seed, simple=True)
            result = _decorate_measure(
                _measure(null_sample, x_grid, c_scale, config, "N1", draw, seed),
                real.sample_id, math.nan,
            )
            result["diagnostic"] = {
                "sample_id": real.sample_id, "draw": draw, "seed": seed,
                **asdict(diag), "acceptance_rate": diag.acceptance_rate,
            }
            results.append(result)
    return results


def _run_n2_chain_task(
    real, chain_id, num_samples, per_chain, x_grid, c_scale, config,
    root_seed, burnin_sweeps, thinning_attempt_sweeps, blas_threads,
):
    seed = stable_seed(root_seed, real.dataset, real.sample_id, "N2", chain_id)
    results = []
    with threadpool_limits(limits=int(blas_threads)):
        iterator = run_fixed_margins_chain(
            real, seed, num_samples=num_samples,
            burnin_sweeps=burnin_sweeps,
            thinning_attempt_sweeps=thinning_attempt_sweeps,
            simple=True,
        )
        for local_draw, (null_sample, diag) in enumerate(iterator):
            draw = chain_id * per_chain + local_draw
            result = _decorate_measure(
                _measure(null_sample, x_grid, c_scale, config, "N2", draw, seed),
                real.sample_id, chain_id,
            )
            result["diagnostic"] = {
                "sample_id": real.sample_id, "chain": chain_id,
                "draw": draw, "seed": seed, **asdict(diag),
                "acceptance_rate": diag.acceptance_rate,
            }
            results.append(result)
    return results


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
    return run_dir / "checkpoints" / "e1" / f"{slug}-{suffix}.json"


def _checkpoint_identity(config: dict, sample) -> dict:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "experiment": "E1",
        "config_hash": config_hash(config),
        "sample_signature": _sample_signature(sample),
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
    }


def _new_checkpoint(identity: dict) -> dict:
    return {
        **identity,
        "complete": False,
        "completed_real": False,
        "completed_n1_draws": [],
        "completed_n2_chains": [],
        "moments": [],
        "curves": [],
        "sampler_diagnostics": [],
        "mixing_diagnostics": [],
        "statistics": [],
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
    return payload


def _append_packages(payload: dict, packages: list[dict]) -> None:
    for package in packages:
        payload["moments"].append(package["moment"])
        payload["curves"].extend(package["curves"])
        payload["sampler_diagnostics"].append(package["diagnostic"])


def _finalize_statistics(payload: dict, real, x_grid, settings) -> None:
    moment_rows = payload["moments"]
    curve_rows = payload["curves"]
    current_moments = pd.DataFrame(
        row for row in moment_rows if row["model"] == "N2"
    )
    current_curves = pd.DataFrame(
        row for row in curve_rows if row["model"] == "N2"
    )
    traces = {
        "W2_per_M": {
            int(chain): group.sort_values("draw").W2.to_numpy()
            / group.sort_values("draw").M.to_numpy()
            for chain, group in current_moments.groupby("chain")
        },
        "kappa3": {
            int(chain): group.sort_values("draw").kappa3.to_numpy()
            for chain, group in current_moments.groupby("chain")
        },
    }
    if not current_curves.empty:
        closest_x = float(x_grid[np.argmin(np.abs(x_grid - 1.0))])
        at_x = current_curves[np.isclose(current_curves.x, closest_x)]
        traces["U_x_near_1"] = {
            int(chain): group.sort_values("draw").U.to_numpy()
            for chain, group in at_x.groupby("chain")
        }
        traces["S_norm_x_near_1"] = {
            int(chain): group.sort_values("draw").S_norm.to_numpy()
            for chain, group in at_x.groupby("chain")
        }

    mixing_rows = []
    for metric, chain_values in traces.items():
        rhat_value, rhat_status = rhat(list(chain_values.values()))
        for chain_id, values in chain_values.items():
            ess_value, ess_status = effective_sample_size(values)
            mixing_rows.append({
                "record_type": "mixing", "sample_id": real.sample_id,
                "model": "degree_size_preserving", "chain": chain_id,
                "metric": metric, "trace_length": len(values),
                "ess": ess_value, "ess_status": ess_status,
                "rhat": rhat_value, "rhat_status": rhat_status,
                "status": "diagnostic_only_not_proof_of_global_mixing",
            })
    payload["mixing_diagnostics"] = mixing_rows

    curve_table = pd.DataFrame(curve_rows)
    real_group = curve_table[curve_table.model == "real"]
    x_min = float(settings.get("integral_x_min", 0.01))
    x_max = float(settings.get("integral_x_max", 10.0))
    real_value = normalized_curve_integral(
        real_group.x, real_group.S_norm, x_min=x_min, x_max=x_max
    )
    statistic_rows = []
    for model in ("N1", "N2"):
        values = [
            normalized_curve_integral(
                group.x, group.S_norm, x_min=x_min, x_max=x_max
            )
            for _, group in curve_table[curve_table.model == model].groupby("draw")
        ]
        z, z_status = standardized_difference(real_value, values)
        statistic_rows.append({
            "dataset": real.dataset, "sample_id": real.sample_id, "model": model,
            "statistic": "normalized_entropy_logx_integral",
            "real_value": real_value, "null_mean": float(np.mean(values)),
            "null_std": float(np.std(values)), "z": z, "z_status": z_status,
            "p_monte_carlo_two_sided": monte_carlo_two_sided_p(real_value, values),
            "B": len(values), "mixing_claim": "not_established_by_sampler_alone",
        })
    payload["statistics"] = statistic_rows


def _draw_chunks(draws: list[int], batch_size: int) -> list[list[int]]:
    return [draws[start:start + batch_size] for start in range(0, len(draws), batch_size)]


def _process_real_sample(
    real, c_scale, config, settings, x_grid, root_seed, checkpoint_path, identity,
    executor, workers, blas_threads, resume,
) -> dict:
    payload = (
        _load_checkpoint(checkpoint_path, identity) if resume else None
    ) or _new_checkpoint(identity)
    if payload.get("complete"):
        return payload

    if not payload.get("completed_real"):
        with threadpool_limits(limits=blas_threads):
            real_result = _decorate_measure(
                _measure(real, x_grid, c_scale, config, "real", 0, root_seed),
                real.sample_id, math.nan,
            )
        payload["moments"].append(real_result["moment"])
        payload["curves"].extend(real_result["curves"])
        payload["completed_real"] = True
        write_json(checkpoint_path, payload)

    B = int(settings.get("samples_per_model", 100))
    completed_n1 = {int(value) for value in payload.get("completed_n1_draws", [])}
    missing_draws = [draw for draw in range(B) if draw not in completed_n1]
    batch_size = max(
        1, int(settings.get("parallel_batch_draws", math.ceil(B / max(1, workers * 4))))
    )
    chunks = _draw_chunks(missing_draws, batch_size)
    if chunks:
        print(
            f"[E1] sample={real.sample_id}: N1 start; remaining={len(missing_draws)}/{B}",
            flush=True,
        )
    if executor is None:
        completed_batches = (
            (
                chunk,
                _run_n1_batch_task(
                    real, chunk, x_grid, c_scale, config, root_seed, blas_threads
                ),
            )
            for chunk in chunks
        )
    else:
        future_map = {
            executor.submit(
                _run_n1_batch_task, real, chunk, x_grid, c_scale, config,
                root_seed, blas_threads,
            ): chunk
            for chunk in chunks
        }
        completed_batches = (
            (future_map[future], future.result()) for future in as_completed(future_map)
        )
    for chunk, packages in completed_batches:
        _append_packages(payload, packages)
        completed_n1.update(chunk)
        payload["completed_n1_draws"] = sorted(completed_n1)
        write_json(checkpoint_path, payload)
        print(f"[E1] sample={real.sample_id}: N1 {len(completed_n1)}/{B}", flush=True)

    chains = int(settings.get("chains", 4))
    per_chain = int(np.ceil(B / chains))
    completed_chains = {
        int(value) for value in payload.get("completed_n2_chains", [])
    }
    missing_chains = [chain for chain in range(chains) if chain not in completed_chains]
    if missing_chains:
        print(
            f"[E1] sample={real.sample_id}: N2 start; "
            f"remaining_chains={len(missing_chains)}/{chains}", flush=True,
        )

    def chain_args(chain_id: int):
        count = max(0, min(per_chain, B - chain_id * per_chain))
        return (
            real, chain_id, count, per_chain, x_grid, c_scale, config,
            root_seed, int(settings.get("burnin_sweeps", 20)),
            int(settings.get("thinning_attempt_sweeps", 5)), blas_threads,
        )

    if executor is None:
        chain_results = (
            (chain, _run_n2_chain_task(*chain_args(chain))) for chain in missing_chains
        )
    else:
        chain_future_map = {
            executor.submit(_run_n2_chain_task, *chain_args(chain)): chain
            for chain in missing_chains
        }
        chain_results = (
            (chain_future_map[future], future.result())
            for future in as_completed(chain_future_map)
        )
    for chain_id, packages in chain_results:
        _append_packages(payload, packages)
        completed_chains.add(chain_id)
        payload["completed_n2_chains"] = sorted(completed_chains)
        write_json(checkpoint_path, payload)
        print(
            f"[E1] sample={real.sample_id}: N2 chains {len(completed_chains)}/{chains}",
            flush=True,
        )

    payload["moments"].sort(
        key=lambda row: (str(row["model"]), int(row["draw"]), str(row.get("chain")))
    )
    payload["curves"].sort(
        key=lambda row: (str(row["model"]), int(row["draw"]), float(row["x"]))
    )
    payload["sampler_diagnostics"].sort(
        key=lambda row: (str(row.get("model")), int(row.get("draw", -1)))
    )
    _finalize_statistics(payload, real, x_grid, settings)
    payload["complete"] = True
    write_json(checkpoint_path, payload)
    return payload


def _checkpoint_frames(paths: list[Path], key: str):
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            continue
        if key == "diagnostics":
            rows = payload.get("sampler_diagnostics", []) + payload.get(
                "mixing_diagnostics", []
            )
        else:
            rows = payload.get(key, [])
        if rows:
            yield pd.DataFrame(rows)


def _persist_outputs(run_dir: Path, checkpoint_paths: list[Path]) -> None:
    write_table_batches(
        _checkpoint_frames(checkpoint_paths, "moments"), run_dir / "moments.parquet"
    )
    write_table_batches(
        _checkpoint_frames(checkpoint_paths, "curves"), run_dir / "curves.parquet"
    )
    write_table_batches(
        _checkpoint_frames(checkpoint_paths, "diagnostics"),
        run_dir / "null_diagnostics.csv",
    )
    write_table_batches(
        _checkpoint_frames(checkpoint_paths, "statistics"),
        run_dir / "null_statistics.csv",
    )


def _select_samples(config: dict, settings: dict, root_seed: int):
    names = settings.get("datasets", config.get("classification", {}).get("datasets", []))
    samples, _ = load_hic_from_config(config, names)
    scholp_names = settings.get("scholp_datasets", [])
    if scholp_names:
        tables, _ = load_scholp_from_config(config, scholp_names)
        for name, table in tables.items():
            lower = name.lower()
            if "congress" in lower:
                candidates = calendar_window_samples(
                    table, "year", int(settings.get("congress_min_unique_edges", 100))
                )
                limit = int(settings.get("congress_initial_num_windows", 8))
                if len(candidates) > limit:
                    chosen = np.linspace(0, len(candidates) - 1, limit).round().astype(int)
                    candidates = [candidates[index] for index in sorted(set(chosen))]
                samples.extend(candidates)
            else:
                samples.append(event_table_to_sample(table, f"{name}:all_events"))
                if "enron" in lower and settings.get("include_enron_quarters", True):
                    samples.extend(calendar_window_samples(table, "quarter"))
    samples, _, _ = clean_samples(samples, policy_from_config(config))
    selected_ids = set(settings.get("sample_ids", []))
    if selected_ids:
        return [sample for sample in samples if sample.sample_id in selected_ids]
    if "max_objects" in settings:
        return samples[: int(settings["max_objects"])]

    selected = []
    for dataset in sorted({sample.dataset for sample in samples}):
        current = [sample for sample in samples if sample.dataset == dataset]
        if current and all(sample.graph_label is not None for sample in current) and len(current) > 20:
            ordered = sorted(current, key=lambda item: (item.M, item.sample_id))
            rng = np.random.default_rng(
                stable_seed(root_seed, dataset, "mechanism_selection")
            )
            for bucket in np.array_split(np.arange(len(ordered)), 4):
                take = min(5, len(bucket))
                selected.extend(
                    ordered[index]
                    for index in sorted(rng.choice(bucket, take, replace=False))
                )
        else:
            selected.extend(current)
    return selected


def _fixed_null_scales(samples) -> dict[str, float]:
    """Apply the predeclared size-only E1 scale protocol."""
    scales = {}
    for dataset in sorted({sample.dataset for sample in samples}):
        current = [sample for sample in samples if sample.dataset == dataset]
        means = {
            sample.sample_id: float(np.mean([len(edge) for edge in sample.hyperedges]))
            for sample in current
        }
        shared = (
            "congress" in dataset.lower()
            or all(sample.graph_label is not None for sample in current)
        )
        if shared:
            common = float(np.median(list(means.values())))
            scales.update({sample.sample_id: common for sample in current})
        else:
            scales.update(means)
    return scales


def run_nulls(config: dict) -> Path:
    run_dir = prepare_run(config, "nulls")
    settings = config.get("null_experiment", {})
    root_seed = int(config.get("seed", 11))
    samples = _select_samples(config, settings, root_seed)
    fixed_scales = _fixed_null_scales(samples)
    x_grid = np.concatenate([
        [0.0], np.logspace(-3, 1, int(settings.get("display_points", 80)))
    ])
    workers = max(1, int(settings.get("parallel_workers", 1)))
    blas_threads = max(1, int(settings.get("blas_threads", 1)))
    resume = bool(settings.get("resume", True))
    print(
        f"[E1] selected_real_samples={len(samples)} workers={workers} "
        f"blas_threads_per_worker={blas_threads}", flush=True,
    )
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    completed_paths: list[Path] = []
    run_started = perf_counter()
    try:
        for index, real in enumerate(samples, start=1):
            sample_started = perf_counter()
            path = _checkpoint_path(run_dir, real)
            identity = _checkpoint_identity(config, real)
            identity["c_scale"] = fixed_scales[real.sample_id]
            checkpoint = _load_checkpoint(path, identity) if resume else None
            if checkpoint is not None and checkpoint.get("complete"):
                completed_paths.append(path)
                print(
                    f"[E1] sample {index}/{len(samples)} {real.sample_id}: "
                    "resumed checkpoint", flush=True,
                )
                continue
            print(
                f"[E1] sample {index}/{len(samples)} {real.sample_id}: "
                f"n={real.num_vertices} M={real.M}", flush=True,
            )
            try:
                _process_real_sample(
                    real, fixed_scales[real.sample_id], config, settings, x_grid,
                    root_seed, path, identity,
                    executor, workers, blas_threads, resume,
                )
                completed_paths.append(path)
                elapsed = perf_counter() - sample_started
                total_elapsed = perf_counter() - run_started
                average = total_elapsed / index
                eta = average * (len(samples) - index)
                print(
                    f"[E1] sample {index}/{len(samples)} complete in {elapsed:.1f}s; "
                    f"ETA≈{eta / 3600:.2f}h; checkpoint saved", flush=True,
                )
            except Exception as exc:
                append_jsonl(run_dir / "failures.jsonl", {
                    "dataset": real.dataset, "sample_id": real.sample_id,
                    "stage": "nulls", "error_type": type(exc).__name__,
                    "reason": str(exc), "recoverable": True,
                })
                print(f"[E1] sample={real.sample_id}: failed: {exc}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    _persist_outputs(run_dir, completed_paths)
    return run_dir
