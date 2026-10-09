"""E1: real structures versus size- and margin-constrained null ensembles."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from hce.core import (
    build_incidence, compute_cumulants, compute_spectrum, compute_walk_moments,
    evaluate_thermodynamics,
)
from hce.evaluation.statistics import (
    monte_carlo_two_sided_p, normalized_curve_integral, standardized_difference,
)
from hce.experiments.common import (
    clean_samples, load_hic_from_config, load_scholp_from_config, policy_from_config,
    prepare_run, spectrum_kwargs,
)
from hce.data.scholp import calendar_window_samples, event_table_to_sample
from hce.nulls import effective_sample_size, rhat, run_fixed_margins_chain, sample_size_null
from hce.utils import append_jsonl, stable_seed, write_table


def _measure(sample, x_grid, c_scale, config, model, draw, seed):
    H = build_incidence(sample)
    moments = compute_walk_moments(
        H, memory_budget_gb=float(config.get("spectrum", {}).get("memory_budget_gb", 8))
    )
    cumulants = compute_cumulants(moments.W1, moments.W2, moments.W3, moments.M)
    spectrum = compute_spectrum(H, **spectrum_kwargs(config))
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


def run_nulls(config: dict) -> Path:
    run_dir = prepare_run(config, "nulls")
    settings = config.get("null_experiment", {})
    root_seed = int(config.get("seed", 11))
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
        samples = [sample for sample in samples if sample.sample_id in selected_ids]
    elif "max_objects" in settings:
        samples = samples[: int(settings["max_objects"])]
    else:
        # For graph-classification sources, select 20 graphs without looking at
        # labels: sort by M, split into four near-equal bins, sample five per bin.
        selected = []
        for dataset in sorted({sample.dataset for sample in samples}):
            current = [sample for sample in samples if sample.dataset == dataset]
            if current and all(sample.graph_label is not None for sample in current) and len(current) > 20:
                ordered = sorted(current, key=lambda item: (item.M, item.sample_id))
                rng = np.random.default_rng(stable_seed(root_seed, dataset, "mechanism_selection"))
                for bucket in np.array_split(np.arange(len(ordered)), 4):
                    take = min(5, len(bucket))
                    selected.extend(ordered[index] for index in sorted(rng.choice(bucket, take, replace=False)))
            else:
                selected.extend(current)
        samples = selected
    x_grid = np.concatenate([[0.0], np.logspace(-3, 1, int(settings.get("display_points", 80)))])
    B = int(settings.get("samples_per_model", 100))
    moment_rows, curve_rows, diagnostic_rows, statistic_rows = [], [], [], []

    for real in samples:
        try:
            sizes = np.asarray([len(edge) for edge in real.hyperedges], dtype=float)
            c_scale = float(np.mean(sizes))
            real_moment, real_curves = _measure(real, x_grid, c_scale, config, "real", 0, root_seed)
            real_moment.update({"parent_sample_id": real.sample_id, "chain": np.nan})
            for curve_row in real_curves:
                curve_row.update({"parent_sample_id": real.sample_id, "chain": np.nan})
            moment_rows.append(real_moment)
            curve_rows.extend(real_curves)
            for draw in range(B):
                seed = stable_seed(root_seed, real.dataset, real.sample_id, "N1", draw)
                null_sample, diag = sample_size_null(real, seed, simple=True)
                row, curves = _measure(null_sample, x_grid, c_scale, config, "N1", draw, seed)
                row.update({"parent_sample_id": real.sample_id, "chain": np.nan})
                for curve_row in curves:
                    curve_row.update({"parent_sample_id": real.sample_id, "chain": np.nan})
                moment_rows.append(row)
                curve_rows.extend(curves)
                diagnostic_rows.append({"sample_id": real.sample_id, "draw": draw,
                                        "seed": seed, **asdict(diag),
                                        "acceptance_rate": diag.acceptance_rate})
            chains = int(settings.get("chains", 4))
            per_chain = int(np.ceil(B / chains))
            collected = 0
            for chain_id in range(chains):
                seed = stable_seed(root_seed, real.dataset, real.sample_id, "N2", chain_id)
                iterator = run_fixed_margins_chain(
                    real, seed, num_samples=per_chain,
                    burnin_sweeps=int(settings.get("burnin_sweeps", 20)),
                    thinning_attempt_sweeps=int(settings.get("thinning_attempt_sweeps", 5)),
                    simple=True,
                )
                for null_sample, diag in iterator:
                    if collected >= B:
                        break
                    row, curves = _measure(null_sample, x_grid, c_scale, config, "N2", collected, seed)
                    row.update({"parent_sample_id": real.sample_id, "chain": chain_id})
                    for curve_row in curves:
                        curve_row.update({"parent_sample_id": real.sample_id, "chain": chain_id})
                    moment_rows.append(row)
                    curve_rows.extend(curves)
                    diagnostic_rows.append({"sample_id": real.sample_id, "chain": chain_id,
                                            "draw": collected, "seed": seed, **asdict(diag),
                                            "acceptance_rate": diag.acceptance_rate})
                    collected += 1
            # Acceptance and invariant checks alone do not establish mixing.
            # Diagnose several predeclared structural/response traces per chain.
            current_moments = pd.DataFrame([
                row for row in moment_rows
                if row.get("parent_sample_id") == real.sample_id and row["model"] == "N2"
            ])
            current_curves = pd.DataFrame([
                row for row in curve_rows
                if row.get("parent_sample_id") == real.sample_id and row["model"] == "N2"
            ])
            traces = {
                "W2_per_M": {int(chain): group.sort_values("draw").W2.to_numpy() /
                              group.sort_values("draw").M.to_numpy()
                              for chain, group in current_moments.groupby("chain")},
                "kappa3": {int(chain): group.sort_values("draw").kappa3.to_numpy()
                           for chain, group in current_moments.groupby("chain")},
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
            for metric, chain_values in traces.items():
                rhat_value, rhat_status = rhat(list(chain_values.values()))
                for chain_id, values in chain_values.items():
                    ess_value, ess_status = effective_sample_size(values)
                    diagnostic_rows.append({
                        "record_type": "mixing", "sample_id": real.sample_id,
                        "model": "degree_size_preserving", "chain": chain_id,
                        "metric": metric, "trace_length": len(values),
                        "ess": ess_value, "ess_status": ess_status,
                        "rhat": rhat_value, "rhat_status": rhat_status,
                        "status": "diagnostic_only_not_proof_of_global_mixing",
                    })
            curve_table = pd.DataFrame(
                row for row in curve_rows if row["sample_id"] == real.sample_id
                or row["sample_id"].startswith(real.sample_id + ":")
            )
            for model in ("N1", "N2"):
                values = []
                for _, group in curve_table[curve_table.model == model].groupby("draw"):
                    values.append(normalized_curve_integral(group.x, group.S_norm))
                real_group = curve_table[curve_table.model == "real"]
                real_value = normalized_curve_integral(real_group.x, real_group.S_norm)
                z, z_status = standardized_difference(real_value, values)
                statistic_rows.append({
                    "dataset": real.dataset, "sample_id": real.sample_id, "model": model,
                    "statistic": "normalized_entropy_logx_integral",
                    "real_value": real_value, "null_mean": float(np.mean(values)),
                    "null_std": float(np.std(values)), "z": z, "z_status": z_status,
                    "p_monte_carlo_two_sided": monte_carlo_two_sided_p(real_value, values),
                    "B": len(values), "mixing_claim": "not_established_by_sampler_alone",
                })
        except Exception as exc:
            append_jsonl(run_dir / "failures.jsonl", {
                "dataset": real.dataset, "sample_id": real.sample_id, "stage": "nulls",
                "error_type": type(exc).__name__, "reason": str(exc), "recoverable": True,
            })
    if moment_rows:
        write_table(pd.DataFrame(moment_rows), run_dir / "moments.parquet")
    if curve_rows:
        write_table(pd.DataFrame(curve_rows), run_dir / "curves.parquet")
    if diagnostic_rows:
        write_table(pd.DataFrame(diagnostic_rows), run_dir / "null_diagnostics.csv")
    if statistic_rows:
        write_table(pd.DataFrame(statistic_rows), run_dir / "null_statistics.csv")
    return run_dir

