"""E4: simultaneous school groups, calibrated windows, and activity controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hce.config import resolve_project_path
from hce.core import (
    build_incidence, compute_cumulants, compute_spectrum, compute_walk_moments,
    evaluate_thermodynamics,
)
from hce.data.school import (
    contacts_to_maximal_groups, cross_class_contact_ratio, parse_contacts, window_groups,
)
from hce.experiments.common import prepare_run, spectrum_kwargs
from hce.utils import append_jsonl, stable_seed, write_table


def _window_measure(sample, c, config, descriptor_x, display_x=None):
    H = build_incidence(sample)
    moments = compute_walk_moments(H)
    cumulants = compute_cumulants(moments.W1, moments.W2, moments.W3, moments.M)
    spectrum = compute_spectrum(H, **spectrum_kwargs(config))
    descriptor_x = np.asarray(descriptor_x, dtype=float)
    curve = evaluate_thermodynamics(spectrum, descriptor_x / c)
    display_x = descriptor_x if display_x is None else np.asarray(display_x, dtype=float)
    if np.array_equal(display_x, descriptor_x):
        display_curve = curve
    else:
        display_curve = evaluate_thermodynamics(spectrum, display_x / c)
    sizes = np.asarray([len(edge) for edge in sample.hyperedges], dtype=float)
    row = {
        **sample.metadata, "sample_id": sample.sample_id,
        "n_active": int(np.sum(np.asarray(H.getnnz(axis=1)).ravel() > 0)),
        "M": sample.M, "I_H": int(H.nnz), "mean_size": float(np.mean(sizes)),
        "CV_size": float(np.std(sizes) / np.mean(sizes)),
        "kappa1": cumulants.kappa1, "kappa2": cumulants.kappa2,
        "kappa3": cumulants.kappa3, "O2_per_M": moments.O2 / moments.M,
        "T3_per_M": moments.T3 / moments.M, "lambda_max": spectrum.lambda_max,
    }
    descriptor = np.concatenate([curve.U_struct / c, curve.normalized_entropy])
    for x, U, S in zip(
        display_x, display_curve.U_struct / c, display_curve.normalized_entropy
    ):
        row[f"U_over_c_x_{x:g}"] = U
        row[f"S_norm_x_{x:g}"] = S
    return row, descriptor, spectrum.lambda_max


def run_temporal(config: dict) -> Path:
    run_dir = prepare_run(config, "temporal")
    settings = config.get("temporal", {})
    source = config.get("sources", {}).get("school", {})
    contacts_path = source.get("contacts_path")
    calibration_end = settings.get("calibration_end_timestamp")
    if not contacts_path or calibration_end is None:
        append_jsonl(run_dir / "failures.jsonl", {
            "stage": "temporal", "error_type": "UnverifiedTimeMapping",
            "reason": "contacts_path and verified calibration_end_timestamp are required",
            "recoverable": True,
        })
        return run_dir
    contacts = parse_contacts(resolve_project_path(config, contacts_path))
    groups = contacts_to_maximal_groups(contacts)
    width = int(settings.get("primary_window_seconds", 600))
    samples = window_groups(groups, width, origin=settings.get("window_origin"),
                            merge_duplicate_edges=True)
    calibration_end = int(calibration_end)
    valid_day1 = [sample for sample in samples
                  if sample.M and int(sample.metadata["time_end"]) <= calibration_end]
    if not valid_day1:
        append_jsonl(run_dir / "failures.jsonl", {
            "stage": "temporal", "error_type": "NoCalibrationWindows",
            "reason": "no non-empty window falls in the verified calibration day",
            "recoverable": True,
        })
        return run_dir
    lambdas = [compute_spectrum(build_incidence(sample), **spectrum_kwargs(config)).lambda_max
               for sample in valid_day1]
    c_school = float(np.median(lambdas))
    descriptor_x = np.logspace(-2, 1, 12)
    display_x = np.asarray(settings.get("display_x", [0.1, 1, 10]), dtype=float)
    measured = []
    for sample in samples:
        if not sample.M:
            measured.append((sample, None, None))
            continue
        try:
            row, descriptor, _ = _window_measure(
                sample, c_school, config, descriptor_x, display_x=display_x
            )
            measured.append((sample, row, descriptor))
        except Exception as exc:
            append_jsonl(run_dir / "failures.jsonl", {
                "sample_id": sample.sample_id, "stage": "temporal_window",
                "error_type": type(exc).__name__, "reason": str(exc), "recoverable": True,
            })
            measured.append((sample, None, None))
    train_descriptors = np.vstack([
        descriptor for sample, row, descriptor in measured
        if descriptor is not None and int(sample.metadata["time_end"]) <= calibration_end
    ])
    scaler = StandardScaler().fit(train_descriptors)
    rows = []
    previous = None
    previous_end = None
    for sample, row, descriptor in measured:
        if row is None:
            rows.append({**sample.metadata, "sample_id": sample.sample_id,
                         "status": "empty_or_failed", "c_school": c_school, "D_t": np.nan})
            previous = previous_end = None
            continue
        transformed = scaler.transform(descriptor.reshape(1, -1))[0]
        contiguous = previous_end == sample.metadata["time_start"]
        distance = float(np.linalg.norm(transformed - previous)) if previous is not None and contiguous else np.nan
        rows.append({**row, "status": "ok", "c_school": c_school, "D_t": distance})
        previous, previous_end = transformed, sample.metadata["time_end"]
    write_table(pd.DataFrame(rows), run_dir / "temporal_windows.parquet")

    # Fixed-M activity control, calibrated from day-one nontrivial windows.
    M_day1 = [sample.M for sample in valid_day1 if sample.M >= 2]
    subsample_rows = []
    if M_day1:
        m_star = max(2, int(np.floor(np.percentile(M_day1, 20))))
        repetitions = int(settings.get("fixed_edge_count_repetitions", 30))
        for sample in samples:
            if sample.M < m_star:
                continue
            for repetition in range(repetitions):
                seed = stable_seed(int(config.get("seed", 11)), sample.sample_id,
                                   "fixed_M", repetition)
                rng = np.random.default_rng(seed)
                indices = sorted(rng.choice(sample.M, m_star, replace=False))
                selected = type(sample)(
                    sample_id=f"{sample.sample_id}:fixedM:{repetition}",
                    dataset=sample.dataset, num_vertices=sample.num_vertices,
                    hyperedges=tuple(sample.hyperedges[idx] for idx in indices),
                    original_vertex_ids=sample.original_vertex_ids,
                    metadata=sample.metadata,
                )
                row, _, _ = _window_measure(
                    selected, c_school, config, display_x, display_x=display_x
                )
                subsample_rows.append({**row, "repetition": repetition,
                                       "seed": seed, "m_star": m_star})
        if subsample_rows:
            write_table(pd.DataFrame(subsample_rows), run_dir / "temporal_fixed_M.parquet")
    return run_dir
