"""E2: structural graph classification with nested scale fitting."""

from __future__ import annotations

from pathlib import Path
import json
import re
from time import perf_counter

import pandas as pd

from hce.config import config_hash, resolve_project_path
from hce.evaluation import NestedCVConfig, find_isomorphism_groups, run_nested_cv
from hce.experiments.common import (
    clean_samples, load_hic_from_config, policy_from_config, prepare_run, spectrum_kwargs,
)
from hce.features import build_graph_cache, netlsd_features
from hce.baselines import run_external_adapter
from hce.utils import append_jsonl, write_json, write_table


METHOD_MAP = {
    "B0": "dummy", "B1": "basic", "B2": "cumulants", "B3": "low_order",
    "B4": "size_thermo", "B5": "gram_thermo", "B6": "basic_gram",
    "B7-edge": "netlsd_edge", "B7-vertex": "netlsd_vertex",
    "U-only": "gram_U", "S-only": "gram_S",
}

EXTERNAL_METHODS = {
    "B8": "HIC-hypergraph-subtree",
    "B9": "HIC-hypergraph-hyedge",
    "B10": "MoCHy-hmotif",
    "B11": "RethinkHGK",
}

CHECKPOINT_VERSION = 1


def _checkpoint_paths(run_dir: Path, dataset: str, requested: str) -> dict[str, Path]:
    slug = lambda value: re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    stem = run_dir / "checkpoints" / f"{slug(dataset)}--{slug(requested)}"
    return {
        "predictions": Path(f"{stem}.predictions.parquet"),
        "metrics": Path(f"{stem}.metrics.csv"),
        "splits": Path(f"{stem}.splits.json"),
        "complete": Path(f"{stem}.complete.json"),
    }


def _load_checkpoint(paths: dict[str, Path], identity: dict):
    required = tuple(paths.values())
    if not all(path.exists() for path in required):
        return None
    marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    if any(marker.get(key) != value for key, value in identity.items()):
        return None
    predictions = pd.read_parquet(paths["predictions"])
    metrics = pd.read_csv(paths["metrics"])
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    return predictions, metrics, splits, marker


def _save_checkpoint(
    paths: dict[str, Path],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    splits: list[dict],
    marker: dict,
) -> None:
    # The completion marker is written last.  A process interrupted before it
    # appears will recompute this method instead of trusting partial files.
    write_table(predictions, paths["predictions"])
    write_table(metrics, paths["metrics"])
    write_json(paths["splits"], splits)
    write_json(paths["complete"], marker)


def _persist_aggregate_outputs(
    run_dir: Path,
    predictions: list[pd.DataFrame],
    metrics: list[pd.DataFrame],
    splits: list[dict],
    baseline_status: list[dict],
    runtime_rows: list[dict],
) -> None:
    if predictions:
        write_table(pd.concat(predictions, ignore_index=True),
                    run_dir / "cv_predictions.parquet")
    if metrics:
        write_table(pd.concat(metrics, ignore_index=True), run_dir / "cv_metrics.csv")
    write_json(run_dir / "split_manifest.json", splits)
    if baseline_status:
        write_table(pd.DataFrame(baseline_status), run_dir / "baseline_status.csv")
    if runtime_rows:
        write_table(pd.DataFrame(runtime_rows), run_dir / "classification_runtime.csv")


def run_classification(config: dict) -> Path:
    run_dir = prepare_run(config, "classification")
    settings = config.get("classification", {})
    names = settings.get("datasets", ["IMDB_dir_genre", "stream_player"])
    samples, manifest = load_hic_from_config(config, names)
    samples, audits, statistics = clean_samples(samples, policy_from_config(config))
    write_json(run_dir / "sources_manifest.json", manifest)
    write_table(audits, run_dir / "preprocess_audit.csv")
    write_table(statistics, run_dir / "sample_statistics.parquet")
    methods = settings.get("methods", list(METHOD_MAP))
    run_config_hash = config_hash(config)
    classifier = settings.get("classifier", "linear_svc")
    if classifier != "linear_svc":
        raise ValueError("classification.classifier must be 'linear_svc'")
    linear_svc_loss = settings.get("linear_svc_loss", "squared_hinge")
    if linear_svc_loss not in {"hinge", "squared_hinge"}:
        raise ValueError("classification.linear_svc_loss is invalid")
    cv_config = NestedCVConfig(
        outer_folds=int(settings.get("outer_folds", 5)),
        inner_folds=int(settings.get("inner_folds", 3)),
        seeds=tuple(int(seed) for seed in settings.get("seeds", [11, 23, 37, 51, 71])),
        C_values=tuple(float(value) for value in settings.get("C", [0.01, 0.1, 1, 10, 100])),
        class_weights=tuple(settings.get("class_weight", [None, "balanced"])),
        linear_svc_max_iter=int(settings.get("linear_svc_max_iter", 20_000)),
        linear_svc_tol=float(settings.get("linear_svc_tol", 1e-4)),
        linear_svc_loss=linear_svc_loss,
    )
    resume = bool(settings.get("resume", True))
    progress_every = max(1, int(settings.get("progress_every_graphs", 250)))
    all_predictions, all_metrics, all_splits = [], [], []
    baseline_status, runtime_rows = [], []
    for dataset in names:
        current = [sample for sample in samples if sample.dataset == dataset]
        source_entry = next(row for row in manifest if row["dataset"] == dataset)
        checkpoint_identity = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "config_hash": run_config_hash,
            "source_sha256": source_entry.get("sha256"),
        }
        try:
            print(f"[E2] dataset={dataset}: building graph caches for {len(current)} graphs",
                  flush=True)
            started = perf_counter()
            caches = []
            for index, sample in enumerate(current, start=1):
                caches.append(build_graph_cache(sample, **spectrum_kwargs(config)))
                if index == 1 or index % progress_every == 0 or index == len(current):
                    print(f"[E2] dataset={dataset}: graph cache {index}/{len(current)}",
                          flush=True)
            runtime_rows.append({
                "dataset": dataset, "method": None, "stage": "graph_cache",
                "elapsed_seconds": perf_counter() - started, "status": "ok",
            })
            print(f"[E2] dataset={dataset}: grouping isomorphic samples", flush=True)
            started = perf_counter()
            groups = find_isomorphism_groups(current)
            runtime_rows.append({
                "dataset": dataset, "method": None, "stage": "isomorphism_grouping",
                "elapsed_seconds": perf_counter() - started, "status": "ok",
            })
            for requested in methods:
                if requested in EXTERNAL_METHODS:
                    entry = settings.get("external_baselines", {}).get(requested)
                    if not entry:
                        baseline_status.append({
                            "dataset": dataset, "baseline": requested,
                            "implementation": EXTERNAL_METHODS[requested], "status": "not_run",
                            "reason": "no frozen author-code checkout/adapter configured",
                        })
                        _persist_aggregate_outputs(
                            run_dir, all_predictions, all_metrics, all_splits,
                            baseline_status, runtime_rows,
                        )
                        continue
                    shared = {}
                    for split in all_splits:
                        if split["dataset"] == dataset:
                            key = (split["repeat"], split["outer_fold"])
                            shared.setdefault(key, {
                                field: value for field, value in split.items() if field != "method"
                            })
                    if not shared:
                        baseline_status.append({
                            "dataset": dataset, "baseline": requested,
                            "implementation": EXTERNAL_METHODS[requested], "status": "not_run",
                            "reason": "external baseline requires at least one local method to freeze splits",
                        })
                        _persist_aggregate_outputs(
                            run_dir, all_predictions, all_metrics, all_splits,
                            baseline_status, runtime_rows,
                        )
                        continue
                    protocol = {
                        "baseline": requested, "implementation": EXTERNAL_METHODS[requested],
                        "samples": [{
                            "sample_id": sample.sample_id,
                            "num_vertices": sample.num_vertices,
                            "hyperedges": [list(edge) for edge in sample.hyperedges],
                            "graph_label": sample.graph_label,
                            "group_id": group,
                        } for sample, group in zip(current, groups)],
                        "split_manifest": list(shared.values()),
                        "C": list(cv_config.C_values),
                    }
                    result = run_external_adapter(
                        requested, resolve_project_path(config, entry["checkout"]),
                        entry["command"], protocol,
                        run_dir / "external" / dataset,
                        expected_commit=entry.get("commit"),
                        timeout_seconds=int(entry.get("timeout_seconds", 86400)),
                    )
                    baseline_status.append({
                        "dataset": dataset, "baseline": requested,
                        "implementation": EXTERNAL_METHODS[requested],
                        "status": result.status,
                        "reason": result.diagnostics.get("reason"),
                        "commit": result.commit,
                    })
                    if result.status == "ok" and result.output_path:
                        payload = json.loads(Path(result.output_path).read_text(encoding="utf-8"))
                        predictions = pd.DataFrame(payload["predictions"])
                        metrics = pd.DataFrame(payload["fold_metrics"])
                        predictions.insert(0, "dataset", dataset)
                        metrics.insert(0, "dataset", dataset)
                        all_predictions.append(predictions)
                        all_metrics.append(metrics)
                    _persist_aggregate_outputs(
                        run_dir, all_predictions, all_metrics, all_splits,
                        baseline_status, runtime_rows,
                    )
                    continue
                method = METHOD_MAP.get(requested, requested)
                checkpoint_paths = _checkpoint_paths(run_dir, dataset, requested)
                checkpoint = (
                    _load_checkpoint(checkpoint_paths, checkpoint_identity)
                    if resume else None
                )
                if checkpoint is not None:
                    predictions, metrics, splits, marker = checkpoint
                    all_predictions.append(predictions)
                    all_metrics.append(metrics)
                    all_splits.extend(splits)
                    runtime_rows.append({
                        "dataset": dataset, "method": method, "stage": "nested_cv",
                        "elapsed_seconds": marker.get("elapsed_seconds"),
                        "status": "resumed",
                    })
                    print(f"[E2] dataset={dataset} method={requested}: resumed checkpoint",
                          flush=True)
                    _persist_aggregate_outputs(
                        run_dir, all_predictions, all_metrics, all_splits,
                        baseline_status, runtime_rows,
                    )
                    continue

                if method in {"netlsd_edge", "netlsd_vertex"}:
                    perspective = "edge" if method == "netlsd_edge" else "vertex"
                    print(
                        f"[E2] dataset={dataset} method={requested}: "
                        f"precomputing NetLSD-{perspective}", flush=True,
                    )
                    precompute_started = perf_counter()
                    for index, cache in enumerate(caches, start=1):
                        netlsd_features(cache, perspective=perspective)
                        if index == 1 or index % progress_every == 0 or index == len(caches):
                            print(
                                f"[E2] dataset={dataset} method={requested}: "
                                f"NetLSD {index}/{len(caches)}", flush=True,
                            )
                    runtime_rows.append({
                        "dataset": dataset, "method": method,
                        "stage": "feature_precompute",
                        "elapsed_seconds": perf_counter() - precompute_started,
                        "status": "ok",
                    })

                def progress(payload, *, _dataset=dataset, _requested=requested):
                    if payload["event"] == "outer_complete":
                        print(
                            f"[E2] dataset={_dataset} method={_requested} "
                            f"repeat={payload['repeat'] + 1}/{payload['repeats']} "
                            f"fold={payload['outer_fold'] + 1}/{payload['outer_folds']} "
                            f"f1={payload['f1_macro']:.4f} C={payload['best_C']:g}",
                            flush=True,
                        )

                print(f"[E2] dataset={dataset} method={requested}: nested CV start",
                      flush=True)
                started = perf_counter()
                try:
                    predictions, metrics, splits = run_nested_cv(
                        caches, method, cv_config, groups=groups, progress=progress,
                    )
                except Exception as exc:
                    append_jsonl(run_dir / "failures.jsonl", {
                        "dataset": dataset, "method": requested,
                        "stage": "classification_method",
                        "error_type": type(exc).__name__, "reason": str(exc),
                        "recoverable": True,
                    })
                    runtime_rows.append({
                        "dataset": dataset, "method": method, "stage": "nested_cv",
                        "elapsed_seconds": perf_counter() - started, "status": "failed",
                    })
                    _persist_aggregate_outputs(
                        run_dir, all_predictions, all_metrics, all_splits,
                        baseline_status, runtime_rows,
                    )
                    print(f"[E2] dataset={dataset} method={requested}: failed: {exc}",
                          flush=True)
                    continue
                elapsed = perf_counter() - started
                predictions.insert(0, "dataset", dataset)
                metrics.insert(0, "dataset", dataset)
                enriched_splits = [
                    {"dataset": dataset, "method": method, **split} for split in splits
                ]
                marker = {
                    **checkpoint_identity,
                    "dataset": dataset, "requested_method": requested,
                    "method": method, "elapsed_seconds": elapsed,
                    "prediction_rows": len(predictions), "metric_rows": len(metrics),
                    "status": "complete",
                }
                _save_checkpoint(
                    checkpoint_paths, predictions, metrics, enriched_splits, marker
                )
                all_predictions.append(predictions)
                all_metrics.append(metrics)
                all_splits.extend(enriched_splits)
                runtime_rows.append({
                    "dataset": dataset, "method": method, "stage": "nested_cv",
                    "elapsed_seconds": elapsed, "status": "ok",
                })
                _persist_aggregate_outputs(
                    run_dir, all_predictions, all_metrics, all_splits,
                    baseline_status, runtime_rows,
                )
                print(
                    f"[E2] dataset={dataset} method={requested}: complete in "
                    f"{elapsed:.1f}s; checkpoint saved", flush=True,
                )
        except Exception as exc:
            append_jsonl(run_dir / "failures.jsonl", {
                "dataset": dataset, "stage": "classification",
                "error_type": type(exc).__name__, "reason": str(exc), "recoverable": True,
            })
    _persist_aggregate_outputs(
        run_dir, all_predictions, all_metrics, all_splits,
        baseline_status, runtime_rows,
    )
    return run_dir
