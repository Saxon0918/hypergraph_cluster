"""E2: structural graph classification with nested scale fitting."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import json

import pandas as pd

from hce.config import resolve_project_path
from hce.evaluation import NestedCVConfig, find_isomorphism_groups, run_nested_cv
from hce.experiments.common import (
    clean_samples, load_hic_from_config, policy_from_config, prepare_run, spectrum_kwargs,
)
from hce.features import build_graph_cache
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
    cv_config = NestedCVConfig(
        outer_folds=int(settings.get("outer_folds", 5)),
        inner_folds=int(settings.get("inner_folds", 3)),
        seeds=tuple(int(seed) for seed in settings.get("seeds", [11, 23, 37, 51, 71])),
        C_values=tuple(float(value) for value in settings.get("C", [0.01, 0.1, 1, 10, 100])),
        class_weights=tuple(settings.get("class_weight", [None, "balanced"])),
    )
    all_predictions, all_metrics, all_splits, baseline_status = [], [], [], []
    for dataset in names:
        current = [sample for sample in samples if sample.dataset == dataset]
        try:
            caches = [build_graph_cache(sample, **spectrum_kwargs(config)) for sample in current]
            groups = find_isomorphism_groups(current)
            for requested in methods:
                if requested in EXTERNAL_METHODS:
                    entry = settings.get("external_baselines", {}).get(requested)
                    if not entry:
                        baseline_status.append({
                            "dataset": dataset, "baseline": requested,
                            "implementation": EXTERNAL_METHODS[requested], "status": "not_run",
                            "reason": "no frozen author-code checkout/adapter configured",
                        })
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
                    continue
                method = METHOD_MAP.get(requested, requested)
                predictions, metrics, splits = run_nested_cv(
                    caches, method, cv_config, groups=groups,
                )
                predictions.insert(0, "dataset", dataset)
                metrics.insert(0, "dataset", dataset)
                all_predictions.append(predictions)
                all_metrics.append(metrics)
                for split in splits:
                    all_splits.append({"dataset": dataset, "method": method, **split})
        except Exception as exc:
            append_jsonl(run_dir / "failures.jsonl", {
                "dataset": dataset, "stage": "classification",
                "error_type": type(exc).__name__, "reason": str(exc), "recoverable": True,
            })
    if all_predictions:
        write_table(pd.concat(all_predictions, ignore_index=True),
                    run_dir / "cv_predictions.parquet")
    if all_metrics:
        write_table(pd.concat(all_metrics, ignore_index=True), run_dir / "cv_metrics.csv")
    write_json(run_dir / "split_manifest.json", all_splits)
    if baseline_status:
        write_table(pd.DataFrame(baseline_status), run_dir / "baseline_status.csv")
    return run_dir
