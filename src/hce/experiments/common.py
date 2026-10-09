"""Shared source resolution, sample loading, auditing and run initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from hce.config import config_hash, resolve_project_path
from hce.data import PreprocessPolicy, parse_hic, parse_scholp, preprocess
from hce.data.preprocess import sample_statistics
from hce.data.scholp import event_table_to_sample
from hce.models import HypergraphSample
from hce.utils import environment_record, sha256_file, write_json, write_table, write_yaml


def policy_from_config(config: dict, variant: str | None = None) -> PreprocessPolicy:
    values = config.get("preprocess", {})
    return PreprocessPolicy(
        min_edge_size=int(values.get("min_edge_size", 1)),
        max_edge_size=values.get("max_edge_size"),
        deduplicate_within_edge=bool(values.get("deduplicate_within_edge", True)),
        merge_duplicate_edges=bool(values.get("merge_duplicate_edges", True)),
        preserve_declared_vertices=bool(values.get("preserve_declared_vertices", True)),
        simplicial_closure=bool(values.get("simplicial_closure", False)),
        variant_id=variant or values.get("variant_id", "simple_primary"),
    )


def spectrum_kwargs(config: dict) -> dict:
    values = config.get("spectrum", {})
    return {
        "side": values.get("side", "auto"),
        "psd_relative_tolerance": float(values.get("psd_relative_tolerance", 1e-10)),
        "memory_budget_gb": float(values.get("memory_budget_gb", 8)),
    }


def prepare_run(config: dict, command: str) -> Path:
    root = resolve_project_path(config, config.get("paths", {}).get("runs", "runs"))
    run_id = config.get("run_id") or f"{command}-{config_hash(config)[:12]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "resolved_config.yaml",
               {k: v for k, v in config.items() if not k.startswith("_")})
    write_json(run_dir / "environment.json", environment_record())
    return run_dir


def load_hic_from_config(config: dict, names: Sequence[str] | None = None):
    sources = config.get("sources", {}).get("hic", {})
    requested = list(sources) if names is None else list(names)
    samples = []
    manifest = []
    for name in requested:
        entry = sources.get(name)
        if entry is None:
            raise FileNotFoundError(f"no sources.hic.{name} entry")
        path = resolve_project_path(
            config, entry["path"] if isinstance(entry, dict) else entry
        )
        index_base = int(entry.get("index_base", 0)) if isinstance(entry, dict) else 0
        parsed = parse_hic(path, name, index_base=index_base)
        samples.extend(parsed)
        manifest.append({
            "dataset": name, "path": str(path.resolve()), "sha256": sha256_file(path),
            "graph_count": len(parsed), "status": "available",
        })
    return samples, manifest


def clean_samples(samples: Sequence[HypergraphSample], policy: PreprocessPolicy):
    valid, audits, stats = [], [], []
    for sample in samples:
        cleaned, audit = preprocess(sample, policy)
        audits.append(audit.to_dict())
        stats.append(sample_statistics(cleaned))
        if cleaned.M:
            valid.append(cleaned)
    return valid, pd.DataFrame(audits), pd.DataFrame(stats)


def load_scholp_from_config(config: dict, names: Sequence[str] | None = None):
    sources = config.get("sources", {}).get("scholp", {})
    requested = list(sources) if names is None else list(names)
    tables, manifest = {}, []
    for name in requested:
        entry = sources.get(name)
        if entry is None:
            raise FileNotFoundError(f"no sources.scholp.{name} entry")
        directory = resolve_project_path(config, entry["directory"])
        table = parse_scholp(
            directory, name, entry["timestamp_unit"],
            entry.get("timestamp_origin"), entry.get("timezone"),
        )
        tables[name] = table
        manifest.append({"dataset": name, "directory": str(directory),
                         "event_count": len(table.events), "status": "available",
                         **table.metadata})
    return tables, manifest

