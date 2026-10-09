"""Data identity and preprocessing audit entry point."""

from __future__ import annotations

import pandas as pd

from hce.config import resolve_project_path
from hce.data import parse_scholp
from hce.experiments.common import (
    clean_samples,
    load_hic_from_config,
    policy_from_config,
    prepare_run,
)
from hce.data.scholp import event_table_to_sample
from hce.utils import append_jsonl, sha256_file, write_json, write_table


def run_audit(config: dict) -> Path:
    run_dir = prepare_run(config, "audit")
    audits = []
    statistics = []
    source_manifest = []
    try:
        hic_samples, hic_manifest = load_hic_from_config(config)
        source_manifest.extend(hic_manifest)
        _, audit_table, stat_table = clean_samples(hic_samples, policy_from_config(config))
        audits.append(audit_table)
        statistics.append(stat_table)
    except Exception as exc:
        append_jsonl(run_dir / "failures.jsonl", {
            "stage": "hic_audit", "error_type": type(exc).__name__, "reason": str(exc),
            "recoverable": True,
        })

    for name, entry in config.get("sources", {}).get("scholp", {}).items():
        try:
            directory = resolve_project_path(config, entry["directory"])
            table = parse_scholp(
                directory, name, entry["timestamp_unit"],
                entry.get("timestamp_origin"), entry.get("timezone"),
            )
            sample = event_table_to_sample(table, f"{name}:all_events")
            _, audit_table, stat_table = clean_samples([sample], policy_from_config(config))
            audits.append(audit_table)
            statistics.append(stat_table)
            source_manifest.append({
                "dataset": name, "directory": str(directory),
                "event_count": len(table.events), "status": "available",
                **table.metadata,
            })
        except Exception as exc:
            append_jsonl(run_dir / "failures.jsonl", {
                "dataset": name, "stage": "scholp_audit",
                "error_type": type(exc).__name__, "reason": str(exc), "recoverable": True,
            })
    write_json(run_dir / "sources_manifest.json", source_manifest)
    if audits:
        write_table(pd.concat(audits, ignore_index=True), run_dir / "preprocess_audit.csv")
    if statistics:
        table = pd.concat(statistics, ignore_index=True)
        write_table(table, run_dir / "sample_statistics.parquet")
        write_table(table, run_dir / "sample_statistics.csv")
    return run_dir

