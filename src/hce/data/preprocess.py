"""Deterministic hypergraph cleaning with an explicit audit trail."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np

from hce.exceptions import InvalidHypergraphError
from hce.models import AuditRecord, HypergraphSample


@dataclass(frozen=True)
class PreprocessPolicy:
    min_edge_size: int = 1
    max_edge_size: int | None = None
    deduplicate_within_edge: bool = True
    merge_duplicate_edges: bool = True
    preserve_declared_vertices: bool = True
    simplicial_closure: bool = False
    variant_id: str = "simple_primary"

    def validate(self) -> None:
        if self.min_edge_size < 1:
            raise ValueError("min_edge_size must be at least one")
        if self.max_edge_size is not None and self.max_edge_size < self.min_edge_size:
            raise ValueError("max_edge_size must be >= min_edge_size")
        if self.simplicial_closure:
            raise ValueError("simplicial closure is outside the core model")


def preprocess(
    sample: HypergraphSample, policy: PreprocessPolicy = PreprocessPolicy()
) -> tuple[HypergraphSample, AuditRecord]:
    policy.validate()
    if sample.num_vertices < 0:
        raise InvalidHypergraphError("num_vertices must be non-negative")
    empty = filtered = duplicate_memberships = 0
    cleaned: List[Tuple[int, ...]] = []
    raw_to_clean: List[int | None] = []
    first_index: Dict[Tuple[int, ...], int] = {}
    merged = 0
    for raw_edge in sample.hyperedges:
        vertices = [int(v) for v in raw_edge]
        if any(v < 0 or v >= sample.num_vertices for v in vertices):
            raise InvalidHypergraphError(
                f"sample {sample.sample_id} has a vertex outside declared range"
            )
        if policy.deduplicate_within_edge:
            unique = sorted(set(vertices))
            duplicate_memberships += len(vertices) - len(unique)
        else:
            if len(vertices) != len(set(vertices)):
                raise InvalidHypergraphError("binary incidence cannot retain duplicate membership")
            unique = sorted(vertices)
        edge = tuple(unique)
        if not edge:
            empty += 1
            raw_to_clean.append(None)
            continue
        if len(edge) < policy.min_edge_size or (
            policy.max_edge_size is not None and len(edge) > policy.max_edge_size
        ):
            filtered += 1
            raw_to_clean.append(None)
            continue
        if policy.merge_duplicate_edges and edge in first_index:
            merged += 1
            raw_to_clean.append(first_index[edge])
            continue
        clean_idx = len(cleaned)
        cleaned.append(edge)
        first_index.setdefault(edge, clean_idx)
        raw_to_clean.append(clean_idx)

    active = set(v for edge in cleaned for v in edge)
    n_declared = sample.num_vertices if policy.preserve_declared_vertices else len(active)
    if not policy.preserve_declared_vertices and active:
        remap = {old: new for new, old in enumerate(sorted(active))}
        cleaned = [tuple(remap[v] for v in edge) for edge in cleaned]
        original_vertex_ids = tuple(str(v) for v in sorted(active))
    else:
        original_vertex_ids = tuple(sample.original_vertex_ids)
    distinct_count = len(set(cleaned))
    processed = HypergraphSample(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        num_vertices=n_declared,
        hyperedges=tuple(cleaned),
        graph_label=sample.graph_label,
        group_id=sample.group_id,
        original_vertex_ids=original_vertex_ids,
        event_times=None if sample.event_times is None else tuple(sample.event_times),
        metadata={
            **sample.metadata,
            "variant_id": policy.variant_id,
            "raw_to_processed_edge": raw_to_clean,
        },
    )
    raw_count = len(sample.hyperedges)
    audit = AuditRecord(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        variant=policy.variant_id,
        status="ok" if cleaned else "invalid_M0",
        n_declared=n_declared,
        n_active=len(active),
        raw_edge_count=raw_count,
        processed_edge_count=len(cleaned),
        distinct_edge_count=distinct_count,
        empty_edges_removed=empty,
        size_filtered_edges=filtered,
        duplicate_edges_merged=merged,
        within_edge_duplicates_removed=duplicate_memberships,
        isolated_vertices=max(0, n_declared - len(active)),
        incidence_count=sum(map(len, cleaned)),
        duplicate_column_ratio=(raw_count - len(set(tuple(sorted(set(e))) for e in sample.hyperedges))) / raw_count
        if raw_count
        else 0.0,
        metadata={
            "min_edge_size": policy.min_edge_size,
            "max_edge_size": policy.max_edge_size,
            "merge_duplicate_edges": policy.merge_duplicate_edges,
        },
    )
    return processed, audit


def sample_statistics(sample: HypergraphSample) -> dict:
    sizes = np.asarray([len(e) for e in sample.hyperedges], dtype=np.float64)
    degree = np.zeros(sample.num_vertices, dtype=np.int64)
    for edge in sample.hyperedges:
        degree[list(edge)] += 1

    # Count connected components of the incidence bipartite graph, including
    # declared isolated vertices as singleton components. A union-find avoids
    # materializing a potentially large projected graph.
    total_nodes = sample.num_vertices + len(sample.hyperedges)
    parent = list(range(total_nodes))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for edge_idx, edge in enumerate(sample.hyperedges):
        edge_node = sample.num_vertices + edge_idx
        for vertex in edge:
            union(vertex, edge_node)
    component_count = len({find(node) for node in range(total_nodes)}) if total_nodes else 0

    def summary(values: np.ndarray, prefix: str) -> dict:
        if values.size == 0:
            return {f"{prefix}_{key}": np.nan for key in (
                "min", "mean", "std", "median", "p25", "p75", "p90", "p95", "max"
            )}
        return {
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values, ddof=0)),
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_p25": float(np.percentile(values, 25)),
            f"{prefix}_p75": float(np.percentile(values, 75)),
            f"{prefix}_p90": float(np.percentile(values, 90)),
            f"{prefix}_p95": float(np.percentile(values, 95)),
            f"{prefix}_max": float(np.max(values)),
        }

    row = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "n_declared": sample.num_vertices,
        "n_active": int(np.sum(degree > 0)),
        "M": len(sample.hyperedges),
        "I_H": int(np.sum(sizes)),
        "isolated_vertices": int(np.sum(degree == 0)),
        "isolated_ratio": float(np.mean(degree == 0)) if degree.size else np.nan,
        "fraction_size1": float(np.mean(sizes == 1)) if sizes.size else np.nan,
        "fraction_size2": float(np.mean(sizes == 2)) if sizes.size else np.nan,
        "fraction_size_gt2": float(np.mean(sizes > 2)) if sizes.size else np.nan,
        "size_cv": float(np.std(sizes) / np.mean(sizes)) if sizes.size and np.mean(sizes) else np.nan,
        "graph_label": sample.graph_label,
        "connected_components": component_count,
    }
    row.update(summary(sizes, "size"))
    row.update(summary(degree.astype(np.float64), "degree"))
    return row

