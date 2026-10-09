"""Candidate hashing followed by type-preserving exact incidence isomorphism."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import networkx as nx

from hce.models import HypergraphSample


def incidence_graph(sample: HypergraphSample) -> nx.Graph:
    graph = nx.Graph()
    for vertex in range(sample.num_vertices):
        graph.add_node(("v", vertex), kind="vertex")
    for edge_idx, edge in enumerate(sample.hyperedges):
        edge_node = ("e", edge_idx)
        graph.add_node(edge_node, kind="hyperedge")
        graph.add_edges_from((("v", vertex), edge_node) for vertex in edge)
    return graph


def _signature(sample: HypergraphSample) -> tuple:
    degree = [0] * sample.num_vertices
    for edge in sample.hyperedges:
        for vertex in edge:
            degree[vertex] += 1
    graph = incidence_graph(sample)
    wl_hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr="kind", iterations=3)
    return (
        sample.num_vertices,
        len(sample.hyperedges),
        tuple(sorted(degree)),
        tuple(sorted(map(len, sample.hyperedges))),
        wl_hash,
    )


def find_isomorphism_groups(samples: Sequence[HypergraphSample]) -> list[str]:
    """Return stable group ids; WL equality is only a filter, never the proof."""
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        buckets[_signature(sample)].append(index)
    groups: list[str | None] = [None] * len(samples)
    kind_match = nx.algorithms.isomorphism.categorical_node_match("kind", None)
    group_counter = 0
    for signature in sorted(buckets, key=repr):
        representatives: list[tuple[str, nx.Graph]] = []
        for index in buckets[signature]:
            graph = incidence_graph(samples[index])
            matched = None
            for group_id, representative in representatives:
                if nx.is_isomorphic(graph, representative, node_match=kind_match):
                    matched = group_id
                    break
            if matched is None:
                matched = f"iso:{group_counter:06d}"
                group_counter += 1
                representatives.append((matched, graph))
            groups[index] = matched
    return [str(group) for group in groups]

