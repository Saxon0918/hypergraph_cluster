"""Strict parser for the graph-level HIC text datasets."""

from __future__ import annotations

from pathlib import Path
from typing import List

from hce.exceptions import DataFormatError
from hce.models import HypergraphSample


def _tokens(line: str, context: str) -> list[str]:
    values = line.strip().split()
    if not values:
        raise DataFormatError(f"empty line while reading {context}")
    return values


def parse_hic(
    path: str | Path,
    dataset_name: str,
    index_base: int = 0,
) -> List[HypergraphSample]:
    """Parse HIC graphs; every token on an edge line is a member index.

    The first edge-line token is deliberately *not* treated as a size prefix.
    index_base defaults to the author loader's zero-based convention and is never
    inferred from observed minima, because vertex zero can be isolated.
    """
    if index_base not in {0, 1}:
        raise ValueError("index_base must be explicitly 0 or 1")
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise DataFormatError("empty HIC file")
    try:
        expected_graphs = int(_tokens(lines[0], "graph count")[0])
    except (ValueError, IndexError) as exc:
        raise DataFormatError("invalid graph count") from exc
    cursor = 1
    samples: List[HypergraphSample] = []
    for graph_idx in range(expected_graphs):
        if cursor >= len(lines):
            raise DataFormatError(f"file ended before graph {graph_idx}")
        header = _tokens(lines[cursor], f"graph {graph_idx} header")
        cursor += 1
        if len(header) != 3:
            raise DataFormatError(
                f"single-label graph {graph_idx} header must contain n M label"
            )
        try:
            n, M, label = map(int, header)
        except ValueError as exc:
            raise DataFormatError(f"non-integer graph {graph_idx} header") from exc
        if n < 0 or M < 0:
            raise DataFormatError("n and M must be non-negative")
        if cursor >= len(lines):
            raise DataFormatError(f"missing vertex-label line for graph {graph_idx}")
        vertex_labels = _tokens(lines[cursor], f"graph {graph_idx} vertex labels")
        cursor += 1
        if len(vertex_labels) != n:
            raise DataFormatError(
                f"graph {graph_idx} declares {n} vertices but has "
                f"{len(vertex_labels)} vertex-label tokens"
            )
        edges = []
        for edge_idx in range(M):
            if cursor >= len(lines):
                raise DataFormatError(
                    f"file ended in graph {graph_idx}, edge {edge_idx}"
                )
            edge_tokens = _tokens(lines[cursor], f"graph {graph_idx} edge {edge_idx}")
            cursor += 1
            try:
                members = tuple(int(value) - index_base for value in edge_tokens)
            except ValueError as exc:
                raise DataFormatError("non-integer hyperedge member") from exc
            if any(v < 0 or v >= n for v in members):
                raise DataFormatError(
                    f"graph {graph_idx} edge {edge_idx} has member outside "
                    f"the declared {index_base}-based range"
                )
            edges.append(members)
        samples.append(
            HypergraphSample(
                sample_id=f"{dataset_name}:{graph_idx:06d}",
                dataset=dataset_name,
                num_vertices=n,
                hyperedges=tuple(edges),
                graph_label=label,
                original_vertex_ids=tuple(str(i + index_base) for i in range(n)),
                metadata={"source_path": str(path), "vertex_labels": vertex_labels},
            )
        )
    trailing = [line for line in lines[cursor:] if line.strip()]
    if trailing:
        raise DataFormatError(f"{len(trailing)} unexpected trailing lines")
    return samples

