"""N1: independent uniform edges with the observed size sequence."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from hce.exceptions import InvalidHypergraphError
from hce.models import HypergraphSample, NullDiagnostics


def sample_size_null(
    sample: HypergraphSample,
    seed: int,
    simple: bool = True,
    max_rejections_per_edge: int = 100_000,
) -> tuple[HypergraphSample, NullDiagnostics]:
    n = sample.num_vertices
    sizes = [len(edge) for edge in sample.hyperedges]
    counts = Counter(sizes)
    for size, count in counts.items():
        if size < 1 or size > n:
            raise InvalidHypergraphError(f"invalid edge size {size} for n={n}")
        if simple and count > math.comb(n, size):
            raise InvalidHypergraphError(
                f"cannot sample {count} distinct size-{size} edges from C({n},{size})"
            )
    rng = np.random.default_rng(seed)
    seen_by_size: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    sampled = []
    attempts = accepted = 0
    for size in sizes:
        rejections = 0
        while True:
            attempts += 1
            edge = tuple(sorted(int(v) for v in rng.choice(n, size=size, replace=False)))
            if simple and edge in seen_by_size[size]:
                rejections += 1
                if rejections >= max_rejections_per_edge:
                    raise RuntimeError(
                        "size-null rejection sampler stalled near a saturated space"
                    )
                continue
            seen_by_size[size].add(edge)
            sampled.append(edge)
            accepted += 1
            break
    null_sample = HypergraphSample(
        sample_id=f"{sample.sample_id}:N1:{seed}",
        dataset=sample.dataset,
        num_vertices=n,
        hyperedges=tuple(sampled),
        graph_label=None,
        original_vertex_ids=tuple(sample.original_vertex_ids),
        metadata={**sample.metadata, "null_model": "size_preserving", "seed": seed},
    )
    diagnostics = NullDiagnostics(
        model="size_preserving",
        attempts=attempts,
        accepted=accepted,
        invariant_ok=sorted(sizes) == sorted(map(len, sampled)),
    )
    return null_sample, diagnostics

