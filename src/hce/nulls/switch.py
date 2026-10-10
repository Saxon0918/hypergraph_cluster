"""N2: binary 2x2 switch chain with fixed vertex degrees and edge sizes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterator

import numpy as np

from hce.models import HypergraphSample, NullDiagnostics


class FixedMarginsChain:
    """A labeled-incidence switch chain advanced by attempted, not accepted, moves."""

    def __init__(self, sample: HypergraphSample, seed: int, simple: bool = True):
        self.sample = sample
        self.simple = simple
        self.rng = np.random.default_rng(seed)
        self.edge_sets = [set(edge) for edge in sample.hyperedges]
        if any(len(edge) != len(raw) for edge, raw in zip(self.edge_sets, sample.hyperedges)):
            raise ValueError("switch chain requires binary hyperedges")
        if simple and len({tuple(sorted(edge)) for edge in self.edge_sets}) != len(self.edge_sets):
            raise ValueError("simple switch chain cannot start from duplicate columns")
        # Each slot belongs to a fixed edge column. Swapping slot values preserves
        # column sizes and the multiset of row endpoints (hence all degrees).
        self.slot_edges = np.asarray(
            [edge_idx for edge_idx, edge in enumerate(self.edge_sets) for _ in edge],
            dtype=np.int64,
        )
        self.slot_vertices = np.asarray(
            [vertex for edge in self.edge_sets for vertex in sorted(edge)], dtype=np.int64
        )
        # Frozenset fingerprints make proposed-column construction and exact
        # duplicate checks O(size) without sorting on every attempted switch.
        self.edge_keys = [frozenset(edge) for edge in self.edge_sets]
        self.fingerprints = set(self.edge_keys)
        self.diag = NullDiagnostics(model="degree_size_preserving", attempts=0, accepted=0)
        self.initial_sizes = np.asarray([len(edge) for edge in self.edge_sets])
        self.initial_degrees = self._degrees()

    def _degrees(self) -> np.ndarray:
        return np.bincount(self.slot_vertices, minlength=self.sample.num_vertices)

    @property
    def incidence_count(self) -> int:
        return len(self.slot_vertices)

    def _attempt_indices(self, left: int, right: int) -> bool:
        self.diag.attempts += 1
        if self.incidence_count < 2:
            self.diag.rejected_same_row_or_column += 1
            return False
        v, u = int(self.slot_vertices[left]), int(self.slot_vertices[right])
        e, f = int(self.slot_edges[left]), int(self.slot_edges[right])
        if v == u or e == f:
            self.diag.rejected_same_row_or_column += 1
            return False
        if u in self.edge_sets[e] or v in self.edge_sets[f]:
            self.diag.rejected_existing_incidence += 1
            return False
        old_e = self.edge_keys[e]
        old_f = self.edge_keys[f]
        new_e = frozenset(u if value == v else value for value in old_e)
        new_f = frozenset(v if value == u else value for value in old_f)
        if self.simple:
            duplicate_e = (
                new_e in self.fingerprints and new_e != old_e and new_e != old_f
            )
            duplicate_f = (
                new_f in self.fingerprints and new_f != old_e and new_f != old_f
            )
            if new_e == new_f or duplicate_e or duplicate_f:
                self.diag.rejected_duplicate_edge += 1
                return False
        self.edge_sets[e].remove(v)
        self.edge_sets[e].add(u)
        self.edge_sets[f].remove(u)
        self.edge_sets[f].add(v)
        self.edge_keys[e] = new_e
        self.edge_keys[f] = new_f
        self.slot_vertices[left], self.slot_vertices[right] = u, v
        if self.simple:
            self.fingerprints.discard(old_e)
            self.fingerprints.discard(old_f)
            self.fingerprints.add(new_e)
            self.fingerprints.add(new_f)
        self.diag.accepted += 1
        return True

    def attempt(self) -> bool:
        if self.incidence_count < 2:
            self.diag.attempts += 1
            self.diag.rejected_same_row_or_column += 1
            return False
        left = int(self.rng.integers(self.incidence_count))
        right_base = int(self.rng.integers(self.incidence_count - 1))
        right = right_base + int(right_base >= left)
        return self._attempt_indices(left, right)

    def advance(self, attempts: int) -> None:
        remaining = int(attempts)
        if self.incidence_count < 2:
            for _ in range(remaining):
                self.attempt()
            return
        # Drawing many uniform ordered pairs at once removes hundreds of
        # millions of tiny ``rng.choice`` allocations in full E1 runs.  Mapping
        # [0, I-2] around ``left`` is exactly uniform over distinct ``right``.
        batch_size = 16_384
        while remaining:
            count = min(batch_size, remaining)
            left = self.rng.integers(self.incidence_count, size=count)
            right = self.rng.integers(self.incidence_count - 1, size=count)
            right += right >= left
            for left_index, right_index in zip(left, right):
                self._attempt_indices(int(left_index), int(right_index))
            remaining -= count

    def snapshot(self, sample_index: int) -> HypergraphSample:
        edges = tuple(tuple(sorted(edge)) for edge in self.edge_keys)
        sizes_ok = np.array_equal(
            np.asarray([len(edge) for edge in edges]), self.initial_sizes
        )
        degrees_ok = np.array_equal(self._degrees(), self.initial_degrees)
        simple_ok = not self.simple or len(set(edges)) == len(edges)
        self.diag.invariant_ok = bool(sizes_ok and degrees_ok and simple_ok)
        if not self.diag.invariant_ok:
            self.diag.status = "invariant_failure"
            raise AssertionError("fixed-margin switch violated an invariant")
        return HypergraphSample(
            sample_id=f"{self.sample.sample_id}:N2:{sample_index:04d}",
            dataset=self.sample.dataset,
            num_vertices=self.sample.num_vertices,
            hyperedges=edges,
            graph_label=None,
            original_vertex_ids=tuple(self.sample.original_vertex_ids),
            metadata={**self.sample.metadata,
                      "null_model": "degree_size_preserving",
                      "chain_sample_index": sample_index},
        )


def run_fixed_margins_chain(
    sample: HypergraphSample,
    seed: int,
    num_samples: int = 100,
    burnin_sweeps: int = 20,
    thinning_attempt_sweeps: int = 5,
    simple: bool = True,
) -> Iterator[tuple[HypergraphSample, NullDiagnostics]]:
    """Yield snapshots after fixed numbers of attempted-incidence sweeps."""
    chain = FixedMarginsChain(sample, seed=seed, simple=simple)
    sweep = max(1, chain.incidence_count)
    chain.advance(int(burnin_sweeps) * sweep)
    for sample_index in range(int(num_samples)):
        if sample_index:
            chain.advance(int(thinning_attempt_sweeps) * sweep)
        snapshot = chain.snapshot(sample_index)
        # Copy diagnostics so later chain updates do not mutate saved rows.
        diagnostic = NullDiagnostics(**asdict(chain.diag))
        yield snapshot, diagnostic

