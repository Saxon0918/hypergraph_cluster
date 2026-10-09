"""Serializable data objects shared by loaders, numerical code, and experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class HypergraphSample:
    sample_id: str
    dataset: str
    num_vertices: int
    hyperedges: Sequence[Tuple[int, ...]]
    graph_label: Optional[int] = None
    group_id: Optional[str] = None
    original_vertex_ids: Sequence[str] = field(default_factory=tuple)
    event_times: Optional[Sequence[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def M(self) -> int:
        return len(self.hyperedges)


@dataclass
class AuditRecord:
    sample_id: str
    dataset: str
    variant: str
    status: str
    n_declared: int
    n_active: int
    raw_edge_count: int
    processed_edge_count: int
    distinct_edge_count: int
    empty_edges_removed: int
    size_filtered_edges: int
    duplicate_edges_merged: int
    within_edge_duplicates_removed: int
    isolated_vertices: int
    incidence_count: int
    duplicate_column_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        meta = row.pop("metadata")
        row.update({f"metadata_{key}": value for key, value in meta.items()})
        return row


@dataclass
class EventTable:
    dataset: str
    vertex_ids: Sequence[int]
    events: Sequence[Tuple[int, ...]]
    times: Sequence[int]
    original_event_indices: Sequence[int]
    timestamp_unit: str
    timestamp_origin: Optional[str]
    timezone: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectrumResult:
    values: np.ndarray
    multiplicities: np.ndarray
    M: int
    n_active: int
    side: str
    lambda_max: float
    tolerance: float
    raw_min_eigenvalue: float
    operator_dimension: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def state_count(self) -> int:
        return int(np.sum(self.multiplicities))

    def expanded_values(self) -> np.ndarray:
        return np.repeat(self.values, self.multiplicities.astype(int))


@dataclass
class MomentResult:
    M: int
    W0: float
    W1: float
    W2: float
    W3: float
    A2: float
    O2: float
    A3: float
    B3: float
    T3: float
    side: str
    ordered_sum_convention: bool = True
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CumulantResult:
    M: int
    kappa1: float
    kappa2: float
    kappa3: float


@dataclass
class ThermoCurves:
    beta: np.ndarray
    logZ: np.ndarray
    U_struct: np.ndarray
    entropy: np.ndarray
    free_energy: np.ndarray
    free_energy_excess: np.ndarray
    spectral_variance: np.ndarray
    heat_capacity: np.ndarray
    normalized_entropy: np.ndarray
    status: Sequence[str]


@dataclass
class InverseResult:
    beta: Optional[float]
    status: str
    residual: Optional[float] = None
    condition_slope: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NullDiagnostics:
    model: str
    attempts: int
    accepted: int
    rejected_same_row_or_column: int = 0
    rejected_existing_incidence: int = 0
    rejected_duplicate_edge: int = 0
    invariant_ok: bool = True
    status: str = "ok"

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0

