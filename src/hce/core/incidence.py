"""Binary incidence construction with strict shape and index validation."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy import sparse

from hce.exceptions import InvalidHypergraphError
from hce.models import HypergraphSample


def build_incidence(sample: HypergraphSample, dtype=np.float64) -> sparse.csr_matrix:
    """Build an n x M binary CSR matrix; duplicate membership is rejected.

    Preprocessing normally removes within-edge duplicates. Rejecting them here keeps
    the low-level routine safe when called directly and prevents count-valued H.
    """
    n = int(sample.num_vertices)
    if n < 0:
        raise InvalidHypergraphError("num_vertices must be non-negative")
    rows = []
    cols = []
    for edge_idx, raw_edge in enumerate(sample.hyperedges):
        edge = tuple(int(v) for v in raw_edge)
        if not edge:
            raise InvalidHypergraphError(f"empty hyperedge at column {edge_idx}")
        if len(set(edge)) != len(edge):
            raise InvalidHypergraphError(
                f"duplicate vertex membership in hyperedge {edge_idx}"
            )
        for vertex in edge:
            if vertex < 0 or vertex >= n:
                raise InvalidHypergraphError(
                    f"vertex {vertex} in edge {edge_idx} outside [0, {n})"
                )
            rows.append(vertex)
            cols.append(edge_idx)
    data = np.ones(len(rows), dtype=dtype)
    H = sparse.coo_matrix(
        (data, (np.asarray(rows), np.asarray(cols))),
        shape=(n, len(sample.hyperedges)),
        dtype=dtype,
    ).tocsr()
    H.sum_duplicates()
    if H.nnz and not np.all(H.data == 1):
        raise InvalidHypergraphError("incidence matrix is not binary")
    return H


def validate_incidence(H: sparse.spmatrix) -> sparse.csr_matrix:
    """Return a float64 CSR copy after validating finite binary entries and M>0."""
    if not sparse.issparse(H):
        H = sparse.csr_matrix(np.asarray(H))
    H = H.tocsr().astype(np.float64, copy=True)
    H.sum_duplicates()
    H.eliminate_zeros()
    if H.ndim != 2:
        raise InvalidHypergraphError("H must be two-dimensional")
    if H.shape[1] == 0:
        raise InvalidHypergraphError("M=0 has no defined spectrum or thermodynamics")
    if not np.all(np.isfinite(H.data)):
        raise InvalidHypergraphError("H contains non-finite values")
    if not np.all(H.data == 1.0):
        raise InvalidHypergraphError("H must be a binary incidence matrix")
    return H


def active_rows(H: sparse.csr_matrix) -> Tuple[sparse.csr_matrix, np.ndarray]:
    mask = np.asarray(H.getnnz(axis=1)).ravel() > 0
    return H[mask], mask

