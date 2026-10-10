"""Reusable dense Gram operators for exact spectra and walk moments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse

from hce.core.incidence import active_rows, validate_incidence
from hce.exceptions import MemoryBudgetError


OperatorSide = Literal["auto", "edge", "vertex"]


@dataclass
class DenseGramOperator:
    """A materialized small- or edge-side Gram operator.

    ``H_active`` is retained because the vertex-side third-order decomposition
    needs column blocks of the incidence matrix.  Building this object once
    allows moments and the exact spectrum to share sparse multiplication and
    dense conversion.
    """

    H_active: sparse.csr_matrix
    matrix: np.ndarray
    side: str
    n_active: int
    M: int

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])


def choose_operator_side(
    n_active: int,
    M: int,
    side: OperatorSide = "auto",
) -> str:
    if side not in {"auto", "edge", "vertex"}:
        raise ValueError("side must be 'auto', 'edge', or 'vertex'")
    if side == "auto":
        return "edge" if M <= n_active else "vertex"
    if side == "vertex" and n_active > M:
        # The vertex operator would contain modes that are not states of G.
        return "edge"
    return side


def ensure_dense_budget(
    dimension: int,
    memory_budget_gb: float,
    workspace_factor: float,
    operation: str,
) -> None:
    estimated = float(workspace_factor) * 8 * int(dimension) * int(dimension)
    budget = float(memory_budget_gb) * (1024**3)
    if estimated > budget:
        raise MemoryBudgetError(
            f"{operation} dimension {dimension} needs an estimated "
            f"{estimated / 1024**3:.2f} GiB, above {memory_budget_gb:.2f} GiB"
        )


def build_dense_gram_operator(
    H,
    side: OperatorSide = "auto",
    memory_budget_gb: float = 8.0,
    workspace_factor: float = 4.0,
    operation: str = "dense Gram computation",
) -> DenseGramOperator:
    """Validate ``H`` and materialize exactly one dense Gram operator."""

    validated = validate_incidence(H)
    H_active, _ = active_rows(validated)
    n_active, M = H_active.shape
    chosen = choose_operator_side(n_active, M, side)
    dimension = M if chosen == "edge" else n_active
    ensure_dense_budget(
        dimension, memory_budget_gb, workspace_factor, operation
    )
    operator_sparse = (
        H_active.T @ H_active if chosen == "edge" else H_active @ H_active.T
    )
    # Binary incidence products are exactly symmetric in float64 at these
    # dimensions, so avoid the former (A + A.T) temporary allocation.
    matrix = operator_sparse.toarray().astype(np.float64, copy=False)
    return DenseGramOperator(
        H_active=H_active,
        matrix=matrix,
        side=chosen,
        n_active=int(n_active),
        M=int(M),
    )
