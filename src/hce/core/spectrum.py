"""Exact Gram spectra, including all zero-mode multiplicity."""

from __future__ import annotations

from typing import Literal

import numpy as np

from hce.core.incidence import active_rows, validate_incidence
from hce.exceptions import MemoryBudgetError, NumericalSpectrumError
from hce.models import SpectrumResult


def _ensure_dense_budget(dimension: int, memory_budget_gb: float) -> None:
    # eigvalsh needs matrix storage plus workspace; use a conservative 4x factor.
    estimated = 4 * 8 * int(dimension) * int(dimension)
    budget = float(memory_budget_gb) * (1024**3)
    if estimated > budget:
        raise MemoryBudgetError(
            f"dense eigendecomposition dimension {dimension} needs an estimated "
            f"{estimated / 1024**3:.2f} GiB, above {memory_budget_gb:.2f} GiB"
        )


def _compress_exact_zeros(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compress only exact equal runs, never bucket nearby positive eigenvalues."""
    if values.size == 0:
        return values, np.empty(0, dtype=np.int64)
    unique, counts = np.unique(values, return_counts=True)
    return unique.astype(np.float64), counts.astype(np.int64)


def compute_spectrum(
    H,
    side: Literal["auto", "edge", "vertex"] = "auto",
    psd_relative_tolerance: float = 1e-10,
    memory_budget_gb: float = 8.0,
    compress_multiplicities: bool = True,
) -> SpectrumResult:
    """Compute the complete eigenvalue multiset of G=H.T@H exactly.

    When the active vertex side is smaller, Q=H@H.T is diagonalized and
    M-n_active additional zero modes are restored. Hence state_count is always M.
    """
    H = validate_incidence(H)
    H_active, _ = active_rows(H)
    n_active, M = H_active.shape
    if side not in {"auto", "edge", "vertex"}:
        raise ValueError("side must be 'auto', 'edge', or 'vertex'")
    chosen = "edge" if side == "auto" and M <= n_active else side
    if side == "auto" and n_active < M:
        chosen = "vertex"
    if chosen == "vertex" and n_active > M:
        # Q would require subtracting states. That is unnecessary and error-prone.
        chosen = "edge"

    dimension = M if chosen == "edge" else n_active
    _ensure_dense_budget(dimension, memory_budget_gb)
    operator_sparse = H_active.T @ H_active if chosen == "edge" else H_active @ H_active.T
    operator = operator_sparse.toarray().astype(np.float64, copy=False)
    operator = (operator + operator.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(operator)
    raw_min = float(eigenvalues[0])
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = float(psd_relative_tolerance) * scale
    if raw_min < -tolerance:
        raise NumericalSpectrumError(
            f"Gram operator has eigenvalue {raw_min:.6g} below PSD tolerance "
            f"{-tolerance:.6g}"
        )
    negative_mask = eigenvalues < 0.0
    clipped_count = int(np.sum(negative_mask))
    eigenvalues[negative_mask] = 0.0
    if chosen == "vertex":
        missing_zero_modes = M - n_active
        if missing_zero_modes < 0:
            raise AssertionError("vertex-side zero-mode accounting became negative")
        if missing_zero_modes:
            eigenvalues = np.concatenate(
                [np.zeros(missing_zero_modes, dtype=np.float64), eigenvalues]
            )
    eigenvalues.sort()
    if eigenvalues.size != M:
        raise AssertionError("complete spectrum must contain exactly M states")

    if compress_multiplicities:
        values, multiplicities = _compress_exact_zeros(eigenvalues)
    else:
        values = eigenvalues
        multiplicities = np.ones(M, dtype=np.int64)
    return SpectrumResult(
        values=values,
        multiplicities=multiplicities,
        M=M,
        n_active=n_active,
        side=chosen,
        lambda_max=float(eigenvalues[-1]),
        tolerance=tolerance,
        raw_min_eigenvalue=raw_min,
        operator_dimension=dimension,
        diagnostics={
            "clipped_negative_eigenvalues": clipped_count,
            "restored_zero_modes": M - dimension if chosen == "vertex" else 0,
            "dense_operator_bytes": int(operator.nbytes),
        },
    )

