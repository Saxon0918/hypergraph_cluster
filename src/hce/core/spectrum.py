"""Exact Gram spectra, including all zero-mode multiplicity."""

from __future__ import annotations

from typing import Literal

import numpy as np

from hce.core.operators import DenseGramOperator, build_dense_gram_operator
from hce.exceptions import NumericalSpectrumError
from hce.models import SpectrumResult


def _compress_exact_zeros(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compress only exact equal runs, never bucket nearby positive eigenvalues."""
    if values.size == 0:
        return values, np.empty(0, dtype=np.int64)
    unique, counts = np.unique(values, return_counts=True)
    return unique.astype(np.float64), counts.astype(np.int64)


def compute_spectrum_from_operator(
    prepared: DenseGramOperator,
    psd_relative_tolerance: float = 1e-10,
    compress_multiplicities: bool = True,
) -> SpectrumResult:
    """Diagonalize a previously materialized Gram operator."""

    eigenvalues = np.linalg.eigvalsh(prepared.matrix)
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
    if prepared.side == "vertex":
        missing_zero_modes = prepared.M - prepared.n_active
        if missing_zero_modes < 0:
            raise AssertionError("vertex-side zero-mode accounting became negative")
        if missing_zero_modes:
            eigenvalues = np.concatenate(
                [np.zeros(missing_zero_modes, dtype=np.float64), eigenvalues]
            )
    eigenvalues.sort()
    if eigenvalues.size != prepared.M:
        raise AssertionError("complete spectrum must contain exactly M states")

    if compress_multiplicities:
        values, multiplicities = _compress_exact_zeros(eigenvalues)
    else:
        values = eigenvalues
        multiplicities = np.ones(prepared.M, dtype=np.int64)
    return SpectrumResult(
        values=values,
        multiplicities=multiplicities,
        M=prepared.M,
        n_active=prepared.n_active,
        side=prepared.side,
        lambda_max=float(eigenvalues[-1]),
        tolerance=tolerance,
        raw_min_eigenvalue=raw_min,
        operator_dimension=prepared.dimension,
        diagnostics={
            "clipped_negative_eigenvalues": clipped_count,
            "restored_zero_modes": (
                prepared.M - prepared.dimension if prepared.side == "vertex" else 0
            ),
            "dense_operator_bytes": int(prepared.matrix.nbytes),
        },
    )


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
    prepared = build_dense_gram_operator(
        H,
        side=side,
        memory_budget_gb=memory_budget_gb,
        workspace_factor=4.0,
        operation="dense eigendecomposition",
    )
    return compute_spectrum_from_operator(
        prepared,
        psd_relative_tolerance=psd_relative_tolerance,
        compress_multiplicities=compress_multiplicities,
    )

