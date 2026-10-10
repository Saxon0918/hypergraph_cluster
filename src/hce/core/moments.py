"""Closed-walk moments, their ordered decompositions, and cumulants."""

from __future__ import annotations

import math

import numpy as np

from hce.core.operators import DenseGramOperator, build_dense_gram_operator
from hce.exceptions import NumericalSpectrumError
from hce.models import CumulantResult, MomentResult


def compute_walk_moments_from_operator(
    prepared: DenseGramOperator,
    order: int = 3,
    decompose: bool = True,
    column_block_size: int = 256,
    cancellation_tolerance: float = 1e-9,
) -> MomentResult:
    """Compute moments from a reusable dense Gram operator."""

    if order not in {1, 2, 3}:
        raise ValueError("order must be 1, 2, or 3")
    H_active = prepared.H_active
    M = prepared.M
    sizes = np.asarray(H_active.sum(axis=0)).ravel().astype(np.float64)
    W0 = float(M)
    W1 = float(np.sum(sizes))
    A2 = float(np.dot(sizes, sizes)) if order >= 2 else math.nan
    A3 = float(np.sum(sizes**3)) if order >= 3 else math.nan
    if order == 1:
        return MomentResult(M, W0, W1, math.nan, math.nan, math.nan, math.nan,
                            math.nan, math.nan, math.nan, "none")

    operator = prepared.matrix
    W2 = float(np.sum(operator * operator))
    O2 = W2 - A2
    if abs(O2) <= cancellation_tolerance * max(1.0, abs(W2)):
        O2 = max(0.0, O2)
    if order == 2:
        return MomentResult(M, W0, W1, W2, math.nan, A2, O2, math.nan,
                            math.nan, math.nan, prepared.side)

    squared = operator @ operator
    W3 = float(np.sum(squared * operator.T))
    if not decompose:
        return MomentResult(M, W0, W1, W2, W3, A2, O2, A3,
                            math.nan, math.nan, prepared.side)

    if prepared.side == "edge":
        b = np.sum(operator * operator, axis=1)
    else:
        b = np.empty(M, dtype=np.float64)
        for start in range(0, M, int(column_block_size)):
            stop = min(M, start + int(column_block_size))
            block = H_active[:, start:stop].toarray()
            q_block = operator @ block
            b[start:stop] = np.sum(block * q_block, axis=0)
    B3 = float(3.0 * np.dot(sizes, b - sizes * sizes))
    T3 = float(W3 - A3 - B3)
    scale = max(1.0, abs(W3), abs(A3) + abs(B3))
    if T3 < 0 and abs(T3) <= cancellation_tolerance * scale:
        T3 = 0.0
    elif T3 < 0:
        raise NumericalSpectrumError(
            f"T3={T3:.6g} is negative beyond cancellation tolerance; "
            "verify with a direct sparse overlap computation"
        )
    return MomentResult(
        M=M,
        W0=W0,
        W1=W1,
        W2=W2,
        W3=W3,
        A2=A2,
        O2=O2,
        A3=A3,
        B3=B3,
        T3=T3,
        side=prepared.side,
        diagnostics={"column_block_size": int(column_block_size)},
    )


def compute_walk_moments(
    H,
    order: int = 3,
    decompose: bool = True,
    memory_budget_gb: float = 8.0,
    column_block_size: int = 256,
    cancellation_tolerance: float = 1e-9,
) -> MomentResult:
    """Compute W0..W3 without a spectral decomposition.

    O2 and T3 use ordered-index sums. T3/6 is the unordered three-edge
    weighted closure; T3 is neither an unweighted triangle count nor a triple
    intersection size.
    """
    prepared = build_dense_gram_operator(
        H,
        side="auto",
        memory_budget_gb=memory_budget_gb,
        workspace_factor=3.0,
        operation="moment computation",
    )
    return compute_walk_moments_from_operator(
        prepared,
        order=order,
        decompose=decompose,
        column_block_size=column_block_size,
        cancellation_tolerance=cancellation_tolerance,
    )


def compute_cumulants(W1: float, W2: float, W3: float, M: int) -> CumulantResult:
    if M <= 0:
        raise ValueError("M must be positive")
    m1 = float(W1) / M
    m2 = float(W2) / M
    m3 = float(W3) / M
    k2 = m2 - m1 * m1
    k3 = m3 - 3.0 * m2 * m1 + 2.0 * m1**3
    tolerance = 1e-12 * max(1.0, abs(m2), abs(m1 * m1))
    if k2 < 0 and abs(k2) <= tolerance:
        k2 = 0.0
    if k2 < 0:
        raise ArithmeticError("computed spectral variance is negative")
    return CumulantResult(M=M, kappa1=m1, kappa2=k2, kappa3=k3)


def approximate_U(cumulants: CumulantResult, betas, order: int = 2) -> np.ndarray:
    """U1 (order=1) or U2 (order=2), named by beta polynomial degree."""
    beta = np.asarray(betas, dtype=np.float64)
    if order == 1:
        return cumulants.kappa1 + beta * cumulants.kappa2
    if order == 2:
        return (
            cumulants.kappa1
            + beta * cumulants.kappa2
            + 0.5 * beta * beta * cumulants.kappa3
        )
    raise ValueError("order must be 1 or 2")


def approximate_entropy(cumulants: CumulantResult, betas, order: int = 3) -> np.ndarray:
    beta = np.asarray(betas, dtype=np.float64)
    result = np.log(cumulants.M) - 0.5 * beta * beta * cumulants.kappa2
    if order == 3:
        result = result - (beta**3) * cumulants.kappa3 / 3.0
    elif order != 2:
        raise ValueError("entropy order must be 2 or 3")
    return result

